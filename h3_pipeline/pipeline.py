from dataclasses import dataclass
from pathlib import Path
import json
import secrets
import time
import uuid

from .client import ComfyClient, ComfyExecutionError
from .config import DEFAULT_PROFILE, RuntimePaths, write_json
from .locking import file_lock
from .media import prepare_audio, prepare_image, prepare_video, verify_output
from .references import FPS, PromptBuilder, ReferenceManager, canvas, frame_count
from .workflow import build_workflow, validate_graph


class GenerationError(RuntimeError):
    def __init__(self, message, report_path):
        super().__init__(message)
        self.report_path = str(report_path)


@dataclass(frozen=True)
class GenerationResult:
    video_path: str
    report_path: str
    workflow_path: str
    seed: int
    reference_mapping: list
    performance_report: dict


def fallback_attempts(megapixels, mode):
    """Bounded retries; do not increase an already smaller requested canvas."""
    attempts = [(megapixels, mode, 0, "requested settings"),
                (megapixels, "match", 1, "OOM cleanup, encoder offload, match references"),
                (megapixels, "match", 2, "stronger dynamic residency headroom")]
    for reduced in (0.5, 0.4):
        if reduced < megapixels:
            attempts.append((reduced, "match", 2, f"reduce canvas to {reduced} MP"))
    return attempts


class MiniMaxH3Pipeline:
    def __init__(self, server_url="http://127.0.0.1:8188", root=None, profile=DEFAULT_PROFILE,
                 username=None, password=None, client=None, progress=print):
        self.paths = RuntimePaths(Path(root).expanduser().resolve()) if root else RuntimePaths.default()
        self.profile = profile
        self.client = client or ComfyClient(server_url, username, password)
        self.progress = progress

    def generate(self, prompt, reference_video=None, character_images=None, clothing_images=None,
                 additional_reference_images=None, reference_audio=None, duration=15.0, megapixels=0.98,
                 seed=None, *, aspect_ratio="16:9", ref_image_size="max", include_video_audio=True,
                 reference_video_start=0.0, scheduler="simple", steps=25, turbo=False,
                 oom_fallback=False, timeout=21600):
        length = frame_count(duration)
        canvas(megapixels, aspect_ratio)
        if ref_image_size not in ("match", "max"):
            raise ValueError("ref_image_size must be match or max")
        if scheduler not in ("simple", "beta", "normal"):
            raise ValueError("scheduler must be simple, beta or normal")
        if not isinstance(turbo, bool):
            raise ValueError("turbo must be a boolean")
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
            raise ValueError("steps must be an integer between 1 and 100")
        if turbo and steps != 4:
            raise ValueError("Turbo requires exactly four steps; disable turbo for base sampling")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if seed is None:
            seed = secrets.randbits(63)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        images = ReferenceManager.images(character_images, clothing_images, additional_reference_images)
        for path in (reference_video, reference_audio):
            if path and not Path(path).expanduser().is_file():
                raise FileNotFoundError(path)
        # Validate user text before uploading any assets.
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        self.paths.create()
        with file_lock(self.paths.root / "locks" / "generation.lock"):
            return self._generate(prompt, images, reference_video, reference_audio, duration, megapixels,
                                  length, seed, aspect_ratio, ref_image_size, include_video_audio,
                                  reference_video_start, scheduler, steps, turbo, oom_fallback, timeout)

    def _generate(self, prompt, images, video, audio, duration, megapixels, length, seed, aspect,
                  mode, include_audio, video_start, scheduler, steps, turbo, oom_fallback, timeout):
        job_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:10]
        job_dir = self.paths.root / "jobs" / job_id
        job_dir.mkdir(parents=True)
        started = time.monotonic()
        report_path = job_dir / "report.json"
        report = {"job_id": job_id, "status": "preparing", "seed": seed,
                  "requested_duration": duration, "actual_duration": length / FPS, "frame_count": length,
                  "profile": self.profile, "requested_megapixels": megapixels, "steps": steps,
                  "turbo": turbo, "scheduler": scheduler, "attempts": []}
        write_json(report_path, report)
        try:
            self.client.check()
            attempts = fallback_attempts(megapixels, mode) if oom_fallback else [(megapixels, mode, 0, "requested settings")]
            for index, (mp, ref_mode, memory_level, reason) in enumerate(attempts):
                width, height = canvas(mp, aspect)
                attempt_id = f"{job_id}-a{index + 1}"
                work = job_dir / f"attempt-{index + 1}"
                work.mkdir()
                prep_start = time.monotonic()
                uploaded_images = []
                for i, image in enumerate(images):
                    prepared = work / f"picture-{i + 1}.png"
                    prepare_image(image.path, prepared, width, height, ref_mode)
                    uploaded_images.append(self.client.upload(prepared, attempt_id))
                video_info = None
                uploaded_video = None
                if video:
                    prepared = work / "reference.mp4"
                    video_info = prepare_video(Path(video).expanduser(), prepared, length / FPS,
                                               width, height, video_start, include_audio)
                    uploaded_video = self.client.upload(prepared, attempt_id)
                uploaded_audio = None
                if audio:
                    prepared = work / "reference-audio.wav"
                    prepare_audio(Path(audio).expanduser(), prepared, length / FPS)
                    uploaded_audio = self.client.upload(prepared, attempt_id)
                paired = bool(video_info and video_info["paired_audio"])
                refs = ReferenceManager.mapping(images, video, paired, audio)
                effective_prompt = PromptBuilder.build(prompt, refs)
                graph = validate_graph(build_workflow(prompt=effective_prompt, images=uploaded_images,
                    video=uploaded_video, audio=uploaded_audio, include_video_audio=paired, width=width, height=height,
                    length=length, seed=seed, job_id=attempt_id, memory_level=memory_level,
                    ref_image_size=ref_mode, profile=self.profile, scheduler=scheduler, steps=steps, turbo=turbo))
                graph["201"]["inputs"]["reference_mapping"] = json.dumps([r.dict() for r in refs])
                workflow_path = work / "workflow-api.json"
                write_json(workflow_path, graph)
                write_json(work / "references.json", [r.dict() for r in refs])
                (work / "prompt.txt").write_text(effective_prompt + "\n")
                attempt = {"attempt": index + 1, "job_id": attempt_id, "reason": reason, "megapixels": mp,
                           "width": width, "height": height, "ref_image_size": ref_mode, "memory_level": memory_level,
                           "preprocessing_seconds": time.monotonic() - prep_start, "reference_video": video_info,
                           "workflow_path": str(workflow_path)}
                report["attempts"].append(attempt)
                report.update(status="running", reference_mapping=[r.dict() for r in refs])
                write_json(report_path, report)
                if self.progress:
                    self.progress(f"Attempt {index + 1}: {width}x{height}, {length} frames, seed {seed}; {reason}")
                attempt_start = time.monotonic()
                try:
                    history = self.client.execute(graph, timeout, self.progress)
                except ComfyExecutionError as exc:
                    attempt.update(status="oom" if exc.is_oom else "failed", error=exc.details,
                                   wall_seconds=time.monotonic() - attempt_start)
                    # Custom-node errors flush server metrics to disk; remote clients also get the details above.
                    telemetry = self.paths.output / "h3" / attempt_id / "telemetry.json"
                    if telemetry.is_file():
                        attempt["server"] = json.loads(telemetry.read_text())
                    write_json(report_path, report)
                    if not exc.is_oom:
                        raise
                    continue
                outputs = history.get("outputs", {}).get("92", {})
                candidates = [item for value in outputs.values() if isinstance(value, list)
                              for item in value if isinstance(item, dict) and item.get("filename", "").endswith(".mp4")]
                if not candidates:
                    raise RuntimeError("ComfyUI finished without an MP4 output")
                video_path = self.client.download(candidates[0], job_dir / "generated.mp4")
                media = verify_output(video_path, length)
                attempt.update(status="completed", wall_seconds=time.monotonic() - attempt_start,
                               server=outputs.get("h3_report", [None])[0])
                total = time.monotonic() - started
                report.update(status="completed", video_path=video_path, total_wall_seconds=total,
                              compute_seconds_per_output_second=total / (length / FPS),
                              effective_megapixels=mp, effective_width=width, effective_height=height,
                              fallback_used=index > 0, media=media)
                write_json(report_path, report)
                return GenerationResult(video_path, str(report_path), str(workflow_path), seed,
                                        [r.dict() for r in refs], report)
            if not oom_fallback:
                raise RuntimeError("CUDA OOM at requested quality; automatic quality reductions are disabled")
            raise RuntimeError("CUDA OOM fallback exhausted, including available 0.5/0.4 MP reductions; duration was preserved")
        except Exception as exc:
            report.update(status="failed", error=str(exc), total_wall_seconds=time.monotonic() - started)
            write_json(report_path, report)
            raise GenerationError(f"{exc} (report: {report_path})", report_path) from exc
