"""Optional single-outfit Wan-Animate replacement comparison.

The external official environment owns its weights and dependencies. No H3
environment changes, automatic large downloads, or claims of 16 GB fit. The
reference must already depict the SAME dancer in the desired outfit.
"""
import json
from pathlib import Path
import subprocess
import time

from .config import write_json
from .dance_media import normalize_dance_video
from .dance_quality import review_section, sha256
from .media import run, executable, probe


def wan_commands(job, output):
    repo = Path(job["wan_repository"])
    ckpt = Path(job["checkpoint_directory"])
    python = job["python"]
    prepared = Path(output) / "processed"
    # Keep preprocessing at 30 fps: upstream saves at cfg.sample_fps=30.
    # The resulting continuous video is normalized to 24 fps only afterwards.
    preprocess = [python, str(repo / "wan/modules/animate/preprocess/preprocess_data.py"),
                  "--ckpt_path", str(ckpt / "process_checkpoint"), "--video_path", str(job["reference_video"]),
                  "--refer_path", str(job["outfit_character_image"]), "--save_path", str(prepared),
                  "--resolution_area", "720", "1280", "--fps", "30", "--replace_flag"]
    generate = [python, str(repo / "generate.py"), "--task", "animate-14B", "--ckpt_dir", str(ckpt),
                "--src_root_path", str(prepared), "--refert_num", "1", "--replace_flag",
                "--offload_model", "True", "--t5_cpu", "--convert_model_dtype",
                "--sample_steps", str(job.get("steps", 20)), "--base_seed", str(job.get("seed", 123456789)),
                "--save_file", str(Path(output) / "wan-raw.mp4")]
    if job.get("relighting", False):
        generate.append("--use_relighting_lora")
    return preprocess, generate


def run_wan_job(job_path, plan_only=False):
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    required = {"wan_repository", "checkpoint_directory", "python", "reference_video", "outfit_character_image", "output_directory"}
    if not required <= job.keys() or job.keys() - (required | {"seed", "steps", "relighting", "pose_model"}):
        raise ValueError(f"Wan job requires {sorted(required)}; optional seed, steps, relighting, pose_model")
    for field in required | {"pose_model"}:
        if job.get(field):
            job[field] = str((job_path.parent / job[field]).resolve())
    if isinstance(job.get("steps", 20), bool) or not isinstance(job.get("steps", 20), int) or not 1 <= job.get("steps", 20) <= 100:
        raise ValueError("Wan steps must be an integer between 1 and 100")
    if isinstance(job.get("seed", 0), bool) or not isinstance(job.get("seed", 0), int) or job.get("seed", 0) < 0:
        raise ValueError("Wan seed must be a nonnegative integer")
    output = Path(job["output_directory"])
    commands = wan_commands(job, output)
    report = {"backend": "Wan2.2-Animate-14B replacement", "status": "planned", "commands": commands,
              "quality_validated": False, "gpu_fit_validated": False,
              "scope": "One outfit comparison; use H3 anchored bridges for wardrobe transformations."}
    if plan_only:
        return report
    for field in ("python", "reference_video", "outfit_character_image"):
        if not Path(job[field]).is_file():
            raise ValueError(f"Missing {field}: {job[field]}")
    repo = Path(job["wan_repository"])
    for path in (repo / "generate.py", repo / "wan/modules/animate/preprocess/preprocess_data.py",
                 Path(job["checkpoint_directory"]) / "process_checkpoint"):
        if not path.exists():
            raise ValueError(f"Missing official Wan environment component: {path}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output_directory to preserve existing comparison results")
    output.mkdir(parents=True, exist_ok=True)
    report["inputs"] = {field: {"path": job[field], "sha256": sha256(job[field])} for field in ("reference_video", "outfit_character_image")}
    started = time.monotonic()
    try:
        report["wan_revision"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        report["wan_source_modified"] = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
        report["status"] = "running"
        write_json(output / "report.json", report)
        for stage, command in zip(("preprocess", "generate"), commands):
            report["stage"] = stage
            write_json(output / "report.json", report)
            with (output / f"{stage}.log").open("w") as log:
                subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
        raw = output / "wan-raw.mp4"
        source_info = probe(job["reference_video"])
        src = next(s for s in source_info["streams"] if s["codec_type"] == "video")
        width, height = src["width"] - src["width"] % 2, src["height"] - src["height"] % 2
        normalized_source = output / "source-24fps.mp4"
        normalize_dance_video(job["reference_video"], normalized_source, width, height)
        count = int(next(s for s in probe(normalized_source)["streams"] if s["codec_type"] == "video")["nb_frames"])
        normalized_result = output / "wan-24fps.mp4"
        normalize_dance_video(raw, normalized_result, width, height)
        got = int(next(s for s in probe(normalized_result)["streams"] if s["codec_type"] == "video")["nb_frames"])
        if got < count:
            raise RuntimeError(f"Wan returned {got} frames, source has {count}; refusing to stretch or repeat dance motion")
        final = output / "generated.mp4"
        run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(normalized_result), "-i", str(normalized_source),
             "-map", "0:v:0", "-map", "1:a:0?", "-vf", f"trim=end_frame={count},setpts=PTS-STARTPTS",
             "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final)])
        report["quality"] = review_section(normalized_source, final, output, pose_model=job.get("pose_model"), mode="pose_warn" if job.get("pose_model") else "preview")
        report.update(status="completed", video_path=str(final), wall_seconds=time.monotonic() - started)
        write_json(output / "report.json", report)
        return report
    except BaseException as exc:
        report.update(status="failed", error=str(exc), wall_seconds=time.monotonic() - started)
        write_json(output / "report.json", report)
        raise
