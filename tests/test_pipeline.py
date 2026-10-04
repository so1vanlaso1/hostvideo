import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h3_pipeline.client import ComfyClient, ComfyExecutionError
from h3_pipeline.cli import benchmark_job, read_job
from h3_pipeline.config import ASSETS, model_files
from h3_pipeline.models import ModelManager
from h3_pipeline.pipeline import MiniMaxH3Pipeline, GenerationError, fallback_attempts
from h3_pipeline.references import ReferenceManager, PromptBuilder, canvas, frame_count
from h3_pipeline.workflow import build_workflow, validate_graph
from h3_pipeline.media import prepare_video, probe, verify_output


class ReferenceTests(unittest.TestCase):
    def test_native_temporal_alignment_and_limits(self):
        self.assertEqual(frame_count(15), 362)
        self.assertEqual(frame_count(5), 124)
        self.assertEqual(frame_count(3), 73)
        for milliseconds in range(200, 15001, 17):
            duration = milliseconds / 1000
            n = max(5, round(duration * 24))
            self.assertEqual(frame_count(duration), n + (5 - n % 17) % 17)
            self.assertLessEqual(frame_count(duration), 362)
        for invalid in (0, 16, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                frame_count(invalid)

    def test_resolution_matches_native_mib_pixel_formula(self):
        self.assertEqual(canvas(0.6), (1056, 608))
        self.assertEqual(canvas(0.4), (864, 480))
        self.assertEqual(canvas(0.98), (1344, 768))
        self.assertEqual(canvas(1.0), (1344, 768))
        for aspect in ("16:9", "9:16", "1:1"):
            for mp in (0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.98, 1):
                w, h = canvas(mp, aspect)
                self.assertEqual((w % 32, h % 32), (0, 0))
                self.assertLessEqual(w * h, 1344 * 768)

    def test_reference_order_and_paired_audio_ordinals(self):
        with tempfile.TemporaryDirectory() as work:
            paths = [Path(work) / name for name in ("face2.png", "face1.png", "top.png", "bottom.png", "style.png")]
            for path in paths:
                path.touch()
            images = ReferenceManager.images(paths[:2], paths[2:4], paths[4:])
            self.assertEqual([r.path for r in images], [str(p.resolve()) for p in paths])
            refs = ReferenceManager.mapping(images, "dance.mp4", True, "voice.wav")
            self.assertEqual([r.tag for r in refs], [f"<Picture {i}>" for i in range(1, 6)] + ["<Audio 1>", "<Video 1>", "<Audio 2>"])
            prompt = PromptBuilder.build("Dance in a studio", refs)
            self.assertIn("<Picture 3> defines clothing", prompt)
            self.assertIn("<Audio 2> defines standalone", prompt)
            with self.assertRaises(ValueError):
                PromptBuilder.build("Use <Picture 9>", refs)
            with self.assertRaises(ValueError):
                ReferenceManager.images([paths[0]] * 10)

    def test_silent_video_does_not_consume_audio_tag(self):
        refs = ReferenceManager.mapping([], "dance.mp4", False, "voice.wav")
        self.assertEqual([r.tag for r in refs], ["<Video 1>", "<Audio 1>"])


class WorkflowTests(unittest.TestCase):
    def test_managed_server_selects_sage_and_setup_pins_same_revision(self):
        from h3_pipeline.server import command
        from h3_pipeline.config import RuntimePaths
        from h3_pipeline.attention import SAGE_REVISION
        cmd = command(RuntimePaths(Path("/tmp/h3-test")))
        self.assertIn("--use-sage-attention", cmd)
        self.assertNotIn("--use-pytorch-cross-attention", cmd)
        setup = (Path(__file__).resolve().parents[1] / "scripts/setup_vast.sh").read_text()
        self.assertIn(f"SAGE_REVISION={SAGE_REVISION}", setup)
        self.assertIn("--no-build-isolation", setup)

    def graph(self, **overrides):
        settings = dict(prompt="test", images=["job/person.png", "job/outfit.png"], video="job/dance.mp4",
                        audio="job/voice.wav", include_video_audio=True, width=1056, height=608,
                        length=362, seed=99, job_id="test-job")
        settings.update(overrides)
        return build_workflow(**settings)

    def test_joint_audio_video_graph_and_order(self):
        graph = validate_graph(self.graph())
        refs = graph["136"]["inputs"]
        self.assertEqual(refs["ref_images.ref_image_0"], ["300", 0])
        self.assertEqual(refs["ref_video_audios.ref_video_audio_0"], ["310", 1])
        self.assertEqual(graph["121"]["inputs"]["video_ready"], ["122", 0])
        self.assertEqual(graph["130"]["inputs"]["audio"], ["121", 0])
        self.assertEqual(graph["124"]["inputs"]["steps"], 25)
        self.assertEqual(graph["124"]["inputs"]["scheduler"], "simple")
        self.assertNotIn("145", graph)
        self.assertEqual(graph["126"]["inputs"]["model"], ["127", 0])
        self.assertEqual(graph["124"]["inputs"]["model"], ["127", 0])
        self.assertEqual(graph["126"]["class_type"], "BasicGuider")
        self.assertEqual(graph["92"]["inputs"]["format.codec"], "h264")

    def test_turbo_is_explicit_and_requires_four_steps(self):
        graph = validate_graph(self.graph(turbo=True, steps=4))
        self.assertEqual(graph["126"]["inputs"]["model"], ["145", 0])
        self.assertEqual(graph["124"]["inputs"]["model"], ["145", 0])
        with self.assertRaisesRegex(ValueError, "four steps"):
            self.graph(turbo=True, steps=25)

    def test_profiles_are_explicit(self):
        for profile in ("primary", "fp8", "int8-encoder", "fp8-int8-encoder"):
            graph = self.graph(profile=profile)
            name = graph["127"]["inputs"]["unet_name"]
            self.assertIn("fp8" if profile.startswith("fp8") else "int8", name)
            self.assertEqual(len(model_files(profile)), 5)
        with self.assertRaises(ValueError):
            self.graph(profile="bf16")

    def test_prompt_only_graph_has_no_dangling_reference_nodes(self):
        graph = validate_graph(self.graph(images=[], video=None, audio=None, include_video_audio=False))
        self.assertNotIn("310", graph)
        self.assertNotIn("ref_videos.ref_video_0", graph["136"]["inputs"])

    def test_pristine_template_and_ui_links(self):
        import hashlib
        provenance = json.loads((ASSETS / "provenance.json").read_text())
        self.assertEqual(hashlib.sha256((ASSETS / "workflows/official_ref2va.json").read_bytes()).hexdigest(), provenance["template_sha256"])
        graph = json.loads((ASSETS / "workflows/ref2va_16gb_ui.json").read_text())
        nodes = {n["id"]: n for n in graph["nodes"]}
        for link, source, source_slot, target, target_slot, kind in graph["links"]:
            self.assertIn(link, nodes[source]["outputs"][source_slot]["links"])
            self.assertEqual(link, nodes[target]["inputs"][target_slot]["link"])
            self.assertIn(nodes[source]["outputs"][source_slot]["type"], kind.split(","))
            self.assertIn(kind, nodes[target]["inputs"][target_slot]["type"].split(","))
        record = nodes[201]["widgets_values"]
        self.assertEqual(len(record), 9)
        self.assertEqual(record[5], nodes[127]["widgets_values"][0])
        self.assertEqual(record[6], nodes[128]["widgets_values"][0])
        self.assertIn(record[7], ("max", "match"))
        visiting, visited = set(), set()
        def visit(node_id):
            self.assertNotIn(node_id, visiting, f"Cycle at {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for link in graph["links"]:
                if link[3] == node_id:
                    visit(link[1])
            visiting.remove(node_id)
            visited.add(node_id)
        for node_id in nodes:
            visit(node_id)


class ClientTests(unittest.TestCase):
    def test_history_oom_is_classified_and_not_confused_with_other_failures(self):
        client = ComfyClient()
        with patch.object(client, "request") as request, patch("h3_pipeline.client.time.sleep"):
            request.side_effect = [type("Response", (), {"json": lambda self: {"prompt_id": "fixture"}})(),
                type("Response", (), {"json": lambda self: {"fixture": {"status": {"messages": [["execution_error", {
                    "exception_type": "torch.OutOfMemoryError", "exception_message": "CUDA out of memory"}]]}}}})()]
            with self.assertRaises(ComfyExecutionError) as caught:
                client.execute({})
            self.assertTrue(caught.exception.is_oom)
            self.assertEqual(request.call_count, 2)
        self.assertFalse(ComfyExecutionError({"exception_type": "MemoryError", "exception_message": "system RAM exhausted"}).is_oom)

    def test_timeout_preserves_running_job_and_reports_prompt_id(self):
        client = ComfyClient()
        response = type("Response", (), {"json": lambda self: {"prompt_id": "still-running"}})()
        with patch.object(client, "request", return_value=response) as request, patch("h3_pipeline.client.time.monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(TimeoutError, "still-running"):
                client.execute({}, timeout=1)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(request.call_args.args, ("POST", "/prompt"))


class DownloadTests(unittest.TestCase):
    def manager(self, root):
        manager = ModelManager(root)
        data = b"small-checkpoint-fixture"
        manager.files = {"fixture": {"path": "vae/fixture.safetensors", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
        return manager, data

    def test_interrupted_download_resumes_and_receipt_verifies(self):
        with tempfile.TemporaryDirectory() as root:
            manager, data = self.manager(root)
            state = {"attempts": 0}
            def fetch(**kwargs):
                state["attempts"] += 1
                cache = Path(kwargs["local_dir"]) / ".partial"
                if state["attempts"] == 1:
                    cache.write_bytes(data[:8])
                    raise ConnectionError("interrupted transfer")
                self.assertEqual(cache.read_bytes(), data[:8])
                path = Path(kwargs["local_dir"]) / kwargs["filename"]
                path.parent.mkdir()
                path.write_bytes(cache.read_bytes() + data[8:])
                return str(path)
            with patch("h3_pipeline.models.shutil.disk_usage", return_value=type("Disk", (), {"free": 20 * 1024**3})()):
                with self.assertRaises(ConnectionError):
                    manager.download(fetch)
                self.assertTrue(manager.download(fetch)[0]["valid"])
                self.assertTrue(manager.download(fetch, verify_existing=True)[0]["valid"])
                self.assertEqual(state["attempts"], 2)

    def test_bad_integrity_and_disk_shortage_fail(self):
        with tempfile.TemporaryDirectory() as root:
            manager, data = self.manager(root)
            with patch("h3_pipeline.models.shutil.disk_usage", return_value=type("Disk", (), {"free": 0})()):
                with self.assertRaisesRegex(RuntimeError, "GiB free"):
                    manager.download(lambda **kwargs: self.fail("must not download"))
            path = Path(root) / "vae/fixture.safetensors"
            path.parent.mkdir()
            path.write_bytes(b"x" * len(data))
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                manager.download()


class FakeClient:
    def __init__(self, errors=()):
        self.graphs = []
        self.errors = list(errors)

    def check(self):
        return {}

    def upload(self, path, subfolder):
        return subfolder + "/" + Path(path).name

    def execute(self, graph, timeout, progress):
        self.graphs.append(graph)
        if self.errors:
            raise self.errors.pop(0)
        return {"outputs": {"92": {"images": [{"filename": "fixture.mp4", "subfolder": "fixture", "type": "output"}], "h3_report": [{"test_fixture": True}]}}}

    def download(self, item, destination):
        Path(destination).write_bytes(b"test-fixture-only")
        return str(destination)


def oom():
    return ComfyExecutionError({"exception_type": "torch.OutOfMemoryError", "exception_message": "CUDA out of memory"})


class PipelineTests(unittest.TestCase):
    def test_bounded_oom_ladder_and_preserved_duration_seed(self):
        client = FakeClient([oom()] * 4)
        with tempfile.TemporaryDirectory() as root, patch("h3_pipeline.pipeline.verify_output", return_value={"test_fixture": True}):
            result = MiniMaxH3Pipeline(root=root, client=client, progress=None).generate("A dance", seed=77, oom_fallback=True)
            self.assertEqual(len(client.graphs), 5)
            self.assertTrue(result.performance_report["fallback_used"])
            self.assertEqual(result.performance_report["effective_megapixels"], 0.4)
            for graph in client.graphs:
                self.assertEqual(graph["136"]["inputs"]["length"], 362)
                self.assertEqual(graph["129"]["inputs"]["noise_seed"], 77)
            self.assertEqual([g["200"]["inputs"]["memory_level"] for g in client.graphs], [0, 1, 2, 2, 2])

    def test_other_errors_are_not_oom_retried(self):
        error = ComfyExecutionError({"exception_type": "ValueError", "exception_message": "shape mismatch"})
        with tempfile.TemporaryDirectory() as root:
            client = FakeClient([error])
            with self.assertRaises(GenerationError) as caught:
                MiniMaxH3Pipeline(root=root, client=client, progress=None).generate("Dance")
            self.assertEqual(len(client.graphs), 1)
            report = json.loads(Path(caught.exception.report_path).read_text())
            self.assertEqual(report["status"], "failed")

    def test_oom_exhaustion_and_benchmark_no_fallback(self):
        self.assertEqual(len(fallback_attempts(0.4, "match")), 3)
        with tempfile.TemporaryDirectory() as root:
            client = FakeClient([oom()] * 8)
            with self.assertRaisesRegex(GenerationError, "fallback exhausted"):
                MiniMaxH3Pipeline(root=root, client=client, progress=None).generate("Dance", megapixels=0.4, oom_fallback=True)
            self.assertEqual(len(client.graphs), 3)
        job = benchmark_job({"prompt": "Dance", "reference_video": "video", "character_images": ["face"], "clothing_images": ["outfit"]}, "smoke")
        self.assertFalse(job["oom_fallback"])
        self.assertNotIn("reference_video", job)
        self.assertEqual(job["duration"], 3)

    def test_quality_default_stops_on_oom_without_reducing_canvas(self):
        with tempfile.TemporaryDirectory() as root:
            client = FakeClient([oom()] * 8)
            with self.assertRaisesRegex(GenerationError, "quality reductions are disabled"):
                MiniMaxH3Pipeline(root=root, client=client, progress=None).generate("Dance")
            self.assertEqual(len(client.graphs), 1)
            settings = client.graphs[0]["136"]["inputs"]
            self.assertEqual((settings["width"], settings["height"]), (1344, 768))
            self.assertEqual(settings["ref_image_size"], "max")

    def test_json_job_asset_paths_and_options(self):
        with tempfile.TemporaryDirectory() as root:
            job = Path(root) / "job.json"
            job.write_text(json.dumps({"prompt": "Dance", "character_images": ["face.png"]}))
            self.assertEqual(read_job(job)["character_images"], [str((Path(root) / "face.png").resolve())])
            job.write_text('{"prompt":"Dance","invalid":1}')
            with self.assertRaises(ValueError):
                read_job(job)


@unittest.skipUnless(os.environ.get("FFMPEG") or __import__("shutil").which("ffmpeg"), "ffmpeg is unavailable")
class MediaTests(unittest.TestCase):
    def fixture(self, directory, audio=True):
        from h3_pipeline.media import executable, run
        source = Path(directory) / ("source-audio.mp4" if audio else "source-silent.mp4")
        command = [executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-c:a", "aac"]
        run(command + ["-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)])
        return source

    def test_cpu_preprocessing_resizes_preserves_pairing_and_timing(self):
        with tempfile.TemporaryDirectory() as work:
            source = self.fixture(work)
            destination = Path(work) / "prepared.mp4"
            info = prepare_video(source, destination, 3, 864, 480, start=0.5)
            self.assertTrue(info["paired_audio"])
            self.assertEqual(info["frames"], 72)
            self.assertEqual(info["usable_frames"], 56)
            streams = probe(destination)["streams"]
            video = next(s for s in streams if s["codec_type"] == "video")
            audio = next(s for s in streams if s["codec_type"] == "audio")
            self.assertLessEqual(video["width"] * video["height"], 864 * 480 + 32 * 864)
            self.assertLess(abs(float(video["duration"]) - float(audio["duration"])), 0.15)

    def test_silent_video_and_audio_disable(self):
        with tempfile.TemporaryDirectory() as work:
            source = self.fixture(work, audio=False)
            self.assertFalse(prepare_video(source, Path(work) / "silent.mp4", 3, 864, 480)["paired_audio"])
            source = self.fixture(work, audio=True)
            out = Path(work) / "disabled.mp4"
            self.assertFalse(prepare_video(source, out, 3, 864, 480, include_audio=False)["paired_audio"])
            self.assertEqual(len(probe(out)["streams"]), 1)

    def test_output_verification_requires_native_audio_and_valid_frames(self):
        from h3_pipeline.media import executable, run
        with tempfile.TemporaryDirectory() as work:
            video = Path(work) / "fixture.mp4"
            run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=size=64x64:rate=24",
                 "-f", "lavfi", "-i", "sine=sample_rate=48000", "-t", str(73 / 24), "-c:v", "libx264", "-c:a", "aac", str(video)])
            self.assertEqual(int(verify_output(video, 73)["streams"][0]["nb_frames"]), 73)
            silent = self.fixture(work, False)
            with self.assertRaisesRegex(RuntimeError, "AND"):
                verify_output(silent)


if __name__ == "__main__":
    unittest.main()
