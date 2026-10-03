"""Optional integration against real, pinned ComfyUI. No weights or GPU needed."""
import asyncio
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

comfy_path = os.environ.get("H3_COMFY_PATH")
if not comfy_path:
    raise SystemExit("Set H3_COMFY_PATH to the pinned ComfyUI checkout with its dependencies installed")
sys.path.insert(0, str(Path(comfy_path).resolve()))
sys.argv = [sys.argv[0], "--cpu", "--fp16-intermediates"]
import comfy.options
comfy.options.enable_args_parsing()
import folder_paths
import nodes
from comfy_api.latest import io, InputImpl
from h3_pipeline import server_nodes as custom
from h3_pipeline import server_runtime as runtime
from h3_pipeline.config import ASSETS, COMFY_REVISION, model_files
from h3_pipeline.workflow import build_workflow
from h3_pipeline import dance_nodes as dance


class NativeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=comfy_path, capture_output=True, text=True, check=True).stdout.strip()
        if revision != COMFY_REVISION:
            raise RuntimeError("Native tests require the pinned ComfyUI revision")
        cls.work = tempfile.TemporaryDirectory(prefix="h3-contract-")
        cls.root = Path(cls.work.name)
        for name in ("input", "output", "temp"):
            (cls.root / name).mkdir()
            getattr(folder_paths, f"set_{name}_directory")(str(cls.root / name))
        for item in [*model_files().values(), *model_files("int8-encoder").values()]:
            path = cls.root / "models" / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()  # Schema validation only; never attempt model loading.
            folder_paths.add_model_folder_path(item["path"].split("/")[0], str(path.parent))
        # Register only the native types used by the official baseline.
        from comfy_extras import nodes_custom_sampler, nodes_video, nodes_audio
        for module in (nodes_custom_sampler, nodes_video, nodes_audio):
            extension = asyncio.run(module.comfy_entrypoint())
            for node in asyncio.run(extension.get_node_list()):
                nodes.NODE_CLASS_MAPPINGS[node.GET_SCHEMA().node_id] = node
        source = Path(__file__).resolve().parents[1] / "custom_nodes"
        for directory in ("h3_lowvram", "h3_native_adapters"):
            if not asyncio.run(nodes.load_custom_node(str(source / directory))):
                raise RuntimeError(f"Custom-node registration failed: {directory}")

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_v1_v3_schemas_and_real_prompt_validation(self):
        import execution
        graph = build_workflow(prompt="A dancer", images=[], video=None, audio=None, include_video_audio=False,
                               width=1056, height=608, length=362, seed=1, job_id="native-test")
        for name in set(node["class_type"] for node in graph.values()):
            cls = nodes.NODE_CLASS_MAPPINGS[name]
            cls.INPUT_TYPES()
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("native-test", graph, None))
        self.assertTrue(valid, json.dumps({"error": error, "nodes": node_errors}, indent=2))

    def test_reference_autogrow_and_dynamic_codec_native_inputs(self):
        import execution
        from PIL import Image
        Image.new("RGB", (64, 64), "red").save(self.root / "input/person.png")
        graph = build_workflow(prompt="Use <Picture 1>", images=["person.png"], video=None, audio=None,
                               include_video_audio=False, width=1056, height=608, length=362, seed=1, job_id="native-reference")
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("native-reference", graph, None))
        self.assertTrue(valid, json.dumps({"error": error, "nodes": node_errors}, indent=2))

    def test_explicit_turbo_graph_still_validates_natively(self):
        import execution
        graph = build_workflow(prompt="A dancer", images=[], video=None, audio=None, include_video_audio=False,
                               width=1344, height=768, length=73, seed=1, job_id="native-turbo", turbo=True, steps=4)
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("native-turbo", graph, None))
        self.assertTrue(valid, json.dumps({"error": error, "nodes": node_errors}, indent=2))

    def test_video_paired_audio_and_standalone_audio_graph_validates_natively(self):
        import execution
        from PIL import Image
        Image.new("RGB", (64, 64), "blue").save(self.root / "input/outfit.png")
        (self.root / "input/schema-only.mp4").touch()
        (self.root / "input/schema-only.wav").touch()
        graph = build_workflow(prompt="Use <Picture 1>, <Video 1>, <Audio 1> and <Audio 2>", images=["outfit.png"],
                               video="schema-only.mp4", audio="schema-only.wav", include_video_audio=True,
                               width=1056, height=608, length=362, seed=1, job_id="native-all-refs")
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("native-all-refs", graph, None))
        self.assertTrue(valid, json.dumps({"error": error, "nodes": node_errors}, indent=2))

    def test_missing_video_reports_only_file_and_keeps_numeric_validation(self):
        import execution
        graph = build_workflow(prompt="A dancer", images=[], video="missing.mp4", audio=None,
                               include_video_audio=True, width=1344, height=768, length=124,
                               seed=1, job_id="native-missing-video")
        _, _, _, errors = asyncio.run(execution.validate_prompt("missing-video", graph, None))
        self.assertEqual([e["extra_info"]["input_name"] for e in errors["310"]["errors"]], ["file"])
        (self.root / "input/numeric-check.mp4").touch()
        graph["310"]["inputs"].update(file="numeric-check.mp4", megapixels=2)
        _, _, _, errors = asyncio.run(execution.validate_prompt("invalid-megapixels", graph, None))
        self.assertEqual([e["extra_info"]["input_name"] for e in errors["310"]["errors"]], ["megapixels"])

    def test_cpu_frames_native_decoder_and_memory_headroom_restore(self):
        import torch
        class VAE:
            output_device = torch.device("cpu")
            called = 0
            def decode(self, latent):
                self.called += 1
                return torch.ones((5, 32, 32, 3), device=self.output_device, dtype=torch.float16)
        vae = VAE()
        manager = runtime.MemoryManager()
        initial = manager.mm.EXTRA_RESERVED_VRAM
        manager.configure(2)
        self.assertGreaterEqual(manager.mm.EXTRA_RESERVED_VRAM, 2 * 1024**3)
        frames = manager.video_decode(vae, {"samples": torch.zeros((1, 24, 2, 2, 2))})
        self.assertEqual(frames.device.type, "cpu")
        self.assertEqual(frames.dtype, torch.float16)
        self.assertEqual(vae.called, 1)
        manager.restore()
        self.assertEqual(manager.mm.EXTRA_RESERVED_VRAM, initial)

    def test_method_instrumentation_restores_after_error(self):
        import contextlib
        class Encoder:
            def encode(self, value):
                raise ValueError("fixture error")
        class Monitor:
            @contextlib.contextmanager
            def stage(self, name):
                yield
        encoder = Encoder()
        with self.assertRaises(ValueError):
            with runtime.measure_method(encoder, "encode", Monitor(), "qwen"):
                encoder.encode(1)
        self.assertNotIn("encode", encoder.__dict__)

    def test_native_video_mux_produces_h264_aac(self):
        from fractions import Fraction
        import torch
        from comfy_api.latest import InputImpl, Types
        from h3_pipeline.media import verify_output
        frames = torch.full((73, 64, 64, 3), 0.5, dtype=torch.float16, device="cpu")
        rate = 48000
        audio = {"waveform": torch.zeros((1, 2, round(73 / 24 * rate))), "sample_rate": rate}
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=audio, frame_rate=Fraction(24)), bit_depth=8, color_space="sRGB")
        destination = self.root / "output/native-mux.mp4"
        video.save_to(str(destination), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
        verify_output(destination, 73)

    def test_profiled_save_executes_as_native_class_clone_and_flushes_telemetry(self):
        from fractions import Fraction
        import torch
        from comfy_api.latest import InputImpl, Types
        context = custom.H3Begin().begin("save-contract", 0)[0]
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=torch.full((22, 32, 32, 3), 0.5, dtype=torch.float16),
            audio={"waveform": torch.zeros((1, 2, 44000)), "sample_rate": 48000}, frame_rate=Fraction(24)),
            bit_depth=8, color_space="sRGB")
        cls = custom.H3SaveVideo.PREPARE_CLASS_CLONE(None)
        result = cls.execute(context=context, video=video, filename_prefix="unused", format={"format": "mp4", "codec": {"codec": "h264"}})
        self.assertEqual(result.ui["h3_report"][0]["status"], "completed")
        self.assertIn("video_encoding", result.ui["h3_report"][0]["stages_seconds"])
        self.assertIsNone(runtime._active)
        self.assertTrue((self.root / "output/h3/save-contract/telemetry.json").is_file())

    def test_reference_video_loader_decodes_fp16_cpu_and_trims_paired_audio(self):
        from h3_pipeline.media import run, executable
        context = custom.H3Begin().begin("load-contract", 0)[0]
        source = self.root / "input/dance.mp4"
        run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=size=128x64:rate=24",
             "-f", "lavfi", "-i", "sine=sample_rate=48000", "-t", "3", "-c:v", "libx264", "-c:a", "aac", str(source)])
        frames, audio, metadata = custom.H3LoadReferenceVideo().load("dance.mp4", 3, 0.4, "16:9", 0, True, context)
        info = json.loads(metadata)
        self.assertEqual(frames.device.type, "cpu")
        self.assertEqual(str(frames.dtype), "torch.float16")
        self.assertEqual(len(frames), 56)
        self.assertEqual(audio["waveform"].shape[-1], round(56 / 24 * audio["sample_rate"]))
        runtime.finish(context)

    def dance_fixture(self, outfits=2, silent=False):
        from PIL import Image
        from h3_pipeline.media import run, executable
        source = self.root / "input/general-dance.mp4"
        command = [executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=96x128:rate=24"]
        if not silent:
            command += ["-f", "lavfi", "-i", "sine=sample_rate=48000", "-c:a", "aac"]
        run(command + ["-t", "2.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)])
        paths = []
        for i in range(outfits):
            name = f"general-outfit-{i}.png"
            Image.new("RGB", (64, 64), (i * 10, 64, 128)).save(self.root / "input" / name)
            paths.append(name)
        return {"reference_video": source.name, "character_image": "Keep original", "background_image": "Keep original",
                "outfit_images": json.dumps(paths), "prompt": "Dance naturally", "seed": 123,
                "megapixels": 0.98, "max_section_seconds": 5, "audio_mode": "source", "profile": "int8-encoder",
                "steps": 25, "turbo": False, "context_frames": 4, "outfit_durations": "[]",
                "start_seconds": 0, "duration_seconds": 0}

    def test_general_dance_workflow_expands_and_validates_more_than_nine_outfits(self):
        import execution
        from comfy_execution.graph_utils import is_link
        job = self.dance_fixture(outfits=12)
        self.assertTrue(dance.H3DanceWorkflow.VALIDATE_INPUTS(**{k: job[k] for k in (
            "reference_video", "character_image", "background_image", "outfit_images")}, outfit_durations="[]"))
        root = {"1": {"class_type": "H3DanceWorkflow", "inputs": job}}
        valid, error, _, errors = asyncio.run(execution.validate_prompt("dance-root", root, None))
        self.assertTrue(valid, json.dumps({"error": error, "nodes": errors}))
        expanded = dance.H3DanceWorkflow().expand(**job)["expand"]
        valid, error, _, errors = asyncio.run(execution.validate_prompt("dance-expanded", expanded, None))
        self.assertTrue(valid, json.dumps({"error": error, "nodes": errors}))
        gates = [n for n in expanded.values() if n["class_type"] == "H3DanceBeginSection"]
        self.assertEqual(len(gates), 12)
        self.assertNotIn("previous_section", gates[0]["inputs"])
        for gate in gates[1:]:
            dependency = gate["inputs"]["previous_section"]
            self.assertTrue(is_link(dependency))
            self.assertEqual(expanded[dependency[0]]["class_type"], "H3DanceSaveSection")
        references = [n for n in expanded.values() if n["class_type"] == "H3ReferenceToVideo"]
        self.assertTrue(all(len([k for k in n["inputs"] if k.startswith("ref_images.")]) <= 4 for n in references))
        # Every selected outfit is represented in its own prompt/mapping, never a
        # competing set of clothing references in a single native H3 render.
        mappings = [json.loads(n["inputs"]["reference_mapping"]) for n in expanded.values() if n["class_type"] == "H3RecordSettings"]
        self.assertEqual(len({Path(m[2]["path"]).name for m in mappings}), 12)

    def test_general_dance_synthetic_save_and_assembly_without_model_weights(self):
        job = self.dance_fixture()
        expanded = dance.H3DanceWorkflow().expand(**job)["expand"]
        gates = [n["inputs"] for n in expanded.values() if n["class_type"] == "H3DanceBeginSection"]
        job_directory = gates[0]["job_directory"]
        report_path = dance.output_path(job_directory) / "report.json"
        report = json.loads(report_path.read_text())
        previous = None
        # Substitute synthetic reference files for inference output. Everything
        # after inference uses the real native encoder and real workflow nodes.
        for section in report["sections"]:
            index = section["index"]
            context = dance.H3DanceBeginSection().begin(job_directory, index, previous)[0]
            video_path = self.root / "input/h3-dance" / report["job_id"] / f"section-{index}.mp4"
            previous = dance.H3DanceSaveSection().save(InputImpl.VideoFromFile(str(video_path)), context, job_directory, index)[0]
        result = dance.H3DanceAssemble().assemble(job_directory, previous)
        from h3_pipeline.media import verify_output
        verify_output(result["result"][1], 60)
        self.assertEqual(json.loads(report_path.read_text())["status"], "completed")
        self.assertTrue(result["ui"]["animated"][0])
        self.assertIsNone(runtime._active)

    def test_general_dance_replacement_paths_and_silent_audio_mapping(self):
        from PIL import Image
        job = self.dance_fixture(outfits=1, silent=True)
        for name in ("person.png", "scene.png"):
            Image.new("RGB", (64, 64), "green").save(self.root / "input" / name)
        job.update(character_image="person.png", background_image="scene.png")
        expanded = dance.H3DanceWorkflow().expand(**job)["expand"]
        records = [n["inputs"] for n in expanded.values() if n["class_type"] == "H3RecordSettings"]
        mapping = json.loads(records[0]["reference_mapping"])
        self.assertEqual([Path(m["path"]).name for m in mapping[:2]], ["person.png", "scene.png"])
        self.assertFalse(any(m["tag"].startswith("<Audio") for m in mapping))
        self.assertIn("Replace the original dancer", records[0]["prompt"])
        self.assertIn("Replace the source background", records[0]["prompt"])
        with self.assertRaises(ValueError):
            dance.input_path("../outside.mp4")
        with self.assertRaises(ValueError):
            dance.output_path("../outside")

    def test_general_dance_runs_through_native_expansion_executor_with_synthetic_video(self):
        import execution
        from types import SimpleNamespace
        from app.assets.manager import NoAssets
        from comfy.cli_args import args
        job = self.dance_fixture()

        def synthetic_graph(**options):
            # Replace inference with the native source decoder/CreateVideo only.
            # Exercise real graph expansion, queue ordering, profiling, save,
            # trimming and final preview without loading any model weights.
            graph = build_workflow(**options)
            graph = {key: value for key, value in graph.items() if key in ("200", "201", "310", "130")}
            graph["130"]["inputs"].update(images=["310", 0], audio=["310", 1])
            return graph

        server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *args: None)
        executor = execution.PromptExecutor(server, execution.CacheType.NONE,
            {"ram": 0, "ram_inactive": 0}, asset_manager=NoAssets(args))
        root = {"1": {"class_type": "H3DanceWorkflow", "inputs": job}}
        with patch.object(dance, "build_workflow", side_effect=synthetic_graph):
            executor.execute(root, "synthetic-dance-expansion", execute_outputs=["1"])
        self.assertTrue(executor.success, str(executor.status_messages))
        self.assertIsNone(runtime._active)
        reports = list((self.root / "output/h3/dance").glob("*/report.json"))
        matching = [json.loads(p.read_text()) for p in reports if json.loads(p.read_text()).get("status") == "completed"]
        self.assertTrue(matching)


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
