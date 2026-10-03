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
from comfy_api.latest import io
from h3_pipeline import server_nodes as custom
from h3_pipeline import server_runtime as runtime
from h3_pipeline.config import ASSETS, COMFY_REVISION, model_files
from h3_pipeline.workflow import build_workflow


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
        for item in model_files().values():
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


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
