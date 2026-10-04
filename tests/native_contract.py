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

    def test_wide_and_tall_saved_graphs_validate(self):
        import execution
        from h3_pipeline.dance_workflow import dance_canvas
        for dimensions in ((2560, 1080), (1080, 2560)):
            width, height = dance_canvas(*dimensions, 0.98)
            graph = build_workflow(prompt="A dancer", images=[], video=None, audio=None,
                include_video_audio=False, width=width, height=height, length=107, seed=1, job_id="wide")
            valid, error, _, errors = asyncio.run(execution.validate_prompt("wide", graph, None))
            self.assertTrue(valid, json.dumps({"error": error, "nodes": errors}))

    def test_shared_turbo_settings_survive_each_section(self):
        for turbo in (True, False):
            graph = build_workflow(prompt="Dance", images=[], video=None, audio=None, include_video_audio=False,
                width=128, height=96, length=22, seed=1, job_id="turbo-metadata", turbo=turbo, steps=4 if turbo else 25)
            for section in range(3):
                context = custom.H3Begin().begin(f"turbo-metadata-{section}", 0)[0]
                settings = {**graph["201"]["inputs"], "context": context}
                custom.H3RecordSettings().record(**settings)
                report = runtime.finish(context)
                self.assertEqual(report["turbo"], turbo)
                self.assertEqual(report["lora"], graph["145"]["inputs"]["lora_name"] if turbo else None)

    def test_stage_failures_and_interruptions_update_owning_report(self):
        from comfy.model_management import InterruptProcessingException
        from h3_pipeline.config import write_json
        for label in ("diffusion_checkpoint_load", "reference_conditioning_and_latents", "h3_sampling",
                      "video_vae_decode", "audio_vae_decode", "video_encoding"):
            for error in (RuntimeError("injected failure"), InterruptProcessingException()):
                path = self.root / "output/h3/dance/failure-test/report.json"
                completed = {"index": 1, "status": "completed", "video_path": "preserved.mp4"}
                write_json(path, {"job_id": "failure-test", "status": "running", "sections": [
                    completed, {"index": 2, "status": "running"}, {"index": 3}]})
                context = runtime.begin("failure-test-section-2", self.root / "output", 0,
                                        report_path=path, section_index=2)
                with self.assertRaises(type(error)):
                    with runtime.stage(context, label):
                        raise error
                report = json.loads(path.read_text())
                status = "interrupted" if isinstance(error, InterruptProcessingException) else "failed"
                self.assertEqual(report["status"], status)
                self.assertEqual(report["failed_section"], 2)
                self.assertEqual(report["sections"][0], completed)
                self.assertEqual(report["sections"][1]["status"], status)
                self.assertEqual(report["sections"][2]["status"], "cancelled")
                self.assertEqual(report["sections"][1]["server"]["failed_stage"], label)
                self.assertIsNone(runtime._active)

    def test_executor_errors_outside_adapters_are_scoped_to_the_prompt(self):
        import execution
        from types import SimpleNamespace
        from app.assets.manager import NoAssets
        from comfy.cli_args import args
        from comfy.model_management import InterruptProcessingException
        from comfy_execution.utils import CurrentNodeContext
        from h3_pipeline.config import write_json
        from h3_pipeline.dance_reports import register_report, _prompt_reports

        class UnprofiledFailure:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"context": ("STRING",)}}
            RETURN_TYPES = ()
            FUNCTION = "execute"
            OUTPUT_NODE = True

            def execute(self, context):
                raise failure

        unrelated = self.root / "output/h3/dance/unrelated/report.json"
        write_json(unrelated, {"status": "planned", "sections": [{"index": 1}]})
        with CurrentNodeContext("unrelated-prompt", "1"):
            register_report(unrelated)
        try:
            for failure in (RuntimeError("outside adapter"), InterruptProcessingException()):
                path = self.root / "output/h3/dance/executor-failure/report.json"
                write_json(path, {"job_id": "executor-failure", "status": "planned", "sections": [{"index": 1}]})
                server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *args: None)
                executor = execution.PromptExecutor(server, execution.CacheType.NONE,
                    {"ram": 0, "ram_inactive": 0}, asset_manager=NoAssets(args))
                graph = {"1": {"class_type": "H3DanceBeginSection", "inputs": {
                    "job_directory": "h3/dance/executor-failure", "section_index": 1}},
                    "2": {"class_type": "H3UnprofiledFailure", "inputs": {"context": ["1", 0]}}}
                with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"H3UnprofiledFailure": UnprofiledFailure}):
                    executor.execute(graph, "failing-prompt", execute_outputs=["2"])
                self.assertFalse(executor.success)
                report = json.loads(path.read_text())
                self.assertEqual(report["status"], "interrupted" if isinstance(failure, InterruptProcessingException) else "failed")
                self.assertEqual(report["failed_section"], 1)
                self.assertEqual(json.loads(unrelated.read_text())["status"], "planned")
                self.assertNotIn("failing-prompt", _prompt_reports)
                self.assertIsNone(runtime._active)
        finally:
            _prompt_reports.pop("unrelated-prompt", None)

    def test_attention_policy_blocks_checkpoint_and_vae_overrides_and_restores(self):
        import torch
        from types import SimpleNamespace
        from comfy.ldm.modules import attention as attn
        from comfy.ldm.minimax import model, vae
        from comfy.text_encoders import llama, qwen_vl
        from h3_pipeline.attention import attention_policy
        from unittest.mock import Mock
        monitor = SimpleNamespace(data={})
        original = model.optimized_attention, vae.optimized_attention, vae.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE
        forbidden = Mock(side_effect=AssertionError("Comfy Kitchen/override attention must not run"))
        q = torch.randn(1, 2, 5, 64)
        with self.assertRaisesRegex(ValueError, "restore probe"):
            with attention_policy(monitor, "reference_conditioning_and_latents"):
                containers = [attn.AttentionTensorContainer(t.clone()) for t in (q, q, q)]
                result = model.optimized_attention(*containers, 2, skip_reshape=True,
                    preferred_attention=SimpleNamespace(function=forbidden),
                    transformer_options={"optimized_attention_override": forbidden})
                self.assertEqual(result.shape, (1, 5, 128))
                self.assertFalse(vae.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE)
                decoder = vae.Attention(2, 64, operations=torch.nn)
                norm = SimpleNamespace(weight=None, eps=1e-6)
                def linear(layer, x, *args, **kwargs):
                    return torch.cat((q.transpose(1, 2).reshape(1, 5, 128),) * 3, -1) if layer is decoder.to_qkv else x
                with patch("comfy.ops.QuantizedTensor", torch.Tensor), patch("comfy.ops.linear_input_act", side_effect=linear), \
                     patch("comfy.quant_ops.ck.int8_attention", forbidden):
                    result = decoder(q.transpose(1, 2).reshape(1, 5, 128), None, norm, None, None)
                self.assertEqual(result.shape, (1, 5, 128))
                for module in (llama, qwen_vl):
                    module.optimized_attention_for_device(q.device)(q, q, q, 2, skip_reshape=True)
                import comfy.ops
                comfy.ops.scaled_dot_product_attention(q, q, q, is_causal=True)
                forbidden.assert_not_called()
                raise ValueError("restore probe")
        self.assertEqual(original, (model.optimized_attention, vae.optimized_attention, vae.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE))
        self.assertEqual(monitor.data["attention"]["effective_calls"], {
            "diffusion": {"pytorch": 1}, "video_vae": {"pytorch": 1},
            "text_encoder": {"pytorch": 2}, "audio_vae": {"pytorch": 1}})

    def test_sage_success_and_native_fallback_are_counted_separately(self):
        import torch
        from types import SimpleNamespace
        from comfy.ldm.modules import attention as attn
        from comfy.ldm.minimax import model
        from h3_pipeline.attention import attention_policy

        # CPU-backed tensor advertising CUDA ONLY to probe dispatch control flow.
        # The Sage kernel is explicitly mocked; this is not a CUDA correctness test.
        class DispatchTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device("cuda")

        q = torch.randn(1, 2, 5, 64, dtype=torch.float16).as_subclass(DispatchTensor)
        monitor = SimpleNamespace(data={})
        with patch.object(attn, "SAGE_ATTENTION_IS_AVAILABLE", True), \
             patch.object(attn, "sageattn", side_effect=[q, RuntimeError("synthetic kernel failure")], create=True):
            with attention_policy(monitor, "h3_sampling"):
                model.optimized_attention(q, q, q, 2, skip_reshape=True)
                model.optimized_attention(q, q, q, 2, skip_reshape=True)
        self.assertEqual(monitor.data["attention"]["effective_calls"], {"diffusion": {"sage": 1, "pytorch": 1}})

    def test_sage_preflight_rejects_unpinned_build_before_kernel_execution(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from h3_pipeline.preflight import attention_smoke
        from h3_pipeline.attention import SAGE_VERSION
        kernel = Mock(side_effect=AssertionError("Unpinned kernel must not run"))
        with patch.dict(sys.modules, {"sageattention": SimpleNamespace(sageattn=kernel)}), \
             patch("h3_pipeline.preflight.attention_metadata", return_value={
                 "sageattention_version": SAGE_VERSION, "sageattention_revision": "wrong"}):
            with self.assertRaisesRegex(RuntimeError, "rerun setup"):
                attention_smoke()
        kernel.assert_not_called()

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
        for kind, expected in (("H3UNETLoader", 1), ("H3CLIPLoader", 1), ("H3VAELoader", 2)):
            self.assertEqual(sum(n["class_type"] == kind for n in expanded.values()), expected)
        for field in ("model", "clip", "vae", "audio_vae"):
            consumers = references if field != "model" else [n for n in expanded.values() if n["class_type"] == "BasicGuider"]
            users = [n["inputs"][field][0] for n in consumers]
            self.assertEqual(len(set(users)), 1, field)
        # Every selected outfit is represented in its own prompt/mapping, never a
        # competing set of clothing references in a single native H3 render.
        mappings = [json.loads(n["inputs"]["reference_mapping"]) for n in expanded.values() if n["class_type"] == "H3RecordSettings"]
        self.assertEqual(len({Path(m[2]["path"]).name for m in mappings}), 12)

    def test_dance_turbo_uses_one_shared_lora_and_has_no_section_dependency_cycle(self):
        import execution
        job = self.dance_fixture(outfits=3)
        job.update(turbo=True, steps=4)
        expanded = dance.H3DanceWorkflow().expand(**job)["expand"]
        valid, error, _, errors = asyncio.run(execution.validate_prompt("dance-turbo-expanded", expanded, None))
        self.assertTrue(valid, json.dumps({"error": error, "nodes": errors}))
        loras = [key for key, n in expanded.items() if n["class_type"] == "H3LoraLoader"]
        self.assertEqual(len(loras), 1)
        for node in expanded.values():
            if node["class_type"] in ("BasicGuider", "H3Scheduler"):
                self.assertEqual(node["inputs"]["model"], [loras[0], 0])

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

    def test_dance_models_load_once_and_are_reused_with_native_cache_disabled(self):
        import execution
        import torch
        from types import SimpleNamespace
        from contextlib import ExitStack
        from app.assets.manager import NoAssets
        from comfy.cli_args import args
        from comfy.model_patcher import ModelPatcher
        job = self.dance_fixture(outfits=3)
        loaded, consumed, tracker_sizes = [], [], []

        class ProbePatcher(ModelPatcher):
            def is_dynamic(self):
                return True

            def set_in_use_by_current_prompt(self, value):
                self.in_current_prompt = value

        def load_model(self, context, **kwargs):
            runtime.current(context)
            patcher = ProbePatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
            loaded.append(patcher)
            return (patcher if "unet_name" in kwargs else SimpleNamespace(patcher=patcher),)

        class UseModels:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
                    "audio_vae": ("VAE",), "images": ("IMAGE",), "audio": ("AUDIO",), "context": ("STRING",)}}
            RETURN_TYPES = ("IMAGE", "AUDIO")
            FUNCTION = "use"

            def use(self, model, clip, vae, audio_vae, images, audio, context):
                runtime.current(context)
                models = (model, clip.patcher, vae.patcher, audio_vae.patcher)
                consumed.append(tuple(id(m) for m in models))
                self_outer.assertTrue(all(m.in_current_prompt for m in models))
                tracker_sizes.append(len(executor.prompt_model_tracker.models))
                return images, audio

        def synthetic_graph(**options):
            graph = build_workflow(**options)
            graph = {key: value for key, value in graph.items() if key in ("200", "201", "310", "130", "127", "128", "119", "120")}
            graph["330"] = {"class_type": "H3TestUseModels", "inputs": {
                "model": ["127", 0], "clip": ["128", 0], "vae": ["119", 0], "audio_vae": ["120", 0],
                "images": ["310", 0], "audio": ["310", 1], "context": ["201", 0]}}
            graph["130"]["inputs"].update(images=["330", 0], audio=["330", 1])
            return graph

        self_outer = self
        server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *args: None)
        executor = execution.PromptExecutor(server, execution.CacheType.NONE,
            {"ram": 0, "ram_inactive": 0}, asset_manager=NoAssets(args))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(nodes.NODE_CLASS_MAPPINGS, {"H3TestUseModels": UseModels}))
            stack.enter_context(patch.object(dance, "build_workflow", side_effect=synthetic_graph))
            for loader in (custom.H3UNETLoader, custom.H3CLIPLoader, custom.H3VAELoader):
                stack.enter_context(patch.object(loader, loader.FUNCTION, load_model))
            executor.execute({"1": {"class_type": "H3DanceWorkflow", "inputs": job}},
                "dance-model-reuse", execute_outputs=["1"])
        self.assertTrue(executor.success, str(executor.status_messages))
        self.assertEqual(len(loaded), 4)
        self.assertEqual(len(consumed), 3)
        self.assertEqual(len(set(consumed)), 1)
        self.assertEqual(tracker_sizes, [4, 4, 4])
        self.assertEqual(executor.prompt_model_tracker.models, {})
        self.assertTrue(all(not m.in_current_prompt for m in loaded))


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
