"""Exercise real graph expansion on CPU with a minimal Comfy graph transport.

This validates dependencies/paths/planning, not native sampling or model fit.
Native schemas are separately tested in native_contract.py on ComfyUI.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from contextlib import contextmanager

from PIL import Image
from h3_pipeline.config import ASSETS
from h3_pipeline.workflow import validate_graph
from h3_pipeline.dance_media import normalize_dance_video, slice_dance_video
from h3_pipeline.dance_workflow import plan_sections, plan_transitions, assembly_plan
from h3_pipeline.media import executable, run, verify_output


class FakeNode:
    def __init__(self, builder, name, kind, inputs):
        self.builder, self.name = builder, name
        builder.graph[name] = {"class_type": kind, "inputs": inputs}
    def set_input(self, name, value):
        self.builder.graph[self.name]["inputs"][name] = value
    def out(self, slot):
        return [self.name, slot]
    def set_override_display_id(self, name):
        pass


class FakeGraph:
    def __init__(self):
        self.graph = {}
    def node(self, kind, id=None, **inputs):
        return FakeNode(self, id or f"auto-{len(self.graph)}", kind, inputs)
    def finalize(self):
        return self.graph


def load_nodes(input_root, output_root):
    folder = types.ModuleType("folder_paths")
    folder.get_input_directory = lambda: str(input_root)
    folder.get_output_directory = lambda: str(output_root)
    folder.get_annotated_filepath = lambda name: str(input_root / name)
    api = types.ModuleType("comfy_api.latest")
    api.InputImpl = api.Types = api.io = api.ui = types.SimpleNamespace()
    graph = types.ModuleType("comfy_execution.graph_utils")
    graph.GraphBuilder = FakeGraph
    graph.is_link = lambda value: isinstance(value, list) and len(value) == 2
    runtime = types.ModuleType("h3_pipeline.server_runtime")
    spec = importlib.util.spec_from_file_location("h3_pipeline._test_dance_nodes", ASSETS.parent / "dance_nodes.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"folder_paths": folder, "torch": types.ModuleType("torch"),
                                "comfy_api": types.ModuleType("comfy_api"), "comfy_api.latest": api,
                                "comfy_execution": types.ModuleType("comfy_execution"), "comfy_execution.graph_utils": graph,
                                "h3_pipeline.server_runtime": runtime}):
        spec.loader.exec_module(module)
    module.register_report = lambda path: None
    return module


class DanceGraphTests(unittest.TestCase):
    def test_auto_refinement_and_bridges_have_real_dependency_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "input", root / "output"
            source.mkdir(); output.mkdir()
            (source / "dance.mp4").write_bytes(b"fixture")
            for image in ("a.png", "b.png"):
                Image.new("RGB", (128, 96)).save(source / image)
            nodes = load_nodes(source, output)
            metadata = {"streams": [{"codec_type": "video", "width": 128, "height": 96, "nb_frames": "240"}]}
            def extract(video, destination, frame):
                Image.new("RGB", (128, 96)).save(destination)
                return str(destination)
            with patch.object(nodes, "probe", return_value=metadata), patch.object(nodes, "normalize_dance_video", return_value=metadata), \
                 patch.object(nodes, "slice_dance_video"), patch.object(nodes, "extract_frame", side_effect=extract):
                result = nodes.H3DanceWorkflow().expand("dance.mp4", "Keep original", "Keep original", '["a.png","b.png"]',
                    "Preserve dance", 1, 0.1, 5, "source", "int8-encoder", 25, False, quality_mode="preview")
            graph = result["expand"]
            validate_graph(graph)
            reports = list(output.rglob("report.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            stable = [s for s in report["sections"] if s["kind"] == "stable"]
            bridges = [s for s in report["sections"] if s["kind"] == "transition"]
            self.assertEqual(len(bridges), 1)
            self.assertEqual(sum(p["frames"] for p in report["assembly_plan"]), 240)
            loaders = [n for n in graph.values() if n["class_type"] in nodes.MODEL_LOADER_TYPES]
            self.assertEqual(len(loaders), 4)  # ONE model set across every pass.
            for s in stable:
                index = s["index"]
                begin = graph[f"section{index}_200"]["inputs"]
                self.assertEqual(begin["anchor_pass"], [f"section{index}_anchor_92", 0])
                self.assertTrue(graph[f"section{index}_126"]["inputs"]["conditioning"][0].startswith(f"section{index}_5"))
            bridge = bridges[0]
            index = bridge["index"]
            self.assertIn("previous_section", graph[f"section{index}_200"]["inputs"])
            for i in (0, 1):
                self.assertEqual(graph[f"section{index}_{500+2*i}"]["inputs"]["dependency"], [f"section{index}_{450+i}", 0])
            # Replay a selected outfit and its dependent bridge. Other outfits
            # remain on disk, and the graph contains no dangling model links.
            for s in report["sections"]:
                clip = reports[0].parent / f"section-{s['index']}" / "kept.mp4"
                clip.write_bytes(b"generated")
                s.update(status="completed", video_path=str(clip))
            report["status"] = "completed"
            reports[0].write_text(json.dumps(report))
            resumed = nodes.H3DanceResume().expand(str(reports[0].parent.relative_to(output)), "[1]", 42)
            validate_graph(resumed["expand"])
            self.assertNotIn(f"section{stable[-1]['index']}_200", resumed["expand"])
            self.assertEqual(resumed["expand"]["section1_129"]["inputs"]["noise_seed"], 42)
            self.assertTrue(list(reports[0].parent.glob("section-1/attempts/*/job-report.json")))

    def test_manifest_cannot_escape_input_or_silently_duplicate_guides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nodes = load_nodes(root, root / "output")
            Image.new("RGB", (32, 32)).save(root / "anchor.png")
            entry = {"frame": 0, "outfit_index": 1, "image": "anchor.png"}
            self.assertEqual(nodes.parse_anchor_manifest(json.dumps([entry]), 24, 1), [entry])
            with self.assertRaises(ValueError):
                nodes.parse_anchor_manifest(json.dumps([entry, entry]), 24, 1)
            with self.assertRaises(ValueError):
                nodes.parse_anchor_manifest(json.dumps([{**entry, "frame": True}]), 24, 1)
            with self.assertRaises(ValueError):
                nodes.parse_anchor_manifest(json.dumps([{**entry, "image": "../outside.png"}]), 24, 1)

    @unittest.skipUnless(os.environ.get("FFMPEG") or shutil.which("ffmpeg"), "ffmpeg is unavailable")
    def test_save_anchors_sections_and_bridge_assembly_with_real_media(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "input", root / "output"
            source.mkdir(); output.mkdir()
            nodes = load_nodes(source, output)
            nodes.Types.VideoContainer = types.SimpleNamespace(MP4="mp4")
            nodes.Types.VideoCodec = types.SimpleNamespace(H264="h264")
            nodes.io.FolderType = types.SimpleNamespace(output="output")
            nodes.ui.SavedResult = lambda *args: args
            nodes.ui.PreviewVideo = lambda *args: types.SimpleNamespace(as_dict=lambda: {"preview": True})
            nodes.InputImpl.VideoFromFile = lambda path: path
            @contextmanager
            def stage(context, name):
                yield None
            nodes.runtime.stage = stage
            nodes.runtime.finish = lambda context: {"test": "CPU media only"}
            nodes.runtime.failure_status = lambda exc: "failed"
            job_directory = "h3/dance/test-job"
            work = output / job_directory
            work.mkdir(parents=True)
            source_file = source / "dance.mp4"
            run([executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=24",
                 "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source_file)])
            sections = plan_sections(72, ["a", "b"], context_frames=4)
            for s in sections:
                s["kind"] = "stable"
            bridges = plan_transitions(sections)
            report = {"job_id": "test-job", "status": "running", "sections": sections + bridges,
                      "assembly_plan": assembly_plan(sections, bridges), "normalized_source": str(source_file),
                      "quality_mode": "preview", "audio_mode": "source", "frame_count": 72}
            report_file = work / "report.json"
            report_file.write_text(json.dumps(report))
            class Video:
                def __init__(self, path):
                    self.path = path
                def save_to(self, destination, **kwargs):
                    shutil.copy2(self.path, destination)
            for s in report["sections"]:
                index = s["index"]
                section_dir = work / f"section-{index}"
                section_dir.mkdir()
                ref = source / f"ref-{index}.mp4"
                slice_dance_video(source_file, ref, s["reference_start_frame"], s["generation_frames"], pad=True)
                if s["kind"] == "stable":
                    nodes.H3DanceSaveAnchors().save(Video(ref), "test", job_directory, index)
                    self.assertTrue(list(section_dir.glob("anchor-*.png")))
                nodes.H3DanceSaveSection().save(Video(ref), "test", job_directory, index)
            last = str((work / f"section-{bridges[-1]['index']}" / "kept.mp4").relative_to(output))
            result = nodes.H3DanceAssemble().assemble(job_directory, last)
            verify_output(result["result"][1], 72)
            self.assertTrue((work / "review.html").is_file())
            self.assertEqual(json.loads(report_file.read_text())["quality_status"], "needs_visual_review")
            # Strict mode must stop BEFORE assembly, while preserving completed
            # section clips. Explicit artifact-bound review can release a flag.
            from h3_pipeline.dance_resume import approve_sections
            report = json.loads(report_file.read_text())
            report.update(status="planned", quality_mode="pose_strict")
            for s in report["sections"]:
                s["quality"]["motion"] = {"status": "pass"}
            report["sections"][0]["quality"]["motion"]["status"] = "fail"
            report_file.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "sections \\[1\\]"):
                nodes.H3DanceAssemble().assemble(job_directory, last)
            self.assertEqual(json.loads(report_file.read_text())["status"], "failed")
            approve_sections(report_file, [1], "Reviewed the fixture")
            with patch.object(nodes, "review_section", return_value={"status": "needs_review", "motion": {"status": "pass"}}):
                nodes.H3DanceAssemble().assemble(job_directory, last)
            self.assertEqual(json.loads(report_file.read_text())["status"], "completed")
