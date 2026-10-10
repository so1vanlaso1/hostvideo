import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from h3_pipeline.dance_workflow import plan_sections, plan_transitions, assembly_plan, guide_frames, section_references, dance_prompt
from h3_pipeline.dance_quality import compare_poses, approved, sha256
from h3_pipeline.dance_resume import selected_sections, approve_sections, archive_section
from h3_pipeline.workflow import build_workflow, add_frame_guides, validate_graph
from h3_pipeline.wan_dance import wan_commands


class TransitionTests(unittest.TestCase):
    def test_bridges_replace_source_frames_without_overlap_or_time_shift(self):
        for total in (96, 304, 720, 1501):
            for count in (1, 2, 3, 5):
                stable = plan_sections(total, [str(i) for i in range(count)])
                bridges = plan_transitions(stable, 1)
                pieces = assembly_plan(stable, bridges)
                self.assertEqual(len(bridges), count - 1)
                timeline = [f for p in pieces for f in range(p["start_frame"], p["end_frame"])]
                self.assertEqual(timeline, list(range(total)))
                for bridge in bridges:
                    left, right = stable[bridge["left_section"] - 1], stable[bridge["right_section"] - 1]
                    self.assertGreaterEqual(bridge["start_frame"], left["start_frame"])
                    self.assertLessEqual(bridge["end_frame"], right["end_frame"])
                    self.assertEqual(bridge["generation_frames"] % 17, 5)
                    self.assertEqual(guide_frames(bridge), [0, bridge["keep_frames"] - 1])
                for a, b in zip(bridges, bridges[1:]):
                    self.assertLessEqual(a["end_frame"], b["start_frame"])

    def test_dense_outfits_reject_unrenderable_transformations(self):
        stable = plan_sections(5, ["a", "b", "c", "d", "e"])
        with self.assertRaisesRegex(ValueError, "too short"):
            plan_transitions(stable)
        self.assertEqual(plan_transitions(stable, 0), [])

    def test_transition_prompt_does_not_require_one_outfit_throughout(self):
        refs = section_references("person", None, None, "new", from_outfit="old")
        text, _ = dance_prompt("Preserve movement", refs, False, False, False, True, transition={"keep_frames": 24})
        self.assertIn("[video editing + reference generation]", text)
        self.assertIn("opening outfit from <Picture 4>", text)
        self.assertNotIn("throughout this section, from first to last frame", text)

    def test_guides_modify_conditioning_without_replacing_latent(self):
        graph = build_workflow(prompt="edit", images=[], video=None, audio=None, include_video_audio=False,
                               width=128, height=96, length=39, seed=1, job_id="guides")
        add_frame_guides(graph, [(12, "opening.png"), (24, "closing.png")])
        validate_graph(graph)
        self.assertEqual(graph["125"]["inputs"]["latent_image"], ["136", 1])
        self.assertEqual(graph["126"]["inputs"]["conditioning"], ["503", 0])
        self.assertEqual(graph["503"]["inputs"]["positive"], ["501", 0])
        with self.assertRaises(ValueError):
            add_frame_guides(graph, [(39, "bad.png")])
        with self.assertRaises(ValueError):
            add_frame_guides(graph, [(0, "a.png"), (0, "b.png")])


class PoseScoreTests(unittest.TestCase):
    def poses(self, frames=24):
        a = np.full((frames, 33, 3), 0.5)
        a[:, :, 2] = 1
        a[:, [11, 12], 1] = 0.3
        a[:, [23, 24], 1] = 0.6
        a[:, :, 0] += np.arange(frames)[:, None] * 0.002
        return a

    def test_exact_motion_passes_and_wrong_travel_fails(self):
        a = self.poses()
        self.assertEqual(compare_poses(a, a)["status"], "pass")
        b = a.copy()
        b[:, :, 0] += 0.2
        self.assertEqual(compare_poses(a, b)["status"], "fail")
        b = a[::-1].copy()
        b[12:, :, 0] += 0.3
        result = compare_poses(a, b)
        self.assertEqual(result["status"], "fail")
        self.assertGreater(result["p90_velocity_error"], 0)

    def test_missing_or_occluded_joints_cannot_pass(self):
        a = self.poses()
        b = a.copy()
        b[:, :, 2] = 0
        self.assertEqual(compare_poses(a, b)["status"], "inconclusive")
        b[:] = np.nan
        self.assertEqual(compare_poses(a, b)["status"], "inconclusive")
        with self.assertRaises(ValueError):
            compare_poses(a, a[:12])

    def test_short_boundary_jump_is_not_diluted_by_clip_percentiles(self):
        a = self.poses(120)
        b = a.copy()
        b[60, :, 0] += 0.3
        result = compare_poses(a, b)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["severe_jump_frames"], [60, 61])


class RerenderTests(unittest.TestCase):
    def test_only_same_outfit_dependents_and_bridges_are_invalidated(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "clip.mp4"
            clip.write_bytes(b"render")
            stable = plan_sections(720, ["a", "b", "c"])
            report = {"sections": stable + plan_transitions(stable)}
            for s in report["sections"]:
                s.update(status="completed", video_path=str(clip))
            chosen = selected_sections(report, [1])
            self.assertEqual({s["index"] for s in stable if s["index"] in chosen}, {s["index"] for s in stable if s["outfit_index"] == 1})
            self.assertIn(len(stable) + 1, chosen)
            self.assertNotIn(len(stable) + 2, chosen)

    def test_approval_is_bound_to_exact_video_and_archive_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            section = work / "section-1"
            section.mkdir()
            clip = section / "kept.mp4"
            clip.write_bytes(b"original")
            report_file = work / "report.json"
            report_file.write_text(json.dumps({"status": "failed", "sections": [{"index": 1, "status": "completed", "video_path": str(clip)}]}))
            report = approve_sections(report_file, [1], "Reviewed pose and clothing")
            self.assertTrue(approved(report["sections"][0]))
            archive = archive_section(work, 1)
            clip.write_bytes(b"replacement")
            self.assertFalse(approved(report["sections"][0]))
            self.assertEqual((archive / "kept.mp4").read_bytes(), b"original")

    def test_wan_uses_separate_environment_and_matching_upstream_fps(self):
        pre, gen = wan_commands({"wan_repository": "/wan", "checkpoint_directory": "/models", "python": "/wan/venv/python",
                                 "reference_video": "dance.mp4", "outfit_character_image": "edited.png"}, "/comparison")
        self.assertEqual(pre[0], gen[0])
        self.assertEqual(pre[pre.index("--fps") + 1], "30")
        self.assertIn("--replace_flag", pre)
        self.assertIn("--replace_flag", gen)
        self.assertNotIn("--use_relighting_lora", gen)
