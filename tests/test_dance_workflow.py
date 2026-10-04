import json
from pathlib import Path
import os
import shutil
import tempfile
import unittest

from h3_pipeline.dance_workflow import dance_canvas, plan_sections, section_references, dance_prompt
from h3_pipeline.dance_media import normalize_dance_video, slice_dance_video, assemble_dance_video, extract_frame
from h3_pipeline.media import executable, run, probe, verify_output


class DancePlanTests(unittest.TestCase):
    def assert_timeline(self, total, outfits, **options):
        sections = plan_sections(total, outfits, **options)
        self.assertEqual(sections[0]["start_frame"], 0)
        self.assertEqual(sections[-1]["end_frame"], total)
        self.assertEqual(sum(s["keep_frames"] for s in sections), total)
        for before, after in zip(sections, sections[1:]):
            self.assertEqual(before["end_frame"], after["start_frame"])
        for section in sections:
            self.assertEqual(section["generation_frames"] % 17, 5)
            self.assertLessEqual(section["generation_frames"], 362)
            self.assertGreaterEqual(section["trim_start_frame"], 0)
            self.assertLessEqual(section["trim_start_frame"] + section["keep_frames"], section["generation_frames"])
        return sections

    def test_arbitrary_outfit_counts_and_non_aligned_tail(self):
        for total in (5, 23, 304, 360, 1501):
            for count in (0, 1, 3, min(12, total)):
                outfits = [f"outfit-{i}.png" for i in range(count)]
                sections = self.assert_timeline(total, outfits)
                self.assertEqual(set(s["outfit_image"] for s in sections), set(outfits) if outfits else {None})
        self.assertEqual(dance_canvas(576, 768, 0.98), (576, 768))
        self.assertEqual(dance_canvas(1920, 1080, 0.98), (1344, 768))

    def test_custom_timeline_and_long_outfit_intervals(self):
        sections = self.assert_timeline(24 * 40, ["one", "two"], outfit_durations=[10, 30], max_section_seconds=15, context_frames=48)
        first = [s for s in sections if s["outfit_index"] == 1]
        self.assertEqual(first[-1]["end_frame"], 240)
        self.assertGreater(len(sections), 2)
        for options in ({"outfit_durations": [1]}, {"outfit_durations": [0, 10]},
                        {"outfit_durations": [1, 1]}, {"context_frames": 49}, {"max_section_seconds": float("nan")}):
            with self.assertRaises(ValueError):
                plan_sections(240, ["one", "two"], **options)
        with self.assertRaisesRegex(ValueError, "more outfits"):
            plan_sections(5, ["outfit"] * 6)

    def test_generation_ceiling_includes_context_and_alignment(self):
        for seconds in (1, 2, 5, 15):
            for context in (0, 4, 12, 48):
                if (int(seconds * 24) - 5) // 17 * 17 + 5 <= 2 * context:
                    with self.assertRaisesRegex(ValueError, "too short"):
                        plan_sections(720, ["outfit"], max_section_seconds=seconds, context_frames=context)
                    continue
                sections = self.assert_timeline(720, ["outfit"], max_section_seconds=seconds, context_frames=context)
                self.assertTrue(all(s["generation_frames"] <= seconds * 24 for s in sections))
                self.assertTrue(all(s["generation_seconds"] == s["generation_frames"] / 24 for s in sections))
        for seconds in (0.2, 0.8):
            with self.assertRaisesRegex(ValueError, "too short"):
                plan_sections(720, [], max_section_seconds=seconds)

    def test_wide_and_tall_canvases_follow_native_limits(self):
        for width, height in ((2560, 1080), (1080, 2560), (100000, 1), (1, 100000)):
            w, h = dance_canvas(width, height, 0.98)
            self.assertLessEqual(max(w, h), 16384)
            self.assertLessEqual(w * h, 1344 * 768)
        for width, height in ((0, 10), (-1, 20)):
            with self.assertRaises(ValueError):
                dance_canvas(width, height, 0.98)

    def test_all_character_background_combinations_and_clothing_roles(self):
        for character in (None, "replacement-person.png"):
            for background in (None, "replacement-scene.png"):
                for outfit in (None, "dress.png"):
                    refs = section_references("source-frame.png", character, background, outfit, "previous.png")
                    text, mapping = dance_prompt("Dance naturally", refs, True, bool(character), bool(background), bool(outfit))
                    self.assertEqual(refs[0].path, character or "source-frame.png")
                    self.assertEqual(refs[1].path, background or "source-frame.png")
                    self.assertIn("exact temporal order", text)
                    self.assertIn("overrides its pose", text)
                    self.assertIn("Replace the original dancer" if character else "Keep the original dancer", text)
                    self.assertIn("Replace the source background" if background else "Keep the original room", text)
                    self.assertIn("clothing photograph's person" if outfit else "Keep the original dancer's clothing", text)
                    self.assertEqual([r.tag for r in mapping[-2:]], ["<Audio 1>", "<Video 1>"])
        refs = section_references("anchor", None, None, None)
        with self.assertRaisesRegex(ValueError, "not connected"):
            dance_prompt("Use <Picture 9>", refs, False, False, False, False)


@unittest.skipUnless(os.environ.get("FFMPEG") or shutil.which("ffmpeg"), "ffmpeg is unavailable")
class DanceMediaTests(unittest.TestCase):
    def fixture(self, folder, audio=True):
        source = Path(folder) / "source.mp4"
        command = [executable("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=30"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-c:a", "aac"]
        run(command + ["-t", "3.1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)])
        return source

    def test_context_trim_and_assembly_keep_every_frame_and_source_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            source = self.fixture(work)
            normalized = work / "normalized.mp4"
            metadata = normalize_dance_video(source, normalized, 128, 96)
            total = int(metadata["streams"][0]["nb_frames"])
            self.assertEqual(total, 74)
            clips = []
            for s in plan_sections(total, ["a", "b", "c"], context_frames=4):
                reference = work / f"reference-{s['index']}.mp4"
                slice_dance_video(normalized, reference, s["reference_start_frame"], s["generation_frames"], pad=True)
                kept = work / f"kept-{s['index']}.mp4"
                slice_dance_video(reference, kept, s["trim_start_frame"], s["keep_frames"])
                verify_output(kept, s["keep_frames"])
                clips.append(kept)
            assembled = work / "assembled.mp4"
            assemble_dance_video(clips, assembled, total, normalized)
            verify_output(assembled, total)
            # Compare decoded source positions, not only counts: this detects dropped,
            # duplicated or restarted movement at cuts. Compression adds small error.
            import numpy as np
            from PIL import Image
            for frame in (0, 23, 24, 48, 49, total - 1):
                extract_frame(normalized, work / "before.png", frame)
                extract_frame(assembled, work / "after.png", frame)
                a = np.asarray(Image.open(work / "before.png"), dtype=float)
                b = np.asarray(Image.open(work / "after.png"), dtype=float)
                self.assertLess(abs(a - b).mean(), 5)
            decoded_audio = lambda path: __import__("subprocess").run(
                [executable("ffmpeg"), "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
                capture_output=True, check=True).stdout
            a = np.frombuffer(decoded_audio(normalized), dtype=np.float32)
            b = np.frombuffer(decoded_audio(assembled), dtype=np.float32)
            n = min(len(a), len(b))
            self.assertGreater(np.corrcoef(a[:n], b[:n])[0, 1], 0.99)

    def test_short_non_aligned_silent_source_pads_context_only(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            source = self.fixture(work, audio=False)
            normalized = work / "normalized.mp4"
            info = normalize_dance_video(source, normalized, 128, 96, start=0.5, duration=0.25)
            self.assertEqual(len(info["streams"]), 1)
            s = plan_sections(6, ["one"], context_frames=12)[0]
            self.assertGreater(s["padding_frames"], 0)
            reference = work / "reference.mp4"
            result = slice_dance_video(normalized, reference, 0, s["generation_frames"], pad=True)
            self.assertEqual(int(result["streams"][0]["nb_frames"]), s["generation_frames"])
            self.assertEqual(len(probe(reference)["streams"]), 1)


if __name__ == "__main__":
    unittest.main()
