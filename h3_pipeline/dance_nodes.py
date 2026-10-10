"""ComfyUI graph expansion for the reusable dance workflow. No client/CLI calls."""
from pathlib import Path
import json
import time
import uuid

import folder_paths
import torch
from comfy_api.latest import InputImpl, Types, io, ui
from comfy_execution.graph_utils import GraphBuilder, is_link

from . import server_runtime as runtime
from .config import DEFAULT_PROFILE, PROFILES, write_json
from .dance_media import normalize_dance_video, extract_frame, slice_dance_video, assemble_dance_video
from .dance_workflow import KEEP_ORIGINAL, dance_canvas, plan_sections, section_references, dance_prompt, plan_transitions, assembly_plan, guide_frames
from .media import prepare_image, probe, verify_output
from .references import FPS
from .workflow import build_workflow, add_frame_guides
from .dance_reports import register_report, fail_report
from .dance_quality import review_section, require_pose, approved, write_review_index, download_pose_model

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_LOADER_TYPES = {"H3UNETLoader", "H3CLIPLoader", "H3VAELoader", "H3LoraLoader"}


def add_section_graph(graph, template, index, shared_models):
    """Load one model set per dance job, even with ComfyUI's cache disabled.

    The native prompt tracker retains dynamic models until the entire expanded
    prompt ends. Separate loaders per section therefore retain separate copies
    of every checkpoint. Link all sections to the first section's loaders.
    """
    copied = {}
    reused = set()
    for key, node in template.items():
        if node["class_type"] in MODEL_LOADER_TYPES:
            if key in shared_models:
                copied[key] = shared_models[key]
                reused.add(key)
                continue
            copied[key] = graph.node(node["class_type"], id=f"models_{key}")
            shared_models[key] = copied[key]
        else:
            copied[key] = graph.node(node["class_type"], id=f"section{index}_{key}")
    for key, node in template.items():
        if key in reused:
            continue  # Keep shared loader dependencies on the first section.
        for field, value in node["inputs"].items():
            copied[key].set_input(field, copied[value[0]].out(value[1]) if is_link(value) else value)
    return copied


def input_files(suffixes):
    root = Path(folder_paths.get_input_directory())
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in suffixes and "h3-dance" not in p.relative_to(root).parts)


def input_path(name):
    root = Path(folder_paths.get_input_directory()).resolve()
    path = Path(folder_paths.get_annotated_filepath(name)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Select an uploaded file in ComfyUI's input directory: {name}")
    return path


def output_path(name):
    root = Path(folder_paths.get_output_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root / "h3" / "dance"):
        raise ValueError("Dance outputs must be inside output/h3/dance")
    return path


def parse_outfits(value):
    value = value.strip()
    outfits = json.loads(value) if value.startswith("[") else [line.strip() for line in value.splitlines() if line.strip()]
    if not isinstance(outfits, list) or any(not isinstance(p, str) or not p for p in outfits):
        raise ValueError("Outfit images must be a JSON array or one uploaded filename per line")
    for name in outfits:
        input_path(name)
    return outfits


def parse_durations(value):
    durations = json.loads(value or "[]")
    if not isinstance(durations, list):
        raise ValueError("Outfit durations must be a JSON array of seconds, or [] for equal time")
    return durations


def parse_anchor_manifest(value, total_frames, outfit_count):
    anchors = json.loads(value or "[]")
    if not isinstance(anchors, list):
        raise ValueError("anchor_manifest must be a JSON array")
    seen = set()
    for a in anchors:
        if not isinstance(a, dict) or set(a) != {"frame", "outfit_index", "image"}:
            raise ValueError("Each anchor requires exactly frame, outfit_index and image")
        for field, maximum in (("frame", total_frames - 1), ("outfit_index", outfit_count)):
            minimum = 0 if field == "frame" else 1
            if isinstance(a[field], bool) or not isinstance(a[field], int) or not minimum <= a[field] <= maximum:
                raise ValueError(f"Invalid anchor {field}")
        input_path(a["image"])
        key = (a["frame"], a["outfit_index"])
        if key in seen:
            raise ValueError("Duplicate anchor for the same outfit and source frame")
        seen.add(key)
    return anchors


class H3DanceWorkflow:
    @classmethod
    def INPUT_TYPES(cls):
        images = [KEEP_ORIGINAL] + input_files(IMAGE_SUFFIXES)
        return {"required": {
            "reference_video": (["Choose dance video"] + input_files(VIDEO_SUFFIXES), {"video_upload": True}),
            "character_image": (images, {"tooltip": "Keep the source dancer, or choose a replacement character."}),
            "background_image": (images, {"tooltip": "Keep the source scene, or choose a replacement background."}),
            "outfit_images": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Upload/select outfits below, in their appearance order. Empty keeps source clothing."}),
            "prompt": ("STRING", {"default": "A realistic full-body dance with natural motion and consistent appearance.", "multiline": True}),
            "seed": ("INT", {"default": 123456789, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
            "megapixels": ("FLOAT", {"default": 0.98, "min": 0.1, "max": 1, "step": 0.01}),
            "max_section_seconds": ("FLOAT", {"default": 5, "min": 0.2, "max": 15, "step": 0.1}),
            "audio_mode": (["source", "generated"], {"tooltip": "Source preserves the dance soundtrack; silent sources use generated music."}),
            "profile": ([DEFAULT_PROFILE] + [p for p in PROFILES if p != DEFAULT_PROFILE],),
            "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "turbo": ("BOOLEAN", {"default": False}),
            "context_frames": ("INT", {"default": 12, "min": 0, "max": 48, "advanced": True}),
            "outfit_durations": ("STRING", {"default": "[]", "multiline": True, "advanced": True, "tooltip": "Optional seconds per outfit, covering the full selected video."}),
            "start_seconds": ("FLOAT", {"default": 0, "min": 0, "advanced": True}),
            "duration_seconds": ("FLOAT", {"default": 0, "min": 0, "advanced": True, "tooltip": "0 uses the rest of the source video."})},
            "optional": {
            "transition_seconds": ("FLOAT", {"default": 1.0, "min": 0, "max": 3, "step": 0.1, "tooltip": "A separate anchored transformation replaces frames around each outfit change. 0 uses cuts."}),
            "anchor_mode": (["auto", "supplied", "off"], {"tooltip": "Auto generates outfit-correct endpoints in a first pass, then refines. Supplied uses edited pose-matched images."}),
            "anchor_manifest": ("STRING", {"default": "[]", "multiline": True, "advanced": True, "tooltip": 'JSON: [{"frame":0,"outfit_index":1,"image":"edited.png"}]. Frames refer to the selected normalized source, not generation context.'}),
            "quality_mode": (["pose_warn", "preview", "pose_strict"], {"tooltip": "CPU pose screening is on by default. Strict blocks failed/inconclusive checks; preview needs no pose dependencies."}),
            "pose_model": ("STRING", {"default": "", "advanced": True, "tooltip": "Server path to a MediaPipe .task file. Empty downloads the verified small model once."})},
            "hidden": {"unique_id": "UNIQUE_ID"}}
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("assembled_video", "output_path")
    FUNCTION = "expand"
    CATEGORY = "H3 Pipeline/Dance"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, reference_video, character_image, background_image, outfit_images, outfit_durations="[]"):
        try:
            input_path(reference_video)
            for name in (character_image, background_image):
                if name != KEEP_ORIGINAL:
                    input_path(name)
            parse_outfits(outfit_images)
            parse_durations(outfit_durations)
        except (ValueError, OSError) as exc:
            return str(exc)
        return True

    def expand(self, reference_video, character_image, background_image, outfit_images, prompt, seed,
               megapixels, max_section_seconds, audio_mode, profile, steps, turbo, context_frames=12,
               outfit_durations="[]", start_seconds=0, duration_seconds=0, unique_id=None,
               transition_seconds=1.0, anchor_mode="auto", anchor_manifest="[]", quality_mode="pose_warn", pose_model=""):
        from PIL import Image
        if not prompt.strip():
            raise ValueError("Enter a nonempty dance prompt")
        if turbo and steps != 4:
            raise ValueError("Turbo requires exactly four steps")
        if anchor_mode not in {"auto", "supplied", "off"} or quality_mode not in {"preview", "pose_warn", "pose_strict"}:
            raise ValueError("Invalid anchor or quality mode")
        if quality_mode.startswith("pose"):
            if not pose_model:
                pose_model = download_pose_model(Path(folder_paths.get_input_directory()) / "h3-dance-quality" / "pose_landmarker_lite.task")["path"]
            require_pose(pose_model)
        source = input_path(reference_video)
        character = None if character_image == KEEP_ORIGINAL else str(input_path(character_image))
        background = None if background_image == KEEP_ORIGINAL else str(input_path(background_image))
        outfits = parse_outfits(outfit_images)
        durations = parse_durations(outfit_durations)
        for name in [character, background] + [str(input_path(p)) for p in outfits]:
            if name:
                with Image.open(name) as image:
                    image.verify()
        metadata = probe(source)
        video = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
        if not video:
            raise ValueError("The reference file has no video stream")
        iw, ih = video["width"], video["height"]
        rotation = next((s.get("rotation", 0) for s in video.get("side_data_list", []) if "rotation" in s), 0)
        if abs(rotation) % 180 == 90:
            iw, ih = ih, iw
        width, height = dance_canvas(iw, ih, megapixels)
        job_id = "dance-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        job_directory = "h3/dance/" + job_id
        work = output_path(job_directory)
        work.mkdir(parents=True)
        inputs = Path(folder_paths.get_input_directory()) / "h3-dance" / job_id
        inputs.mkdir(parents=True)
        normalized = inputs / "source.mp4"
        normalized_info = normalize_dance_video(source, normalized, width, height, start_seconds, duration_seconds or None)
        total_frames = int(next(s for s in normalized_info["streams"] if s["codec_type"] == "video")["nb_frames"])
        paired = any(s["codec_type"] == "audio" for s in normalized_info["streams"])
        sections = plan_sections(total_frames, outfits, durations, max_section_seconds, context_frames)
        for s in sections:
            s["kind"] = "stable"
        transitions = plan_transitions(sections, transition_seconds)
        for s in transitions:
            if s["generation_seconds"] > max_section_seconds:
                raise ValueError("Transition render exceeds max_section_seconds; shorten transition_seconds")
        supplied = parse_anchor_manifest(anchor_manifest, total_frames, max(1, len(outfits)))
        for a in supplied:
            with Image.open(input_path(a["image"])) as image:
                if image.size != (width, height):
                    raise ValueError(f"Edited anchor {a['image']} must match the output canvas {width}x{height}; cropping would change the pose")
        if anchor_mode == "supplied":
            for s in sections:
                required = {s["start_frame"], s["end_frame"] - 1}
                available = {a["frame"] for a in supplied if a["outfit_index"] == s["outfit_index"]}
                if not required <= available:
                    raise ValueError(f"Supplied anchor mode requires core endpoint images for section {s['index']}: {sorted(required - available)}")
        anchor = extract_frame(normalized, inputs / "source-anchor.png", 0)
        report = {"job_id": job_id, "status": "planned", "reference_video": str(source),
                  "character_image": character, "background_image": background, "outfit_images": outfits,
                  "frame_count": total_frames, "duration": total_frames / FPS, "width": width, "height": height,
                  "source_start_seconds": start_seconds, "seed": seed, "steps": steps, "turbo": turbo,
                  "profile": profile, "audio_mode": "source" if paired and audio_mode == "source" else "generated",
                  "max_section_seconds": max_section_seconds, "context_frames": context_frames,
                  "normalized_source": str(normalized), "sections": sections,
                  "transitions": transitions, "assembly_plan": assembly_plan(sections, transitions),
                  "anchor_mode": anchor_mode, "anchor_manifest": supplied, "prompt": prompt,
                  "quality_mode": quality_mode, "pose_model": pose_model,
                  "transition_seconds": transition_seconds,
                  "assembly": "Exact source timeline; dedicated anchored bridges replace outfit boundaries. Visual review required."}
        report["sections"] = sections + transitions
        write_json(work / "report.json", report)
        register_report(work / "report.json")
        graph = GraphBuilder()
        shared_models = {}
        previous = None
        for section in sections + transitions:
            index = section["index"]
            section_dir = work / f"section-{index}"
            section_dir.mkdir()
            reference = inputs / f"section-{index}.mp4"
            slice_dance_video(normalized, reference, section["reference_start_frame"], section["generation_frames"], pad=True)
            previous_frame = None
            refs = section_references(anchor, character, background,
                                      str(input_path(section["outfit_image"])) if section["outfit_image"] else None,
                                      previous_frame, str(input_path(section["from_outfit_image"])) if section.get("from_outfit_image") else None)
            effective_prompt, mapping = dance_prompt(prompt, refs, paired, bool(character), bool(background), bool(outfits), reference,
                                                      transition=section if section["kind"] == "transition" else None)
            # Preprocess selected references once; the continuity frame is produced
            # later by the preceding section and loaded only after its gate.
            image_paths = []
            for i, ref in enumerate(refs):
                if previous_frame and ref.path == str(previous_frame):
                    image_paths.append(str(previous_frame))
                else:
                    prepared = inputs / f"section-{index}-picture-{i + 1}.png"
                    prepare_image(ref.path, prepared, width, height, "max")
                    image_paths.append(str(prepared))
            template = build_workflow(prompt=effective_prompt, images=image_paths,
                video=str(reference.relative_to(folder_paths.get_input_directory())), audio=None,
                include_video_audio=paired, width=width, height=height, length=section["generation_frames"],
                seed=seed, job_id=f"{job_id}-section-{index}", profile=profile, steps=steps, turbo=turbo)
            template["200"] = {"class_type": "H3DanceBeginSection", "inputs": {
                "job_directory": job_directory, "section_index": index}}
            # The loader's 0.1 MP floor does not upscale a smaller source; retain
            # small source canvases while satisfying its existing public schema.
            template["310"]["inputs"]["megapixels"] = max(0.1, width * height / 1024**2)
            template["201"]["inputs"]["reference_mapping"] = json.dumps([r.dict() for r in mapping])
            for node in template.values():
                if node["class_type"] == "LoadImage":
                    node["class_type"] = "H3DanceLoadImage"
                    node["inputs"]["context"] = ["200", 0]
            template["92"] = {"class_type": "H3DanceSaveSection", "inputs": {
                "video": ["130", 0], "context": ["201", 0], "job_directory": job_directory, "section_index": index}}
            guides = []
            if section["kind"] == "transition":
                # Images are extracted at the very same source timestamps from
                # adjacent stable renders, rather than their unrelated ends.
                before = sections[section["left_section"] - 1]
                after = sections[section["right_section"] - 1]
                for frame, parent in ((section["start_frame"], before), (section["end_frame"] - 1, after)):
                    path = section_dir / f"bridge-anchor-{frame}.png"
                    node_id = str(450 + len(guides))
                    template[node_id] = {"class_type": "H3DanceExtractGuide", "inputs": {
                        "video_path": str(work / f"section-{parent['index']}" / "generated.mp4"),
                        "frame": frame - parent["reference_start_frame"], "image_path": str(path), "context": ["200", 0]}}
                    guides.append((frame - section["reference_start_frame"], path))
                add_frame_guides(template, guides)
                for i in range(len(guides)):
                    template[str(500 + 2*i)]["inputs"]["dependency"] = [str(450 + i), 0]
            else:
                chosen = [a for a in supplied if a["outfit_index"] == section["outfit_index"] and
                          section["reference_start_frame"] <= a["frame"] < section["reference_start_frame"] + section["generation_frames"]]
                guides = [(a["frame"] - section["reference_start_frame"], input_path(a["image"])) for a in chosen]
                # Carry a pose from the preceding same-outfit render's FUTURE
                # context at this core's opening timestamp. Never anchor old
                # clothes into a new outfit or mistake last-frame for t=0.
                if index > 1 and anchor_mode != "off":
                    before = sections[index - 2]
                    frame = section["start_frame"]
                    if before["outfit_index"] == section["outfit_index"] and frame < before["reference_start_frame"] + before["generation_frames"] and not any(a["frame"] == frame for a in chosen):
                        image_path = section_dir / "continuity-guide.png"
                        template["450"] = {"class_type": "H3DanceExtractGuide", "inputs": {
                            "video_path": str(work / f"section-{before['index']}" / "generated.mp4"),
                            "frame": frame - before["reference_start_frame"], "image_path": str(image_path), "context": ["200", 0]}}
                        guides.append((frame - section["reference_start_frame"], image_path))
                if anchor_mode == "auto":
                    # The prepass uses the same outfit and source timeline.
                    # It is not an independent image edit, so pose QA must still
                    # screen its anchors before they are reused.
                    import copy
                    prepass = copy.deepcopy(template)
                    prepass["200"]["inputs"]["phase"] = "anchor"
                    prepass["92"]["class_type"] = "H3DanceSaveAnchors"
                    if guides:
                        add_frame_guides(prepass, guides)
                        if "450" in prepass:
                            prepass[str(500 + 2*(len(guides)-1))]["inputs"]["dependency"] = ["450", 0]
                    write_json(section_dir / "anchor-workflow-api.json", prepass)
                    copied_anchor = add_section_graph(graph, prepass, f"{index}_anchor", shared_models)
                    if previous:
                        copied_anchor["200"].set_input("previous_section", previous)
                    existing = {g[0] for g in guides}
                    guides += [(f, section_dir / f"anchor-{f}.png") for f in guide_frames(section) if f not in existing]
                if guides:
                    add_frame_guides(template, guides)
                    if "450" in template:
                        continuity_index = next(i for i, (_, p) in enumerate(guides) if str(p).endswith("continuity-guide.png"))
                        template[str(500 + 2*continuity_index)]["inputs"]["dependency"] = ["450", 0]
            section["guides"] = [{"frame": f, "image": str(p)} for f, p in guides]
            write_json(section_dir / "workflow-api.json", template)
            write_json(section_dir / "references.json", [r.dict() for r in mapping])
            (section_dir / "prompt.txt").write_text(effective_prompt + "\n")
            copied = add_section_graph(graph, template, index, shared_models)
            if section["kind"] == "stable" and anchor_mode == "auto":
                copied["200"].set_input("anchor_pass", copied_anchor["92"].out(0))
            if previous:
                copied["200"].set_input("previous_section", previous)
            previous = copied["92"].out(0)
        write_json(work / "report.json", report)
        assembled = graph.node("H3DanceAssemble", job_directory=job_directory, last_section=previous)
        if unique_id:
            assembled.set_override_display_id(unique_id)
        write_json(work / "expanded-workflow-api.json", graph.finalize())
        return {"result": (assembled.out(0), assembled.out(1)), "expand": graph.finalize()}


class H3DanceBeginSection:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_directory": ("STRING",), "section_index": ("INT", {"min": 1})},
                "optional": {"previous_section": ("STRING", {"forceInput": True}),
                             "anchor_pass": ("STRING", {"forceInput": True}), "phase": ("STRING", {"default": "render"})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "begin"
    CATEGORY = "H3 Pipeline/Dance"

    def begin(self, job_directory, section_index, previous_section=None, anchor_pass=None, phase="render"):
        work = output_path(job_directory)
        if previous_section and not output_path(previous_section).is_file():
            raise RuntimeError("The preceding dance section has not been saved")
        if anchor_pass and not output_path(anchor_pass).is_file():
            raise RuntimeError("The anchor pass has not been saved")
        report = json.loads((work / "report.json").read_text())
        register_report(work / "report.json")
        try:
            context = runtime.begin(f"{report['job_id']}-section-{section_index}-{phase}", folder_paths.get_output_directory(), 0,
                                    report_path=work / "report.json", section_index=section_index)
        except BaseException as exc:
            fail_report(work / "report.json", str(exc), runtime.failure_status(exc), section_index, "section_start")
            raise
        report["status"] = "running"
        report["sections"][section_index - 1]["status"] = "running"
        report["sections"][section_index - 1]["phase"] = phase
        write_json(work / "report.json", report)
        return (context,)


class H3DanceLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("STRING",), "context": ("STRING", {"forceInput": True})},
                "optional": {"dependency": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "H3 Pipeline/Dance"

    def load(self, image, context, dependency=None):
        from PIL import Image, ImageOps
        import numpy as np
        path = Path(image).resolve()
        input_root = Path(folder_paths.get_input_directory()).resolve()
        output_root = Path(folder_paths.get_output_directory()).resolve() / "h3" / "dance"
        if not (path.is_relative_to(input_root) or path.is_relative_to(output_root)):
            raise ValueError("Reference images must be inside ComfyUI input or dance output")
        with runtime.stage(context, "reference_image_preprocessing"):
            with Image.open(path) as source:
                pixels = np.array(ImageOps.exif_transpose(source).convert("RGB"))
            return (torch.from_numpy(pixels).to(dtype=torch.float16).div_(255).unsqueeze(0),)


class H3DanceExtractGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING",), "image_path": ("STRING",),
                             "frame": ("INT", {"min": 0}), "context": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "extract"
    CATEGORY = "H3 Pipeline/Dance"

    def extract(self, video_path, image_path, frame, context):
        root = Path(folder_paths.get_output_directory()).resolve() / "h3" / "dance"
        if any(not Path(p).resolve().is_relative_to(root) for p in (video_path, image_path)):
            raise ValueError("Bridge guides must belong to the dance output directory")
        with runtime.stage(context, "bridge_guide_extraction"):
            extract_frame(video_path, image_path, frame)
        return (image_path,)


class H3DanceSaveAnchors:
    @classmethod
    def INPUT_TYPES(cls):
        return H3DanceSaveSection.INPUT_TYPES()
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    CATEGORY = "H3 Pipeline/Dance"

    def save(self, video, context, job_directory, section_index):
        work = output_path(job_directory)
        report = json.loads((work / "report.json").read_text())
        section = report["sections"][section_index - 1]
        folder = work / f"section-{section_index}"
        raw = folder / "anchor-pass.mp4"
        try:
            with runtime.stage(context, "anchor_video_encoding"):
                video.save_to(str(raw), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=16)
                verify_output(raw, section["generation_frames"])
                for frame in guide_frames(section):
                    extract_frame(raw, folder / f"anchor-{frame}.png", frame)
                if report.get("quality_mode", "preview").startswith("pose"):
                    source_core, anchor_core = folder / "source-core.mp4", folder / "anchor-core.mp4"
                    slice_dance_video(report["normalized_source"], source_core, section["start_frame"], section["keep_frames"])
                    slice_dance_video(raw, anchor_core, section["trim_start_frame"], section["keep_frames"])
                    quality = review_section(source_core, anchor_core, folder / "anchor-review", pose_model=report["pose_model"], mode=report["quality_mode"])
                    section["anchor_quality"] = quality
                    write_json(work / "report.json", report)
                    if report["quality_mode"] == "pose_strict" and quality["motion"]["status"] != "pass":
                        raise RuntimeError("Anchor pass failed pose screening; rerender this section or supply corrected edited anchors")
            section.update(status="anchored", anchor_video_path=str(raw), anchor_server=runtime.finish(context))
            write_json(work / "report.json", report)
        except BaseException as exc:
            fail_report(work / "report.json", str(exc), runtime.failure_status(exc), section_index, "anchor_generation")
            raise
        return (str(raw.relative_to(Path(folder_paths.get_output_directory()).resolve())),)


class H3DanceSaveSection:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "context": ("STRING", {"forceInput": True}),
                             "job_directory": ("STRING",), "section_index": ("INT", {"min": 1})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    CATEGORY = "H3 Pipeline/Dance"

    def save(self, video, context, job_directory, section_index):
        work = output_path(job_directory)
        report = json.loads((work / "report.json").read_text())
        section = report["sections"][section_index - 1]
        folder = work / f"section-{section_index}"
        try:
            with runtime.stage(context, "video_encoding"):
                raw, kept = folder / "generated.mp4", folder / "kept.mp4"
                video.save_to(str(raw), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=16)
                verify_output(raw, section["generation_frames"])
                slice_dance_video(raw, kept, section["trim_start_frame"], section["keep_frames"])
                verify_output(kept, section["keep_frames"])
                extract_frame(kept, folder / "last-frame.png", section["keep_frames"] - 1)
            with runtime.stage(context, "dance_quality_review"):
                source_core = folder / "source-core.mp4"
                slice_dance_video(report["normalized_source"], source_core, section["start_frame"], section["keep_frames"])
                section["quality"] = review_section(source_core, kept, folder, pose_model=report.get("pose_model"), mode=report.get("quality_mode", "preview"))
                section.pop("approval", None)
            section.update(status="completed", video_path=str(kept), server=runtime.finish(context))
            write_json(work / "report.json", report)
        except BaseException as exc:
            fail_report(work / "report.json", str(exc), runtime.failure_status(exc), section_index, "video_encoding")
            raise
        return (str(kept.relative_to(Path(folder_paths.get_output_directory()).resolve())),)


class H3DanceAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_directory": ("STRING",), "last_section": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("VIDEO", "STRING")
    FUNCTION = "assemble"
    CATEGORY = "H3 Pipeline/Dance"
    OUTPUT_NODE = True

    def assemble(self, job_directory, last_section):
        work = output_path(job_directory)
        report = json.loads((work / "report.json").read_text())
        destination = work / "generated.mp4"
        clips = [work / f"section-{s['index']}" / "kept.mp4" for s in report["sections"]]
        try:
            if not output_path(last_section).is_file():
                raise RuntimeError("The final dance section is missing")
            if any(s.get("status") != "completed" for s in report["sections"]):
                raise RuntimeError("Every dance section must complete before assembly")
            if report.get("quality_mode") == "pose_strict":
                flagged = [s["index"] for s in report["sections"] if s.get("quality", {}).get("motion", {}).get("status") != "pass" and not approved(s)]
                if flagged:
                    raise RuntimeError(f"Pose checks failed or were inconclusive in sections {flagged}. Review their previews and approve or rerender before assembly")
            if report.get("assembly_plan"):
                pieces = []
                for i, piece in enumerate(report["assembly_plan"]):
                    dest = work / f"assembly-piece-{i}.mp4"
                    slice_dance_video(work / f"section-{piece['section_index']}" / "kept.mp4", dest, piece["clip_start_frame"], piece["frames"])
                    pieces.append(dest)
                clips = pieces
            media = assemble_dance_video(clips, destination, report["frame_count"],
                report["normalized_source"] if report["audio_mode"] == "source" else None)
            report.update(status="completed", video_path=str(destination), media=media)
            report["final_quality"] = review_section(report["normalized_source"], destination, work,
                pose_model=report.get("pose_model"), mode=report.get("quality_mode", "preview"))
            report["quality_flags"] = [s["index"] for s in report["sections"] if s.get("quality", {}).get("status") == "flagged" and not approved(s)]
            report["quality_status"] = "flagged" if report["final_quality"]["status"] == "flagged" or report["quality_flags"] else "needs_visual_review"
            report["review_index"] = write_review_index(report, work)
            write_json(work / "report.json", report)
        except BaseException as exc:
            fail_report(work / "report.json", str(exc), runtime.failure_status(exc), stage="assembly")
            raise
        preview = ui.PreviewVideo([ui.SavedResult(destination.name, job_directory, io.FolderType.output)]).as_dict()
        return {"result": (InputImpl.VideoFromFile(str(destination)), str(destination)), "ui": preview}


class H3DanceResume:
    """Rerender selected sections without repeating accepted independent outfits."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_directory": ("STRING",), "sections": ("STRING", {"default": "[]"}),
                             "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})}}
    RETURN_TYPES = ("VIDEO", "STRING")
    FUNCTION = "expand"
    CATEGORY = "H3 Pipeline/Dance"
    OUTPUT_NODE = True

    def expand(self, job_directory, sections="[]", seed=-1):
        from .dance_resume import selected_sections, archive_section, archive_job
        work = output_path(job_directory)
        report = json.loads((work / "report.json").read_text())
        if report["status"] in {"running", "planned"}:
            raise RuntimeError("This job is active; inspect ComfyUI before resubmitting")
        selected = selected_sections(report, json.loads(sections))
        if seed < -1:
            raise ValueError("seed must be -1 or a nonnegative integer")
        graph, shared, previous = GraphBuilder(), {}, None
        register_report(work / "report.json")
        archive_job(work)
        # Preserve audit evidence BEFORE replacing a selected render.
        for section in report["sections"]:
            if section["index"] in selected:
                archive_section(work, section["index"])
                section["status"] = "planned"
                section.pop("approval", None)
        report.update(status="planned")
        for field in ("error", "failed_stage", "failed_section", "video_path", "final_quality", "quality_status"):
            report.pop(field, None)
        write_json(work / "report.json", report)
        for section in report["sections"]:
            index = section["index"]
            if index not in selected:
                continue
            folder = work / f"section-{index}"
            template = json.loads((folder / "workflow-api.json").read_text())
            if seed != -1:
                template["129"]["inputs"]["noise_seed"] = seed
                template["201"]["inputs"]["seed"] = seed
                section["seed"] = seed
            if section.get("kind") != "transition" and report.get("anchor_mode") == "auto":
                prepass = json.loads((folder / "anchor-workflow-api.json").read_text())
                if seed != -1:
                    prepass["129"]["inputs"]["noise_seed"] = seed
                    prepass["201"]["inputs"]["seed"] = seed
                anchor = add_section_graph(graph, prepass, f"{index}_anchor", shared)
                if previous:
                    anchor["200"].set_input("previous_section", previous)
                final = add_section_graph(graph, template, index, shared)
                final["200"].set_input("anchor_pass", anchor["92"].out(0))
                write_json(folder / "anchor-workflow-api.json", prepass)
            else:
                final = add_section_graph(graph, template, index, shared)
            if previous:
                final["200"].set_input("previous_section", previous)
            previous = final["92"].out(0)
            write_json(folder / "workflow-api.json", template)
        write_json(work / "report.json", report)
        last = previous or str(Path(report["sections"][-1]["video_path"]).relative_to(Path(folder_paths.get_output_directory()).resolve()))
        assembled = graph.node("H3DanceAssemble", job_directory=job_directory, last_section=last)
        write_json(work / "resume-workflow-api.json", graph.finalize())
        return {"result": (assembled.out(0), assembled.out(1)), "expand": graph.finalize()}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3DanceWorkflow, H3DanceBeginSection, H3DanceLoadImage, H3DanceExtractGuide, H3DanceSaveAnchors, H3DanceSaveSection, H3DanceAssemble, H3DanceResume)}
