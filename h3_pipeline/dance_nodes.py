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
from .dance_workflow import KEEP_ORIGINAL, dance_canvas, plan_sections, section_references, dance_prompt
from .media import prepare_image, probe, verify_output
from .references import FPS
from .workflow import build_workflow
from .dance_reports import register_report, fail_report

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
               outfit_durations="[]", start_seconds=0, duration_seconds=0, unique_id=None):
        from PIL import Image
        if not prompt.strip():
            raise ValueError("Enter a nonempty dance prompt")
        if turbo and steps != 4:
            raise ValueError("Turbo requires exactly four steps")
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
        anchor = extract_frame(normalized, inputs / "source-anchor.png", 0)
        report = {"job_id": job_id, "status": "planned", "reference_video": str(source),
                  "character_image": character, "background_image": background, "outfit_images": outfits,
                  "frame_count": total_frames, "duration": total_frames / FPS, "width": width, "height": height,
                  "source_start_seconds": start_seconds, "seed": seed, "steps": steps, "turbo": turbo,
                  "profile": profile, "audio_mode": "source" if paired and audio_mode == "source" else "generated",
                  "max_section_seconds": max_section_seconds, "context_frames": context_frames,
                  "normalized_source": str(normalized), "sections": sections,
                  "assembly": "Chronological cuts, exact core frames; context/padding excluded. Model pose fidelity requires visual review."}
        write_json(work / "report.json", report)
        register_report(work / "report.json")
        graph = GraphBuilder()
        shared_models = {}
        previous = None
        for section in sections:
            index = section["index"]
            section_dir = work / f"section-{index}"
            section_dir.mkdir()
            reference = inputs / f"section-{index}.mp4"
            slice_dance_video(normalized, reference, section["reference_start_frame"], section["generation_frames"], pad=True)
            previous_frame = work / f"section-{index - 1}" / "last-frame.png" if index > 1 else None
            refs = section_references(anchor, character, background,
                                      str(input_path(section["outfit_image"])) if section["outfit_image"] else None,
                                      previous_frame)
            effective_prompt, mapping = dance_prompt(prompt, refs, paired, bool(character), bool(background), bool(outfits), reference)
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
            write_json(section_dir / "workflow-api.json", template)
            write_json(section_dir / "references.json", [r.dict() for r in mapping])
            (section_dir / "prompt.txt").write_text(effective_prompt + "\n")
            copied = add_section_graph(graph, template, index, shared_models)
            if previous:
                copied["200"].set_input("previous_section", previous)
            previous = copied["92"].out(0)
        assembled = graph.node("H3DanceAssemble", job_directory=job_directory, last_section=previous)
        if unique_id:
            assembled.set_override_display_id(unique_id)
        write_json(work / "expanded-workflow-api.json", graph.finalize())
        return {"result": (assembled.out(0), assembled.out(1)), "expand": graph.finalize()}


class H3DanceBeginSection:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_directory": ("STRING",), "section_index": ("INT", {"min": 1})},
                "optional": {"previous_section": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "begin"
    CATEGORY = "H3 Pipeline/Dance"

    def begin(self, job_directory, section_index, previous_section=None):
        work = output_path(job_directory)
        if previous_section and not output_path(previous_section).is_file():
            raise RuntimeError("The preceding dance section has not been saved")
        report = json.loads((work / "report.json").read_text())
        register_report(work / "report.json")
        try:
            context = runtime.begin(f"{report['job_id']}-section-{section_index}", folder_paths.get_output_directory(), 0,
                                    report_path=work / "report.json", section_index=section_index)
        except BaseException as exc:
            fail_report(work / "report.json", str(exc), runtime.failure_status(exc), section_index, "section_start")
            raise
        report["status"] = "running"
        report["sections"][section_index - 1]["status"] = "running"
        write_json(work / "report.json", report)
        return (context,)


class H3DanceLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("STRING",), "context": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "H3 Pipeline/Dance"

    def load(self, image, context):
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
            media = assemble_dance_video(clips, destination, report["frame_count"],
                report["normalized_source"] if report["audio_mode"] == "source" else None)
            report.update(status="completed", video_path=str(destination), media=media)
            write_json(work / "report.json", report)
        except BaseException as exc:
            fail_report(work / "report.json", str(exc), runtime.failure_status(exc), stage="assembly")
            raise
        preview = ui.PreviewVideo([ui.SavedResult(destination.name, job_directory, io.FolderType.output)]).as_dict()
        return {"result": (InputImpl.VideoFromFile(str(destination)), str(destination)), "ui": preview}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3DanceWorkflow, H3DanceBeginSection, H3DanceLoadImage, H3DanceSaveSection, H3DanceAssemble)}
