"""Separate one-pass dance alternative with endpoint-anchored join renders."""
import math
from .dance_nodes import *

def plan_transitions(sections, seconds=1.0):
    """Separate bridge renders overwrite (never insert) source-timeline frames.

    Clamp each half-window to its adjacent core. This makes bridges disjoint
    even for very short outfit intervals and keeps their endpoints available in
    the corresponding stable renders, including at the ends of the source.
    """
    if not math.isfinite(seconds) or not 0 <= seconds <= 3:
        raise ValueError("transition_seconds must be between 0 and 3")
    if seconds == 0:
        return []
    half = max(1, round(seconds * FPS / 2))
    bridges = []
    for before, after in zip(sections, sections[1:]):
        left = min(half, max(0, (before["keep_frames"] - 1) // 2))
        right = min(half, max(0, (after["keep_frames"] - 1) // 2))
        if not left or not right:
            raise ValueError("Outfit intervals are too short for a transformation; increase their duration or set transition_seconds=0")
        start, end = before["end_frame"] - left, after["start_frame"] + right
        # Both endpoints are REAL source frames, already rendered in the old
        # and new outfit. Padding is outside the retained bridge.
        needed = end - start
        length = max(5, needed) + (5 - max(5, needed) % 17) % 17
        bridges.append({"index": len(sections) + len(bridges) + 1, "kind": "transition",
                        "outfit_index": after["outfit_index"], "outfit_image": after["outfit_image"],
                        "from_outfit_image": before["outfit_image"],
                        "left_section": before["index"], "right_section": after["index"],
                        "start_frame": start, "end_frame": end, "keep_frames": needed,
                        "reference_start_frame": start, "trim_start_frame": 0,
                        "generation_frames": length, "generation_seconds": length / FPS,
                        "padding_frames": max(0, start + length - sections[-1]["end_frame"]),
                        "boundary_frame": before["end_frame"]})
    return bridges


class H3DanceWorkflowFast:
    REFERENCE_SCENE = False
    PHOTO_OUTFIT = False

    @classmethod
    def INPUT_TYPES(cls):
        images = [KEEP_ORIGINAL] + input_files(IMAGE_SUFFIXES)
        return {"required": {
            "reference_video": (["Choose dance video"] + input_files(VIDEO_SUFFIXES), {"video_upload": True}),
            "character_image": (images, {"tooltip": "Keep the source dancer, or choose a replacement character. Facial identity comes from this image; expressions and head movement follow the dance video."}),
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
            "anchor_mode": (["boundary"], {"tooltip": "One pass per section, then short anchored bridges at all section joins."}),
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
               transition_seconds=1.0, anchor_mode="boundary", anchor_manifest="[]", quality_mode="pose_warn", pose_model=""):
        if not prompt.strip():
            raise ValueError("Enter a nonempty dance prompt")
        if turbo and steps != 4:
            raise ValueError("Turbo requires exactly four steps")
        if anchor_mode not in {"boundary"} or quality_mode not in {"preview", "pose_warn", "pose_strict"}:
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
        if self.REFERENCE_SCENE:
            with Image.open(character) as image:
                iw, ih = ImageOps.exif_transpose(image).size
        width, height = dance_canvas(iw, ih, megapixels)
        job_id = "dance-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        job_directory = "h3/dance/" + job_id
        work = output_path(job_directory)
        work.mkdir(parents=True)
        inputs = Path(folder_paths.get_input_directory()) / "h3-dance" / job_id
        inputs.mkdir(parents=True)
        normalized = inputs / "source.mp4"
        normalized_info = normalize_dance_video(source, normalized, width, height, start_seconds, duration_seconds or None,
                                                preserve_aspect=self.REFERENCE_SCENE)
        total_frames = int(next(s for s in normalized_info["streams"] if s["codec_type"] == "video")["nb_frames"])
        paired = any(s["codec_type"] == "audio" for s in normalized_info["streams"])
        sections = plan_sections(total_frames, outfits, durations, max_section_seconds, context_frames)
        for s in sections:
            s["kind"] = "stable"
        transition_seconds = transition_seconds or 1.0
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
        report = {"pipeline": "boundary-refinement-v1", "job_id": job_id, "status": "planned", "reference_video": str(source),
                  "character_image": character, "background_image": background, "outfit_images": outfits,
                  "reference_scene": self.REFERENCE_SCENE,
                  "photo_outfit": self.PHOTO_OUTFIT,
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
                                                      transition=section if section["kind"] == "transition" and section.get("from_outfit_image") != section.get("outfit_image") else None,
                                                      reference_scene=self.REFERENCE_SCENE, photo_outfit=self.PHOTO_OUTFIT)
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


class H3SceneDanceMultipleOutfitsFast(H3DanceWorkflowFast):
    REFERENCE_SCENE = True

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = schema["required"]
        required.pop("character_image")
        required.pop("background_image")
        schema["required"] = {
            "reference_image": (["Choose character and scene image"] + input_files(IMAGE_SUFFIXES),
                                {"tooltip": "This picture defines facial identity, character, scene, lighting and camera composition. The dance video supplies facial expressions, blinking, mouth movement and head motion."}),
            **required}
        required["outfit_images"][1]["tooltip"] = "Upload clothes in appearance order. Each image supplies only the outfit, never its model or background."
        schema["optional"]["quality_mode"] = (["preview", "pose_warn", "pose_strict"],
            {"tooltip": "Preview creates side-by-side reviews. Pose scores can flag intentional changes in framing or body proportions."})
        return schema

    @classmethod
    def VALIDATE_INPUTS(cls, reference_image, reference_video, outfit_images, outfit_durations="[]"):
        try:
            input_path(reference_image)
            if not parse_outfits(outfit_images):
                return "Add at least one outfit image."
        except (ValueError, OSError) as exc:
            return str(exc)
        return super().VALIDATE_INPUTS(reference_video, reference_image, reference_image, outfit_images, outfit_durations)

    def expand(self, reference_image, reference_video, outfit_images, **settings):
        input_path(reference_image)
        if not parse_outfits(outfit_images):
            raise ValueError("Add at least one outfit image.")
        return super().expand(reference_video=reference_video, character_image=reference_image,
                              background_image=reference_image, outfit_images=outfit_images, **settings)


class H3SceneDanceSingleOutfitFast(H3SceneDanceMultipleOutfitsFast):
    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"] = {
            ("outfit_image" if name == "outfit_images" else name):
                (["Choose outfit image"] + input_files(IMAGE_SUFFIXES), {"tooltip": "Wear this outfit throughout the dance."})
                if name == "outfit_images" else value
            for name, value in schema["required"].items() if name != "outfit_durations"}
        schema["optional"].pop("transition_seconds")
        return schema

    @classmethod
    def VALIDATE_INPUTS(cls, reference_image, reference_video, outfit_image):
        return super().VALIDATE_INPUTS(reference_image, reference_video, json.dumps([outfit_image]))

    def expand(self, reference_image, reference_video, outfit_image, **settings):
        return super().expand(reference_image, reference_video, json.dumps([outfit_image]),
                              outfit_durations="[]", transition_seconds=0, **settings)


class H3PhotoDanceFast(H3SceneDanceSingleOutfitFast):
    """Animate a dressed character in their photographed scene using dance motion."""
    PHOTO_OUTFIT = True

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"].pop("outfit_image")
        schema["required"]["reference_image"][1]["tooltip"] = (
            "Full-body photo defining the character, their existing clothes, background, lighting and viewpoint. "
            "The dance video supplies motion and expressions only.")
        schema["required"]["prompt"][1]["default"] = (
            "Animate the photographed character dancing in their own scene. Preserve their existing outfit "
            "throughout. Follow the sample video's dance and expressions.")
        return schema

    @classmethod
    def VALIDATE_INPUTS(cls, reference_image, reference_video):
        return super().VALIDATE_INPUTS(reference_image, reference_video, reference_image)

    def expand(self, reference_image, reference_video, **settings):
        # One photo supplies all three fixed appearance roles in every section.
        return super().expand(reference_image, reference_video, reference_image, **settings)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (H3DanceWorkflowFast, H3PhotoDanceFast, H3SceneDanceSingleOutfitFast, H3SceneDanceMultipleOutfitsFast)}
