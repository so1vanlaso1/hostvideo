import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .config import ASSETS, DEFAULT_PROFILE, PROFILES, RuntimePaths, write_json


def parser():
    p = argparse.ArgumentParser(prog="h3", description="MiniMax H3 native Ref2VA, one 16 GB GPU")
    p.add_argument("--root", default=str(RuntimePaths.defa…21747 tokens truncated…total_frames) > 1.01:
            raise ValueError("Outfit durations must add up to the full reference video duration (within one frame)")
        boundaries, elapsed = [0], 0
        for duration in outfit_durations:
            elapsed += duration
            boundaries.append(round(elapsed * FPS))
        boundaries[-1] = total_frames
    else:
        boundaries = [i * total_frames // count for i in range(count + 1)]
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Each outfit must have at least one frame on the timeline")
    # Round DOWN to the native grid before budgeting context. No generation
    # window, including context and padding, may exceed the requested ceiling.
    frame_budget = min(math.floor(max_section_seconds * FPS), MAX_FRAMES)
    max_generation = frame_budget - (frame_budget - 5) % 17
    max_core = max_generation - 2 * context_frames
    if max_generation < 5 or max_core < 1:
        raise ValueError("max_section_seconds is too short for the requested context_frames and H3's five-frame minimum")
    sections = []
    for outfit_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        chunks = math.ceil((end - start) / max_core)
        for chunk in range(chunks):
            core_start = start + chunk * (end - start) // chunks
            core_end = start + (chunk + 1) * (end - start) // chunks
            needed = max(5, core_end - core_start + 2 * context_frames)
            length = needed + (5 - needed % 17) % 17
            reference_start = min(max(0, core_start - context_frames), max(0, total_frames - length))
            sections.append({"index": len(sections) + 1, "outfit_index": outfit_index + 1,
                             "outfit_image": outfits[outfit_index] if outfits else None,
                             "start_frame": core_start, "end_frame": core_end,
                             "keep_frames": core_end - core_start,
                             "reference_start_frame": reference_start, "generation_frames": length,
                             "generation_seconds": length / FPS,
                             "trim_start_frame": core_start - reference_start,
                             "padding_frames": max(0, reference_start + length - total_frames)})
    return sections


def section_references(anchor, character, background, outfit, previous_frame=None, from_outfit=None):
    images = []
    def add(path, role):
        images.append(Reference(f"<Picture {len(images) + 1}>", role, str(path)))
    add(character or anchor, "replacement character identity" if character else "original dancer identity from the source video")
    add(background or anchor, "replacement background scene only" if background else "original background scene only")
    if outfit:
        add(outfit, "clothing design only; ignore the clothing model's identity and background")
    if from_outfit:
        add(from_outfit, "opening outfit clothing design only; ignore the clothing model's identity and background")
    if previous_frame:
        add(previous_frame, "appearance continuity from the preceding generated section; ignore its outfit and pose")
    return images


def dance_prompt(prompt, images, paired_audio, replace_character, replace_background, has_outfit, reference_video="reference.mp4", transition=None,
                 reference_scene=False, photo_outfit=False):
    refs = list(images)
    if paired_audio:
        refs.append(Reference("<Audio 1>", "source dance soundtrack paired with <Video 1>", str(reference_video)))
    refs.append(Reference("<Video 1>", "the exact chronological dance excerpt, including body choreography, facial performance and head movement for this section", str(reference_video)))
    character = ("Replace the original dancer with the person in <Picture 1>. Do not borrow the original dancer's face or hair."
                 if replace_character else "Keep the original dancer from <Video 1> and <Picture 1>, including face, hair, skin tone and body proportions.")
    background = ("Replace the source background with the scene in <Picture 2>; ignore any people in this scene reference."
                  if replace_background else "Keep the original room, background objects, lighting and camera framing from <Video 1> and <Picture 2>.")
    clothing = ("Wear the complete outfit from <Picture 3> throughout this section, from first to last frame. Transfer only its garments, colors, cut and details. Ignore the clothing photograph's person and location. Do not mix outfits."
                if has_outfit else "Keep the original dancer's clothing from <Video 1>, even when replacing the character or background.")
    if transition:
        clothing = ("Begin wearing the complete opening outfit from <Picture 4>. While continuing the source choreography, "
                    f"transform the garments once into the complete outfit from <Picture 3> between 0.000 and {(transition['keep_frames'] - 1) / FPS:.3f} seconds. "
                    "Finish in the new outfit and retain it through the remaining context. A continuous fully clothed fashion transformation, "
                    "no alternating outfits, no repeated transformation, no body or face morph. The anchored endpoint images define the exact poses and outfits.")
    summary = ("[video editing + reference generation] Edit <Video 1>. Preserve its original choreography and timing. "
               "Change only the explicitly selected clothing, character or background.")
    retention = ("Preserve the frame-by-frame dance sequence, foot contacts, body position, movement direction, camera trajectory and shot timing. "
                 "Identity and scene remain preserved unless a replacement is explicitly selected.")
    motion = ("Follow <Video 1> in its exact temporal order: matching dance poses, gestures, footwork, movement direction, speed and camera motion. "
              "Start with its opening pose and finish with its final pose. Do not invent a different dance or restart the choreography.")
    if reference_scene:
        summary = ("[reference generation] Animate the person in <Picture 1> dancing in that picture's own scene, "
               "wearing the supplied clothes and following the choreography, facial performance and timing of <Video 1>.")
        character = ("Preserve the person in <Picture 1>: the same facial identity, hair, skin tone and body proportions, with expressions animated from <Video 1>. "
                     "Replace only their clothing using the outfit reference. Ignore every other reference person's identity.")
        background = ("<Picture 1> and <Picture 2> are the same character-and-scene photograph. Preserve its environment, "
                      "background objects, layout, lighting, viewpoint and camera composition throughout the dance. "
                      "Animate the existing person; do not leave a second static copy in the background. Keep the camera fixed.")
        retention = ("The photograph controls identity, scene and framing. The clothing image controls garments only. "
                     "<Video 1> controls body choreography, facial expressions, gaze, blinking, mouth movement, head movement and their timing; "
                     "ignore its dancer's facial identity, hair, skin tone, body proportions, clothes, location, camera movement and cuts. "
                     "Any black padding in the motion reference is preprocessing, not part of the target scene.")
        motion = ("Follow <Video 1> in its exact temporal order: matching dance poses, gestures, footwork, movement direction and speed. "
                  "Transfer the movements into the photographed scene while maintaining the photograph's viewpoint. "
                  "Start with the reference dance's opening pose and finish with its final pose. Do not restart or invent a different dance.")
    if photo_outfit:
        summary = ("[reference generation] Animate the dressed person in <Picture 1> in that photograph's own scene. "
                   "Retain the clothes they already wear and follow only the dance and facial performance of <Video 1>.")
        character = ("Preserve the photographed person's facial identity, hair, skin tone and body proportions. "
                     "Animate expressions and head movement from <Video 1> without copying its dancer's identity.")
        clothing = ("<Picture 1>, <Picture 2> and <Picture 3> are the same photograph. Keep the complete outfit "
                    "already worn by this person, including garment shapes, colors, patterns, shoes and accessories, "
                    "from first to last frame and across every section. No clothing changes or transformations. "
                    "Ignore all clothing in <Video 1>.")
        retention = ("The photograph controls character identity, existing outfit, background, lighting and framing. "
                     "<Video 1> controls body choreography, facial expressions, gaze, blinking, mouth movement, head movement and their timing. "
                     "Ignore the video's dancer identity, clothing, scene, camera movement and cuts. "
                     "Any black padding in the motion reference is preprocessing, not part of the target scene.")
    text = ["subject_definitions:", *[f"{r.tag} defines {r.role}." for r in refs],
            "summary:\n" + summary, "retention_analysis:\n" + retention, "detailed_description:"]
    text += [character, background, clothing,
             motion,
             facial_motion_prompt("<Picture 1>"),
             "Maintain the same selected character and scene throughout. One continuous shot in this section, coherent hands and limbs, full-body framing, no extra people, labels or clothing-photo backgrounds."]
    if images[-1].role.startswith("appearance continuity"):
        text.append(f"Use {images[-1].tag} only to maintain the generated character's appearance and scene across the cut. The current outfit reference overrides its old clothes; <Video 1> overrides its pose, facial expression, gaze and mouth movement at the current timestamp.")
    text.append("Generate synchronized audio following <Audio 1>." if paired_audio else "Generate suitable synchronized dance music.")
    # Validate references without adding the generic single-outfit prompt rules.
    PromptBuilder.build(prompt, refs)
    text.append(prompt.strip())
    return "\n\n".join(text), refs
