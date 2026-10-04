"""Planning and prompts for the general ComfyUI wardrobe-change workflow."""
import math

from .references import FPS, MAX_FRAMES, MAX_PIXELS, Reference, PromptBuilder

KEEP_ORIGINAL = "Keep original"


def dance_canvas(width, height, megapixels):
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (width, height)):
        raise ValueError("Source dimensions must be positive integers")
    if not math.isfinite(megapixels) or not 0.1 <= megapixels <= 1:
        raise ValueError("megapixels must be between 0.1 and 1")
    scale = min(1, 16384 / width, 16384 / height,
                math.sqrt(min(MAX_PIXELS, megapixels * 1024**2) / (width * height)))
    w, h = max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)
    while w * h > MAX_PIXELS:
        if w >= h:
            w -= 32
        else:
            h -= 32
    return w, h


def plan_sections(total_frames, outfits, outfit_durations=(), max_section_seconds=5, context_frames=12):
    """Keep every source frame once; H3 alignment applies only to context windows."""
    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames < 5:
        raise ValueError("The dance video must contain at least five frames at 24 fps")
    if not math.isfinite(max_section_seconds) or not 0.2 <= max_section_seconds <= 15:
        raise ValueError("max_section_seconds must be between 0.2 and 15")
    if isinstance(context_frames, bool) or not isinstance(context_frames, int) or not 0 <= context_frames <= 48:
        raise ValueError("context_frames must be an integer between 0 and 48")
    count = max(1, len(outfits))
    if count > total_frames:
        raise ValueError("There are more outfits than source frames; use a longer video to show every outfit")
    if outfit_durations:
        if len(outfit_durations) != count:
            raise ValueError("Supply one outfit duration for each outfit")
        if any(isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0 for d in outfit_durations):
            raise ValueError("Outfit durations must be positive finite numbers in seconds")
        if abs(sum(outfit_durations) * FPS - total_frames) > 1.01:
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


def section_references(anchor, character, background, outfit, previous_frame=None):
    images = []
    def add(path, role):
        images.append(Reference(f"<Picture {len(images) + 1}>", role, str(path)))
    add(character or anchor, "replacement character identity" if character else "original dancer identity from the source video")
    add(background or anchor, "replacement background scene only" if background else "original background scene only")
    if outfit:
        add(outfit, "clothing design only; ignore the clothing model's identity and background")
    if previous_frame:
        add(previous_frame, "appearance continuity from the preceding generated section; ignore its outfit and pose")
    return images


def dance_prompt(prompt, images, paired_audio, replace_character, replace_background, has_outfit, reference_video="reference.mp4"):
    refs = list(images)
    if paired_audio:
        refs.append(Reference("<Audio 1>", "source dance soundtrack paired with <Video 1>", str(reference_video)))
    refs.append(Reference("<Video 1>", "the exact chronological dance excerpt for this section", str(reference_video)))
    character = ("Replace the original dancer with the person in <Picture 1>. Do not borrow the original dancer's face or hair."
                 if replace_character else "Keep the original dancer from <Video 1> and <Picture 1>, including face, hair, skin tone and body proportions.")
    background = ("Replace the source background with the scene in <Picture 2>; ignore any people in this scene reference."
                  if replace_background else "Keep the original room, background objects, lighting and camera framing from <Video 1> and <Picture 2>.")
    clothing = ("Wear the complete outfit from <Picture 3> throughout this section, from first to last frame. Transfer only its garments, colors, cut and details. Ignore the clothing photograph's person and location. Do not mix outfits."
                if has_outfit else "Keep the original dancer's clothing from <Video 1>, even when replacing the character or background.")
    text = [f"{r.tag} defines {r.role}." for r in refs]
    text += [character, background, clothing,
             "Follow <Video 1> in its exact temporal order: matching dance poses, gestures, footwork, movement direction, speed and camera motion. Start with its opening pose and finish with its final pose. Do not invent a different dance or restart the choreography.",
             "Maintain the same selected character and scene throughout. One continuous shot in this section, coherent hands and limbs, full-body framing, no extra people, labels or clothing-photo backgrounds."]
    if images[-1].role.startswith("appearance continuity"):
        text.append(f"Use {images[-1].tag} only to maintain the generated character's appearance and scene across the cut. The current outfit reference overrides its old clothes; <Video 1> overrides its pose.")
    text.append("Generate synchronized audio following <Audio 1>." if paired_audio else "Generate suitable synchronized dance music.")
    # Validate references without adding the generic single-outfit prompt rules.
    PromptBuilder.build(prompt, refs)
    text.append(prompt.strip())
    return "\n\n".join(text), refs
