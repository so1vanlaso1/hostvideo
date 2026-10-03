from dataclasses import dataclass, asdict
import math
from pathlib import Path
import re

FPS = 24
MAX_FRAMES = 362
MAX_PIXELS = 1344 * 768
ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1)}


def frame_count(duration):
    if not math.isfinite(duration) or not 0.2 <= duration <= 15:
        raise ValueError("duration must be between 0.2 and 15 seconds")
    # Exact expression from the pinned official ComfyMathExpression node.
    n = max(5, round(duration * FPS))
    return n + (5 - n % 17) % 17


def canvas(megapixels, aspect_ratio="16:9"):
    if not math.isfinite(megapixels) or not 0.1 <= megapixels <= 1:
        raise ValueError("megapixels must be between 0.1 and 1.0")
    if aspect_ratio not in ASPECTS:
        raise ValueError(f"aspect_ratio must be one of {tuple(ASPECTS)}")
    w, h = ASPECTS[aspect_ratio]
    scale = math.sqrt(megapixels * 1024 * 1024 / (w * h))
    width, height = round(w * scale / 32) * 32, round(h * scale / 32) * 32
    # A nominal 1.0 MP rounds above H3's trained area. Use its native cap.
    while width * height > MAX_PIXELS:
        if width / height >= w / h:
            width -= 32
        else:
            height -= 32
    return max(32, width), max(32, height)


def reference_canvas(width, height, output_width, output_height, mode="match"):
    if mode not in ("match", "max"):
        raise ValueError("ref_image_size must be match or max")
    scale = min(1.0, math.sqrt(output_width * output_height / (width * height))) if mode == "match" else min(1.0, 2048 / min(width, height))
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


@dataclass(frozen=True)
class Reference:
    tag: str
    role: str
    path: str

    def dict(self):
        return asdict(self)


class ReferenceManager:
    @staticmethod
    def images(character_images=(), clothing_images=(), additional_reference_images=()):
        refs = []
        for role, paths in (("character identity and appearance", character_images or ()),
                            ("clothing and garment details", clothing_images or ()),
                            ("additional visual style and scene reference", additional_reference_images or ())):
            if isinstance(paths, (str, Path)):
                raise ValueError("Image groups must be lists, not a single path")
            for path in paths:
                p = Path(path).expanduser().resolve()
                if not p.is_file():
                    raise FileNotFoundError(p)
                refs.append(Reference(f"<Picture {len(refs) + 1}>", role, str(p)))
        if len(refs) > 9:
            raise ValueError("H3 supports at most nine reference images")
        return refs

    @staticmethod
    def mapping(images, video=None, paired_audio=False, audio=None):
        refs = list(images)
        audio_index = 1
        if video:
            # Native H3 presents paired audio before that video to Qwen.
            if paired_audio:
                refs.append(Reference("<Audio 1>", "soundtrack, rhythm and audio atmosphere paired with <Video 1>", str(video)))
                audio_index += 1
            refs.append(Reference("<Video 1>", "choreography, body movement, timing and camera movement", str(video)))
        if audio:
            refs.append(Reference(f"<Audio {audio_index}>", "standalone sound, music or voice reference", str(audio)))
        return refs


class PromptBuilder:
    @staticmethod
    def build(prompt, references):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        tags = {r.tag for r in references}
        unknown = set(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", prompt)) - tags
        if unknown:
            raise ValueError(f"Prompt contains references that are not connected: {sorted(unknown)}")
        descriptions = [f"{r.tag} defines {r.role}." for r in references]
        identity = [r.tag for r in references if r.role.startswith("character")]
        clothing = [r.tag for r in references if r.role.startswith("clothing")]
        instructions = []
        if identity:
            instructions.append(f"Use {', '.join(identity)} for the same person's face, hair, identity and body proportions throughout the shot.")
        if clothing:
            instructions.append(f"Dress the person in the clothing from {', '.join(clothing)}; maintain garment shape, color, patterns and details throughout the shot.")
        if "<Video 1>" in tags:
            instructions.append("Follow <Video 1> for choreography, pose sequence, timing and camera motion.")
        if any(r.tag.startswith("<Audio") for r in references):
            instructions.append("Generate native synchronized audio guided by the assigned audio references.")
        else:
            instructions.append("Generate native synchronized audio appropriate to the scene.")
        return "\n\n".join(descriptions + instructions + [prompt.strip(), "One continuous shot, natural body motion, coherent hands and limbs, consistent identity and outfit."])
