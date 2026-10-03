from dataclasses import dataclass
from pathlib import Path
import json
import os

ASSETS = Path(__file__).resolve().parent / "assets"
COMFY_REVISION = "e9027f2b30f37bb3052714eb08fcf479542f4fc0"
TEMPLATE_REVISION = "0e5c5efb32ba6f3365d6da07da64aaf668157042"
MODEL_REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"
MODEL_REPO = "Comfy-Org/MiniMax-H3"
PROFILES = ("primary", "fp8", "int8-encoder", "fp8-int8-encoder")
DEFAULT_PROFILE = "int8-encoder"


def manifest():
    return json.loads((ASSETS / "models.json").read_text())


def model_files(profile=DEFAULT_PROFILE):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PROFILES}")
    keys = ["diffusion_fp8" if profile.startswith("fp8") else "diffusion",
            "encoder_int8" if "int8-encoder" in profile else "encoder",
            "video_vae", "audio_vae", "lora"]
    return {key: manifest()["files"][name] for key, name in zip(
        ["diffusion", "encoder", "video_vae", "audio_vae", "lora"], keys)}


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @classmethod
    def default(cls):
        return cls(Path(os.environ.get("H3_ROOT", "/data/minimax-h3")).expanduser().resolve())

    @property
    def comfy(self):
        return self.root / "ComfyUI"

    @property
    def models(self):
        return self.root / "models"

    @property
    def input(self):
        return self.root / "input"

    @property
    def output(self):
        return self.root / "output"

    def create(self):
        for name in ("models", "input", "output", "logs", "locks", "user", "temp", "jobs", "private"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "private").chmod(0o700)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
