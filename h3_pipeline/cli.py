import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .config import ASSETS, DEFAULT_PROFILE, PROFILES, RuntimePaths, write_json


def parser():
    p = argparse.ArgumentParser(prog="h3", description="MiniMax H3 native Ref2VA, one 16 GB GPU")
    p.add_argument("--root", default=str(RuntimePaths.default().root), help="Persistent runtime root (or H3_ROOT)")
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="Check RAM, GPU, CUDA and optional quantization/SageAttention kernels")
    pre.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    pre.add_argument("--kernels", action="store_true")
    pre.add_argument("--local", action="store_true", help="Allow a CPU development machine")
    pre.add_argument("--output", type=Path)
    download = sub.add_parser("download-models", help="Download and verify only the selected five checkpoints")
    download.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    download.add_argument("--verify-existing", action="store_true")
    start = sub.add_parser("serve", help="Start private ComfyUI with the 16 GB memory profile")
    start.add_argument("--port", type=int, default=8188)
    start.add_argument("--profile", choices=PROFILES, default=os.environ.get("H3_PROFILE", DEFAULT_PROFILE))
    start.add_argument("--print-command", action="store_true")
    for name in ("generate", "benchmark"):
        job = sub.add_parser(name)
        job.add_argument("--job", required=True, type=Path, help="JSON job; asset paths are relative to this file")
        job.add_argument("--server", default=os.environ.get("H3_SERVER_URL", "http://127.0.0.1:8188"))
        job.add_argument("--profile", choices=PROFILES, default=os.environ.get("H3_PROFILE", DEFAULT_PROFILE))
        if name == "benchmark":
            job.add_argument("--stage", choices=("smoke", "five-second", "target", "identity-motion", "clothing", "resolution"), default="smoke")
            job.add_argument("--megapixels", type=float, help="Required for optional resolution stage: 0.7, 0.8, 0.98")
    export = sub.add_parser("export-workflows", help="Copy the browser and API workflow files")
    export.add_argument("--directory", required=True, type=Path)
    return p


def read_job(path):
    job = json.loads(path.read_text())
    allowed = {"prompt", "reference_video", "character_images", "clothing_images", "additional_reference_images", "reference_audio",
               "duration", "megapixels", "seed", "aspect_ratio", "ref_image_size", "include_video_audio", "reference_video_start", "scheduler", "steps", "turbo", "oom_fallback", "timeout"}
    unknown = job.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown job options: {sorted(unknown)}")
    for field in ("reference_video", "reference_audio"):
        if job.get(field):
            job[field] = str((path.parent / Path(job[field]).expanduser()).resolve())
    for field in ("character_images", "clothing_images", "additional_reference_images"):
        if job.get(field) is not None:
            if not isinstance(job[field], list):
                raise ValueError(f"{field} must be a list")
            job[field] = [str((path.parent / Path(p).expanduser()).resolve()) for p in job[field]]
    return job


def benchmark_job(job, stage, megapixels=None):
    job = dict(job)
    job["oom_fallback"] = False  # A failed requested size must not count as a passing benchmark.
    job.setdefault("steps", 25)
    job.setdefault("turbo", False)
    if stage == "smoke":
        job.update(duration=3.0, megapixels=0.98)
    elif stage == "five-second":
        job.update(duration=5.0, megapixels=0.98)
    else:
        job.update(duration=15.0, megapixels=0.98)
    if stage in ("smoke", "five-second", "target"):
        for field in ("reference_video", "reference_audio", "character_images", "clothing_images", "additional_reference_images"):
            job.pop(field, None)
        job["prompt"] = "A full-body dancer performs a simple dance in a studio. Natural motion, soft lighting and synchronized rhythmic music."
    if stage == "identity-motion":
        job.pop("clothing_images", None)
        if not job.get("reference_video") or not job.get("character_images"):
            raise ValueError("identity-motion benchmark requires a character image and reference video")
    if stage in ("clothing", "resolution"):
        if not all(job.get(k) for k in ("reference_video", "character_images", "clothing_images")):
            raise ValueError("This benchmark requires dance video, character image and clothing image")
    if stage == "resolution":
        if megapixels not in (0.7, 0.8, 0.98):
            raise ValueError("Choose --megapixels 0.7, 0.8 or 0.98; test each separately")
        job["megapixels"] = megapixels
    return job


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            from .preflight import preflight
            report = preflight(args.root, args.profile, args.kernels, not args.local)
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
        if args.command == "download-models":
            from .models import ModelManager
            print(json.dumps(ModelManager(Path(args.root) / "models", args.profile).download(verify_existing=args.verify_existing), indent=2))
        elif args.command == "serve":
            from .server import serve
            result = serve(args.root, args.port, args.print_command, args.profile)
            if result:
                print(result)
        elif args.command == "export-workflows":
            import shutil
            args.directory.mkdir(parents=True, exist_ok=True)
            for name in ("ref2va_16gb_ui.json", "ref2va_api.json", "ref2va_quality_5s_098mp.json", "dance_general_ui.json"):
                shutil.copy2(ASSETS / "workflows" / name, args.directory / name)
            print(args.directory.resolve())
        elif args.command in ("generate", "benchmark"):
            from .pipeline import MiniMaxH3Pipeline
            job = read_job(args.job.resolve())
            if args.command == "benchmark":
                job = benchmark_job(job, args.stage, args.megapixels)
            pipeline = MiniMaxH3Pipeline(args.server, args.root, args.profile,
                                        os.environ.get("H3_HTTP_USERNAME"), os.environ.get("H3_HTTP_PASSWORD"),
                                        progress=lambda message: print(message, file=sys.stderr, flush=True))
            result = pipeline.generate(**job)
            print(json.dumps(asdict(result), indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"h3: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
