from pathlib import Path
import os
import subprocess
import sys

from .config import RuntimePaths, COMFY_REVISION


def command(paths, port=8188):
    return [sys.executable, str(paths.comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
            "--models-directory", str(paths.models), "--input-directory", str(paths.input),
            "--output-directory", str(paths.output), "--temp-directory", str(paths.root / "temp"),
            "--user-directory", str(paths.root / "user"), "--enable-dynamic-vram", "--fp16-intermediates",
            "--reserve-vram", "1", "--use-pytorch-cross-attention", "--cache-none", "--preview-method", "none"]


def serve(root, port=8188, print_command=False, profile="primary"):
    paths = RuntimePaths(Path(root).expanduser().resolve())
    cmd = command(paths, port)
    if print_command:
        import shlex
        return shlex.join(cmd)
    if not paths.comfy.is_dir():
        raise RuntimeError("ComfyUI is not installed; run scripts/setup_vast.sh first")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.comfy, capture_output=True, text=True, check=True).stdout.strip()
    if actual != COMFY_REVISION:
        raise RuntimeError("ComfyUI revision differs from the validated node contract; rerun setup or explicitly validate the update")
    paths.create()
    from .preflight import preflight
    from .config import write_json
    report = preflight(paths.root, profile=profile, kernels=True)
    write_json(paths.root / "locks" / "preflight.json", report)
    if not report["passed"]:
        raise RuntimeError("Server preflight failed: " + "; ".join(report["errors"]))
    os.execv(sys.executable, cmd)
