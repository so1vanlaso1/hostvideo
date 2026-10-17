from pathlib import Path
import os
import subprocess
import sys

from .config import DEFAULT_PROFILE, RuntimePaths, COMFY_REVISION
from .hardware import device_info, runtime_policy


def command(paths, port=8188, hardware=None):
    if hardware is None:
        try:
            import torch
        except ImportError:
            hardware = {}
        else:
            hardware = device_info(torch)
    policy = runtime_policy(hardware)
    memory_flags = ["--enable-dynamic-vram" if policy["memory_mode"] == "dynamic" else "--disable-dynamic-vram",
                    "--reserve-vram", str(policy["reserve_vram_gib"])]
    if policy["disable_pinned_memory"]:
        memory_flags.append("--disable-pinned-memory")
    return [sys.executable, str(paths.comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
            "--models-directory", str(paths.models), "--input-directory", str(paths.input),
            "--output-directory", str(paths.output), "--temp-directory", str(paths.root / "temp"),
            "--user-directory", str(paths.root / "user"), *memory_flags, "--fp16-intermediates",
            "--use-sage-attention", "--cache-none", "--preview-method", "none"]


def serve(root, port=8188, print_command=False, profile=DEFAULT_PROFILE):
    paths = RuntimePaths(Path(root).expanduser().resolve())
    if print_command:
        import shlex
        return shlex.join(command(paths, port))
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
    cmd = command(paths, port, hardware=report)
    print("H3 memory policy:", report["runtime_policy"], flush=True)
    os.execv(sys.executable, cmd)
