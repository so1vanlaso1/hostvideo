"""Hardware policy shared by setup, startup and inference telemetry."""
import math
import os
import platform


ARM_MACHINES = {"aarch64", "arm64"}


def is_gb10(gpu_model, capability=(), machine=None):
    machine = (machine or platform.machine()).lower()
    return "GB10" in gpu_model.upper() or (
        machine in ARM_MACHINES and tuple(capability) == (12, 1))


def device_info(torch):
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    name = torch.cuda.get_device_name(device)
    capability = list(torch.cuda.get_device_capability(device))
    gb10 = is_gb10(name, capability)
    return {"gpu_model": name, "capability": capability,
            "total_vram_bytes": props.total_memory, "gb10": gb10,
            "shared_memory": gb10 or bool(getattr(props, "is_integrated", False))}


def runtime_policy(hardware, environ=None):
    environ = os.environ if environ is None else environ
    shared = hardware.get("shared_memory", False)
    mode = environ.get("H3_MEMORY_MODE", "auto")
    if mode not in {"auto", "dynamic", "static"}:
        raise ValueError("H3_MEMORY_MODE must be auto, dynamic or static")
    if mode == "auto":
        mode = "dynamic" if hardware.get("gb10") or not shared else "static"
    reserve = float(environ.get("H3_RESERVE_VRAM", "12" if shared else "1"))
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("H3_RESERVE_VRAM must be a finite nonnegative number of GiB")
    return {"memory_mode": mode, "reserve_vram_gib": reserve,
            "disable_pinned_memory": shared}
