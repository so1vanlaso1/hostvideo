from pathlib import Path
import os
import platform
import shutil
import subprocess

from .config import DEFAULT_PROFILE, model_files
from .attention import SAGE_VERSION, SAGE_REVISION, metadata as attention_metadata


def allocated_ram():
    import psutil
    limits = [psutil.virtual_memory().total]
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
            if value != "max":
                limits.append(int(value))
        except (OSError, ValueError):
            pass
    return min(limits)


def kernel_smoke(profile):
    """Small operations only. Force CUDA dispatch so eager fallback cannot mask failure."""
    import torch
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor
    required = [("int8_convrot", "TensorWiseINT8Layout", {"per_channel": True, "convrot": True, "convrot_groupsize": 256})]
    if profile.startswith("fp8"):
        required.append(("fp8", "TensorCoreFP8Layout", {}))
    if "int8-encoder" not in profile:
        required.append(("nvfp4", "TensorCoreNVFP4Layout", {}))
    results = {"backends": str(ck.list_backends())}
    with torch.inference_mode(), ck.registry.use_backend("cuda"):
        for name, layout, kwargs in required:
            x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
            quantized = QuantizedTensor.from_float(weight, layout, **kwargs)
            if name == "nvfp4":
                x = QuantizedTensor.from_float(x, layout)
            output = torch.nn.functional.linear(x, quantized)
            torch.cuda.synchronize()
            if tuple(output.shape) != (32, 128) or not torch.isfinite(output).all().item():
                raise RuntimeError(f"{name} kernel produced invalid output")
            results[name] = "passed"
            del x, weight, quantized, output
    torch.cuda.empty_cache()
    return results


def attention_smoke():
    """Exercise actual Sage CUDA kernels and compare to FP32 SDPA on small H3 shapes."""
    import torch
    from sageattention import sageattn
    results = attention_metadata()
    if results["sageattention_version"] != SAGE_VERSION or results["sageattention_revision"] != SAGE_REVISION:
        raise RuntimeError(f"Expected SageAttention {SAGE_VERSION} at {SAGE_REVISION}; rerun setup")
    results["checks"] = []
    with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1234)
        for dtype in (torch.float16, torch.bfloat16):
            for heads, dim in ((56, 128), (32, 64)):
                q, k, v = [torch.randn(1, heads, 257, dim, device="cuda", dtype=dtype) for _ in range(3)]
                output = sageattn(q, k, v, tensor_layout="HND", is_causal=False, smooth_k=False)
                expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
                torch.cuda.synchronize()
                if output.shape != q.shape or not torch.isfinite(output).all().item():
                    raise RuntimeError("SageAttention produced invalid output")
                relative_rmse = ((output.float() - expected).square().mean() / expected.square().mean()).sqrt().item()
                if relative_rmse > 0.1:
                    raise RuntimeError(f"SageAttention relative RMSE {relative_rmse:.4f} exceeds 0.1")
                results["checks"].append({"dtype": str(dtype), "heads": heads, "head_dim": dim,
                                           "relative_rmse": relative_rmse, "status": "passed"})
    results["effective_calls"] = {"diffusion_shape": {"sage": 2}, "video_vae_shape": {"sage": 2}}
    torch.cuda.empty_cache()
    return results


def preflight(root, profile=DEFAULT_PROFILE, kernels=False, require_gpu=True):
    import psutil
    path = Path(root)
    existing = path
    while not existing.exists():
        existing = existing.parent
    ram = allocated_ram()
    cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    report = {"platform": platform.platform(), "python": platform.python_version(), "root": str(path),
              "allocated_ram_bytes": ram, "allocated_cpu_cores": cpu, "disk_free_bytes": shutil.disk_usage(existing).free,
              "models_bytes": sum(m["size"] for m in model_files(profile).values()), "profile": profile,
              "warnings": [], "errors": [], "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
    if ram < 32 * 1024**3:
        report["warnings"].append("Less than 32 GiB allocated RAM: below guide minimum; choose a larger Vast RAM allocation")
    elif ram < 64 * 1024**3:
        report["warnings"].append("64 GiB allocated system RAM is preferred for model offloading")
    if cpu < 8:
        report["warnings"].append("Fewer than eight allocated CPU cores; preprocessing/encoding may be slow")
    if not report["ffmpeg"] or not report["ffprobe"]:
        report["errors"].append("ffmpeg and ffprobe must be installed")
    if require_gpu and platform.system() != "Linux":
        report["errors"].append("Inference deployment requires Linux with an NVIDIA GPU")
    try:
        import torch
        report.update(torch_version=torch.__version__, cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            report.update(gpu_model=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                          total_vram_bytes=torch.cuda.get_device_properties(0).total_memory)
            # A real operation catches wheels missing this GPU's architecture.
            (torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda")).sum().item()
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
            report["driver_version"] = driver
            if int(driver.split(".")[0]) < 580:
                report["errors"].append("CUDA 13.0 requires an R580-or-newer compatible host driver")
            if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) < 13:
                report["errors"].append("Use the pinned CUDA 13.0 PyTorch environment before ConvRot benchmarking")
            if "5060 Ti" not in report["gpu_model"]:
                report["warnings"].append("GPU differs from the planned RTX 5060 Ti; record its benchmarks separately")
            if report["total_vram_bytes"] < 15 * 1024**3:
                report["errors"].append("This deployment targets the 16 GB GPU variant")
            if kernels and not report["errors"]:
                report["kernels"] = kernel_smoke(profile)
                report["attention"] = attention_smoke()
        elif require_gpu:
            report["errors"].append("PyTorch cannot access CUDA")
    except Exception as exc:
        if require_gpu:
            report["errors"].append(f"GPU/kernel preflight failed: {exc}")
        else:
            report["warnings"].append(f"GPU checks unavailable locally: {exc}")
    report["passed"] = not report["errors"]
    return report
