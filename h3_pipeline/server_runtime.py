"""Server-side telemetry and memory policy, imported only inside ComfyUI."""
from contextlib import contextmanager
from pathlib import Path
import gc
import json
import re
import threading
import time

from .config import COMFY_REVISION, MODEL_REVISION, write_json
from .attention import attention_policy, metadata as attention_metadata


class PerformanceMonitor:
    def __init__(self, job_id, directory, memory_level):
        import psutil
        import torch
        self.torch = torch
        self.process = psutil.Process()
        self.psutil = psutil
        self.path = Path(directory) / "h3" / job_id / "telemetry.json"
        self.started = time.perf_counter()
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.data = {"job_id": job_id, "status": "running", "memory_level": memory_level,
                     "comfy_revision": COMFY_REVISION, "model_revision": MODEL_REVISION,
                     "stages_seconds": {}, "model_activity": [], "system_ram_peak_bytes": 0,
                     "system_ram_measurement": "process tree RSS, sampled every 250 ms",
                     "gpu_peak_allocated_bytes": 0, "gpu_peak_reserved_bytes": 0,
                     "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
        self.data["attention"] = attention_metadata()
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)
            self.data.update(gpu_model=torch.cuda.get_device_name(device),
                             gpu_total_vram_bytes=torch.cuda.get_device_properties(device).total_memory)
        self.thread = threading.Thread(target=self._poll, daemon=True, name="h3-metrics")
        self.thread.start()

    def _poll(self):
        while not self.stop_event.wait(0.25):
            self.snapshot()

    def snapshot(self):
        with self.lock:
            rss = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except self.psutil.Error:
                    pass
            self.data["system_ram_peak_bytes"] = max(self.data["system_ram_peak_bytes"], rss)
            if self.torch.cuda.is_available():
                self.data["gpu_peak_allocated_bytes"] = max(self.data["gpu_peak_allocated_bytes"], self.torch.cuda.max_memory_allocated())
                self.data["gpu_peak_reserved_bytes"] = max(self.data["gpu_peak_reserved_bytes"], self.torch.cuda.max_memory_reserved())

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
            # Synchronize only at meaningful stage boundaries for accurate CUDA timing.
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
        except BaseException as exc:
            self.data.update(status=failure_status(exc), failed_stage=name, error=str(exc) or type(exc).__name__)
            raise
        finally:
            with self.lock:
                stages = self.data["stages_seconds"]
                stages[name] = stages.get(name, 0) + time.perf_counter() - started
            self.snapshot()
            write_json(self.path, self.data)

    def finish(self, status="completed"):
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.snapshot()
        self.data["status"] = status
        self.data["total_wall_seconds"] = time.perf_counter() - self.started
        duration = self.data.get("actual_duration")
        if duration:
            self.data["compute_seconds_per_output_second"] = self.data["total_wall_seconds"] / duration
        write_json(self.path, self.data)
        return dict(self.data)


class MemoryManager:
    def __init__(self):
        import comfy.model_management as mm
        self.mm = mm
        self.initial_reserve = mm.EXTRA_RESERVED_VRAM
        import comfy_aimdo.control as control
        self.control = control
        try:
            self.initial_dynamic_headroom = control.get_simple_vram_headroom()
        except RuntimeError:
            self.initial_dynamic_headroom = None  # Native static/CPU backend has no aimdo context.

    def configure(self, level):
        self.mm.EXTRA_RESERVED_VRAM = max(self.initial_reserve, (2 if level >= 2 else 1) * 1024**3)
        if self.initial_dynamic_headroom is not None:
            self.control.set_simple_vram_headroom(max(self.initial_dynamic_headroom, self.mm.EXTRA_RESERVED_VRAM))
        if level:
            gc.collect()
            self.mm.soft_empty_cache()

    def restore(self):
        self.mm.EXTRA_RESERVED_VRAM = self.initial_reserve
        if self.initial_dynamic_headroom is not None:
            self.control.set_simple_vram_headroom(self.initial_dynamic_headroom)

    def offload_encoder(self, clip):
        device = self.mm.get_torch_device()
        patcher = clip.patcher
        targets = [loaded for loaded in self.mm.current_loaded_models
                   if loaded.model is not None and (loaded.model is patcher or loaded.model.clone_base_uuid == patcher.clone_base_uuid)]
        resident = sum(loaded.model.loaded_size() for loaded in targets)
        if resident:
            # Ask for exactly the encoder's currently resident bytes, protecting other models.
            keep = [loaded for loaded in self.mm.current_loaded_models if loaded not in targets]
            free = self.mm.get_free_memory(device)
            self.mm.free_memory(free + resident, device, keep_loaded=keep)

    @staticmethod
    def video_decode(vae, samples):
        import torch
        from nodes import VAEDecode
        old = vae.output_device
        vae.output_device = torch.device("cpu")
        try:
            images = VAEDecode().decode(vae, samples)[0]
            # Native H3 decode writes directly to its preallocated CPU output buffer.
            return images.detach().to(device="cpu", dtype=torch.float16)
        finally:
            vae.output_device = old

    @staticmethod
    def audio_to_cpu(audio):
        return {**audio, "waveform": audio["waveform"].detach().cpu()}


_active = None
_memory = None


def begin(job_id, directory, level, *, report_path=None, section_index=None):
    global _active, _memory
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job_id):
        raise ValueError("job_id must contain 1-100 letters, digits, underscores or hyphens")
    if level not in (0, 1, 2):
        raise ValueError("memory_level must be 0, 1 or 2")
    if _active:
        _active.data["error"] = "Superseded by another H3 context"
        finish(_active.data["job_id"], "failed")
    if _memory:
        _memory.restore()
    _memory = MemoryManager()
    try:
        _memory.configure(level)
        _active = PerformanceMonitor(job_id, directory, level)
    except BaseException:
        _memory.restore()
        _memory = None
        raise
    from comfy_execution.utils import get_executing_context
    execution_context = get_executing_context()
    if execution_context:
        _active.data["prompt_id"] = execution_context.prompt_id
    if report_path is not None:
        _active.data.update(dance_report=str(report_path), section_index=section_index)
    return job_id


def failure_status(exc):
    from comfy.model_management import InterruptProcessingException
    return "interrupted" if isinstance(exc, (InterruptProcessingException, KeyboardInterrupt)) else "failed"


def current(context):
    if _active is None or _active.data["job_id"] != context:
        raise RuntimeError("Missing H3Begin context; execute the provided graph with caching disabled")
    return _active


def memory():
    return _memory


@contextmanager
def stage(context, name):
    monitor = current(context)
    try:
        with monitor.stage(name), model_activity(monitor), attention_policy(monitor, name):
            yield monitor
    except BaseException as exc:
        # Do not keep the traceback or tensor objects in the metrics record.
        finish(context, failure_status(exc))
        raise


def finish(context, status="completed"):
    global _active
    monitor = current(context)
    try:
        result = monitor.finish(status)
        if status != "completed" and result.get("dance_report"):
            from .dance_reports import fail_report
            fail_report(result["dance_report"], result.get("error", status), status,
                        result["section_index"], result.get("failed_stage"), result)
    finally:
        try:
            memory().restore()
        finally:
            _active = None
    return result


@contextmanager
def model_activity(monitor):
    """Scoped observation of native management; restore functions even on failure."""
    import comfy.model_management as mm
    originals = {}
    for name in ("load_models_gpu", "free_memory"):
        original = getattr(mm, name)
        originals[name] = original

        def observed(*args, _name=name, _original=original, **kwargs):
            start = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                event = {"operation": _name, "seconds": time.perf_counter() - start}
                if _name == "free_memory":
                    event["requested_bytes"] = args[0] if args else kwargs.get("memory_required")
                with monitor.lock:
                    monitor.data["model_activity"].append(event)
        setattr(mm, name, observed)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(mm, name, original)


@contextmanager
def measure_method(obj, name, monitor, label):
    """Measure Qwen/VAE calls within the unchanged native Ref2VA node."""
    original = getattr(obj, name)
    had_instance_value = name in obj.__dict__
    previous = obj.__dict__.get(name)

    def measured(*args, **kwargs):
        with monitor.stage(label):
            return original(*args, **kwargs)
    setattr(obj, name, measured)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)
