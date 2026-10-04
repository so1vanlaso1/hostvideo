"""Scoped attention policy for the pinned native H3 implementation.

Quantized linear/ConvRot kernels remain native. Only attention dispatch is
overridden, for the duration of an H3 inference stage, then restored.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from importlib.metadata import distribution, PackageNotFoundError
import json

SAGE_VERSION = "2.2.0"
SAGE_REVISION = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
POLICY = {"diffusion": "sage", "video_vae": "sage", "text_encoder": "pytorch",
          "audio_vae": "pytorch", "checkpoint_attention_overrides": False}
_component = ContextVar("h3_attention_component", default=None)


def metadata():
    try:
        package = distribution("sageattention")
        installed = package.version
        source = json.loads(package.read_text("direct_url.json") or "{}")
    except PackageNotFoundError:
        installed = None
        source = {}
    return {"selected": dict(POLICY), "sageattention_version": installed,
            "sageattention_revision": source.get("vcs_info", {}).get("commit_id"), "effective_calls": {}}


@contextmanager
def component(name):
    token = _component.set(name)
    try:
        yield
    finally:
        _component.reset(token)


@contextmanager
def attention_policy(monitor, stage):
    if stage not in {"h3_sampling", "reference_conditioning_and_latents", "video_vae_decode", "audio_vae_decode"}:
        yield
        return
    from comfy.ldm.modules import attention as attn
    from comfy.ldm.minimax import model, vae
    from comfy.text_encoders import llama, qwen_vl
    import comfy.ops
    import comfy_kitchen
    import torch

    calls = monitor.data.setdefault("attention", metadata())["effective_calls"]

    def record(backend):
        name = _component.get() or "unknown"
        counts = calls.setdefault(name, {})
        counts[backend] = counts.get(backend, 0) + 1

    def observed(original, backend):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            record(backend)  # Count successful dispatch, not merely selection.
            return result
        return call

    pytorch = observed(attn.attention_pytorch, "pytorch")

    def dispatch(name, use_sage):
        def call(q, k, v, *args, **kwargs):
            # The native wrapper otherwise gives checkpoint metadata and custom
            # transformer overrides precedence over the server's backend flag.
            kwargs.pop("preferred_attention", None)
            kwargs["_inside_attn_wrapper"] = True
            tensor = q.peek() if isinstance(q, attn.AttentionTensorContainer) else q
            with component(name):
                if use_sage and tensor.device.type == "cuda" and tensor.dtype in (torch.float16, torch.bfloat16):
                    if not attn.SAGE_ATTENTION_IS_AVAILABLE:
                        raise RuntimeError("H3 requires SageAttention; rerun scripts/setup_vast.sh and kernel preflight")
                    return attn.attention_sage(q, k, v, *args, **kwargs)
                return pytorch(q, k, v, *args, **kwargs)
        return call

    def replace(stack, obj, name, value):
        original = getattr(obj, name)
        setattr(obj, name, value)
        stack.callback(setattr, obj, name, original)

    with ExitStack() as stack:
        replace(stack, model, "optimized_attention", dispatch("diffusion", True))
        replace(stack, vae, "optimized_attention", dispatch("video_vae", True))
        # The quantized video VAE has a separate direct Comfy Kitchen branch.
        replace(stack, vae, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False)
        replace(stack, attn, "attention_pytorch", pytorch)  # Observe Sage fallbacks.
        if hasattr(attn, "sageattn"):
            replace(stack, attn, "sageattn", observed(attn.sageattn, "sage"))
        text_attention = dispatch("text_encoder", False)
        for module in (llama, qwen_vl):
            replace(stack, module, "optimized_attention_for_device", lambda *a, **k: text_attention)
        replace(stack, comfy_kitchen, "flash_attention_decode_is_available", lambda *a, **k: False)
        if stage in {"reference_conditioning_and_latents", "audio_vae_decode"}:
            original_sdpa = comfy.ops.scaled_dot_product_attention

            def audio_sdpa(*args, **kwargs):
                if _component.get() is not None:
                    return original_sdpa(*args, **kwargs)
                with component("audio_vae"):
                    return observed(original_sdpa, "pytorch")(*args, **kwargs)

            replace(stack, comfy.ops, "scaled_dot_product_attention", audio_sdpa)
        yield
