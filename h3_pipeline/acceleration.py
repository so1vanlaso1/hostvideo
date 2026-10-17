"""Opt-in H3 residual caching, composed with native sparse attention patches.

FirstBlockCache follows the Sol-Engine residual reuse algorithm. It changes
sampling numerics and must be checked visually for each kind of motion.
"""
import torch
from comfy.patcher_extension import CallbacksMP


class H3FirstBlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "threshold": ("FLOAT", {"default": 0.12, "min": 0, "max": 1, "step": 0.01}),
            "start_step": ("INT", {"default": 2, "min": 0, "max": 100}),
            "end_dense_steps": ("INT", {"default": 2, "min": 0, "max": 100}),
            "max_consecutive_skips": ("INT", {"default": 2, "min": 1, "max": 10})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3 Pipeline"

    def patch(self, model, threshold, start_step, end_dense_steps, max_consecutive_skips):
        m = model.clone()
        state = {}
        blocks = m.get_model_object("diffusion_model").blocks
        previous = dict(m.model_options["transformer_options"].get("patches_replace", {}).get("dit", {}))

        def reset(*args):
            state.clear()

        def execute(index, args, extra):
            old = previous.get(("double_block", index))
            return extra["original_block"](args) if old is None else old(args, extra)

        def first(args, extra):
            options = args["transformer_options"]
            sigma = float(options["sigmas"][0])
            if not state or sigma > state["sigma"]:
                reset()
                state.update(step=-1, sigma=sigma, residual=None, tail=None, consecutive=0,
                             computed=0, skipped=0)
            if state["step"] == -1 or sigma < state["sigma"]:
                state["step"] += 1
            state["sigma"] = sigma
            before = args["img"].clone()  # Native blocks update the stream in place.
            out = execute(0, args, extra)
            residual = out["img"] - before
            total = options["sample_sigmas"].shape[-1] - 1
            skip = False
            ratio = None
            if (threshold > 0 and state["residual"] is not None and state["tail"] is not None
                    and start_step <= state["step"] < total - end_dense_steps
                    and state["consecutive"] < max_consecutive_skips):
                # Check actual target-video rows; packed text/reference magnitudes
                # would otherwise hide changes in the generated dancer.
                a, b, _ = next(s for s in options["minimax_h3_layout"].segments if s[2] == "video")
                old = state["residual"][a:b].float()
                delta = (residual[a:b].float() - old).abs().mean(-1)
                ratio = float((delta / old.abs().mean(-1).clamp_min(1e-8)).mean())
                skip = ratio <= threshold
            state.update(residual=residual, skip=skip,
                         consecutive=state["consecutive"] + 1 if skip else 0,
                         after_first=None if skip else out["img"].clone())
            state["skipped" if skip else "computed"] += 1
            from . import server_runtime
            if server_runtime._active is not None:
                server_runtime._active.data["first_block_cache"] = {
                    "threshold": threshold, "computed": state["computed"], "skipped": state["skipped"],
                    "last_ratio": ratio, "max_consecutive_skips": max_consecutive_skips}
            return out

        def middle(index):
            def call(args, extra):
                return {"img": args["img"]} if state["skip"] else execute(index, args, extra)
            return call

        def last(args, extra):
            if state["skip"]:
                return {"img": args["img"] + state["tail"]}
            out = execute(len(blocks) - 1, args, extra)
            state["tail"] = out["img"] - state["after_first"]
            state["after_first"] = None
            return out

        m.set_model_patch_replace(first, "dit", "double_block", 0)
        for index in range(1, len(blocks) - 1):
            m.set_model_patch_replace(middle(index), "dit", "double_block", index)
        m.set_model_patch_replace(last, "dit", "double_block", len(blocks) - 1)
        m.add_callback_with_key(CallbacksMP.ON_CLEANUP, "h3_first_block_cache", reset)
        return (m,)
