"""Thin native-node adapters. No H3 architecture, sampling or VAE code is copied."""
from contextlib import ExitStack
from pathlib import Path
import json
import tempfile
import time
import uuid

import folder_paths
import nodes
import torch
from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, BasicScheduler
from comfy_extras.nodes_audio import VAEDecodeAudio, LoadAudio
from comfy_extras.nodes_video import SaveVideo

from . import server_runtime as runtime
from .media import prepare_image, prepare_video
from .references import canvas, PromptBuilder, Reference


def schema_for(native, node_id):
    schema = native.define_schema()
    schema.node_id = node_id
    schema.display_name = node_id
    schema.category = "H3 Pipeline"
    schema.inputs.append(io.String.Input("context", force_input=True))
    return schema


class H3Begin:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job_id": ("STRING", {"default": "browser"}),
                             "memory_level": ("INT", {"default": 0, "min": 0, "max": 2})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "begin"
    CATEGORY = "H3 Pipeline"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def begin(self, job_id, memory_level):
        if job_id == "browser":
            job_id = "browser-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        return (runtime.begin(job_id, folder_paths.get_output_directory(), memory_level),)


class H3RecordSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"context": ("STRING", {"forceInput": True}),
                             "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION}),
                             "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION}),
                             "length": ("INT", {"default": 362, "min": 5, "max": 362}),
                             "prompt": ("STRING", {"default": "", "multiline": True}),
                             "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
                             "diffusion": ("STRING", {"default": "minimax_h3_ref2va_pruned_int8_convrot.safetensors"}),
                             "encoder": ("STRING", {"default": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"}),
                             "ref_image_size": (["max", "match"],),
                             "reference_mapping": ("STRING", {"default": "[]", "multiline": True})},
                "optional": {"turbo": ("BOOLEAN", {"default": False}),
                             "lora": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "record"
    CATEGORY = "H3 Pipeline"

    def record(self, context, turbo=False, lora="", **settings):
        from .references import MAX_PIXELS
        if settings["width"] * settings["height"] > MAX_PIXELS:
            raise ValueError("Canvas exceeds native H3 1344x768 pixel area")
        if settings["length"] % 17 != 5:
            raise ValueError("Frame count must follow H3's 17k+5 rule")
        settings["reference_mapping"] = json.loads(settings["reference_mapping"])
        settings.update(actual_duration=settings["length"] / 24, frame_count=settings["length"],
                        megapixels=settings["width"] * settings["height"] / 1024**2,
                        sampler="res_multistep", guidance=1, batch_size=1,
                        video_vae="minimax_h3_video_vae_int8_convrot.safetensors",
                        audio_vae="minimax_h3_audio_vae_fp32.safetensors",
                        lora=lora or None, turbo=turbo)
        runtime.current(context).data.update(settings)
        return (context,)


def profiled_loader(native, node_id, label):
    """Keep native V1 loader schemas, model selection and execution."""
    method = native.FUNCTION

    class Loader(native):
        CATEGORY = "H3 Pipeline"

        @classmethod
        def INPUT_TYPES(cls):
            inputs = native.INPUT_TYPES()
            inputs["required"]["context"] = ("STRING", {"forceInput": True})
            return inputs

    def load(self, context, **kwargs):
        with runtime.stage(context, label) as monitor:
            for field, target in (("unet_name", "diffusion"), ("clip_name", "encoder"), ("lora_name", "lora")):
                if field in kwargs:
                    monitor.data[target] = kwargs[field]
                    if target == "lora":
                        monitor.data["turbo"] = bool(kwargs.get("strength_model", 0))
            if "vae_name" in kwargs:
                target = "audio_vae" if "audio_vae" in kwargs["vae_name"] else "video_vae"
                monitor.data[target] = kwargs["vae_name"]
            quantization = {}
            for key in ("diffusion", "encoder", "video_vae", "audio_vae", "lora"):
                name = monitor.data.get(key) or ""
                for kind in ("nvfp4_awq", "int8_convrot", "fp8_scaled", "fp32", "bf16"):
                    if kind in name:
                        quantization[key] = kind
                        break
            monitor.data["quantization"] = quantization
            return getattr(super(Loader, self), method)(**kwargs)
    setattr(Loader, method, load)
    Loader.__name__ = node_id
    return Loader


H3UNETLoader = profiled_loader(nodes.UNETLoader, "H3UNETLoader", "diffusion_checkpoint_load")
H3CLIPLoader = profiled_loader(nodes.CLIPLoader, "H3CLIPLoader", "qwen_checkpoint_load")
H3VAELoader = profiled_loader(nodes.VAELoader, "H3VAELoader", "vae_checkpoint_load")
H3LoraLoader = profiled_loader(nodes.LoraLoaderModelOnly, "H3LoraLoader", "lora_load")


class H3ReferenceToVideo(MiniMaxH3ReferenceToVideo):
    @classmethod
    def define_schema(cls):
        return schema_for(MiniMaxH3ReferenceToVideo, "H3ReferenceToVideo")

    @classmethod
    def execute(cls, context, **kwargs):
        with runtime.stage(context, "reference_conditioning_and_latents") as monitor, ExitStack() as stack:
            for key, label in (("clip", "qwen_encode"), ("vae", "reference_video_vae_encode"), ("audio_vae", "reference_audio_vae_encode")):
                obj = kwargs.get(key)
                if obj is not None:
                    stack.enter_context(runtime.measure_method(obj, "encode_from_tokens_scheduled" if key == "clip" else "encode", monitor, label))
            result = super().execute(**kwargs)
            # The encoder is only explicitly evicted on an OOM recovery attempt.
            if monitor.data["memory_level"]:
                runtime.memory().offload_encoder(kwargs["clip"])
            return result


class H3DanceAddGuide(MiniMaxH3AddGuide):
    @classmethod
    def define_schema(cls):
        return schema_for(MiniMaxH3AddGuide, "H3DanceAddGuide")

    @classmethod
    def execute(cls, context, **kwargs):
        # Use the same attention/offloading policy as reference VAE encoding.
        with runtime.stage(context, "reference_conditioning_and_latents"):
            return super().execute(**kwargs)


class H3Sampler(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        return schema_for(SamplerCustomAdvanced, "H3Sampler")

    @classmethod
    def execute(cls, context, **kwargs):
        with runtime.stage(context, "h3_sampling") as monitor:
            monitor.data["steps"] = len(kwargs["sigmas"]) - 1
            return super().execute(**kwargs)


class H3Scheduler(BasicScheduler):
    @classmethod
    def define_schema(cls):
        return schema_for(BasicScheduler, "H3Scheduler")

    @classmethod
    def execute(cls, context, **kwargs):
        with runtime.stage(context, "sampling_schedule") as monitor:
            monitor.data.update(scheduler=kwargs["scheduler"], steps=kwargs["steps"], denoise=kwargs["denoise"])
            return super().execute(**kwargs)


class H3VideoDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "context": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "H3 Pipeline"

    def decode(self, samples, vae, context):
        with runtime.stage(context, "video_vae_decode") as monitor:
            images = runtime.memory().video_decode(vae, samples)
            monitor.data["decoded_video_device"] = str(images.device)
            monitor.data["decoded_video_dtype"] = str(images.dtype)
            return (images,)


class H3AudioDecode(VAEDecodeAudio):
    @classmethod
    def define_schema(cls):
        schema = schema_for(VAEDecodeAudio, "H3AudioDecode")
        # Force video decode/CPU transfer to finish before audio decode begins.
        schema.inputs.append(io.Image.Input("video_ready"))
        return schema

    @classmethod
    def execute(cls, context, video_ready, **kwargs):
        with runtime.stage(context, "audio_vae_decode"):
            return io.NodeOutput(runtime.memory().audio_to_cpu(super().execute(**kwargs)[0]))


class H3SaveVideo(SaveVideo):
    @classmethod
    def define_schema(cls):
        return schema_for(SaveVideo, "H3SaveVideo")

    @classmethod
    def execute(cls, context, video, filename_prefix, **kwargs):
        with runtime.stage(context, "video_encoding"):
            # Each browser run gets a separate output folder, even with an unchanged graph.
            result = super().execute(video=video, filename_prefix=f"h3/{context}/clip", **kwargs)
        report = runtime.finish(context)
        preview = result.ui.as_dict() if hasattr(result.ui, "as_dict") else result.ui
        return io.NodeOutput(*result.args, ui={**preview, "h3_report": [report]})


class H3LoadReferenceVideo:
    @classmethod
    def INPUT_TYPES(cls):
        # Recursive listing supports uploads from the Python client as well as browser files.
        files = [str(p.relative_to(folder_paths.get_input_directory())) for p in Path(folder_paths.get_input_directory()).rglob("*")
                 if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")]
        return {"required": {"file": (sorted(files), {"video_upload": True}),
                             "duration": ("FLOAT", {"default": 15, "min": 0.2, "max": 16}),
                             "megapixels": ("FLOAT", {"default": 0.98, "min": 0.1, "max": 1}),
                             "aspect_ratio": (["16:9", "9:16", "1:1"],),
                             "start": ("FLOAT", {"default": 0, "min": 0}),
                             "include_audio": ("BOOLEAN", {"default": True}),
                             "context": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "paired_audio", "metadata")
    FUNCTION = "load"
    CATEGORY = "H3 Pipeline"

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        return folder_paths.exists_annotated_filepath(file) or f"Missing video {file}"

    def load(self, file, duration, megapixels, aspect_ratio, start, include_audio, context):
        import av
        import numpy as np
        with runtime.stage(context, "reference_video_preprocessing") as monitor:
            width, height = canvas(megapixels, aspect_ratio)
            with tempfile.TemporaryDirectory(prefix="h3-reference-", dir=folder_paths.get_temp_directory()) as work:
                prepared = Path(work) / "reference.mp4"
                info = prepare_video(folder_paths.get_annotated_filepath(file), prepared, duration, width, height, start, include_audio)
                count = info["usable_frames"]
                # One FP16 CPU allocation; never stack a second full float32 video tensor.
                frames = torch.empty((count, info["height"], info["width"], 3), dtype=torch.float16, device="cpu")
                with av.open(str(prepared)) as container:
                    decoded = 0
                    for i, frame in enumerate(container.decode(video=0)):
                        if i >= count:
                            break
                        array = frame.to_ndarray(format="rgb24")
                        frames[i].copy_(torch.from_numpy(array))
                        frames[i].div_(255)
                        decoded += 1
                    if decoded != count:
                        raise RuntimeError(f"Decoded {decoded} frames, expected {count}")
                audio = None
                if info["paired_audio"]:
                    # Native audio loader uses the exact same normalized clip as the frame decoder.
                    from comfy_extras.nodes_audio import load
                    waveform, rate = load(str(prepared))
                    audio = {"waveform": waveform[:, :round(count / 24 * rate)].unsqueeze(0).cpu(), "sample_rate": rate}
                info["source"] = file
                info.pop("prepared", None)
                monitor.data["reference_video"] = info
                return frames, audio, json.dumps(info)


class H3LoadReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": nodes.LoadImage.INPUT_TYPES()["required"]["image"],
                             "megapixels": ("FLOAT", {"default": 0.98, "min": 0.1, "max": 1}),
                             "ref_image_size": (["max", "match"],), "aspect_ratio": (["16:9", "9:16", "1:1"],),
                             "context": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load"
    CATEGORY = "H3 Pipeline"

    def load(self, image, megapixels, ref_image_size, aspect_ratio, context):
        from PIL import Image
        import numpy as np
        with runtime.stage(context, "reference_image_preprocessing"):
            width, height = canvas(megapixels, aspect_ratio)
            with tempfile.TemporaryDirectory(prefix="h3-image-", dir=folder_paths.get_temp_directory()) as work:
                destination = Path(work) / "reference.png"
                prepare_image(folder_paths.get_annotated_filepath(image), destination, width, height, ref_image_size)
                with Image.open(destination) as source:
                    pixels = torch.from_numpy(np.array(source)).to(dtype=torch.float16).div_(255).unsqueeze(0)
            return pixels, image


class H3ReferencePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"default": "A realistic full-body dance with cinematic lighting.", "multiline": True}),
                             "video_metadata": ("STRING", {"forceInput": True}),
                             "character_image": ("STRING", {"default": "character.png"}),
                             "clothing_image": ("STRING", {"default": "outfit.png"})}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "reference_mapping")
    FUNCTION = "build"
    CATEGORY = "H3 Pipeline"

    def build(self, prompt, video_metadata, character_image, clothing_image):
        metadata = json.loads(video_metadata)
        refs = [Reference("<Picture 1>", "character identity and appearance", character_image),
                Reference("<Picture 2>", "clothing and garment details", clothing_image)]
        if metadata["paired_audio"]:
            refs.append(Reference("<Audio 1>", "soundtrack, rhythm and atmosphere paired with <Video 1>", metadata["source"]))
        refs.append(Reference("<Video 1>", "choreography, body movement, timing and camera movement", metadata["source"]))
        return PromptBuilder.build(prompt, refs), json.dumps([r.dict() for r in refs])


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3Begin, H3RecordSettings, H3UNETLoader, H3CLIPLoader, H3VAELoader, H3LoraLoader,
    H3VideoDecode, H3LoadReferenceVideo, H3LoadReferenceImage, H3ReferencePrompt)}


class H3Extension(ComfyExtension):
    async def get_node_list(self):
        return [H3ReferenceToVideo, H3DanceAddGuide, H3Sampler, H3Scheduler, H3AudioDecode, H3SaveVideo]


async def comfy_entrypoint():
    return H3Extension()
