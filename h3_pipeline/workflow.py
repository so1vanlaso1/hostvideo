from copy import deepcopy
import json

from .config import ASSETS, DEFAULT_PROFILE, model_files


def build_workflow(*, prompt, images, video, audio, include_video_audio, width, height, length,
                   seed, job_id, memory_level=0, ref_image_size="max", profile=DEFAULT_PROFILE,
                   scheduler="simple", steps=25, turbo=False):
    if scheduler not in ("simple", "beta", "normal"):
        raise ValueError("scheduler must be simple, beta or normal")
    if not isinstance(turbo, bool):
        raise ValueError("turbo must be a boolean")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer between 1 and 100")
    if turbo and steps != 4:
        raise ValueError("Turbo requires exactly four steps; disable turbo for base sampling")
    graph = json.loads((ASSETS / "workflows" / "ref2va_api.json").read_text())
    files = model_files(profile)
    for node, key, field in (("127", "diffusion", "unet_name"), ("128", "encoder", "clip_name"),
                              ("119", "video_vae", "vae_name"), ("120", "audio_vae", "vae_name")):
        graph[node]["inputs"][field] = files[key]["path"].split("/", 1)[1]
    graph["200"]["inputs"].update(job_id=job_id, memory_level=memory_level)
    graph["124"]["inputs"].update(scheduler=scheduler, steps=steps)
    if turbo:
        graph["145"] = {"class_type": "H3LoraLoader", "inputs": {
            "model": ["127", 0], "lora_name": files["lora"]["path"].split("/", 1)[1],
            "strength_model": 1.0, "context": ["201", 0]}}
    else:
        graph.pop("145", None)
    for node in ("124", "126"):
        graph[node]["inputs"]["model"] = ["145" if turbo else "127", 0]
    graph["129"]["inputs"]["noise_seed"] = seed
    conditioning = graph["136"]["inputs"]
    conditioning.update(prompt=prompt, width=width, height=height, length=length, ref_image_size=ref_image_size)
    for key in list(conditioning):
        if key.startswith("ref_") and key != "ref_image_size":
            del conditioning[key]
    for i, image in enumerate(images):
        node_id = str(300 + i)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        conditioning[f"ref_images.ref_image_{i}"] = [node_id, 0]
    if video:
        graph["310"] = {"class_type": "H3LoadReferenceVideo", "inputs": {
            "file": video, "duration": length / 24, "megapixels": width * height / (1024 * 1024),
            "aspect_ratio": "16:9" if width >= height else "9:16", "start": 0.0,
            "include_audio": include_video_audio, "context": ["200", 0]}}
        conditioning["ref_videos.ref_video_0"] = ["310", 0]
        if include_video_audio:
            conditioning["ref_video_audios.ref_video_audio_0"] = ["310", 1]
    if audio:
        graph["311"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
        conditioning["ref_audios.ref_audio_0"] = ["311", 0]
    graph["201"]["inputs"].update(width=width, height=height, length=length, prompt=prompt, seed=seed,
                                        diffusion=files["diffusion"]["path"], encoder=files["encoder"]["path"],
                                        ref_image_size=ref_image_size, turbo=turbo,
                                        lora=files["lora"]["path"].split("/", 1)[1] if turbo else "")
    graph["92"]["inputs"]["filename_prefix"] = f"h3/{job_id}/clip"
    return graph


def validate_graph(graph):
    """Structural checks before posting; the server validates native schemas."""
    for node_id, node in graph.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list):
                if len(value) != 2 or str(value[0]) not in graph or not isinstance(value[1], int):
                    raise ValueError(f"Invalid connection {node_id}.{key}: {value}")
    if "145" in graph and graph["124"]["inputs"]["steps"] != 4:
        raise ValueError("Turbo graph must use four steps")
    return graph
