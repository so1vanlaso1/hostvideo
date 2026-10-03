from copy import deepcopy
import json

from .config import ASSETS, model_files


def build_workflow(*, prompt, images, video, audio, include_video_audio, width, height, length,
                   seed, job_id, memory_level=0, ref_image_size="match", profile="primary", scheduler="beta"):
    if scheduler not in ("beta", "normal"):
        raise ValueError("scheduler must be beta or normal")
    graph = json.loads((ASSETS / "workflows" / "ref2va_api.json").read_text())
    files = model_files(profile)
    for node, key, field in (("127", "diffusion", "unet_name"), ("128", "encoder", "clip_name"),
                              ("119", "video_vae", "vae_name"), ("120", "audio_vae", "vae_name"),
                              ("145", "lora", "lora_name")):
        graph[node]["inputs"][field] = files[key]["path"].split("/", 1)[1]
    graph["200"]["inputs"].update(job_id=job_id, memory_level=memory_level)
    graph["124"]["inputs"]["scheduler"] = scheduler
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
                                        ref_image_size=ref_image_size)
    graph["92"]["inputs"]["filename_prefix"] = f"h3/{job_id}/clip"
    return graph


def validate_graph(graph):
    """Structural checks before posting; the server validates native schemas."""
    for node_id, node in graph.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list):
                if len(value) != 2 or str(value[0]) not in graph or not isinstance(value[1], int):
                    raise ValueError(f"Invalid connection {node_id}.{key}: {value}")
    if graph["124"]["inputs"]["steps"] != 4:
        raise ValueError("Turbo graph must use four steps")
    return graph
