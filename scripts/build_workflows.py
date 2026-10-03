#!/usr/bin/env python3
"""Rebuild derivatives from the pristine pinned official UI graph."""
from pathlib import Path
import copy
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_pipeline.config import ASSETS, model_files, write_json


def api_graph():
    models = model_files()
    graph = {}
    def add(node_id, class_type, **inputs):
        graph[str(node_id)] = {"class_type": class_type, "inputs": inputs}
    context = ["201", 0]
    add(200, "H3Begin", job_id="example", memory_level=0)
    add(201, "H3RecordSettings", context=["200", 0], width=1056, height=608, length=362,
        prompt="A realistic dance with native synchronized audio.", seed=1,
        diffusion=models["diffusion"]["path"], encoder=models["encoder"]["path"], ref_image_size="match", reference_mapping="[]")
    add(127, "H3UNETLoader", unet_name=Path(models["diffusion"]["path"]).name, weight_dtype="default", context=context)
    add(128, "H3CLIPLoader", clip_name=Path(models["encoder"]["path"]).name, type="minimax", device="default", context=context)
    add(119, "H3VAELoader", vae_name=Path(models["video_vae"]["path"]).name, context=context)
    add(120, "H3VAELoader", vae_name=Path(models["audio_vae"]["path"]).name, context=context)
    add(145, "H3LoraLoader", model=["127", 0], lora_name=Path(models["lora"]["path"]).name, strength_model=1.0, context=context)
    add(136, "H3ReferenceToVideo", clip=["128", 0], vae=["119", 0], audio_vae=["120", 0],
        prompt="A realistic dance with native synchronized audio.", width=1056, height=608, length=362, ref_image_size="match", context=context)
    add(123, "KSamplerSelect", sampler_name="res_multistep")
    add(124, "H3Scheduler", model=["145", 0], scheduler="beta", steps=4, denoise=1.0, context=context)
    add(126, "BasicGuider", model=["145", 0], conditioning=["136", 0])
    add(129, "RandomNoise", noise_seed=1)
    add(125, "H3Sampler", noise=["129", 0], guider=["126", 0], sampler=["123", 0], sigmas=["124", 0], latent_image=["136", 1], context=context)
    add(122, "H3VideoDecode", samples=["125", 0], vae=["119", 0], context=context)
    add(121, "H3AudioDecode", samples=["125", 0], vae=["120", 0], video_ready=["122", 0], context=context)
    add(130, "CreateVideo", images=["122", 0], audio=["121", 0], fps=24.0, bit_depth=8, color_space="sRGB")
    add(92, "H3SaveVideo", video=["130", 0], filename_prefix="h3/example/clip", context=context,
        **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"})
    return graph


def ui_graph():
    graph = json.loads((ASSETS / "workflows" / "official_ref2va.json").read_text())
    graph["id"] = "cc03804e-2e63-4a04-bfc5-2fcd2e88ba12"
    graph["revision"] = 1
    by_id = {n["id"]: n for n in graph["nodes"]}
    swaps = {92: "H3SaveVideo", 119: "H3VAELoader", 120: "H3VAELoader", 121: "H3AudioDecode",
             122: "H3VideoDecode", 124: "H3Scheduler", 125: "H3Sampler", 127: "H3UNETLoader", 128: "H3CLIPLoader",
             136: "H3ReferenceToVideo", 137: "H3LoadReferenceImage", 139: "H3LoadReferenceImage", 145: "H3LoraLoader"}
    for node_id, name in swaps.items():
        node = by_id[node_id]
        node["type"] = name
        node["properties"].pop("cnr_id", None)
        node["properties"].pop("ver", None)
        node["properties"]["Node name for S&R"] = name
    by_id[115]["widgets_values"] = ["16:9 (Widescreen)", 0.6, 32]
    by_id[115]["widgets_values_named"].update(megapixels=0.6)
    by_id[132]["widgets_values"] = [15.0]
    by_id[132]["widgets_values_named"] = {"value": 15.0}
    by_id[124]["widgets_values"][0] = "beta"
    by_id[124]["widgets_values_named"]["scheduler"] = "beta"
    by_id[146]["widgets_values"] = [True]
    by_id[146]["widgets_values_named"] = {"value": True}
    by_id[138]["widgets_values"] = ["A realistic full-body dance in cinematic lighting. Preserve the dancer's identity and the exact referenced clothing."]
    by_id[138]["widgets_values_named"] = {"value": by_id[138]["widgets_values"][0]}
    by_id[92]["widgets_values"] = ["h3/browser/clip", "mp4", "h264", "auto"]
    by_id[92]["widgets_values_named"] = {"filename_prefix": "h3/browser/clip", "format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}

    def new(node_id, kind, widgets, outputs, pos):
        node = {"id": node_id, "type": kind, "pos": pos, "size": [420, 250], "flags": {},
                "order": len(graph["nodes"]), "mode": 0, "inputs": [],
                "outputs": [{"name": name, "type": typ, "links": [], "slot_index": i} for i, (name, typ) in enumerate(outputs)],
                "properties": {"Node name for S&R": kind}, "widgets_values": widgets}
        graph["nodes"].append(node)
        by_id[node_id] = node
        return node

    def disconnect(target, field):
        node = by_id[target]
        found = next((i for i in node["inputs"] if i["name"] == field), None)
        if found and found.get("link"):
            old_id = found["link"]
            old = next(l for l in graph["links"] if l[0] == old_id)
            source = by_id[old[1]]["outputs"][old[2]]
            source["links"].remove(old_id)
            graph["links"] = [l for l in graph["links"] if l[0] != old_id]
            found["link"] = None

    link_id = graph["last_link_id"]
    def connect(source, slot, target, field, typ, widget=False):
        nonlocal link_id
        disconnect(target, field)
        target_node = by_id[target]
        item = next((i for i in target_node["inputs"] if i["name"] == field), None)
        if item is None:
            item = {"name": field, "type": typ, "link": None}
            if widget:
                item["widget"] = {"name": field}
            target_node["inputs"].append(item)
        link_id += 1
        item["link"] = link_id
        by_id[source]["outputs"][slot].setdefault("links", [])
        if by_id[source]["outputs"][slot]["links"] is None:
            by_id[source]["outputs"][slot]["links"] = []
        by_id[source]["outputs"][slot]["links"].append(link_id)
        graph["links"].append([link_id, source, slot, target, target_node["inputs"].index(item), typ])

    new(200, "H3Begin", ["browser", 0], [("context", "STRING")], [-2100, 4300])
    by_id[200]["widgets_values_named"] = {"job_id": "browser", "memory_level": 0}
    settings = api_graph()["201"]["inputs"]
    new(201, "H3RecordSettings", [1056, 608, 362, "", 1, settings["diffusion"], settings["encoder"], "match", "[]"], [("context", "STRING")], [-900, 4300])
    by_id[201]["widgets_values_named"] = {key: value for key, value in settings.items() if key != "context"}
    new(310, "H3LoadReferenceVideo", ["dance.mp4", 15.0, 0.6, "16:9", 0.0, True],
        [("frames", "IMAGE"), ("paired_audio", "AUDIO"), ("metadata", "STRING")], [-1900, 6500])
    by_id[310]["widgets_values_named"] = {"file": "dance.mp4", "duration": 15.0, "megapixels": 0.6, "aspect_ratio": "16:9", "start": 0.0, "include_audio": True}
    new(312, "H3ReferencePrompt", [by_id[138]["widgets_values"][0], "character.png", "outfit.png"],
        [("prompt", "STRING"), ("reference_mapping", "STRING")], [-950, 6500])
    by_id[312]["widgets_values_named"] = {"prompt": by_id[138]["widgets_values"][0], "character_image": "character.png", "clothing_image": "outfit.png"}
    new(314, "PrimitiveInt", [1, "randomize"], [("INT", "INT")], [-1450, 4300])
    connect(200, 0, 201, "context", "STRING")
    for node_id in swaps:
        connect(200 if node_id in (137, 139) else 201, 0, node_id, "context", "STRING")
    connect(200, 0, 310, "context", "STRING")
    connect(310, 0, 136, "ref_videos.ref_video_0", "IMAGE")
    connect(310, 1, 136, "ref_video_audios.ref_video_audio_0", "AUDIO")
    connect(310, 2, 312, "video_metadata", "STRING")
    connect(138, 0, 312, "prompt", "STRING", True)
    connect(312, 0, 136, "prompt", "STRING", True)
    connect(312, 0, 201, "prompt", "STRING", True)
    connect(312, 1, 201, "reference_mapping", "STRING", True)
    connect(115, 0, 201, "width", "INT", True)
    connect(115, 1, 201, "height", "INT", True)
    connect(131, 0, 201, "length", "INT", True)
    connect(314, 0, 129, "noise_seed", "INT", True)
    connect(314, 0, 201, "seed", "INT", True)
    connect(132, 0, 310, "duration", "FLOAT", True)
    connect(122, 0, 121, "video_ready", "IMAGE")
    for node_id, filename, prompt_field in ((137, "character.png", "character_image"), (139, "outfit.png", "clothing_image")):
        by_id[node_id]["widgets_values"] = [filename, 0.6, "match", "16:9"]
        by_id[node_id]["widgets_values_named"] = {"image": filename, "megapixels": 0.6, "ref_image_size": "match", "aspect_ratio": "16:9"}
        # Native LoadImage's unused mask output becomes the path output of this CPU loader.
        by_id[node_id]["outputs"][1] = {"name": "path", "type": "STRING", "links": [], "slot_index": 1}
        connect(node_id, 1, 312, prompt_field, "STRING", True)
    graph.update(last_node_id=314, last_link_id=link_id)
    graph.setdefault("extra", {})["h3_pipeline"] = {"instructions": "Use the provided startup script. Set the same megapixels/aspect on the Resolution Selector and reference loaders. Upload character.png, outfit.png and dance.mp4. Turbo is enabled."}
    return graph


if __name__ == "__main__":
    write_json(ASSETS / "workflows" / "ref2va_api.json", api_graph())
    write_json(ASSETS / "workflows" / "ref2va_16gb_ui.json", ui_graph())
    print("Built API and UI workflows from the pinned official template")
