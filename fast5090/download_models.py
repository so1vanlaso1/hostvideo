#!/usr/bin/env python3
"""Resumable precise model download for standalone Ref2VA workflow."""
import argparse, pathlib
from huggingface_hub import hf_hub_download
# Paths are the official/cited model releases, not unverified third-party mirrors.
BASE=[
 ("Comfy-Org/MiniMax-H3","diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors","diffusion_models"),
 ("Comfy-Org/MiniMax-H3","text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","text_encoders"),
 ("Comfy-Org/MiniMax-H3","vae/minimax_h3_video_vae_fp16.safetensors","vae"),
 ("Comfy-Org/MiniMax-H3","vae/minimax_h3_audio_vae_fp32.safetensors","vae")
]
PDD=("alibaba-pai/MiniMax-H3-Acc-LoRAs","MiniMax-H3-Ref2VA-Acc-8Step.safetensors","pdd_acc")
VSA=[
 ("lightx2v/Minimax-h3-Turbo","minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors","loras"),
 ("barelymining/ComfyUI-MiniMax-H3-FastVideo","fasth3_vsa_gate.safetensors","loras")
]
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--root",type=pathlib.Path,required=True)
 p.add_argument("--preset",choices=["pdd8","vsa4"],default="pdd8")
 a=p.parse_args()
 for repo,filename,folder in BASE+([PDD] if a.preset=="pdd8" else VSA):
  dest=a.root/"models"/folder
  dest.mkdir(parents=True,exist_ok=True)
  print(f"Fetching {repo}/{filename}",flush=True)
  local=hf_hub_download(repo_id=repo,filename=filename,local_dir=str(dest) if "/" not in filename else None)
  # hf_hub_download returns a cached validated file, but folder placement is explicit.
  target=dest/pathlib.Path(filename).name
  if pathlib.Path(local).resolve()!=target.resolve():
   import shutil
   shutil.copy2(local,target)
  print("Ready:",target,flush=True)
if __name__=="__main__":
 main()
