#!/usr/bin/env bash
set -euo pipefail
# Fresh Ubuntu x86_64 RTX 5090 CUDA-capable host, system NVIDIA driver already installed.
ROOT="${FAST5090_ROOT:-$HOME/fast5090-runtime}"
PROFILE="${1:-pdd8}"
[[ "$PROFILE" == pdd8 || "$PROFILE" == vsa4 ]] || { echo "Usage: bash fast5090/setup.sh pdd8|vsa4"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA driver/GPU missing"; exit 1; }
mkdir -p "$ROOT"
if [[ ! -d "$ROOT/ComfyUI/.git" ]]; then git clone https://github.com/Comfy-Org/ComfyUI.git "$ROOT/ComfyUI"; fi
python3 -m venv "$ROOT/venv"
PY="$ROOT/venv/bin/python"
"$PY" -m pip install --upgrade pip
# Use already installed matching CUDA Torch if present; otherwise install CUDA 13 wheel for Blackwell.
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install -r "$ROOT/ComfyUI/requirements.txt" "huggingface_hub>=0.34,<2"
if [[ ! -d "$ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-PDD-Acc" ]]; then
    git clone https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc "$ROOT/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-PDD-Acc"
fi
if [[ "$PROFILE" == vsa4 && ! -d "$ROOT/ComfyUI/custom_nodes/ComfyUI-Ref2VA-VSA" ]]; then
    git clone https://github.com/Kablex/ComfyUI-Ref2VA-VSA "$ROOT/ComfyUI/custom_nodes/ComfyUI-Ref2VA-VSA"
fi
"$PY" "$(dirname "$0")/download_models.py" --root "$ROOT/ComfyUI" --preset "$PROFILE"
echo "Launch: $PY $ROOT/ComfyUI/main.py --listen 127.0.0.1 --port 8188"
echo "Upload a reference image to $ROOT/ComfyUI/input/reference.png before running fast5090/run.py."
