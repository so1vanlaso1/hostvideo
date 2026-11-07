# Lean RTX 5090 MiniMax H3, independent from old hostvideo stack

This is an **experimental, unbenchmarked** standalone ComfyUI Ref2VA setup. It does not run `setup.sh` or modify the original pipeline.

## Two separate modes

- `pdd8` (default): 8-step Ref2VA PDD distillation, quantized 19.53 GiB Ref2VA transformer, NVFP4 text encoder, FP16 VAE, native audio. PDD Apply supplies its own sigma schedule and head bank (not a standard LoRA). No other distillation or cache nodes.
- `vsa4` (optional): Ref2VA Turbo 4-step + 75% VSA, author-reported ~72s warm for 5s/1344×768 on RTX 4090. Experimental visual-fidelity trade-off; do not treat it as RTX 5090 timing.

**Warning:** even the DiT+text encoder checkpoints exceed 32 GB in total. Sequential loading / offloading is expected. This setup does not guarantee no memory swap or 15s/1080p fit. Do not assume output is human-realistic just because sampling finishes.

## Install (fresh NVIDIA RTX 5090 Ubuntu host)

Start from a working GPU driver (`nvidia-smi`), supported Python and internet access. `setup.sh` creates a separate virtual environment under `~/fast5090-runtime`.

```bash
git clone https://github.com/so1vanlaso1/hostvideo.git
cd hostvideo
git checkout prototype/rtx5090-lean-ref2va
bash fast5090/setup.sh pdd8
~/fast5090-runtime/venv/bin/python ~/fast5090-runtime/ComfyUI/main.py --listen 127.0.0.1 --port 8188
```

In another shell, place `reference.png` into `~/fast5090-runtime/ComfyUI/input/` and run:

```bash
python3 fast5090/run.py --preset pdd8 --image reference.png --frames 124
```

The launcher validates node availability and returns an output manifest from ComfyUI history. Video files are stored under the ComfyUI output folder.

For VSA, run `bash fast5090/setup.sh vsa4` after stopping ComfyUI, then start it again and use `python3 fast5090/run.py --preset vsa4`. The VSA module requires compiled Sol-Attention-capable comfy-kitchen; **a successful package installation is not proof that its kernels will work**. Resolve backend errors against the upstream project's README before using it.

## Why start with 5 seconds?

Use a stable seed and reference image across runs. Compare realism, face structure, skin, eyes, hand motion, body mechanics, background movement, flicker, and audio, then record cold/warm generation time and memory use. Only then increase clip length to 10 or 15 seconds. Higher resolution does not repair temporal artifacts.

## Future work

A reference video must be wired through the native H3 reference-video preprocessing node. This starter launcher only wires **one image**; it does **not** yet reproduce exact source dance choreography or outfit swapping. Its purpose is to validate fast, believable Ref2VA output before adding more conditioning and complexity.

The PDD custom-node code is under Apache-2.0; H3 weights have separate license terms. Check license suitability, including geographic and commercial restrictions.

Sources:
- https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc
- https://github.com/Kablex/ComfyUI-Ref2VA-VSA
- https://github.com/MiniMax-AI/awesome-minimax-h3-integration
