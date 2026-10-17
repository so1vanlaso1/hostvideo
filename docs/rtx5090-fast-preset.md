# RTX 5090 Ref2VA fast preset (opt-in)

This path **preserves** the 25-step Ref2VA quality workflow and all dance/outfit reference modes. No FL2VA or FastH3 text-only checkpoint is substituted.

## What changes

- A new importable browser graph: `ref2va_rtx5090_fast_5s.json`.
- It uses the **existing and supported** Ref2VA Turbo **4-step v0.1** LoRA, `beta` scheduler, 1344×768, five seconds, and the smaller NVFP4 Qwen3-VL text encoder.
- A dedicated 5090 smoke-test job: `examples/rtx5090-fast.json`.
- The GPU preflight recognizes the 5090 without emitting the old 5060-Ti mismatch warning.
- No unverified 8-step Ref2VA LoRA is installed or referenced. LightX2V's 768p **8-step** release is **FL2VA only**, not Ref2VA.

## Launch in a separate RTX 5090 environment

On a Linux x86-64 CUDA 13 development host with an RTX 5090 and sufficient disk space:

```bash
git clone https://github.com/so1vanlaso1/hostvideo.git
cd hostvideo
# For this branch before merge:
git checkout feature/rtx5090-ref2va-fast-preset
bash setup.sh --profile primary
```

On an existing instance, **stop the managed server before rerunning setup**, and back up current outputs first. An existing 5060-Ti container may have supervisor managing ComfyUI; use its supervisor controls rather than starting a second server.

`primary` chooses the 19.5 GiB pruned INT8 Ref2VA DiT + approximately 14.6 GiB NVFP4 encoder. **Together they exceed 32 GB**; dynamic residency/offload remains necessary, but the smaller encoder avoids the larger INT8 Qwen memory footprint. Neither checkpoint is guaranteed to remain simultaneously GPU-resident. The diffusion checkpoint and the encoder can be loaded sequentially.

```bash
# CLI fast path (prompt only, no reference files needed)
source .h3-vast.env
source "$H3_ROOT/venv/bin/activate"
h3 generate --profile primary --job examples/rtx5090-fast.json
```

For browser generation, open ComfyUI using SSH forwarding and import `ref2va_rtx5090_fast_5s.json` from the exported workflows directory. Supply a character image, outfit image, and reference dance video; change nothing else for the first test. The browser preset does not enable an external Sol-Attention extension, which has not been validated with this pinned ComfyUI build.

## Compare fairly

Render the **same** 5-second job, seed, resolution, images, and audio under:
1. quality: 25 steps, Turbo off, `simple` schedule;
2. fast: 4 steps, Turbo on, `beta` schedule.

The different schedules are part of the intended respective recipes; report that distinction alongside the result. Record total wall time, `h3_sampling`, `qwen_encode`, `video_vae_decode`, max CUDA allocated/reserved, RAM RSS, and video/audio correctness from the generated `report.json`.

**Quality check:** compare face identity, outfit details, choreography, hands, frame stability, and sound. 4-step Turbo is distilled for ~544p, so 1344×768 is an extrapolation; reduce `megapixels` to 0.5 if artifacts outweigh speed gains. Check short output before trying 15 seconds (362 frames).

## Important limitations

This is an *untested GPU preset*, not an RTX 5090 speed claim. The RTX 5060-Ti logs in this repository do not establish 5090 performance, quality, or memory fit. Keep `H3_PROFILE=primary` in both ComfyUI runtime and CLI. Do not assume `--reserve-vram 1` means all 32 GB are available to the model. Report only measured improvements after a real GPU run.

Official LoRA table: https://github.com/ModelTC/Minimax-H3-Turbo
