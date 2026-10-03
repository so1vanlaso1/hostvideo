# Running Hostvideo on this Vast instance

Runtime: `/workspace/minimax-h3`

Source: `/workspace/hostvideo`

Browser: http://118.100.214.170:61232/ (public, no authentication)

Open the browser URL directly; no login or token is required. ComfyUI is excluded from Caddy authentication on external port 10100. Anyone with the address can submit GPU jobs. Vast rental charges still apply. Private alternative from your own computer:

```bash
ssh -p 61198 -L 8188:127.0.0.1:8188 root@118.100.214.170
```

Then open http://localhost:8188.

The server is managed by supervisor and starts automatically on container startup:

```bash
supervisorctl status hostvideo
supervisorctl restart hostvideo
tail -f /var/log/portal/hostvideo.log
```

In ComfyUI’s Workflows browser, select **`ref2va_quality_5s_098mp.json`**. Upload your character, outfit, and dance references before queueing. This starts at 5 seconds and native 1344×768 resolution (0.98 MP), with 25 base steps, `res_multistep` / `simple`, Turbo disabled, INT8 Qwen3-VL encoder, `max` image references and H.264 CRF 16. The main `ref2va_16gb_ui.json` workflow uses the same quality settings at 15 seconds. Reload the workflow from the browser to replace settings already open in a tab. Previous workflow files are backed up under `/workspace/minimax-h3/backups/pre-quality-workflows/`. Outputs are under `/workspace/minimax-h3/output/`; CLI jobs and downloaded outputs are under `/workspace/minimax-h3/jobs/`.

For CLI use:

```bash
export H3_ROOT=/workspace/minimax-h3
source "$H3_ROOT/venv/bin/activate"
cd /workspace/hostvideo
h3 generate --job examples/job.json
```

Fill the asset paths in the job first. Setup logs and validation logs are in `/workspace/minimax-h3/logs/`. Dependency and GPU validation reports are in `/workspace/minimax-h3/locks/`.

This instance has no persistent host volume. Stop/start preserves the container disk; recycle/destroy deletes it. Download accepted outputs before recycle/destroy.

Earlier speed validation: all five checkpoint SHA256 checks, GPU and quantization kernel checks, 19 project tests, and 8 native integration tests passed. The no-reference smoke render produced a 73-frame, 864×480 MP4 with generated audio in 86.8 seconds with no fallback. The test video is `/workspace/minimax-h3/jobs/20261003-162148-4b88bb68a3/generated.mp4`. Reference quality and 15-second memory fit remain unmeasured.

The Python/CLI default is also quality: `int8-encoder`, 0.98 MP, 25 steps, Turbo off, `max` references, and a six-hour timeout. Automatic OOM resolution/reference reductions are disabled. The main diffusion model and video VAE remain INT8 ConvRot; audio VAE remains FP32. Public access persists in `/workspace/.env` via `AUTH_EXCLUDE=10100`; proxy header handling is configured with `CADDY_HEADER_UP_LOCALHOST=8188`.

Quality validation: the new INT8 encoder SHA256 verification and CUDA quantization kernels passed, along with 21 project tests and 9 native integration tests. A no-reference 22-frame render completed at 1344×768 with 25 base steps, Turbo off, generated audio and no fallback in 155.2 seconds. Report: `/workspace/minimax-h3/jobs/20261003-163828-77cdb88478/report.json`. Five-second reference quality and 15-second memory fit are still unmeasured.
