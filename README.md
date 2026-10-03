# MiniMax H3 Ref2VA — Vast.ai / RTX 5060 Ti 16 GB

This implementation follows `guide.txt`: one reference-driven clip, approximately 15 seconds, native generated audio, Comfy-Org checkpoints, four-step Ref2V Turbo and CPU offloading. It provides a ComfyUI browser graph and a Python/CLI client. **The 15-second fit and reference quality must be measured on your Vast GPU.** Local tests do not establish GPU performance.

## 1. Rent and upload

Choose a **Linux x86-64 Ubuntu/PyTorch SSH template**, with one **RTX 5060 Ti 16 GB**, a compatible **R580-or-newer host driver**, **64 GB allocated RAM preferred** (32 GB guide minimum), eight CPU cores and a **150 GB or larger NVMe volume mounted at `/data`**. Avoid a template that automatically starts another ComfyUI server. The NVIDIA driver belongs to the Vast host.

Install these basic tools in the remote container if absent:

```bash
apt-get update
apt-get install -y git curl ca-certificates ffmpeg tmux python3 python3-venv util-linux
```

From your computer, use the SSH host/port shown by Vast:

```bash
scp -P SSH_PORT -r /Users/nhonyqua/vidgen/Sourcecode root@SSH_HOST:/data/h3-source
ssh -p SSH_PORT root@SSH_HOST
```

Then on Vast:

```bash
cd /data/h3-source
export H3_ROOT=/data/minimax-h3
bash scripts/setup_vast.sh --download-models
source "$H3_ROOT/venv/bin/activate"
```

Setup pins ComfyUI, the official workflow and model revisions. It installs Python 3.11, PyTorch **2.13.0 / CUDA 13.0**, torchvision 0.28.0, native ComfyUI dependencies and this package. If Python 3.11 is absent, a separate uv bootstrap environment supplies it. Setup saves the resolved dependency lock, checks CUDA execution and runs small quantization kernel checks **before downloading weights**. No host driver or system Python is replaced.

The default download contains exactly five checkpoints, approximately **39.1 GiB total**:

| Component | Checkpoint |
|---|---|
| Diffusion | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| LoRA | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` |

Downloads resume through Hugging Face's local cache and are verified against pinned LFS SHA256 hashes. Existing wrong-sized or corrupt models are reported for inspection; they are not deleted. `h3 download-models --verify-existing` rehashes completed files. Keep the download cache when resuming an interrupted download.

The runtime root contains `ComfyUI/`, `venv/`, `models/`, `input/`, `output/`, `jobs/`, `logs/`, `locks/`, `temp/`, `user/` and owner-only `private/`. Keep `/data/h3-source` present because the package and custom nodes are installed from it. A mounted volume is still not an off-host backup.

## 2. Start ComfyUI and authenticated browser access

Start ComfyUI in a persistent terminal:

```bash
tmux new -s h3-comfy
cd /data/h3-source
export H3_ROOT=/data/minimax-h3
bash scripts/start_comfy.sh 2>&1 | tee "$H3_ROOT/logs/comfy.log"
```

Detach with `Ctrl-B`, then `D`. The server listens on **127.0.0.1:8188**, uses dynamic VRAM, one GiB reserved VRAM, FP16/CPU intermediates, PyTorch attention and disabled node-result caching. Native H3 VAE chunking stays intact. INT8 ConvRot must not use Comfy Kitchen attention; Comfy Kitchen's quantization kernels are still used by the checkpoint loaders.

Install ngrok v3 from its [official Linux instructions](https://ngrok.com/download/linux). In the remote terminal, configure secrets privately:

```bash
cd /data/h3-source
export H3_ROOT=/data/minimax-h3
"$H3_ROOT/venv/bin/python" scripts/configure_ngrok.py
tmux new -s h3-ngrok
bash scripts/start_ngrok.sh
```

Open the printed HTTPS URL and use the username/password you selected. The enforced Basic Auth traffic policy protects browser and API access, including WebSockets. The ngrok agent runs in the same container as ComfyUI. Do not expose ports 8188 or 4040 through Vast; keep the original Host header. `NGROK_BIN` can select an installed binary and `H3_PORT` can select another private ComfyUI port. Your ngrok plan must support the configured traffic policy.

To resume after Vast restarts, verify the mounted volume and rerun `start_comfy.sh` and `start_ngrok.sh` in separate tmux sessions. Setup is repeatable and preserves models, outputs and private credentials. Do not rerun setup while generation is active. To rebuild an environment on another host, recreate the venv and use the saved package lock; do not copy a venv across hosts.

## 3. Browser workflow

Load **`h3_pipeline/assets/workflows/ref2va_16gb_ui.json`** in ComfyUI. Setup also copies it to `user/default/workflows`.

1. Upload/select a character image, clothing image and dance video in the corresponding reference loaders.
2. Edit the main text prompt. The prompt builder adds explicit identity, clothing, motion and soundtrack assignments.
3. Start with **3 seconds / 0.4 MP**. Set the same megapixels/aspect in the Resolution Selector and all reference loaders. The duration is shared with video preprocessing.
4. Keep Turbo enabled, four steps, `res_multistep`, `beta`, guidance 1, `match` sizing and one batch. The BasicGuider node is the native no-CFG path equivalent to guidance 1.
5. Queue one generation and inspect the saved MP4 and audio before increasing duration/resolution.

The browser graph is prepared for the primary one-character, one-outfit, one-video use case. The Python API supports up to nine ordered images and optional standalone audio. Additional browser references require wiring the native autogrow inputs and updating the prompt mapping; use the Python API when you want this done automatically.

The browser video loader performs trimming/resizing with ffmpeg **before decoding** and allocates one FP16 CPU frame buffer. The paired soundtrack is cut to the same usable video duration. A silent video supplies no audio reference. `<Audio 1>` is the dance soundtrack when present; standalone audio becomes `<Audio 2>`. Audio references guide newly generated audio; this does not copy the source song unchanged into the final MP4.

Browser jobs produce `output/h3/browser-TIMESTAMP-ID/clip_*.mp4` and `telemetry.json`. Automatic bounded OOM retries belong to the Python/CLI runner. For a browser OOM, change `memory_level` to 1 then 2, use `match`, and reduce all canvas/reference settings to 0.5 and 0.4 MP before reducing duration.

## 4. Command-line and Python generation

Place your assets in `examples/assets/` beside the sample `examples/job.json`, or edit its paths. Paths in a job file are relative to that file. Then run on Vast:

```bash
source /data/minimax-h3/venv/bin/activate
cd /data/h3-source
h3 generate --job examples/job.json
```

The runner uploads preprocessed files to the private ComfyUI API. It saves each attempt's prompt, reference mapping and API graph under `jobs/`, then downloads and verifies the successful MP4. The result includes `video_path`, `report_path`, `workflow_path`, resolved `seed`, `reference_mapping` and `performance_report`.

```python
from h3_pipeline import MiniMaxH3Pipeline

result = MiniMaxH3Pipeline().generate(
    prompt="A realistic full-body dance in a softly lit studio.",
    reference_video="dance.mp4",
    character_images=["character.png"],
    clothing_images=["outfit.png"],
    reference_audio=None,
    duration=15.0,
    megapixels=0.6,
    seed=123456789,
)
print(result.video_path)
```

Other keyword options: `additional_reference_images`, `aspect_ratio` (`16:9`, `9:16`, `1:1`), `ref_image_size` (`match`, `max`), `include_video_audio` (default true), `reference_video_start`, `scheduler` (`beta`, `normal`), `oom_fallback` (default true), and `timeout` in seconds (default 7200).

Image order is character, clothing, additional, retaining list order within each group. Reference tags never silently move. A prompt referring to an unconnected tag fails clearly. If a dance video is silent, a manually written `<Audio 1>` tag is only valid when standalone reference audio is supplied.

The native frame grid is `17k+5`: 3 seconds → 73 frames; 5 seconds → 124; 15 seconds → **362**, or **15.0833 seconds at 24 fps**. A 0.6 MP landscape canvas resolves to **1056×608** using ComfyUI's 1024-based megapixel calculation. Resolution is snapped to 32 and capped at 1344×768 pixel area. Reference video lengths crop down to the valid temporal grid; a 15-second/360-frame source has 345 usable frames. It is not stretched or padded.

On **CUDA OOM only**, the runner retries the same seed/duration with cleanup, encoder offloading and `match`, then stronger residency headroom, then 0.5 and 0.4 MP when smaller than the original request. Maximum five attempts for the default canvas. A change in reference resolution can change conditioning; the seed is preserved but identical output is not promised. Compatibility errors do not trigger resolution retries. A timeout reports the ComfyUI prompt ID and leaves the potentially running job intact—inspect it before resubmitting.

For a remote client, install this package locally, set `server_url` to your ngrok URL, and provide `username`/`password` to `MiniMaxH3Pipeline`. CLI credentials are read from `H3_HTTP_USERNAME` and `H3_HTTP_PASSWORD`; do not put them in job JSON. You can choose a local client `root` for downloaded outputs. The same-host client lock prevents overlapping client retries; ComfyUI serializes execution from all clients.

## 5. GPU acceptance and compatibility profiles

Run these stages in order, inspecting the preceding result before continuing:

```bash
h3 benchmark --job examples/job.json --stage smoke
h3 benchmark --job examples/job.json --stage five-second
h3 benchmark --job examples/job.json --stage target
h3 benchmark --job examples/job.json --stage identity-motion
h3 benchmark --job examples/job.json --stage clothing
```

The first three stages omit references to establish basic execution and memory fit. Identity/motion adds the character and dance video; clothing adds the outfit. Benchmarks disable OOM resolution fallback so a smaller render cannot count as a pass for the requested size. Optional resolution tests are separate commands with `--stage resolution --megapixels 0.7`, `0.8` or `0.98` after the clothing target succeeds. Test `normal` scheduling by editing the job and changing no other parameter.

Acceptance: the requested 362-frame output plays with generated synchronized audio; character face/hair/body remain consistent; garments retain colors, shape and major details; motion resembles the dance reference; hands and limbs remain coherent. Record defects and pass/fail judgments in `ACCEPTANCE.md`; software completion alone does not establish visual quality.

Reports include models/revisions, source and output dimensions, frame count/duration, Qwen encode, reference VAE encode, sampling, video/audio decode and encoding times, native model load/free activity, peak CUDA allocated/reserved memory, and sampled process RAM. Stage timings include nested native operations and must not be summed as independent durations. CPU memory is sampled, so brief peaks may be missed. Total client wall time includes preprocessing, uploads and all retries.

Compatibility checkpoints require explicit selection and their own measurements:

```bash
h3 download-models --profile fp8
H3_PROFILE=fp8 bash scripts/start_comfy.sh
h3 generate --job examples/job.json --profile fp8
```

Profiles: `primary`, `fp8` (diffusion only), `int8-encoder` (encoder only), `fp8-int8-encoder` (both). Set the same `H3_PROFILE` for setup/start and CLI profile for generation. The INT8 encoder is approximately 25.3 GiB on disk and increases RAM pressure. The video VAE remains INT8 ConvRot in every profile, so its kernels still need to pass. Profiles never download BF16 diffusion or an unrelated encoder. Browser checkpoint changes must be made explicitly in its loader nodes.

Retrieve accepted outputs before stopping/destroying the rental:

```bash
scp -P SSH_PORT -r root@SSH_HOST:/data/minimax-h3/jobs ./h3-jobs-backup
scp -P SSH_PORT -r root@SSH_HOST:/data/minimax-h3/output ./h3-output-backup
```

Stop the Vast instance in its dashboard after jobs finish and outputs are backed up. Stopping ComfyUI alone does not stop rental billing. Keep private credentials out of shareable backups.

## Local development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/build_workflows.py
```

Media tests require ffmpeg/ffprobe; set `FFMPEG`/`FFPROBE` to absolute binary paths if they are not on PATH. Optional native contract tests run against the pinned ComfyUI checkout with its dependencies installed:

```bash
H3_COMFY_PATH=/path/to/ComfyUI .venv/bin/python tests/native_contract.py
```

The pristine official graph, its source license, SHA256 and provenance are retained under `h3_pipeline/assets/`. The long-video optimization repository was studied for memory ideas; no restricted runtime code or chaining implementation is included. No model weights or generated demo videos are bundled.
