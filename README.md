# MiniMax H3 Ref2VA — Vast.ai / RTX 5060 Ti 16 GB

This implementation generates reference-driven clips with native audio using Comfy-Org checkpoints and CPU offloading. Quality defaults supersede the original speed settings in `guide.txt`: native 1344×768 output, 25 base sampling steps, Turbo disabled, INT8 Qwen3-VL encoder, and `max` image references. It provides ComfyUI browser workflows and a Python/CLI client. **The 15-second fit and reference quality must be measured on your Vast GPU.** Local tests do not establish visual quality.

## 1. Clone and run one script on Vast

Choose a **Linux x86-64 Ubuntu/Debian SSH template with the CUDA 13.0 development toolkit (`nvcc`)**, an **RTX 5060 Ti 16 GB**, an **R580-or-newer host driver**, **64 GB allocated RAM preferred**, eight CPU cores and **150 GB or more disk**. Avoid templates that automatically start another ComfyUI server. The NVIDIA driver belongs to the Vast host.

SSH into your instance, clone this repository and run:

```bash
git clone https://github.com/so1vanlaso1/hostvideo.git
cd hostvideo
bash setup.sh
```

If `git` is missing from the template, install it first with `apt-get update && apt-get install -y git`.

That one script installs missing system tools (as root or with passwordless sudo), prepares Python 3.11, installs the pinned ComfyUI/PyTorch/SageAttention environment, checks CUDA kernels, exports browser workflows, downloads/verifies the models and starts ComfyUI in a detached **tmux** session. ComfyUI keeps running when you disconnect SSH. The first run downloads roughly 50 GiB of models and compiles SageAttention, so allow time for both. Setup reports success only after the API responds.

Storage is selected automatically: a mounted `/data` volume first, then `/workspace` if it exists, otherwise `.runtime/` inside the checkout. Settings are saved in the ignored `.h3-vast.env` file, so later helper commands use the same root, profile and port. Setup logs go to `logs/setup.log` under the runtime root. If setup is interrupted, rerun `bash setup.sh`: completed models are reused and partial Hugging Face downloads resume. Stop an already running server before rerunning setup.

Optional overrides:

```bash
bash setup.sh --root /workspace/minimax-h3 --port 8188 --profile int8-encoder
bash setup.sh --no-start       # Install and download without starting
bash setup.sh --skip-download  # Install only; skip downloads and startup
bash setup.sh --help
```

`H3_ROOT`, `H3_PROFILE` and `H3_PORT` environment overrides still work. `bash scripts/setup_vast.sh --download-models` remains supported; downloads and startup are now enabled by default. Setup retains the pinned ComfyUI/model revisions and Python **3.11**, PyTorch **2.13.0 / CUDA 13.0**, torchvision **0.28.0** and SageAttention **2.2.0**. It supplies Python through uv when needed, saves the resolved dependency lock and runs CUDA/quantization/attention checks **before downloading weights**. No host driver or system Python is replaced.

The default download contains exactly five checkpoints, approximately **49.8 GiB total**:

| Component | Checkpoint |
|---|---|
| Diffusion | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| Optional Turbo LoRA (disabled) | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` |

Downloads resume through Hugging Face's local cache and are verified against pinned LFS SHA256 hashes. Existing wrong-sized or corrupt models are reported for inspection; they are not deleted. `h3 download-models --verify-existing` rehashes completed files. Keep the download cache when resuming an interrupted download.

The runtime root contains `ComfyUI/`, `venv/`, `models/`, `input/`, `output/`, `jobs/`, `logs/`, `locks/`, `temp/`, `user/` and owner-only `private/`. Keep the cloned source directory present because the package and custom nodes are installed from it. `/workspace` and the checkout fallback use the container disk; destruction/recycling can erase them. A mounted volume is still not an off-host backup.

## 2. Managed ComfyUI and browser access

From the cloned repository on Vast:

```bash
bash scripts/vast.sh status
bash scripts/vast.sh logs
bash scripts/vast.sh stop
bash scripts/vast.sh start
bash scripts/vast.sh restart
```

`restart` stops the current process, including any running generation. The tmux service survives an SSH disconnect; after an instance/container restart, run `bash scripts/vast.sh start` again. For a foreground process or your own supervisor service, use `bash scripts/start_comfy.sh`.

For browser access, run this on **your own computer**, using the SSH host/port shown by Vast:

```bash
ssh -p SSH_PORT -L 8188:127.0.0.1:8188 root@SSH_HOST
```

Keep that SSH connection open and visit **http://localhost:8188**. If you chose another `--port`, replace the final `8188` in the forwarding command with that remote port. Setup does not require ngrok or a Vast portal configuration. The optional ngrok helpers remain available: run `source .h3-vast.env`, then `python3 scripts/configure_ngrok.py`, then `bash scripts/start_ngrok.sh` after installing ngrok. The helper creates a browser login policy.

The previously configured instance uses supervisor and its existing portal URL; see [INSTANCE.md](INSTANCE.md) for that instance's commands. Use supervisor to stop it before running setup there.

The server listens internally on **127.0.0.1:8188**, uses dynamic VRAM, one GiB reserved VRAM, FP16/CPU intermediates, SageAttention and disabled node-result caching. Setup builds SageAttention 2.2.0 from pinned upstream revision `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`; this version was unavailable on PyPI when checked. Its [upstream build requirements](https://github.com/thu-ml/SageAttention#installation) include a CUDA development toolkit. Setup verifies the imported source path, package revision and representative FP16/BF16 CUDA attention against FP32 SDPA before startup. The editable package must resolve to `h3_pipeline/__init__.py` inside your cloned checkout.

H3 adapters enforce SageAttention for CUDA diffusion and video-VAE attention, and PyTorch SDPA for Qwen text/vision conditioning and the FP32 audio VAE. Checkpoint attention metadata and the video VAE's direct Comfy Kitchen attention branch cannot override this policy. Quantization/ConvRot kernels and native VAE chunking remain enabled. CPU/unsupported-dtype calls use PyTorch; native Sage mask/precision/error fallbacks are retained and counted. Each section's `server.attention` telemetry records the policy, installed Sage revision, and successful backend calls by component; a selected policy alone is not proof of GPU dispatch. GPU performance and quality still require the acceptance checks.

For automatic startup after container restarts, optionally register `scripts/start_comfy.sh` with your template's supervisor. The foreground script reads the saved setup settings; stop the tmux service before switching to supervisor.

## 3. Browser workflow

For reusable dance videos with wardrobe changes, load **`h3_pipeline/assets/workflows/dance_general_ui.json`**. This develops the latest three-section dance example into a configurable ComfyUI workflow; the existing Python/CLI generation API stays unchanged.

1. Upload/select the dance video in **H3DanceWorkflow**.
2. Leave **character_image** and **background_image** at **Keep original**, or upload/select either replacement independently.
3. Use **Add outfit images (multiple)**, or select an existing uploaded image and add it. Each image defines one complete outfit. The order in **outfit_images** is the appearance order; edit that list to reorder/remove outfits. An empty list preserves the original clothing.
4. Queue once. The workflow generates all sections sequentially, then previews the assembled video automatically.

There is no nine-outfit limit: each render uses only the current outfit, fixed character/scene references, and the preceding section's appearance reference. Every outfit receives equal time by default. Optionally enter **outfit_durations**, a JSON array of seconds adding up to the selected video's duration. **start_seconds** and **duration_seconds** select a source excerpt; zero duration uses the rest of the video. Long intervals split into renders capped by **max_section_seconds** (five seconds by default). This caps the complete inference window, including both context margins and temporal padding: a five-second ceiling allows at most 107 frames on H3's `17k+5` grid, leaving at most 83 retained frames with the default 12-frame context on each side. The report exposes `generation_frames` and `generation_seconds` for every section. Settings too small to hold the context and native minimum are rejected. Each outfit must occupy at least one source-timeline frame.

All sections share one set of model loaders, including the optional Turbo LoRA. This prevents checkpoint RAM from accumulating per section when ComfyUI runs with caching disabled. Restart ComfyUI after deploying this fix; an already expanded, running job retains its original graph.

The workflow normalizes the source to 24 fps once, preserves its aspect ratio with a 32-pixel grid and output area cap, and uses matching chronological excerpts for every section. Extra context frames surround cuts; native `17k+5` alignment and any end padding apply to generation windows only. Assembly removes that context/padding and keeps each timeline frame exactly once. It uses cuts, not crossfades that would blend two dancers/outfits. **audio_mode=source** retains the source soundtrack continuously; silent sources use generated music. **generated** uses each rendered section's audio.

Each queue creates a separate `output/h3/dance/dance-.../` folder containing `generated.mp4`, `report.json`, the expanded API graph, and each section's prompt, references, raw output, trimmed clip and continuity frame. Prepared uploads live under `input/h3-dance/`. The report records the exact outfit/source frame intervals and Turbo/LoRA settings for every section. Handled stage failures and executor interruptions mark the main report and active section `failed` or `interrupted`, preserve completed sections, and cancel remaining sections. A hard process kill cannot flush a report. No partial sequence is assembled or automatically resubmitted.

The workflow follows H3's reference-video conditioning rather than guaranteeing exact pose tracking. Shared references and appearance continuity help, but visual identity, background preservation, clothing accuracy and seamless movement across cuts require a GPU render and review. No new model weights are needed beyond the existing selected profile. Restart ComfyUI after updating the source/custom nodes, reload the browser so its upload controls appear, and import the new workflow. `h3 export-workflows` and setup now include it.

For the existing single-clip workflow:

Load **`h3_pipeline/assets/workflows/ref2va_16gb_ui.json`** in ComfyUI. Setup also copies it to `user/default/workflows`.

1. Upload/select a character image, clothing image and dance video in the corresponding reference loaders.
2. Edit the main text prompt. The prompt builder adds explicit identity, clothing, motion and soundtrack assignments.
3. Start with **5 seconds / 0.98 MP** using `ref2va_quality_5s_098mp.json` on this instance. The main workflow defaults to 15 seconds. Keep matching megapixels/aspect settings in the Resolution Selector and reference loaders; duration is shared with video preprocessing.
4. Keep Enable Lightning LoRA **off**, 25 base steps, `res_multistep`, `simple`, `max` image reference sizing and one batch. BasicGuider is the native no-CFG path equivalent to guidance 1. H.264 output uses CRF 16.
5. Queue one generation and inspect the saved MP4 and audio before increasing duration/resolution.

The browser graph is prepared for the primary one-character, one-outfit, one-video use case. The Python API supports up to nine ordered images and optional standalone audio. Additional browser references require wiring the native autogrow inputs and updating the prompt mapping; use the Python API when you want this done automatically.

The browser video loader performs trimming/resizing with ffmpeg **before decoding** and allocates one FP16 CPU frame buffer. The paired soundtrack is cut to the same usable video duration. A silent video supplies no audio reference. `<Audio 1>` is the dance soundtrack when present; standalone audio becomes `<Audio 2>`. Audio references guide newly generated audio; this does not copy the source song unchanged into the final MP4.

Browser jobs produce `output/h3/browser-TIMESTAMP-ID/clip_*.mp4` and `telemetry.json`. Automatic OOM quality reductions are disabled by default. For a browser OOM, try a shorter clip or increase `memory_level` to 1 then 2 before choosing smaller resolution or references explicitly.

## 4. Command-line and Python generation

Place your assets in `examples/assets/` beside the sample `examples/job.json`, or edit its paths. Paths in a job file are relative to that file. Then run on Vast:

```bash
# Run from the cloned repository.
source .h3-vast.env
source "$H3_ROOT/venv/bin/activate"
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
    megapixels=0.98,
    seed=123456789,
)
print(result.video_path)
```

Other keyword options: `additional_reference_images`, `aspect_ratio` (`16:9`, `9:16`, `1:1`), `ref_image_size` (`match`, `max`), `include_video_audio` (default true), `reference_video_start`, `scheduler` (`simple` default, `beta`, `normal`), `steps` (default 25), `turbo` (default false), `oom_fallback` (default false), and `timeout` in seconds (default 21600). To explicitly use the older speed mode, pass `turbo=True, steps=4`; setting the step count alone does not enable the LoRA.

Image order is character, clothing, additional, retaining list order within each group. Reference tags never silently move. A prompt referring to an unconnected tag fails clearly. If a dance video is silent, a manually written `<Audio 1>` tag is only valid when standalone reference audio is supplied.

The native frame grid is `17k+5`: 3 seconds → 73 frames; 5 seconds → 124; 15 seconds → **362**, or **15.0833 seconds at 24 fps**. The default 0.98 MP landscape canvas resolves to **1344×768** using ComfyUI's 1024-based megapixel calculation. Resolution is snapped to 32 and capped at 1344×768 pixel area. Reference video lengths crop down to the valid temporal grid; a 15-second/360-frame source has 345 usable frames. It is not stretched or padded.

Only when explicitly enabled with `oom_fallback=True`, on **CUDA OOM** the runner retries the same seed/duration with cleanup, encoder offloading and `match`, then stronger residency headroom, then 0.5 and 0.4 MP when smaller than the original request. Maximum five attempts for the default canvas. A change in reference resolution can change conditioning; the seed is preserved but identical output is not promised. Compatibility errors do not trigger resolution retries. A timeout reports the ComfyUI prompt ID and leaves the potentially running job intact—inspect it before resubmitting.

For a remote client, install this package locally, set `server_url` to the public ComfyUI URL. This instance does not require a username/password. For an independently authenticated proxy, provide `username`/`password` to `MiniMaxH3Pipeline`. CLI credentials are read from `H3_HTTP_USERNAME` and `H3_HTTP_PASSWORD`; do not put them in job JSON. You can choose a local client `root` for downloaded outputs. The same-host client lock prevents overlapping client retries; ComfyUI serializes execution from all clients.

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

Profiles: `int8-encoder` (quality default), `primary` (4-bit encoder), `fp8` (FP8 diffusion / 4-bit encoder), `fp8-int8-encoder` (FP8 diffusion / INT8 encoder). Set the same `H3_PROFILE` for setup/start and CLI profile for generation. The INT8 encoder is approximately 25.3 GiB on disk and increases RAM pressure. The video VAE remains INT8 ConvRot in every profile, so its kernels still need to pass. Profiles never download BF16 diffusion or an unrelated encoder. Browser checkpoint changes must be made explicitly in its loader nodes.

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
PYTHONPATH="$PWD" H3_COMFY_PATH=/path/to/ComfyUI .venv/bin/python tests/native_contract.py
```

The pristine official graph, its source license, SHA256 and provenance are retained under `h3_pipeline/assets/`. The long-video optimization repository was studied for memory ideas; no restricted runtime code or chaining implementation is included. No model weights or generated demo videos are bundled.
