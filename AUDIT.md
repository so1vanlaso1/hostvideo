# Codebase and workflow audit — 4 October 2026

## Current status

All seven code/documentation findings from the original audit have implementations and local regression coverage where applicable. Existing shared-model-loader changes were preserved. **Live CUDA compatibility, deployed attention dispatch, memory fit and visual acceptance remain unverified.** The source fixes have not been deployed by this remediation pass.

Local validation: **52 Python tests passed** (29 client/planning/media tests and 23 integration tests against pinned ComfyUI `e9027f2b30f37bb3052714eb08fcf479542f4fc0`). Browser upload-control tests, shell syntax, Python/JSON parsing, dependency consistency, generated-workflow consistency and whitespace checks also passed.

## Findings and corrections

### 1. SageAttention default — implemented; GPU validation pending

`h3_pipeline/server.py` now selects `--use-sage-attention`. `scripts/setup_vast.sh` builds SageAttention 2.2.0 from pinned upstream revision `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, using the installed Torch environment without build isolation. Setup requires a CUDA 13 development toolkit and C++ compiler, limits build parallelism, and excludes this separately pinned CUDA extension from ordinary dependency-lock replay.

The source revision is pinned because the documented 2.2.0 PyPI release returned HTTP 404 during remediation. The selected source declares support for compute capability 12.0. See [upstream installation requirements](https://github.com/thu-ml/SageAttention#installation) and [the pinned build source](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/setup.py).

`preflight.attention_smoke()` verifies the installed package version and source commit, runs actual Sage kernels with FP16/BF16 and diffusion/video-VAE head dimensions, checks finite output and compares relative RMSE against FP32 SDPA. Managed startup refuses to continue if these checks fail. This check is implemented, but has **not run on CUDA here**; the source build's compatibility with the target Torch/CUDA/GPU combination is not yet established.

### 2. Attention-policy bypasses — fixed in adapters; actual-checkpoint validation pending

`h3_pipeline/attention.py` scopes dispatch changes to H3 inference stages and restores upstream functions afterward, including on errors:

- Diffusion and video VAE select Sage for CUDA FP16/BF16 attention.
- Checkpoint `preferred_attention` and transformer attention overrides cannot supersede that policy.
- The quantized video VAE's direct Comfy Kitchen attention branch is disabled during the stage.
- Qwen text/vision conditioning and FP32 audio VAE use PyTorch SDPA. Comfy Kitchen weight/ConvRot kernels remain enabled.
- CPU/unsupported-dtype calls use PyTorch. Native Sage mask/precision/error fallbacks remain available.

Per-section `server.attention` telemetry records the selected policy, package version/source revision and successful backend-call counts by component. An empty count means no corresponding dispatch was observed. CPU tests exercise native diffusion containers, the video VAE branch, text/audio dispatch, restoration, and separately counted Sage success/PyTorch fallback with an explicitly mocked Sage kernel. These tests verify control flow, not GPU numerics or full-model quality.

### 3. Section-duration ceiling — fixed

`plan_sections()` now rounds the generation budget down to H3's `17k+5` grid **before** allocating context. Core frames, context and temporal padding all fit inside `max_section_seconds`. Impossible context/duration combinations produce an actionable error.

With a five-second ceiling, the largest generation window is 107 frames (4.4583 seconds). With 12 context frames on each side, at most 83 frames remain for the retained core. The original 720-frame reproduction now uses nine sections, each generating 107 frames, while preserving every source frame exactly once. Reports expose `generation_frames` and `generation_seconds`. The independent single-clip API retains its existing upward alignment behavior.

### 4. Failed/interrupted dance reports — fixed for handled execution failures

Section runtime contexts now identify their owning report. Loading, conditioning, sampling, decoding and saving exceptions propagate status, failed section/stage, error and telemetry into the main report. Completed sections and their output paths remain intact; future sections become cancelled.

A prompt-ID-scoped observer of pinned ComfyUI executor terminal events also handles unprofiled node failures and interruption outside the stage wrappers. Tests inject ordinary errors and interrupts both inside stages and through the real native executor, verify monitor cleanup, and verify that another prompt's report is unchanged. Assembly validation failures also update the report.

A hard kill, power loss or inaccessible filesystem cannot flush in-process telemetry; existing reports are not a durable external process-health monitor.

### 5. Wrong deployment source directory — fixed

The README uploads the current checkout rather than the older `Sourcecode` directory. Setup verifies that `h3_pipeline.__file__` resolves to the installed checkout. Native-test reproduction explicitly sets `PYTHONPATH` so an older editable install cannot silently select another source tree.

### 6. Wide/tall API graph schema — fixed

`H3RecordSettings` dimension limits now match native ComfyUI. The pixel-area cap remains enforced, and `dance_canvas()` also bounds extreme dimensions to the native per-axis maximum. Native prompt validation passes for both 1536×672 and 672×1536 graphs.

### 7. Turbo metadata after section one — fixed

`build_workflow()` supplies Turbo and LoRA settings to every section's settings record. Later sections no longer rely on the shared LoRA loader running again to restore those fields. Tests cover multiple consecutive sections with Turbo both enabled and disabled. Shared checkpoint loaders remain in place.

## Verification performed

- 29/29 client, planning, download-fixture, workflow and media tests passed.
- 23/23 pinned native integration tests passed, including all new regression cases.
- Native cache-disabled execution still verifies four model allocations, identical model objects across three sections and prompt-tracker release at completion.
- Browser upload controls passed mocked-DOM tests.
- All repository Python and JSON parsed; all four bundled workflow artifacts matched their generators without rewriting.
- Shell syntax, local package dependency consistency and `git diff --check` passed.

Local environment: macOS ARM64, the existing Python 3.14 validation environment, this checkout explicitly on `PYTHONPATH`, and bundled local ffmpeg/ffprobe binaries. Synthetic tests explicitly bypass full-model inference. Historical generated job artifacts were preserved.

## Remaining external validation

Both documented endpoints were rechecked during remediation: SSH `118.100.214.170:61198` and HTTP `118.100.214.170:61232` refused connections. No remote files were changed and no GPU work was submitted.

Once the instance is reachable, the outstanding checks are:

1. Install this checkout with `scripts/setup_vast.sh`; verify the source path and saved Sage/CUDA preflight report, then restart the managed server.
2. Render with actual checkpoints and inspect effective diffusion, video-VAE, text and audio attention counts, including any fallbacks.
3. Measure post-fix multi-section RAM/VRAM peaks and full-quality 15-second fit.
4. Review identity, clothing, motion/audio continuity and full reference quality under `ACCEPTANCE.md`.

The original shared-loader fix is locally regression-tested, but neither those tests nor historical outputs close the remaining GPU and visual acceptance checks.
