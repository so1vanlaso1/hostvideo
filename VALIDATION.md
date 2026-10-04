# Local validation — 4 October 2026

## Audit remediation — 4 October 2026

**52 Python tests passed:** 29 client/planning/media tests and 23 pinned native integration tests. New coverage verifies the complete generation-window ceiling, wide/tall graph schemas, per-section Turbo metadata, native attention-policy bypass prevention and restoration, observed Sage/PyTorch fallback counts, pinned-build rejection, stage failures/interruption, and prompt-scoped executor failure reporting. The existing shared-model-loader regression remains passing.

JavaScript upload-control tests, shell syntax, Python/JSON parsing, dependency consistency, all four workflow generator comparisons and whitespace checks passed. Native tests used this checkout explicitly on `PYTHONPATH` to avoid the older editable-install location.

Managed startup now selects SageAttention. Setup builds a pinned upstream source revision and requires CUDA 13 development tools; preflight implements representative FP16/BF16 Sage correctness checks. **No Sage CUDA build or kernel execution was validated locally.** Both documented remote endpoints refused connections. This remediation was not deployed, and actual-checkpoint dispatch, memory peaks and visual acceptance remain pending. See [AUDIT.md](AUDIT.md) for the fixes and outstanding checks.

## Model RAM reuse fix — 4 October 2026

The dance expansion now shares one diffusion loader, one encoder loader, two VAE loaders and, when enabled, one Turbo LoRA across all sections. This removes the separate checkpoint copies retained by ComfyUI's prompt model tracker. The original ten-section job peaked at 55.6 GB in section 1 and 105.5 GB in section 2; its process exceeded 148 GB during section 3. That job was stopped at the user's request, with sections 1 and 2 preserved.

**42 tests passed locally:** 26 client/planning/media tests and 16 pinned native integration tests. All 16 native integration tests also passed on the Linux instance. The regression test executes three sections through the actual cache-disabled executor, verifies exactly four model allocations, verifies identical model objects in every section and verifies tracker release at prompt completion. A 12-section graph and the Turbo graph also validate with shared loaders. Compilation and whitespace checks passed.

The fix was deployed and ComfyUI restarted. The newly submitted ten-section browser job contains four model loaders, compared with 40 in the stopped job. A short real-model GPU verification job was interrupted externally before completion, so post-fix multi-section RAM peaks and full-quality completion remain unmeasured.

## General dance workflow — 4 October 2026

**40 Python tests passed:** 26 client/planning/media tests and 14 integration checks against the pinned ComfyUI checkout. The JavaScript upload-control checks, syntax/compilation, workflow export and whitespace checks also passed.

The general workflow validates with 12 outfits and independent character/background replacements. A synthetic video ran through ComfyUI's actual expansion executor, sequential section gates, native video encoding, exact core trimming and assembled preview. Frame-position comparisons checked for omitted, duplicated or restarted movement; decoded source-audio correlation exceeded 0.99. Short/silent sources, model-only end padding, custom outfit timings and long intervals were also checked. The local ComfyUI server registered the new nodes and served their browser extension. Browser visual inspection was unavailable because the browser tool blocked the local preview URL; the upload controls were tested with mocked DOM/API inputs.

These tests use synthetic media and bypass model inference explicitly. They verify workflow/assembly behavior, not GPU memory fit, visual fidelity, exact pose tracking or seamless wardrobe transitions. The existing generation pipeline and saved example artifacts remain unchanged.

## Previous baseline

**27 tests passed:** 19 client/pipeline/media tests and eight integration checks against the actual pinned ComfyUI source. Also passed: shell syntax checks, Python compilation, package dependency consistency and wheel packaging with bundled workflows/model manifest.

Local environment: macOS ARM64, Python 3.14.7, PyTorch 2.13.0, torchvision 0.28.0, Comfy Kitchen 0.2.37, Comfy Aimdo 0.5.5. The deployed interpreter is Python 3.11 with CUDA 13.0; Linux/CUDA execution remains pending.

Verified behavior:

- Stable image order and paired/standalone audio tags; invalid reference tags rejected.
- Native temporal alignment and resolution snapping/capping.
- API graph validation by ComfyUI, including native autogrow reference inputs and dynamic MP4 codec inputs.
- Real registration of both custom-node entrypoints.
- Bounded OOM retries with preserved seed/duration; ordinary errors and timeouts do not trigger unsafe retries or interrupt another job.
- Interrupted-transfer recovery, disk-space checks and checkpoint integrity handling using small test fixtures.
- Synthetic video preprocessing, silent-video handling and paired audio timing.
- Native CPU frame decoding in FP16, memory-policy restoration and instrumentation restoration after an exception.
- Native H.264/AAC muxing and profiled video export with JSON telemetry.
- Pristine upstream workflow hash and derivative UI graph connection consistency.

No model weights were downloaded. No MiniMax H3 video was generated locally. No CUDA kernel, target VRAM fit, generation-time benchmark or visual reference-quality claim is verified by these tests. See `ACCEPTANCE.md` for the Vast GPU checks.

Reproduction commands are in `README.md`. Tests use synthetic media and explicitly marked client fixtures, never fabricated model outputs.
