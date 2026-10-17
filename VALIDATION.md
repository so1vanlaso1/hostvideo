# Local validation

## ARM64 GB10 preparation — 17 October 2026

The CUDA 12.8/13.x baseline was committed and pushed to `cuda12-compatible` at `787be16` before ARM work began. The `arm-gb10` branch adds native Linux aarch64/arm64 setup with CUDA 13, pinned ARM64 PyTorch/torchvision wheels, a native Comfy Kitchen build for visible GPU targets and the existing pinned SageAttention revision's GB10 kernels. CPU architecture, Python ABI and build-helper hashes now participate in extension reuse decisions.

GB10/integrated-memory startup uses estimate-based loading, disabled pinned host buffers and 12 GiB initial headroom. Memory overrides survive setup and detached tmux startup. Preflight reports one container-bounded shared pool, rejects incompatible ARM runtimes and invalid reserve budgets, and retains actual CUDA/quantization/Sage correctness checks before downloading weights. Telemetry identifies shared-memory accounting and tracks minimum system-available memory. Optional MediaPipe CPU pose screening defaults off on ARM.

**76 local tests passed with no skips**, using the existing ffmpeg/ffprobe binaries under `../tmp/tools`. **All 25 native integration tests passed** against pinned ComfyUI with this checkout explicitly on `PYTHONPATH`. New regression coverage includes ARM source-build selection and `121a` target propagation, CUDA 12 rejection on ARM, CPU-only Torch rejection before extension/model work, CPU-architecture lock invalidation with unchanged toolkit/GPU capability, saved memory-policy overrides, shared-pool accounting and serializable pressure telemetry. Shell syntax, Python compilation, dependency consistency and whitespace checks passed. Both Comfy Kitchen patches apply cleanly to the pinned upstream revision. Published Python 3.11 aarch64 CUDA 13 wheels were confirmed for the pinned Torch/torchvision pair, Comfy Kitchen and Comfy Aimdo.

Reproduction (the service tests substitute GPU/build tools; native tests use synthetic media without model inference):

```bash
FFMPEG=/path/to/ffmpeg FFPROBE=/path/to/ffprobe /path/to/client-venv/bin/python -m unittest discover -s tests
PYTHONPATH="$PWD" H3_COMFY_PATH=/path/to/pinned/ComfyUI FFMPEG=/path/to/ffmpeg FFPROBE=/path/to/ffprobe /path/to/native-venv/bin/python tests/native_contract.py
```

No GB10 instance was available, and no deployment was requested. Native ARM64 CUDA compilation, actual checkpoint inference, peak shared-memory fit, runtime comparisons and visual quality remain unverified until target acceptance. Run the staged commands in [the GB10 guide](docs/gb10.md) when the instance is ready; the branch does not claim measured GPU performance.

## CUDA 12.8+ setup conversion — 17 October 2026

Setup now selects PyTorch 2.11.0/cu128 for CUDA 12.8+ in the 12.x family and retains PyTorch 2.13.0/cu130 for CUDA 13.x. CUDA 12 builds Comfy Kitchen 0.2.37 at pinned revision `be003b7c23c5b01328657955b8bc5d3f073d868e`, patching its Linux cuBLASLt loader to use major 12 and building native GPU targets instead of the newer default family targets. Both paths build pinned SageAttention and retain quantization/attention kernel preflight. Build fingerprints prevent old freeze files or CUDA extensions from being reused across migrations.

**55 tests passed; seven media tests skipped because ffmpeg is unavailable.** Local checks cover toolkit and driver boundaries, selection of matching Torch wheels, successful reruns, migrations in both directions and interrupted migrations. Shell syntax, Python compilation and whitespace checks passed. The compatibility patch applies cleanly to the pinned upstream source. These checks do not compile CUDA or execute GPU kernels on this macOS machine; run `bash setup.sh --skip-download` on the target Linux GPU to perform those checks before model downloads.

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
