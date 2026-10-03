# Local validation — 3 October 2026

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
