# Vast RTX 5060 Ti 16 GB acceptance record

Quality defaults: INT8 ConvRot diffusion and video VAE, INT8 Qwen3-VL encoder, FP32 audio VAE, 25 base steps, Turbo disabled, `res_multistep` / `simple`, native 1344×768 output, `max` image references, H.264 CRF 16. Automatic OOM quality reductions are disabled.

**4 October remediation status:** the code fixes and local regression tests are recorded in [AUDIT.md](AUDIT.md). The documented instance endpoints currently refuse connections. The results below are historical and do not validate the new SageAttention policy, source build or post-fix multi-section memory use.

Historical status: model download and SHA256 verification, GPU quantization kernels, and native workflow validation passed. Full visual/reference acceptance remains pending. Public ComfyUI page and browser API return HTTP 200 without credentials.

| Stage | Requested settings | Runtime/OOM | Audio/video | Quality observations | Pass/fail |
|---|---|---|---|---|---|
| Execution check | 22 frames, 0.98 MP, 25 base steps, no references | 155.2 seconds, no OOM or fallback | Verified H.264 + AAC, 1344×768, 24 fps | Too short for visual acceptance | Execution passed |
| Smoke | 3 seconds, 0.98 MP, 25 base steps | Pending | Pending | Pending | Pending |
| Five seconds | 5 seconds, 0.98 MP, 25 base steps | Sampling reached step 3 without OOM; intentionally stopped for shorter validation | Not decoded | About 59 seconds per step on this GPU; no completed quality result | Pending |
| Target | 15 seconds, 0.98 MP, 25 base steps | Pending | Pending | Pending | Pending |
| Identity/motion | Target + character + dance/audio | Pending | Pending | Pending | Pending |
| Clothing | Target + character + clothing + dance/audio | Pending | Pending | Pending | Pending |

The target passes only at its recorded effective resolution. Smaller renders demonstrate execution, not full-duration or reference quality. Benchmark commands disable fallback. Inspect identity, clothing, motion and anatomical consistency in your reference-driven results before accepting them.

The earlier speed smoke test used 73 frames at 864×480, four Turbo steps and the 4-bit encoder. It completed with audio in 86.8 seconds; see `/workspace/minimax-h3/jobs/20261003-162148-4b88bb68a3/report.json`. It does not validate the new quality settings.

Quality validation logs: `/workspace/minimax-h3/logs/quality-smoke-validation.log`. The intentionally interrupted five-second check is recorded in `/workspace/minimax-h3/jobs/20261003-163351-b62a8cfd74/report.json`.

Completed quality execution check: `/workspace/minimax-h3/jobs/20261003-163828-77cdb88478/report.json` and `generated.mp4`. All 21 project tests and 9 native integration tests passed. This 0.917-second clip verifies execution only; it does not establish full-length/reference quality.
