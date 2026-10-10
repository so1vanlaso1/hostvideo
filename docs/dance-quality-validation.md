# Validation — 2026-10-11

The modified pipeline passed **53 local tests, with zero failures, errors or
skips**, in WSL Ubuntu with Python 3.14.4. The Linux test snapshot converted shell
scripts and JSON fixtures to LF, matching the target server's Git checkout.
The live Windows source and pristine template were not rewritten for these tests.
Logs remain in `.runtime/linux-test-results.txt`.

Coverage includes generation-window limits; exact source-frame coverage;
disjoint bridge replacement; source-audio preservation; guide coordinates after
context trimming; dependencies across anchor/refinement passes; shared model
loaders; real MP4 saves and assembly; synchronized review previews; strict section
blocking and artifact-bound approval; rerender dependencies and archives; and
the optional Wan command adapter.

Media tests use synthetic clips as stand-ins for model output. Graph expansion
uses a CPU Comfy transport stub. They establish software behavior and media
alignment, not compatibility or quality of native model sampling. A native
guide-schema test was added to `tests/native_contract.py`, but the pinned ComfyUI
environment was unavailable locally and that contract test was **not run**.
No new GPU generation or Wan generation was performed.

Python compilation, browser JavaScript syntax, `git diff --check`, and the local
Windows environment's `pip check` also passed.

## Pose screening on archived dance outputs

The actual MediaPipe CPU detector was run on matching reference/generated clips
from `jobs/dance-three-outfits-20261003-171604`. Windows Python 3.12 and
MediaPipe 1.1.0 were used with the verified version-1 Lite pose model.

| Archived section | Matching frames | Confident-frame coverage | Median pose error (source torso lengths) | Motion screen |
|---|---:|---:|---:|---|
| 1 | 90 | 100% | 0.0594 | pass |
| 2 | 107 | 100% | 0.4087 | fail |
| 3 | 107 | 100% | 0.0749 | pass |

Section 2 had 77 frames above the initial 0.35 pose-error threshold. A synchronized
preview was generated and inspected: at core frame 48, the source dancer's hands
are raised while the generated dancer holds them at her waist. This supports
the motion flag, but the measurements are screening evidence rather than a
validated perceptual metric. Passing sections can still have garment flicker,
hand errors or other defects.

Input hashes, model hash, metric summaries and test scope are recorded in
[`dance-quality-validation.json`](dance-quality-validation.json). Detailed pose
traces are under `.runtime/quality/`; the inspected old-section preview is
`.runtime/quality/archived-section-2-review/review.mp4`.

The new generation pipeline still needs a target-GPU run and visual review of
its final synchronized preview before claiming improved choreography or seamless
wardrobe changes. Automatic first-pass anchors can be wrong; short bridge windows
are experimental. The extra anchor pass increases compute time and archived
rerenders increase disk use. See [setup and usage](dance-quality.md).
