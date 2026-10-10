# Dance motion and outfit transitions

Update the source, restart ComfyUI, reload the browser and import
`h3_pipeline/assets/workflows/dance_general_ui.json`. Queue the node as before.
Defaults now use **auto anchors**, **one-second transitions**, **CPU pose warnings**,
25 base steps, Turbo disabled and synchronized review previews. Updated setup
installs the quality dependencies. For an existing server, install the extra
below before restarting ComfyUI, or choose `quality_mode=preview`.

The source is normalized to 24 fps once. Sections, guides and reviews use that
timeline. Context/padding are discarded. Bridges replace existing frames rather
than inserting time, and source audio remains continuous when selected.

## Generation

1. Render stable outfit sections as source-video edits.
2. In `anchor_mode=auto`, extract outfit-correct opening, middle and closing core
   frames from a first pass, then refine with native H3 guides at those exact
   timestamps. This adds a complete sampling pass for each stable section.
3. Same-outfit chunks reuse a preceding render's future-context frame at the new
   chunk's actual opening timestamp. Old clothes are never anchored into a new
   outfit. Shared character/background references remain fixed.
4. Render each outfit boundary separately with both outfit references, the
   chronological source excerpt and endpoint guides from adjacent stable renders
   at the SAME source timestamps. Only this render transforms clothes.
5. Save side-by-side `review.mp4` files for every section and the final video.
   Source is on the left, generated result on the right.

Automatic anchors are generated candidates, not verified source-pose edits. A
bad first pass can reinforce its own error. Screen and review them, or supply
corrected edited anchors. Guides constrain selected frames; they do not
guarantee exact intermediate choreography. Native sampling, GPU fit and visual
quality still require a target-GPU render. Short native windows are below H3's
usual training duration, so bridge quality remains experimental.

## CPU pose screening

In the ComfyUI server's activated environment:

```bash
pip install -e '.[dance-quality]'
h3 dance-quality-setup --directory /workspace/minimax-h3/models/dance-quality
```

Set `pose_model` to the downloaded `pose_landmarker_lite.task`, or leave it empty
to download the verified small model once under the server input directory.
Select:

| quality_mode | Behavior |
|---|---|
| preview | Synchronized visual reviews without pose dependencies. |
| pose_warn | Flag anchor/section motion mismatches; assemble a review draft. |
| pose_strict | Reject failed/inconclusive anchor passes; block flagged sections from assembly until reviewed or rerendered. |

The small model download is SHA256-verified. Screening runs on CPU and adds time.
Generation checkpoints and precision stay unchanged. Optional dependencies can
bring OpenCV: run `pip check` after installation in the server environment.

Body joints are compared on matching frames without time warping or separately
centering dancers, preserving evidence of wrong travel. Errors are scaled by
source torso length. Reports include confident-frame coverage, median pose error,
velocity mismatch and bad-frame indices. Initial heuristic thresholds: 0.35
torso lengths for pose error, 0.25 for the 90th-percentile per-frame velocity
error, 65% confident-frame coverage. More than 15% bad comparable frames flags a
clip. A severe one-frame jump (median velocity error above three times the
velocity threshold) also flags a clip, even when diluted in a long sequence.
Calibrate these for your scenes: fast motion, occlusion, skirts and changed
body proportions can cause false flags. Missing detection is **inconclusive**.
These scores do not establish garment, face, hand or transformation quality.

## Edited source-pose anchors

Upload a source frame edited into the desired outfit while preserving its pose,
identity and room. It must match the output canvas exactly. The browser's
**Add edited pose/outfit anchor** button records its time and outfit number.
Alternatively enter `anchor_manifest`:

```json
[
  {"frame": 0, "outfit_index": 1, "image": "dance-assets/edited-start.png"},
  {"frame": 59, "outfit_index": 1, "image": "dance-assets/edited-end.png"}
]
```

Frames refer to the selected normalized source excerpt, not generation context.
Outfit numbers are one-based. `auto` prioritizes supplied images and generates
remaining core anchors. `supplied` requires both core endpoints of every stable
section. `off` disables the first pass; explicit supplied anchors still apply.
Raw source frames with original clothing can reintroduce the old outfit.
Set `transition_seconds=0` to retain cuts instead of transformation renders.

## Review and rerender

Use these report commands **on the server**, where the files exist:

```bash
h3 dance-review --report /workspace/minimax-h3/output/h3/dance/dance-ID/report.json
h3 dance-review --report /workspace/minimax-h3/output/h3/dance/dance-ID/report.json \
  --approve 2 --note "Compared motion, clothing stability and boundary poses"
```

Approval is explicit and tied to the video SHA256; replacing it invalidates the
approval. Releasing a pose flag is a visual judgment, not an automatic pass.

```bash
# On the server, or on a client with --server COMFY_URL:
h3 dance-resume --job-directory h3/dance/dance-ID --sections 2 --seed 987654321
# Empty selection retries incomplete/unapproved flagged sections. If all
# sections are completed/accepted, it only assembles again.
h3 dance-resume --job-directory h3/dance/dance-ID
```

Rerendering refreshes downstream chunks of the same outfit and dependent bridges.
Other outfits stay intact. Old clips, graphs, guides, reviews and the old report
are archived under `section-N/attempts/` before replacement. No automatic OOM
reductions, unbounded retries or overlapping job submission are introduced.
Do not resume an executing job: a timeout does not prove ComfyUI stopped. A hard
kill can leave `running` in a report; inspect the server before correcting it.

`status=completed` means execution completed. `quality_status` remains
`needs_visual_review` or `flagged`. Review the final synchronized video as well;
section-level pose checks do not establish seamless joins. Crossfades and
interpolation are not used to conceal incorrect movement.

## Optional Wan-Animate comparison

Install the [official Wan2.2 environment and Animate weights](https://github.com/Wan-Video/Wan2.2#run-wan-animate)
separately. The adapter does not download large models or claim 16 GB GPU fit.
Supply a full-body image of the SAME dancer already in the desired outfit, rather
than a product photograph featuring someone else.

Edit `examples/wan-dance-job.json`, then run:

```bash
h3 dance-wan --job examples/wan-dance-job.json --plan-only
h3 dance-wan --job examples/wan-dance-job.json
```

This runs official replacement preprocessing (pose, face, mask, background) and
generation with CPU offloading. It records upstream revision, input hashes,
commands, logs and reviews in a new empty output directory. Preprocessing and
generation stay at upstream 30 fps; source and result normalize to 24 fps for
comparison. Short outputs are rejected instead of stretched. Source audio is
retained. This is a single-outfit backend comparison, not an automatic model
switch or wardrobe-transformation model; use H3 bridges for the outfit sequence.

References: [MiniMax reference prompting](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md),
[pinned H3 guides](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/comfy_extras/nodes_minimax_h3.py),
[MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
