# Photo, dance video and clothes

Two ComfyUI presets are in `h3_pipeline/assets/workflows/` and exported by
`h3 export-workflows --directory PATH`:

| Workflow | Inputs | Clothing behavior |
| --- | --- | --- |
| `dance_scene_single_outfit_ui.json` | One character-and-scene picture, dance video, one clothing picture | The same outfit throughout the dance, including across render chunks. |
| `dance_scene_multiple_outfits_ui.json` | One character-and-scene picture, dance video, ordered clothing pictures | Each outfit occupies an interval, with generated transformations between outfits by default. |

After installing the updated custom nodes, restart ComfyUI and reload the browser.
Open a preset from the workflow browser, or drag its JSON onto the canvas. Use the
three numbered upload buttons in order, then queue once. Existing files can also
be selected from the dropdowns; multiple outfits can be added with **Add selected
existing outfit**. Input placeholders deliberately require your own files.

On this instance, both presets are installed in
`/workspace/minimax-h3/user/default/workflows/`, and the updated nodes are already
loaded. Refresh the browser before opening them. ComfyUI runs privately on
`127.0.0.1:8188` under supervisor; use `supervisorctl restart comfyui_h3` for future
node updates. Service logs are at `/var/log/portal/comfyui_h3.log`.

The reference picture supplies the character's face, hair and body, together with
the scene, background objects, lighting and camera viewpoint. The same picture is
used for both identity and background in every render. A full-body picture with
space around the character gives the dance room to move.

The dance video supplies choreography, facial performance, chronological timing
and, when present, the original soundtrack. The character picture supplies facial
identity and proportions; the video supplies expressions, blinks, gaze, eyebrow
movement, smiles, mouth and lip movement, and head turns and tilts at the matching
moments. The photo's expression is free to change. Continuity images preserve
identity without holding the previous expression or head pose. Mouth movement
follows the visible video performance, including when source audio is absent;
the prompt does not ask the face to sing or speak to background music.

The source dancer's identity, clothes, location and camera movement are
excluded by the prompt. Output dimensions follow the picture's aspect ratio,
rounded to H3's 32-pixel grid and capped by `megapixels`; the motion video is fitted
with padding when its aspect ratio differs, keeping its full frame without
stretching the dancer. The prompt excludes this padding from the output scene.
Clothes pictures supply garment design, color and details only.

For multiple outfits, edit `outfit_images` to set their order. An empty
`outfit_durations` list divides the selected video equally. For a 12-second dance
and three outfits, `[4, 4, 4]` changes outfits at 4 and 8 seconds; `[3, 5, 4]`
changes at 3 and 8 seconds. Durations must add up to the selected video length.
`transition_seconds=1` renders a transformation around each boundary;
`transition_seconds=0` uses cuts. Transformations replace timeline frames, so
they do not lengthen the dance or shift its audio. Very short outfit intervals
may require cuts or shorter transitions. You can add more clothing images;
there is no fixed three-outfit limit.

`start_seconds` and `duration_seconds` select an excerpt. Zero duration uses the
rest of the source. Try a short excerpt first. `max_section_seconds=5` caps each
generation window, including context; it does not truncate the full dance.
Long intervals split automatically. Defaults retain 25 steps, Turbo off, automatic
outfit-correct anchors and refinement. `quality_mode=preview` saves synchronized
reviews without downloading a pose detector. Optional pose scores can flag
intentional differences in framing and body proportions.

Results are under `output/h3/dance/dance-.../`: `generated.mp4`, side-by-side
`review.mp4`, `report.json`, prompts and section graphs. The source is on the left
of each review, the generated result on the right.

These are generative workflows: background pixels, exact dance poses, identity,
garment details and smooth outfit transformations are not guaranteed. Review an
actual render, especially at chunk and outfit boundaries. Automatic anchors can
reinforce mistakes from their first pass. The photo is an identity/scene reference,
not a forced first frame, so its original pose and clothes are free to change.
Facial transfer uses H3's video reference and explicit prompt guidance; there is
no dedicated face retargeting model or facial landmark constraint. Check facial
identity and expression timing in `review.mp4`. Body pose scores do not validate
facial performance. A hidden, blurred or very small source face limits what can
be transferred.

The reference roles follow the [MiniMax full-reference prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

Validation: 21 dance planning/media tests, 25 native ComfyUI integration tests and
the JavaScript upload-control checks passed. Both installed presets were checked
against the running server's node schemas. Native checks cover the actual expanded
graphs, shared photo references, photo-derived dimensions, fitted motion footage,
outfit boundaries and frame counts using synthetic media. No character/dance/clothes
assets were available in the input folder, so visual generation quality has not
been tested for these presets.
