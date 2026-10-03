The current render keeps one dancer reference for three outfit-controlled dance sections in the requested upload order:

1. 0–3.75 seconds: white collared shirt, patterned tie and dark trousers (`images.jpeg`).
2. 3.75–8.208 seconds: cream blouse and short skirt (`qNh0S3SXyDGQM6VzOO0kxAlI8ttZ7nOi8w5TWdqp-1.jpg`).
3. 8.208–12.667 seconds: light-gray zip-front top and long skirt (`Set-do-nu-poly-cotton-2da-ao-coc-tay-keo-khoa-chan-vay-dai-SFB3106.webp`).

The fourth uploaded outfit is reserved. `dancer-identity.png` is a still from the uploaded dance video. Each section uses this same identity image, its own corresponding excerpt of the dance, and one clothing image. The prompts assign clothing photographs only to garment design and preserve the original dancer and room. The sections join with outfit cuts. Final visual fidelity remains subject to the generative model.

Reusable ComfyUI presets are already installed in the workflow browser:

- `dance_outfit_1_white_shirt.json`
- `dance_outfit_2_cream_set.json`
- `dance_outfit_3_gray_long_skirt.json`

All three presets passed native ComfyUI prompt validation with their selected files. Each uses the source video's 576×768 portrait resolution, seed 123456789, INT8 encoder, 25 base sampling steps, Turbo disabled, and max-size image references. Browser reruns receive new output folders.

`latest-job.json` identifies the current render directory. Its `report.json` records the status of each section; every section stores its exact `workflow-api.json`, `references.json`, and `prompt.txt`. Completion writes `generated.mp4` to the render directory and this project folder. `run_sections.py` reproduces the three-section generation and join:

```bash
/workspace/minimax-h3/venv/bin/python /workspace/minimax-h3/projects/dance-outfit-sequence/run_sections.py
```

The full-shot workflow is archived as `full-shot-12s-workflow.json`. Dense attention over a long reference and several outfits made that version impractical on this GPU, so its render was interrupted and replaced with the section render.
