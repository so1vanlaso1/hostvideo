from h3_pipeline import MiniMaxH3Pipeline

pipeline = MiniMaxH3Pipeline()  # H3_ROOT, default /data/minimax-h3
result = pipeline.generate(
    prompt="A realistic full-body dance in a studio with soft cinematic lighting.",
    reference_video="/data/minimax-h3/input/dance.mp4",
    character_images=["/data/minimax-h3/input/character.png"],
    clothing_images=["/data/minimax-h3/input/outfit.png"],
    seed=123456789,
)
print(result.video_path)
print(result.report_path)
