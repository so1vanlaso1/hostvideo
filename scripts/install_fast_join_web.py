from pathlib import Path
p=Path('/workspace/minimax-h3/ComfyUI/custom_nodes/h3_dance_fast')
(p/'web').mkdir(exist_ok=True)
s=Path('/workspace/hostvideo/custom_nodes/h3_lowvram/web/dance_workflow.js').read_text()
for name in ['H3DanceWorkflow','H3PhotoDance','H3SceneDanceSingleOutfit','H3SceneDanceMultipleOutfits']:s=s.replace('"'+name+'"','"'+name+'Fast"')
s=s.replace('hostvideo.generalDance','hostvideo.fastJoinDance')
(p/'web/dance_fast.js').write_text(s)
(p/'__init__.py').write_text('from h3_pipeline.dance_fast_nodes import NODE_CLASS_MAPPINGS\nWEB_DIRECTORY = "./web"\n')
