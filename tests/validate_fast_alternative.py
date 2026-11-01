import importlib.util,json,sys,tempfile,uuid
from pathlib import Path
from unittest.mock import patch
from PIL import Image
root=Path('/workspace/hostvideo')
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'tests'))
from test_dance_graph import load_nodes
from h3_pipeline.workflow import validate_graph
with tempfile.TemporaryDirectory() as folder:
    inp,out=Path(folder)/'input',Path(folder)/'output';inp.mkdir();out.mkdir()
    (inp/'dance.mp4').write_bytes(b'fixture');Image.new('RGB',(128,96)).save(inp/'photo.png')
    base=load_nodes(inp,out)
    with patch.dict(sys.modules,{'h3_pipeline.dance_nodes':base}):
        spec=importlib.util.spec_from_file_location('h3_pipeline.dance_fast_nodes',root/'h3_pipeline/dance_fast_nodes.py')
        fast=importlib.util.module_from_spec(spec);spec.loader.exec_module(fast)
    fast.register_report=lambda path:None
    presets=Path('/workspace/minimax-h3/user/default/workflows')
    for filename,cls in [('dance_photo',fast.H3PhotoDanceFast),('dance_scene_single_outfit',fast.H3SceneDanceSingleOutfitFast),('dance_scene_multiple_outfits',fast.H3SceneDanceMultipleOutfitsFast)]:
        doc=json.loads((presets/(filename+'_ui.json')).read_text());node=doc['nodes'][0]
        schema=cls.INPUT_TYPES();fields=list(schema['required'])+list(schema['optional'])
        settings=dict(zip(fields,node['widgets_values']));settings['anchor_mode']='boundary'
        node['type']=cls.__name__;node['properties']['Node name for S&R']=cls.__name__;node['title']='Fast joins: '+node.get('title',filename)
        node['widgets_values']=[settings[k] for k in fields]
        doc['id']=str(uuid.uuid4())
        note='FASTER ALTERNATIVE: one full-quality pass per section, then short endpoint-anchored bridges at every join. Same steps and resolution. Review identity, clothing and joins. Original workflows are unchanged. New nodes require ComfyUI restart after the running job finishes.'
        doc['nodes'][1]['widgets_values']=[note]
        (presets/(filename+'_fast_joins_ui.json')).write_text(json.dumps(doc,indent=2)+'\n')
        settings.update(reference_image='photo.png',reference_video='dance.mp4')
        if 'outfit_image' in settings:settings['outfit_image']='photo.png'
        if 'outfit_images' in settings:settings['outfit_images']='["photo.png"]'
        metadata={'streams':[{'codec_type':'video','width':128,'height':96,'nb_frames':'240'}]}
        with patch.object(fast,'probe',return_value=metadata),patch.object(fast,'normalize_dance_video',return_value=metadata),patch.object(fast,'slice_dance_video'),patch.object(fast,'extract_frame',return_value='photo.png'):
            graph=cls().expand(**settings)['expand']
        validate_graph(graph)
        assert not any(n['class_type']=='H3DanceSaveAnchors' for n in graph.values())
        report=json.loads(sorted(out.rglob('report.json'),key=lambda p:p.stat().st_mtime)[-1].read_text())
        stable=[s for s in report['sections'] if s['kind']=='stable'];bridges=report['transitions']
        assert len(bridges)==len(stable)-1
        assert sum(p['frames'] for p in report['assembly_plan'])==240
        assert all(len(s['guides'])==2 for s in bridges)
        print(filename, 'PASS',len(stable),'sections',len(bridges),'bridges, exact timeline, valid graph structure')
