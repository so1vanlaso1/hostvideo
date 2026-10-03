from pathlib import Path
import json
import time

from h3_pipeline.client import ComfyClient
from h3_pipeline.config import write_json
from h3_pipeline.media import prepare_image, prepare_video, verify_output
from h3_pipeline.workflow import build_workflow

project = Path('/workspace/minimax-h3/projects/dance-outfit-sequence')
assets = json.loads((project / 'assets.json').read_text())
job_id = 'dance-outfits-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime())
work = Path('/workspace/minimax-h3/jobs') / job_id
work.mkdir()
write_json(project / 'latest-job.json', {'job_id': job_id, 'directory': str(work)})
width, height, length, seed = 864, 1152, 294, 123456789
outfits = assets['outfits_in_upload_order'][:3]
prompt = '''Create a photorealistic vertical dance fashion video, one uninterrupted shot lasting 12.25 seconds.
<Picture 1> shows the dancer extracted directly from <Video 1>. Use this exact dancer throughout: preserve her face, long dark hair, skin tone, body proportions and apparent age. <Video 1> defines the same person's identity, dance movements, gestures, room, lighting, camera position and framing. Keep the same home interior and the same dancer, with natural hands and coherent limbs.
<Picture 2>, <Picture 3> and <Picture 4> are CLOTHING DESIGN references only. Ignore the faces, hair, bodies, locations, logos and product text of the people in these clothing photographs. Transfer only their garments to the dancer from <Video 1>. Keep the dancer's own hair and face through both outfit transitions.
Follow this precise wardrobe timeline, exactly three distinct complete outfits in this order:
0.0–4.0 seconds: the dancer already wears the outfit in <Picture 2> from the opening frame: white short-sleeved collared blouse, gray patterned necktie and dark high-waisted wide-leg cropped trousers. She dances naturally and the complete first outfit is clearly visible.
At approximately 4.0 seconds, on a dance turn and musical beat, her clothing switches cleanly to the complete outfit in <Picture 3> while her face, hair, body, position, dance and background remain continuous.
4.0–8.0 seconds: cream long-sleeved ruffled cropped blouse with a wide collar and matching cream short skirt from <Picture 3>. Show this outfit clearly while she continues the choreography.
At approximately 8.0 seconds, on the next dance turn and beat, her clothing switches cleanly to the complete outfit in <Picture 4>, maintaining the same dancer and continuous movement.
8.0–12.25 seconds: light-gray short-sleeved zip-front top with red and white piping and matching light-gray long skirt from <Picture 4>. She finishes the dance wearing only this third outfit.
The two wardrobe switches are fully clothed fashion transitions. Each interval has one coherent outfit matching its assigned photograph; do not combine garment pieces from different references. Keep the framing wide enough to show the blouse, skirt or trousers and dancing legs throughout. No additional person, no face swap, no scene change, no split screen, no labels, no subtitles, no clothing-photo background.
Generate native synchronized dance music guided by the rhythm and atmosphere of <Audio 1>, paired with <Video 1>.'''
(work / 'prompt.txt').write_text(prompt + '\n')
(project / 'prompt.txt').write_text(prompt + '\n')
report = {'job_id': job_id, 'status': 'preparing', 'seed': seed, 'width': width, 'height': height,
          'frame_count': length, 'duration': length / 24, 'outfit_order': outfits,
          'video_source': assets['video'], 'identity_source': assets['identity_image']}
write_json(work / 'report.json', report)
started = time.monotonic()
client = ComfyClient()
try:
    client.check()
    uploads = []
    references = []
    for i, path in enumerate([assets['identity_image']] + [o['path'] for o in outfits]):
        prepared = work / f'picture-{i+1}.png'
        prepare_image(path, prepared, width, height, 'max')
        uploads.append(client.upload(prepared, job_id))
        references.append({'tag': f'<Picture {i+1}>', 'role': 'dancer identity from video' if i == 0 else f'outfit {i}: {outfits[i-1]["description"]}', 'path': path})
    video_info = prepare_video(assets['video'], work / 'reference.mp4', length / 24, width, height, 0, True)
    assert video_info['paired_audio'], 'This prompt expects the dance soundtrack'
    video = client.upload(work / 'reference.mp4', job_id)
    references += [{'tag':'<Audio 1>','role':'soundtrack paired with <Video 1>','path':assets['video']},
                   {'tag':'<Video 1>','role':'dancer identity, dance, room and camera','path':assets['video']}]
    graph = build_workflow(prompt=prompt, images=uploads, video=video, audio=None, include_video_audio=True,
        width=width, height=height, length=length, seed=seed, job_id=job_id, profile='int8-encoder',
        ref_image_size='max', scheduler='simple', steps=25, turbo=False)
    graph['201']['inputs']['reference_mapping'] = json.dumps(references)
    write_json(work / 'workflow-api.json', graph)
    write_json(work / 'references.json', references)
    write_json(project / 'workflow-api.json', graph)
    write_json(project / 'references.json', references)
    report.update(status='running', reference_video=video_info, workflow_path=str(work / 'workflow-api.json'))
    write_json(work / 'report.json', report)
    print(f'Generating {job_id}: {width}x{height}, {length} frames, three outfits, 25 steps', flush=True)
    history = client.execute(graph, timeout=21600, progress=lambda s: print(s, flush=True))
    candidates = [item for value in history.get('outputs', {}).get('92', {}).values() if isinstance(value, list)
                  for item in value if isinstance(item, dict) and item.get('filename', '').endswith('.mp4')]
    if not candidates:
        raise RuntimeError('No generated MP4 found in completed output')
    destination = client.download(candidates[0], work / 'generated.mp4')
    media = verify_output(destination, length)
    report.update(status='completed', video_path=destination, media=media,
                  server=history['outputs']['92'].get('h3_report', [None])[0], wall_seconds=time.monotonic()-started)
    write_json(work / 'report.json', report)
    write_json(project / 'report.json', report)
    print('COMPLETED ' + destination, flush=True)
except Exception as exc:
    report.update(status='failed', error=str(exc), wall_seconds=time.monotonic()-started)
    write_json(work / 'report.json', report)
    write_json(project / 'report.json', report)
    raise
