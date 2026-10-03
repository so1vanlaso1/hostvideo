from pathlib import Path
import json
import shutil
import time

from h3_pipeline.client import ComfyClient
from h3_pipeline.config import write_json
from h3_pipeline.media import prepare_image, prepare_video, verify_output, run, executable, probe
from h3_pipeline.workflow import build_workflow

project=Path('/workspace/minimax-h3/projects/dance-outfit-sequence')
assets=json.loads((project/'assets.json').read_text())
job_id='dance-three-outfits-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime())
work=Path('/workspace/minimax-h3/jobs')/job_id
work.mkdir()
write_json(project/'latest-job.json',{'job_id':job_id,'directory':str(work)})
client=ComfyClient()
width,height,seed=576,768,123456789
segments=[(0,90),(90/24,107),(197/24,107)]
outfits=assets['outfits_in_upload_order'][:3]
report={'job_id':job_id,'status':'preparing','width':width,'height':height,'duration':304/24,
        'frame_count':304,'seed':seed,'steps':25,'outfit_order':outfits,'sections':[],
        'method':'Three wardrobe-controlled dance sections with outfit cuts at 3.75s and 8.208s'}
write_json(work/'report.json',report)
started=time.monotonic()
try:
 client.check()
 for index,((start,length),outfit) in enumerate(zip(segments,outfits),1):
  section=work/f'section-{index}'
  section.mkdir()
  section_id=f'{job_id}-outfit-{index}'
  uploads=[]
  for i,path in enumerate((assets['identity_image'],outfit['path']),1):
   prepared=section/f'picture-{i}.png'
   prepare_image(path,prepared,width,height,'max')
   uploads.append(client.upload(prepared,section_id))
  info=prepare_video(assets['video'],section/'reference.mp4',length/24,width,height,start,True)
  video=client.upload(section/'reference.mp4',section_id)
  prompt=f'''A photorealistic dance fashion video. Use exactly the same woman as <Picture 1>, which is a still extracted from <Video 1>. Preserve this dancer's exact face, long dark hairstyle, skin tone, body proportions and apparent age throughout.
<Video 1> supplies this same dancer, her choreography, gestures and body movements, the home interior, lighting, static camera and full-body portrait framing. Closely follow this excerpt of her dance. Keep her face and hair unchanged and keep the original room consistent.
<Picture 2> supplies CLOTHING ONLY: {outfit["description"]}. Dress the dancer from <Picture 1> and <Video 1> in this exact complete outfit from the first frame through the last frame of this section. Match the garment color, fabric, cut and distinctive details. Ignore the photographed clothing model's face, hair, body and background. Do not borrow her identity. Replace the original dancer's white overalls completely with this referenced outfit.
The dancer dances naturally with continuous movement. Keep the entire outfit and legs visible. One woman, coherent hands and limbs, no new person, no scene cuts within this section, no different face, no text, no labels, no watermarks, no clothing photo background.
Generate synchronized dance music guided by <Audio 1>, paired with <Video 1>.'''
  refs=[{'tag':'<Picture 1>','role':'same dancer identity from video','path':assets['identity_image']},
        {'tag':'<Picture 2>','role':f'outfit {index} clothing only','path':outfit['path']},
        {'tag':'<Audio 1>','role':'dance soundtrack','path':assets['video']},
        {'tag':'<Video 1>','role':f'dancer and choreography from {start:.3f}s','path':assets['video']}]
  graph=build_workflow(prompt=prompt,images=uploads,video=video,audio=None,include_video_audio=True,
    width=width,height=height,length=length,seed=seed,job_id=section_id,profile='int8-encoder',
    ref_image_size='max',scheduler='simple',steps=25,turbo=False)
  graph['201']['inputs']['reference_mapping']=json.dumps(refs)
  write_json(section/'workflow-api.json',graph)
  write_json(section/'references.json',refs)
  (section/'prompt.txt').write_text(prompt+'\n')
  entry={'section':index,'job_id':section_id,'start':start,'frame_count':length,'duration':length/24,
         'outfit':outfit,'status':'running','reference_video':info,'workflow_path':str(section/'workflow-api.json')}
  report['sections'].append(entry)
  report['status']='running'
  write_json(work/'report.json',report)
  write_json(project/'report.json',report)
  print(f'Section {index}/3: {outfit["description"]}; {length} frames at {width}x{height}',flush=True)
  history=client.execute(graph,timeout=21600,progress=lambda s:print(s,flush=True))
  output=history.get('outputs',{}).get('92',{})
  candidates=[item for value in output.values() if isinstance(value,list) for item in value
              if isinstance(item,dict) and item.get('filename','').endswith('.mp4')]
  if not candidates: raise RuntimeError('No generated video returned')
  dest=client.download(candidates[0],section/'generated.mp4')
  verify_output(dest,length)
  entry.update(status='completed',video_path=dest,server=output.get('h3_report',[None])[0])
  write_json(work/'report.json',report)
  print('Section completed: '+dest,flush=True)
 (work/'concat.txt').write_text('\n'.join(f"file 'section-{i}/generated.mp4'" for i in range(1,4))+'\n')
 final=work/'generated.mp4'
 run([executable('ffmpeg'),'-nostdin','-v','error','-y','-f','concat','-safe','0','-i',str(work/'concat.txt'),
      '-map','0:v:0','-map','0:a:0','-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p',
      '-af','aresample=async=1:first_pts=0','-c:a','aac','-ar','48000','-ac','2','-movflags','+faststart',str(final)])
 media=verify_output(final,304)
 shutil.copy2(final,project/'generated.mp4')
 report.update(status='completed',video_path=str(final),project_video_path=str(project/'generated.mp4'),
               media=media,wall_seconds=time.monotonic()-started)
 write_json(work/'report.json',report)
 write_json(project/'report.json',report)
 print('COMPLETED '+str(final),flush=True)
except Exception as exc:
 report.update(status='failed',error=str(exc),wall_seconds=time.monotonic()-started)
 write_json(work/'report.json',report)
 write_json(project/'report.json',report)
 raise
