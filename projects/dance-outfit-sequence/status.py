from pathlib import Path
import json
import re

project=Path('/workspace/minimax-h3/projects/dance-outfit-sequence')
report=json.loads((project/'report.json').read_text())
log=Path('/var/log/portal/hostvideo.log').read_text(errors='replace').split('got prompt')[-1]
progress=re.findall(r'(\d+)%\|[^\r\n]*',log)
active=next((s for s in report.get('sections',[]) if s['status']=='running'),None)
status={'status':report['status'],'completed_sections':[s['section'] for s in report.get('sections',[]) if s['status']=='completed'],
        'active_section':active['section'] if active else None,'sampling_percent':int(progress[-1]) if progress else None,
        'video_path':report.get('video_path')}
(project/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
if progress: print(re.findall(r'\d+%\|[^\r\n]*',log)[-1])
