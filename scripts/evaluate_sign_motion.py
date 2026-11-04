import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apps.api.store import all_of
from apps.api.versions import load_result
from packages.pipeline.signmatch import dtw,usable
from packages.pipeline.signmetric_v3 import motion_sequence
items=[t for t in all_of('sign_template') if usable(t)];results={};expanded={};excluded=[]
for t in items:
    try:
        if t['video_id'] not in results:results[t['video_id']]=load_result(t['video_id'])
        expanded[t['id']]=motion_sequence(results[t['video_id']],t['start_ms'],t['end_ms'])
    except (ValueError,FileNotFoundError) as e:excluded.append({'id':t['id'],'reason':str(e)})
details=[];start=time.perf_counter()
for q in items:
    if q['id'] not in expanded:continue
    others=[t for t in items if t['video_id']!=q['video_id'] and t['id'] in expanded]
    if not any(t['meaning']==q['meaning'] for t in others):continue
    old=[];new=[]
    for t in others:
        try:old.append((dtw(q['sequence'],t['sequence']),t['meaning']));new.append((dtw(expanded[q['id']],expanded[t['id']]),t['meaning']))
        except ValueError:continue
    if old and new:details.append({'id':q['id'],'expected':q['meaning'],'before':min(old)[1],'after':min(new)[1]})
report={'eligible':len(details),'before_correct':sum(x['expected']==x['before'] for x in details),'after_correct':sum(x['expected']==x['after'] for x in details),'seconds':time.perf_counter()-start,'excluded':excluded,'details':details,'deployed':False,'note':'Paired leave-source-video-out nearest candidate check; exploratory development evaluation, not camera accuracy.'}
out=ROOT/'evaluation/results/sign-motion-comparison.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in report.items() if k not in ('details','excluded')})
