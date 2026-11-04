from pathlib import Path
import sys,time,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from apps.api.store import all_of,DATA
from packages.pipeline.analyze import analyze_video
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--video-id');args=p.parse_args()
    videos=all_of('video');v=next((v for v in videos if v['id']==args.video_id),videos[-1] if videos else None)
    if v is None:raise SystemExit('No real uploaded video; cannot perform real-video acceptance')
    out=DATA/'upgrade-checks'/args.phase/v['id'];start=time.perf_counter()
    r=analyze_video(v['path'],out,v['id'],lambda stage,progress:print(stage,progress,flush=True))
    summary={'phase':args.phase,'video_id':v['id'],'runtime_seconds':round(time.perf_counter()-start,3),'duration_ms':r['duration_ms'],
             'coverage':r['coverage'],'actions':len(r['actions']),'trajectory_quality':r.get('trajectory_quality'),
             'limitations':'No ground-truth labels: semantic accuracy and comparative improvement are unmeasured.'}
    (out/'check-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summary,ensure_ascii=False),flush=True)
