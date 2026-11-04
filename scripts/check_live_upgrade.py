from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from apps.api.store import all_of
video=all_of('video')[-1];base='http://127.0.0.1:8000';checks=[]
with httpx.Client(timeout=60) as client:
    for attempt in range(2):
        start=time.perf_counter();r=client.post(base+'/api/v1/videos/'+video['id']+'/analysis-jobs');r.raise_for_status();job=r.json()
        for _ in range(180):
            response=client.get(base+'/api/v1/jobs/'+job['id']);response.raise_for_status();state=response.json()
            if state['status'] in ['completed','failed','cancelled']:break
            time.sleep(.5)
        assert state['status']=='completed',state
        response=client.get(base+'/api/v1/videos/'+video['id']+'/result');response.raise_for_status();result=response.json()
        candidates=result.get('automatic_actions',result['actions']);assert candidates and result['tutorial_steps']
        source=candidates[0]['assets']['source'];asset=client.get(base+source);asset.raise_for_status()
        checks.append({'attempt':attempt+1,'seconds':round(time.perf_counter()-start,3),'video_id':video['id'],'job_id':job['id'],'revision':result['result_revision'],'checkpoint':result['checkpoint_info'],'actions':len(result['actions']),'new_candidates':len(candidates),'coverage':result['coverage'],'legacy_edits_protected':bool(result.get('manual_tutorial_protected'))})
        print(json.dumps(checks[-1]),flush=True)
    assert checks[1]['checkpoint']['vision_reused']
    versions=client.get(base+'/api/v1/videos/'+video['id']+'/versions').json();assert len(versions)>=2
    assert client.get(base+'/api/v1/ai/capabilities').status_code==200
    no_labels=client.post(base+'/api/v1/datasets/exports');assert no_labels.status_code==409
out=Path('data/live-upgrade-check.json');out.write_text(json.dumps(checks,indent=2),encoding='utf8')
