import sys, traceback,time,json
from .store import get,update_job,DATA
from packages.pipeline.analyze import analyze_video,Cancelled
from .versions import load_result,commit_result
from .jobs import release_job

def run(job_id):
    job=get(job_id,'job');video=get(job['video_id'],'video')
    try:
        try:previous=load_result(video['id'])
        except FileNotFoundError:previous=None
        update_job(job_id,status='processing')
        output=DATA/'assets'/video['id']/'versions'/job_id
        result=analyze_video(video['path'],output,video['id'],
                             lambda stage,progress:update_job(job_id,stage=stage,progress=progress,checkpoint_stage=stage,heartbeat=time.time()),
                             lambda:bool(get(job_id,'job').get('cancel_requested')),
                             checkpoint_directory=DATA/'assets'/video['id']/'checkpoints',scene_overrides=[s for s in previous.get('scenes',[]) if s.get('classification_method')=='manual'] if previous else [])
        if get(job_id,'job').get('cancel_requested'):raise Cancelled()
        # Asset URL rewrite binds every image to its immutable analysis run.
        text=json.dumps(result,ensure_ascii=False).replace('/assets/'+video['id']+'/', '/assets/'+video['id']+'/versions/'+job_id+'/')
        result=json.loads(text);result['analysis_version']=job_id
        if previous:
            result['lyrics']=previous.get('lyrics',[])
            protected=previous.get('manual_tutorial_protected') or previous.get('schema_version')=='1.0' or any(a.get('edited_by_human') or a.get('review_status')=='approved' for a in previous['actions'])
            if protected:
                result['automatic_actions']=result['actions'];result['actions']=previous['actions'];result['manual_tutorial_protected']=True;result['warnings'].append('检测已更新，人工编辑的教程结构受保护。新自动分段另存 automatic_actions，可显式应用或恢复历史版本。')
        from packages.pipeline.composer import compose_step
        result['tutorial_steps']=[compose_step(a,result) for a in result['actions']]
        commit_result(video['id'],result,'analysis:'+job_id,previous.get('result_revision',0) if previous else 0)
        update_job(job_id,status='completed',stage='completed',progress=100,error=None)
    except Cancelled:update_job(job_id,status='cancelled',stage='cancelled',error=None)
    except Exception as exc:
        traceback.print_exc()
        update_job(job_id,status='failed',stage='failed',error=str(exc)[:600])
    finally:release_job(job_id)
if __name__=='__main__':run(sys.argv[1])
