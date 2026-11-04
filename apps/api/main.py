from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import json,subprocess,sys,os,re,time,asyncio
import cv2
from .config import load_local_config
load_local_config()
from fastapi import FastAPI,UploadFile,HTTPException,Header
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,model_validator
from .store import ROOT,DATA,put,get,all_of,update_job
from packages.pipeline.motion import lyric_overlap
from packages.pipeline.scene import pipeline_for
from .versions import load_result,commit_result,versions,restore,migrate
from .editing import save_result as _save_result
from .jobs import reserve_job,release_job,pid_alive

PROCESSES={}
@asynccontextmanager
async def lifespan(app):
    migrate()
    for job in all_of('job'):
        if job['status'] in ['queued','processing']:
            if pid_alive(job.get('pid')):continue
            if job.get('recovery_count',0)>=2:
                update_job(job['id'],status='failed',stage='interrupted',error='自动恢复次数已达上限，请手动重试');release_job(job['id']);continue
            update_job(job['id'],status='queued',stage='recovering',recovery_count=job.get('recovery_count',0)+1,cancel_requested=False)
            launch_job(job['id'])
    yield
    for p in PROCESSES.values():
        if p.poll() is None:p.terminate()

app=FastAPI(title='SignFlow · 手语知音',lifespan=lifespan)
class ProjectIn(BaseModel):name:str=Field(min_length=1,max_length=100)
class ScenePatch(BaseModel):
    framing_type:str=Field(pattern='^(hands_only|upper_body|full_body|partial_body|unknown)$')
    expected_revision:int|None=None
class ActionPatch(BaseModel):
    start_ms:int=Field(ge=0)
    end_ms:int=Field(gt=0)
    keyframe_ms:int=Field(ge=0)
    title:str=Field(max_length=150)
    description:str=Field(max_length=3000)
    review_status:str=Field(default='pending',pattern='^(pending|approved|rejected)$')
    expected_revision:int|None=None
    @model_validator(mode='after')
    def times(self):
        if not self.start_ms<=self.keyframe_ms<self.end_ms:raise ValueError('要求 start ≤ keyframe < end')
        return self
class Lyric(BaseModel):
    id:str=Field(default_factory=lambda:'lyric_'+uuid4().hex)
    start_ms:int=Field(ge=0)
    end_ms:int=Field(gt=0)
    text:str=Field(max_length=1000)
    @model_validator(mode='after')
    def times(self):
        if self.end_ms<=self.start_ms:raise ValueError('歌词结束需晚于开始')
        return self
def require(ident,kind):
    r=get(ident,kind)
    if not r:raise HTTPException(404,'记录不存在')
    return r
def result_path(ident):require(ident,'video');return DATA/'assets'/ident/'result.json'
def result(ident):
    require(ident,'video')
    if any(j['video_id']==ident and j['status'] in ['queued','processing'] for j in all_of('job')):raise HTTPException(409,'分析进行中，请完成后编辑')
    try:return load_result(ident)
    except FileNotFoundError:raise HTTPException(409,'分析尚未完成')
def save_result(ident,r):return _save_result(ident,r,'human_edit')
def launch_job(job_id):
    log=(DATA/(job_id+'.log')).open('wb')
    try:
        PROCESSES[job_id]=subprocess.Popen([sys.executable,'-m','apps.api.worker',job_id],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        update_job(job_id,pid=PROCESSES[job_id].pid)
    finally:log.close()

@app.get('/api/v1/health')
def health():return {'status':'ok','mode':'local-single-user','models':{p.name:p.exists() for p in [ROOT/'models'/n for n in ['hand_landmarker.task','pose_landmarker.task','face_landmarker.task']]}}
@app.post('/api/v1/projects',status_code=201)
def create_project(body:ProjectIn):
    ident='project_'+uuid4().hex;return put(ident,'project',{'id':ident,'name':body.name,'created_at':time.time()})
@app.get('/api/v1/projects')
def projects():return all_of('project')
@app.post('/api/v1/projects/{project_id}/videos',status_code=201)
async def upload(project_id:str,file:UploadFile):
    require(project_id,'project');suffix=Path(file.filename or '').suffix.lower()
    if suffix not in ['.mp4','.mov','.webm','.avi','.mkv']:raise HTTPException(415,'支持 MP4/MOV/WebM/AVI/MKV')
    ident='video_'+uuid4().hex;folder=DATA/'videos';folder.mkdir(exist_ok=True);path=folder/(ident+suffix)
    size=0
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>80*1024*1024:raise HTTPException(413,'首版最大 80MB')
                out.write(chunk)
        cap=cv2.VideoCapture(str(path))
        try:
            fps=cap.get(5);count=cap.get(7);duration=round(count/fps*1000) if fps>0 else 0
            if duration<=0 or duration>120000 or cap.get(3)*cap.get(4)>3840*2160:raise HTTPException(422,'请使用 120 秒以内、最高 4K 的可解码视频')
            width,height=int(cap.get(3)),int(cap.get(4))
        finally:cap.release()
        # Normalize decoder-supported MOV/MP4 codecs to browser-playable CFR H.264.
        import imageio_ffmpeg
        normalized=folder/(ident+'.normalized.mp4')
        try:
            await asyncio.to_thread(subprocess.run,[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(path),'-map','0:v:0','-map','0:a?','-c:v','libx264','-preset','fast','-crf','23','-r','30','-c:a','aac','-movflags','+faststart',str(normalized)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except (subprocess.SubprocessError,OSError):
            normalized.unlink(missing_ok=True);raise HTTPException(422,'视频转码失败，请换用 H.264/AAC 的 MP4')
        original_path=str(path) if os.getenv('GESTUREFLOW_RETAIN_ORIGINAL','1')=='1' else None
        if original_path is None:path.unlink()
        path=normalized
        cap=cv2.VideoCapture(str(path))
        try:fps=cap.get(5);duration=round(cap.get(7)/fps*1000)
        finally:cap.release()
        return put(ident,'video',{'id':ident,'project_id':project_id,'filename':file.filename,'path':str(path),'original_path':original_path,'retention_policy':'retain' if original_path else 'normalized_only','duration_ms':duration,'width':width,'height':height,'fps':fps,'url':f'/api/v1/videos/{ident}/media'})
    except BaseException:
        path.unlink(missing_ok=True);raise
    finally:await file.close()
@app.get('/api/v1/videos/{ident}/media')
def media(ident:str):
    record=require(ident,'video')
    from .media import video_file
    path=video_file(ident,record)
    if not path:raise HTTPException(404,'视频文件已不在数据目录，请重新上传')
    return FileResponse(path,media_type='video/mp4')
@app.get('/api/v1/videos/{ident}')
def video(ident:str):
    v=require(ident,'video');return {k:value for k,value in v.items() if k not in ['path','original_path']}
@app.post('/api/v1/videos/{ident}/analysis-jobs',status_code=202)
def start_job(ident:str,idempotency_key:str|None=Header(default=None)):
    require(ident,'video');jid='job_'+uuid4().hex
    try:job,created=reserve_job({'id':jid,'video_id':ident,'status':'queued','stage':'queued','progress':0,'error':None,'key':idempotency_key,'cancel_requested':False})
    except ValueError as e:raise HTTPException(409,str(e))
    if not created:return job
    try:launch_job(jid)
    except Exception as e:update_job(jid,status='failed',error=str(e));release_job(jid);raise HTTPException(500,'启动分析进程失败')
    return job
@app.get('/api/v1/jobs/{ident}')
def job(ident:str):
    j=require(ident,'job');p=PROCESSES.get(ident)
    if j['status'] in ['queued','processing'] and ((p and p.poll() is not None) or (not p and not pid_alive(j.get('pid')))):
        j=update_job(ident,status='failed',stage='failed',error='分析进程意外退出，请重试并检查本地任务日志')
        release_job(ident)
    return j
@app.post('/api/v1/jobs/{ident}/cancel')
def cancel(ident:str):
    j=require(ident,'job')
    if j['status'] not in ['queued','processing']:return j
    j=update_job(ident,cancel_requested=True)
    p=PROCESSES.get(ident)
    if p and p.poll() is None:p.terminate();p.wait(timeout=10);j=update_job(ident,status='cancelled',stage='cancelled');release_job(ident)
    return j
@app.get('/api/v1/videos/{ident}/result')
def get_result(ident:str):return result(ident)
@app.get('/api/v1/videos/{ident}/scenes')
def scenes(ident:str):return result(ident)['scenes']
@app.get('/api/v1/videos/{ident}/actions')
def actions(ident:str):return result(ident)['actions']
@app.get('/api/v1/videos/{ident}/lyrics')
def lyrics(ident:str):return result(ident)['lyrics']
@app.put('/api/v1/videos/{ident}/lyrics')
def update_lyrics(ident:str,body:list[Lyric],expected_revision:int|None=None):
    r=result(ident)
    if expected_revision is not None and expected_revision!=r.get('result_revision',0):raise HTTPException(409,'版本已更新，请刷新')
    if len(body)>1000 or any(l.end_ms>r['duration_ms'] for l in body):raise HTTPException(422,'歌词数量或时间超出范围')
    if len({l.id for l in body})!=len(body):raise HTTPException(422,'歌词 ID 不可重复')
    r['lyrics']=sorted([l.model_dump() for l in body],key=lambda l:l['start_ms'])
    for a in r['actions']:a['lyric_ids']=[m['lyric_id'] for m in lyric_overlap(a,r['lyrics'])]
    save_result(ident,r);return r['lyrics']
@app.patch('/api/v1/videos/{ident}/scenes/{scene_id}')
def patch_scene(ident:str,scene_id:str,body:ScenePatch):
    r=result(ident);s=next((s for s in r['scenes'] if s['id']==scene_id),None)
    if body.expected_revision is not None and body.expected_revision!=r.get('result_revision',0):raise HTTPException(409,'版本已更新，请刷新')
    if s is None:raise HTTPException(404,'镜头不存在')
    s.update(body.model_dump(exclude={'expected_revision'}));s['pipeline']=pipeline_for(s['framing_type'],s['person_count']);s['review_status']='approved';s['classification_method']='manual';save_result(ident,r);return s
@app.patch('/api/v1/videos/{ident}/actions/{action_id}')
def patch_action(ident:str,action_id:str,body:ActionPatch):
    r=result(ident);a=next((a for a in r['actions'] if a['id']==action_id),None)
    if body.expected_revision is not None and body.expected_revision!=r.get('result_revision',0):raise HTTPException(409,'版本已更新，请刷新')
    if a is None:raise HTTPException(404,'动作不存在')
    if body.end_ms>r['duration_ms']:raise HTTPException(422,'动作超出视频时长')
    # Keyframe edits must regenerate both source and diagram; currently fixed.
    if body.keyframe_ms!=a['keyframe_ms']:raise HTTPException(422,'当前版本暂不支持改变代表帧时间')
    others=[x for x in r['actions'] if x['id']!=action_id]
    if any(min(x['end_ms'],body.end_ms)>max(x['start_ms'],body.start_ms) for x in others):raise HTTPException(422,'动作时间与其他步骤重叠，请先调整相邻边界')
    a.update(body.model_dump(exclude={'expected_revision'}));a['edited_by_human']=True;a['source_analysis_version']=r.get('analysis_version');a['lyric_ids']=[m['lyric_id'] for m in lyric_overlap(a,r['lyrics'])];save_result(ident,r);return a
@app.get('/api/v1/videos/{ident}/export')
def export(ident:str):return Response(json.dumps(result(ident),ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename="gestureflow.json"'})
@app.get('/assets/{ident}/{name}')
def asset(ident:str,name:str):
    require(ident,'video')
    if not re.fullmatch(r'action_\d+(?:_start|_end)?\.(jpg|svg)',name):raise HTTPException(404)
    path=DATA/'assets'/ident/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)
@app.get('/assets/{ident}/versions/{version_id}/{name}')
def versioned_asset(ident:str,version_id:str,name:str):
    require(ident,'video')
    if not re.fullmatch(r'job_[a-f0-9]+',version_id) or not re.fullmatch(r'action_\d+(?:_start|_end)?\.(jpg|svg)',name):raise HTTPException(404)
    path=DATA/'assets'/ident/'versions'/version_id/name
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path)
@app.get('/api/v1/videos/{ident}/versions')
def history(ident:str):require(ident,'video');return versions(ident)
@app.post('/api/v1/videos/{ident}/versions/{version_id}/restore')
def restore_version(ident:str,version_id:str):
    result(ident)
    try:return restore(ident,version_id)
    except ValueError as e:raise HTTPException(409,str(e))
@app.post('/api/v1/videos/{ident}/apply-automatic-actions')
def apply_automatic(ident:str):
    r=result(ident)
    if not r.get('automatic_actions'):raise HTTPException(409,'没有待应用的新自动分段')
    r['actions']=r.pop('automatic_actions');r.pop('manual_tutorial_protected',None);return save_result(ident,r)
@app.post('/api/v1/jobs/{ident}/retry')
def retry_job(ident:str):
    previous=require(ident,'job')
    if previous['status'] in ['queued','processing']:raise HTTPException(409,'任务仍在运行')
    return start_job(previous['video_id'],None)
@app.get('/api/v1/sign/status')
def sign_status():
    return {'language':'Chinese Sign Language','recognition_ready':False,'assessment_ready':False,'standards':[],
            'message':'用户示范库与动作检索可用；通用中国手语词汇模型尚未接入。'}
WEB=ROOT/'apps'/'web'/'dist'
from .sign_routes import router as sign_router
app.include_router(sign_router)
from .live_routes import router as live_router
app.include_router(live_router)
from .coach_routes import router as coach_router
app.include_router(coach_router)
if WEB.exists():app.mount('/',StaticFiles(directory=WEB,html=True),name='web')
