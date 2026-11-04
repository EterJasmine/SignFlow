from uuid import uuid4
import threading
from fastapi import APIRouter,HTTPException,Response
from pydantic import BaseModel,Field
from .media import video_file
from .store import DATA,all_of,put,get,delete
from .versions import load_result
from packages.pipeline.signmatch import sequence,dtw,retrieve,usable
from packages.pipeline.sheet import preview_strip
router=APIRouter(prefix='/api/v1/sign')
# Thumbnails are deterministic for a given demonstration and analysis run, so the
# server renders each one once instead of on every list refresh.
_FRAME_CACHE={}
_FRAME_LOCK=threading.Lock()
_FRAME_HEADERS={'Cache-Control':'private, max-age=300'}
class Selection(BaseModel):
    video_id:str
    start_ms:int=Field(default=0,ge=0)
    end_ms:int=Field(gt=0)
class Template(Selection):
    meaning:str=Field(min_length=1,max_length=100)
    source_note:str=Field(default='',max_length=500)
def brief(record):
    data={k:v for k,v in record.items() if k!='sequence'}
    data['video_available']=bool(video_file(record.get('video_id',''),get(record.get('video_id',''),'video')))
    return data
def extract(body,allow_draft=False):
    if not get(body.video_id,'video'):raise HTTPException(404,'视频不存在')
    try:
        r=load_result(body.video_id)
        if not body.start_ms<body.end_ms<=r['duration_ms']:raise ValueError('动作区间超出视频或为空')
        try:return sequence(r,body.start_ms,body.end_ms)
        except ValueError as e:
            if allow_draft:return {'draft_reason':str(e)}
            raise
    except FileNotFoundError:raise HTTPException(409,'请先完成视频分析')
    except ValueError as e:raise HTTPException(422,str(e))
@router.get('/templates')
def templates():return [brief(t) for t in all_of('sign_template')]
@router.post('/templates')
def create(body:Template):
    seq=extract(body,allow_draft=True)
    if not body.meaning.strip():raise HTTPException(422,'请填写含义')
    ident='sign_'+uuid4().hex
    record={**body.model_dump(),'meaning':body.meaning.strip(),'id':ident,'sequence':seq,'provenance':'user_labeled_demonstration','standard_verified':False,'recognition_ready':'rows' in seq,'draft_reason':seq.get('draft_reason')}
    put(ident,'sign_template',record)
    return brief(record)
@router.delete('/templates/{ident}')
def remove(ident:str):
    record=get(ident,'sign_template')
    if not record:raise HTTPException(404,'示范不存在或已被删除')
    delete(ident,'sign_template')
    # 视频文件不随示范一起删：同一条视频可能还承载其它词，也被结果缓存引用。
    remaining=[t for t in all_of('sign_template') if t['video_id']==record['video_id']]
    return {'deleted':ident,'video_id':record['video_id'],'orphan_video':not remaining}
@router.get('/templates/{ident}/frames')
def frames(ident:str,count:int=3,height:int=200):
    """Stills across one saved demonstration, so it can be previewed without opening the video."""
    record=get(ident,'sign_template')
    if not record:raise HTTPException(404,'示范不存在或已被删除')
    if not video_file(record['video_id'],get(record['video_id'],'video')):
        raise HTTPException(410,'源视频已不在本地，无法生成预览截图')
    try:r=load_result(record['video_id'])
    except FileNotFoundError:raise HTTPException(409,'这条视频的分析结果不在了，无法生成预览截图')
    size=max(60,min(height,400));number=max(1,min(count,5))
    key=(ident,number,size,r.get('result_version_id') or r.get('analysis_version') or '')
    with _FRAME_LOCK:
        cached=_FRAME_CACHE.get(key)
    if cached is not None:
        return Response(cached,media_type='image/png',headers=_FRAME_HEADERS)
    try:png=preview_strip(r,record['start_ms'],record['end_ms'],DATA/'assets',number,size)
    except ValueError as e:raise HTTPException(422,str(e))
    with _FRAME_LOCK:
        if len(_FRAME_CACHE)>=64:_FRAME_CACHE.clear()
        _FRAME_CACHE[key]=png
    return Response(png,media_type='image/png',headers=_FRAME_HEADERS)
@router.get('/sources')
def sources():
    """Videos already used as demonstration sources, so one video can be revisited."""
    groups={}
    for t in all_of('sign_template'):
        g=groups.setdefault(t['video_id'],{'video_id':t['video_id'],'count':0,'meanings':[]})
        g['count']+=1;g['meanings'].append(t['meaning'])
    for vid,g in groups.items():
        v=get(vid,'video') or {}
        g['filename']=v.get('filename') or v.get('original_name') or vid
        g['duration_ms']=v.get('duration_ms')
    return sorted(groups.values(),key=lambda g:-g['count'])
@router.get('/candidates/{video_id}')
def candidates(video_id:str):
    """Segmented actions of one analysed video, each pre-checked for labelability."""
    if not get(video_id,'video'):raise HTTPException(404,'视频不存在')
    try:r=load_result(video_id)
    except FileNotFoundError:raise HTTPException(409,'请先完成视频分析')
    used=[(t['start_ms'],t['end_ms']) for t in all_of('sign_template') if t['video_id']==video_id]
    items=[]
    for index,a in enumerate(r.get('actions') or []):
        taken=any(a['start_ms']<end and start<a['end_ms'] for start,end in used)
        try:
            sequence(r,a['start_ms'],a['end_ms']);usable,reason=True,None
        except ValueError as e:
            usable,reason=False,str(e)
        items.append({'start_ms':a['start_ms'],'end_ms':a['end_ms'],'title':a.get('title') or '',
                      'label':a.get('title') or '区间 %d'%(index+1),'taken':taken,
                      'usable':usable,'reason':reason})
    quality=(r.get('coverage') or {}).get('hand_quality') or {}
    return {'video_id':video_id,'duration_ms':r['duration_ms'],'actions':items,
            'usable_count':sum(1 for a in items if a['usable']),
            'suspect_rate':quality.get('suspect_rate'),
            'framing':(r.get('scenes') or [{}])[0].get('framing_type')}

@router.post('/match')
def match(body:Selection):
    seq=extract(body);templates=[]
    for t in all_of('sign_template'):
        if not usable(t):continue
        # 只跳过区间自重叠的示范，同一条视频里的其它词仍然可以参与匹配。
        if t['video_id']==body.video_id and t['start_ms']<body.end_ms and body.start_ms<t['end_ms']:continue
        templates.append(t)
    skipped=len(all_of('sign_template'))-len(templates)
    result=retrieve(seq,templates)
    return {**result,'skipped':skipped,'notice':'示范检索原型；阈值尚未校准，距离不是词义概率或规范性分数。与所选区间重叠的示范不参与匹配。'}
