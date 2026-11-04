"""Evidence-driven learning coach; deterministic tool orchestration, no LLM required."""
import time
from uuid import uuid4
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from .store import get,put,all_of
from packages.pipeline.coach_llm import explain,capabilities
from packages.pipeline.signmatch import sequence,dtw,usable
router=APIRouter(prefix='/api/v1/sign/coach')
class Goal(BaseModel):meaning:str=Field(min_length=1,max_length=100)
class Attempt(BaseModel):diagnostic_id:str

def decide(coverage,distance,previous):
    if coverage<.35:return {'action':'adjust_capture','feedback':'这次有些动作没看清。请让双手完整入镜、找个明亮的位置，再试一次。'}
    if distance is None:return {'action':'review_demo','feedback':'这次还没法判断。先重看一遍示范，再完整做一次试试。'}
    change=None if previous is None else previous-distance
    if distance<.18:
        return {'action':'repeat_to_confirm','feedback':'这次和示范比较接近。再完整做一次，看看能不能保持住。','change':change}
    return {'action':'review_demo','feedback':'这次和示范还有些差别。先慢放看一遍，注意完整过程，再跟着做一次。','change':change}

@router.get('/sessions')
def sessions():return sorted(all_of('sign_coach_session'),key=lambda s:s['created_at'],reverse=True)[:20]

@router.post('/sessions')
def start(body:Goal):
    meaning=body.meaning.strip()
    choices=[t for t in all_of('sign_template') if t['meaning']==meaning and usable(t)]
    if not choices:raise HTTPException(422,'这个词还没有可用示范，请先在示范库添加完整动作')
    template=max(choices,key=lambda t:len(t['sequence']['rows']))
    ident='coach_'+uuid4().hex
    return put(ident,'sign_coach_session',{'id':ident,'meaning':meaning,'template_id':template['id'],'video_id':template['video_id'],
        'start_ms':template['start_ms'],'end_ms':template['end_ms'],'created_at':time.time(),'attempts':[],
        'next_action':'watch_demo','feedback':'先观看这条用户示范，然后使用实时区的「标记动作开始／结束并识别」完成一次练习。',
        'trace':[{'tool':'find_demonstrations','evidence':str(len(choices))+'条可用示范'},{'tool':'choose_demonstration','evidence':template['id']} ]})

@router.post('/sessions/{ident}/attempts')
def evaluate(ident:str,body:Attempt):
    session=get(ident,'sign_coach_session');record=get(body.diagnostic_id,'sign_live_check')
    if not session:raise HTTPException(404,'练习会话不存在')
    if not record:raise HTTPException(404,'动作诊断不存在或已过期，请重新做一次')
    if record['created_at']<session['created_at']:raise HTTPException(422,'请在选择练习词后完成一次新动作')
    if any(a['diagnostic_id']==body.diagnostic_id for a in session['attempts']):return session
    features=record.get('features',{});rows=features.get('rows',[])
    handcols=[i for i,n in enumerate(features.get('names',[])) if 'finger_' in n]
    coverage=sum(sum(bool(r['mask'][i]) for i in handcols) for r in rows)/max(1,len(rows)*len(handcols))
    distances=[]
    try:
        seq=sequence({'features':features},record['start_ms'],record['end_ms'])
        for t in all_of('sign_template'):
            if t['meaning']!=session['meaning'] or not usable(t):continue
            try:distances.append(dtw(seq,t['sequence']))
            except ValueError:continue
    except ValueError:pass
    distances.sort();distance=sum(distances[:3])/len(distances[:3]) if distances else None
    previous=next((a['distance'] for a in reversed(session['attempts']) if a['distance'] is not None),None)
    decision=decide(coverage,distance,previous)
    explanation=explain({'target_word':session['meaning'],'valid_feature_ratio':coverage,'target_distance':distance,
        'previous_distance':previous,'next_action':decision['action'],'reference':'user_labeled_demonstration'},decision['feedback'])
    session['explanation']=explanation
    decision['feedback']=explanation['feedback']
    session['attempts'].append({'diagnostic_id':body.diagnostic_id,'coverage':coverage,'distance':distance,'created_at':time.time(),'decision':decision})
    session['attempts']=session['attempts'][-30:]
    session.update(next_action=decision['action'],feedback=decision['feedback'],trace=[{'tool':'inspect_capture','evidence':'有效手指特征占比 %.0f%%'%(coverage*100)},
        {'tool':'compare_target_demonstrations','evidence':('距离 %.3f'%distance) if distance is not None else '不可比较'},
        {'tool':'choose_next_practice','evidence':decision['action']}])
    return put(ident,'sign_coach_session',session)


@router.get('/capabilities')
def coach_capabilities():return capabilities()


@router.get('/review')
def practice_review():
    from packages.pipeline.practice_review import build_review
    return build_review(all_of('sign_coach_session'))

@router.post('/review')
def generate_review():
    from packages.pipeline.practice_review import build_review
    from packages.pipeline.coach_llm import summarize_review
    review=summarize_review(build_review(all_of('sign_coach_session')))
    put('latest_practice_review','sign_practice_review',{**review,'created_at':time.time()})
    return review
