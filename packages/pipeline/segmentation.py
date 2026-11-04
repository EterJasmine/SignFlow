import numpy as np
from .motion import motion_steps as baseline_motion_steps

# 手型维度的识别键：角度 + 指尖半径 + 拇指间距，跨帧可控、对镜头距离不太敏感
SHAPE_KEYS=('_mcp_angle','_pip_angle','_dip_angle','_tip_radius','_thumb_distance')
SEGMENTATION_VERSION='segmentation-2.2'
# 手型换态阈值：在 5 条已标视频上扫过 0.02–0.20，0.10–0.14 是平台，取中值
DEFAULT_SHAPE_DELTA=0.12


def _shape_columns(names):
    return [i for i,n in enumerate(names) if any(k in n for k in SHAPE_KEYS)]


def shape_state_boundaries(rows,names,delta,min_step_ms=600,window=2,min_dims=10):
    """对手型序列做「平台—跳变」检测，补上停顿时才切分的漏检。

    第 i 个采样点的分数 = 之前 window 帧与前之后 window 帧的逐维中位数之差。
    差值大于 delta 说明手型换了一个稳定态；边界取跳变起点（中位数窗口往前挪半个窗）。
    """
    cols=_shape_columns(names)
    n=len(rows)
    if n<2*window+2 or not cols:
        return []
    frame_ms=(rows[-1]['timestamp_ms']-rows[0]['timestamp_ms'])/max(n-1,1)
    scored=[]
    for i in range(window,n-window):
        neighborhood=rows[i-window:i+window]
        if any(r['scene_id']!=rows[i]['scene_id'] for r in neighborhood):continue
        if any(b['timestamp_ms']-a['timestamp_ms']>350 for a,b in zip(neighborhood,neighborhood[1:])):continue
        diffs=[]
        for c in cols:
            before=[r['values'][c] for r in rows[i-window:i] if r['mask'][c]]
            after=[r['values'][c] for r in rows[i:i+window] if r['mask'][c]]
            if len(before)>=max(1,window-1) and len(after)>=max(1,window-1):
                diffs.append(abs(float(np.median(before))-float(np.median(after))))
        if len(diffs)>=min_dims:
            scored.append((float(np.mean(diffs)),i))
    picked=[];min_gap=max(min_step_ms,400)
    for score,i in sorted(scored,reverse=True):
        if score<delta:
            break
        t=rows[i]['timestamp_ms']
        if not picked or all(abs(t-p)>=min_gap for p in picked):
            picked.append(t)
    if not picked:
        return []
    shift=int(round(window*frame_ms))
    return sorted({max(0,t-shift) for t in picked})


def _merge_boundaries(points,min_gap_ms=350):
    kept=[]
    for t in sorted(set(points)):
        if kept and t-kept[-1]<min_gap_ms:
            continue
        kept.append(t)
    return kept


def segment_actions(frames,scenes,duration_ms,features,min_step_ms=600,max_step_ms=6000,shape_delta=DEFAULT_SHAPE_DELTA):
    names=features['names'];rows=features['rows'];scores=[];phases=[]
    relevant=[i for i,n in enumerate(names) if any(k in n for k in ['angle','radius','thumb_distance','rotation','hand_distance','face_distance','chest_distance'])]
    for i,row in enumerate(rows):
        if not i or row['scene_id']!=rows[i-1]['scene_id']:scores.append(None);continue
        old=rows[i-1];dt=(row['timestamp_ms']-old['timestamp_ms'])/1000
        diffs=[abs(row['values'][j]-old['values'][j])/max(dt,.001) for j in relevant if row['mask'][j] and old['mask'][j]]
        motion=[row['values'][j] for j,n in enumerate(names) if n.endswith('_speed') and row['mask'][j]]
        scores.append((float(np.mean(diffs)) if diffs else 0.)+.15*max(motion,default=0) if diffs or motion else None)
    finite=[s for s in scores if s is not None];threshold=max(.18,float(np.median(finite))*.7) if finite else .18
    active=[s is not None and s>threshold for s in scores]
    # Fill one-frame inactive holes inside a continuous movement; preserve brief changes.
    for i in range(1,len(active)-1):
        if not active[i] and active[i-1] and active[i+1] and rows[i+1]['timestamp_ms']-rows[i-1]['timestamp_ms']<350:active[i]=True
    boundaries={0,duration_ms,*[s['start_ms'] for s in scenes]};last=0
    for i,row in enumerate(rows):
        if scores[i] is None:phase='preparation'
        elif active[i]:phase='transition'
        elif i and active[i-1]:phase='release'
        else:phase='hold'
        phases.append({'timestamp_ms':row['timestamp_ms'],'phase':phase,'change_score':scores[i],'quality':row['values'][names.index('valid_ratio')]})
        # Start a new teaching unit after a stable hold, never at peak speed.
        if i>=3 and active[i] and not any(active[i-3:i]):
            t=rows[i-1]['timestamp_ms']
            if t-last>=min_step_ms and duration_ms-t>=min_step_ms:boundaries.add(t);last=t
    # 手型换态也能起一步：连续动作里没有停顿，光看起停会系统性欠分割
    if shape_delta>0:
        boundaries.update(shape_state_boundaries(rows,names,shape_delta,min_step_ms=min_step_ms))
    base=sorted(boundaries)
    for start,end in zip(base,base[1:]):
        cursor=start
        while end-cursor>max_step_ms:
            target=cursor+max_step_ms
            options=[(scores[i] if scores[i] is not None else float('inf'),r['timestamp_ms']) for i,r in enumerate(rows) if target-750<=r['timestamp_ms']<=target and r['timestamp_ms']-cursor>=min_step_ms]
            chosen=min(options)[1] if options else target;boundaries.add(chosen);cursor=chosen
    ordered=_merge_boundaries(boundaries)
    actions=[]
    for n,(start,end) in enumerate(zip(ordered,ordered[1:])):
        indices=[i for i,r in enumerate(rows) if start<=r['timestamp_ms']<end]
        if not indices:continue
        def quality(i):
            detection=rows[i]['values'][names.index('valid_ratio')]
            stability=1/(1+(scores[i] or 0));sharp=frames[i].get('sharpness',0)
            return .5*detection+.35*stability+.15*min(1,sharp/300)
        best=max(indices,key=quality);first=indices[0];last=indices[-1]
        actions.append({'id':f'action_{n+1:04}','start_ms':start,'end_ms':end,'keyframe_ms':rows[best]['timestamp_ms'],
                        'start_frame_ms':rows[first]['timestamp_ms'],'end_frame_ms':rows[last]['timestamp_ms'],
                        'transition_frames_ms':[rows[i]['timestamp_ms'] for i in indices if active[i]][::max(1,len(indices)//4)][:4],
                        'title':f'动作步骤 {n+1}','description':'请核对动作事实。','motion_type':'candidate','person_track_id':None,'lyric_ids':[],
                        'assets':{'source':None,'diagram':None,'illustration':None},'review_status':'pending','segmentation_method':'multifeature_hold_v2' if shape_delta<=0 else 'multifeature_hold_v2+shape_state'})
    return actions,phases,{'version':SEGMENTATION_VERSION,'threshold':threshold,'min_step_ms':min_step_ms,'max_step_ms':max_step_ms,'shape_delta':shape_delta,'manual_acceptance':'not_evaluated'}
