import math
import numpy as np

def lyric_overlap(action, lyrics):
    return [{'lyric_id':l['id'], 'overlap_ms':min(action['end_ms'],l['end_ms'])-max(action['start_ms'],l['start_ms'])}
            for l in lyrics if min(action['end_ms'],l['end_ms']) > max(action['start_ms'],l['start_ms'])]

def motion_steps(frames, scenes, duration_ms,width=1,height=1):
    """Baseline velocity boundaries. Never differentiate across a missing track."""
    values=[]
    for a,b in zip(frames, frames[1:]):
        old={t['track_id']:t for t in a['tracks']}
        distances=[]
        dt=(b['timestamp_ms']-a['timestamp_ms'])/1000
        if a['scene_id']==b['scene_id'] and dt>0:
            for track in b['tracks']:
                if track['track_id']=='hands_unassigned':continue
                prev=old.get(track['track_id'])
                if not prev:
                    continue
                for side in ['left','right','unknown']:
                    h=track['hands'].get(side); ph=prev['hands'].get(side)
                    if h and ph:
                        if side=='unknown' and not h.get('identity_continuous'):continue
                        p,q=h['landmarks'][0],ph['landmarks'][0]
                        distances.append(math.hypot((p['x']-q['x'])*width,(p['y']-q['y'])*height)/max(width,height)/dt)
        values.append((b['timestamp_ms'],float(np.mean(distances)) if distances else None))
    observed=[x for _,x in values if x is not None]
    threshold=float(np.percentile(observed,75)) if observed else float('inf')
    boundaries={0,duration_ms,*[s['start_ms'] for s in scenes]}
    last=0
    for i,(t,value) in enumerate(values):
        previous=values[i-1][1] if i else None
        following=values[i+1][1] if i+1<len(values) else None
        if value is not None and value>max(.12,threshold) and previous is not None and following is not None and value>=previous and value>=following and t-last>=800 and duration_ms-t>=600:
            boundaries.add(t);last=t
    boundaries=sorted(boundaries)
    # Long unsegmented spans are kept intact; no invented motion boundaries.
    actions=[]
    for i,(start,end) in enumerate(zip(boundaries,boundaries[1:])):
        if end<=start:continue
        candidates=[f for f in frames if start<=f['timestamp_ms']<end]
        # Prefer well-observed frames near the step's middle over the first frame.
        middle=(start+end)/2
        best=max(candidates,key=lambda f:(sum(len(t['hands']) for t in f['tracks']),-abs(f['timestamp_ms']-middle)),default=None)
        actions.append({'id':f'action_{i+1:04}', 'start_ms':start,'end_ms':end,
                        'keyframe_ms':best['timestamp_ms'] if best else start,'title':f'动作步骤 {i+1}',
                        'description':'请核对原片并填写手型、位置与移动方向。',
                        'motion_type':'candidate','person_track_id':None,'lyric_ids':[],
                        'assets':{'source':None,'diagram':None,'illustration':None},'review_status':'pending',
                        'segmentation_method':'wrist_velocity_peaks_v1'})
    return actions
