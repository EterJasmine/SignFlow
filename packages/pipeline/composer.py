from typing import Literal
import hashlib,json,math
from pydantic import BaseModel,Field
from .geometry import distance,shoulder_scale
from .motion import lyric_overlap

class TeachingText(BaseModel):
    title:str=Field(min_length=1,max_length=150)
    description:str=Field(min_length=1,max_length=3000)
    caution:str=Field(default='请对照原视频核对手型与朝向。',max_length=1000)
class TutorialStep(BaseModel):
    id:str
    action_id:str
    start_ms:int
    end_ms:int
    poses:dict
    facts:dict
    text:TeachingText
    assets:dict
    layout:str
    lyric_ids:list[str]
    source_version:str
    facts_sha256:str
    review_status:Literal['pending','approved','rejected']='pending'
    generation_source:str='rules'
    generation_error:str|None=None

def facts_for(action,result):
    frames=result.get('processed_frames',result['frames']);frame=min(frames,key=lambda f:abs(f['timestamp_ms']-action['keyframe_ms']))
    hands=[];bodies={t['track_id']:t for t in frame['tracks'] if t['pose_landmarks']}
    for track in frame['tracks']:
        for side,hand in track['hands'].items():
            body=bodies.get(hand.get('owner_track_id'));p=hand['landmarks'];reference='unknown'
            if body:
                pose=body['pose_landmarks'];scale=shoulder_scale(pose,result['width'],result['height']);chest={'x':(pose[11]['x']+pose[12]['x'])/2,'y':(pose[11]['y']+pose[12]['y'])/2}
                if distance(p[0],pose[0],result['width'],result['height'])/scale<.7:reference='face'
                elif distance(p[0],chest,result['width'],result['height'])/scale<.8:reference='chest'
            hands.append({'track_id':track['track_id'],'side':side,'identity_status':hand.get('identity_status','unknown'),'body_reference':reference,'trajectory_quality':hand.get('trajectory_quality'),'landmarks':p})
    scene=next((s for s in result['scenes'] if s['start_ms']<=action['keyframe_ms']<s['end_ms']),{})
    relation=None
    if len(hands)==2:
        p,q=hands[0]['landmarks'][0],hands[1]['landmarks'][0]
        relation={'wrist_distance_pixels':distance(p,q,result['width'],result['height']),'coordinate_system':'pixel_aspect_correct','interpretation':'距离为二维投影，不能单独确定交叉遮挡关系'}
    return {'framing_type':scene.get('framing_type','unknown'),'hands':hands,'two_hand_relation':relation,'trajectories':action.get('trajectories',[]),'rotation':action.get('rotation_facts',[]),'coordinate_metadata':result.get('coordinate_metadata',{}),'evidence':'observed_and_explicitly_processed_landmarks','no_lyrics_in_motion_facts':True}

def compose_step(action,result):
    facts=facts_for(action,result);kind=facts['framing_type'];layout={'hands_only':'hands_closeup','upper_body':'upper_body_figure','full_body':'full_body_figure','partial_body':'visible_region'}.get(kind,'source_frame')
    digest=hashlib.sha256(json.dumps(facts,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return TutorialStep(id='step_'+action['id'],action_id=action['id'],start_ms=action['start_ms'],end_ms=action['end_ms'],
                        poses={'start_ms':action.get('start_frame_ms',action['start_ms']),'key_ms':action['keyframe_ms'],'end_ms':action.get('end_frame_ms',action['end_ms']-1)},facts=facts,
                        text=TeachingText(title=action['title'],description=action['description']),assets=action['assets'],layout=layout,
                        lyric_ids=[m['lyric_id'] for m in lyric_overlap(action,result.get('lyrics',[]))],source_version=result.get('analysis_version',result.get('pipeline_version','unknown')),facts_sha256=digest,review_status=action.get('review_status','pending')).model_dump()

def diagram_viewbox(svg,frame,kind,width,height):
    if kind not in ['hands_only','partial_body','upper_body']:return svg,{'viewbox':[0,0,width,height]}
    pts=[]
    for tr in frame['tracks']:
        for h in tr['hands'].values():pts.extend(h['landmarks'])
        if kind!='hands_only':pts.extend(p for p in tr['pose_landmarks'] if (p.get('visibility') or 0)>.6)
    pts=[p for p in pts if 0<=p['x']<=1 and 0<=p['y']<=1 and not p.get('low_quality')]
    if not pts:return svg,{'viewbox':[0,0,width,height]}
    left=max(0,min(p['x']*width for p in pts)-width*.05);top=max(0,min(p['y']*height for p in pts)-height*.05)
    right=min(width,max(p['x']*width for p in pts)+width*.05);bottom=min(height,max(p['y']*height for p in pts)+height*.05)
    box=[left,top,max(1,right-left),max(1,bottom-top)]
    return svg.replace(f'viewBox="0 0 {width} {height}"','viewBox="'+' '.join(str(round(v,2)) for v in box)+'"'),{'viewbox':box,'source_coordinates':'full_frame_pixels','crop_only_no_pose_change':True}
