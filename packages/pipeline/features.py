"""Versioned, masked 2-D features. No model z from different networks is fused."""
from .quality import usable_for_recognition
import math
import numpy as np
from .geometry import angle,distance,palm_scale,shoulder_scale,valid

FINGERS=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
HAND_NAMES=[f'finger_{i}_{name}' for i in range(5) for name in ['mcp_angle','pip_angle','dip_angle','tip_radius','thumb_distance']]+['wrist_x','wrist_y','rotation','speed','tip_speed','acceleration','curvature','amplitude','face_distance','chest_distance']
NAMES=[f'hand_{i}_{n}' for i in range(2) for n in HAND_NAMES]+['hand_distance','approach_speed','synchrony','valid_ratio','tracking_break','jitter']
VERSION='features-2.3'
SLOT_ORDER={'left':0,'right':1}

def slot_key(item):
    """两只手在特征向量里的固定槽位。

    槽位必须只由「哪只手」决定，不能由「先检测到哪只手」决定：轨迹号是检测创建
    顺序，示范视频和实时画面之间没有任何对应关系，按它排序会让同一动作左右对调，
    自距离从 0 直接跳到 0.3 以上。模型给的左右手判定与视角、镜像方向无关，最稳；
    解剖侧依赖躯干检测，实时里经常是 unknown，只作退路。
    """
    tr,side,hand=item
    label=(hand.get('model_handedness') or '').strip().lower()
    return (SLOT_ORDER.get(label,2),SLOT_ORDER.get(side,2),tr['track_id'])

def extract_features(frames,width,height,fixed_slots=False):
    rows=[];previous={};last_pair=None
    for f in frames:
        t=f['timestamp_ms'];hands=[(tr,side,h) for tr in f['tracks'] for side,h in tr['hands'].items()]
        hands.sort(key=slot_key)
        if fixed_slots:
            assigned=[None,None]
            for item in hands:
                label=(item[2].get('model_handedness') or item[1]).strip().lower()
                target=SLOT_ORDER.get(label)
                if target is not None and assigned[target] is None:assigned[target]=item
            hands=assigned
        bodies={tr['track_id']:tr for tr in f['tracks'] if len(tr['pose_landmarks'])==33}
        values=[];hand_info=[];breaks=0;jitter=[]
        for slot in range(2):
            if slot>=len(hands) or hands[slot] is None:values.extend([None]*len(HAND_NAMES));continue
            tr,side,hand=hands[slot];ident=(f['scene_id'],tr['track_id'],side)
            if not usable_for_recognition(hand):
                values.extend([None]*len(HAND_NAMES));previous.pop(ident,None);breaks+=1;continue
            p=hand['landmarks'];scale=palm_scale(p,width,height)
            body=bodies.get(hand.get('owner_track_id') or tr.get('person_track_id'));pose=body['pose_landmarks'] if body else []
            body_scale=shoulder_scale(pose,width,height) if pose else scale
            static=[]
            for wrist,mcp,pip,dip,tip in FINGERS:
                if not all(valid(p[k]) or p[k].get('interpolated') for k in [wrist,mcp,pip,dip,tip]):static.extend([None]*5);continue
                a=[angle(p[wrist],p[mcp],p[pip],width,height),angle(p[mcp],p[pip],p[dip],width,height),angle(p[pip],p[dip],p[tip],width,height)]
                static.extend([v/180 if v is not None else None for v in a]+[distance(p[0],p[tip],width,height)/scale,distance(p[4],p[tip],width,height)/scale])
            wrist=p[0];rotation=math.atan2((p[9]['y']-wrist['y'])*height,(p[9]['x']-wrist['x'])*width)
            old=previous.get(ident);dt=(t-old['t'])/1000 if old else 0
            continuous=old and 0<dt<=.35 and hand.get('identity_status')!='unknown'
            speed=tip_speed=accel=curvature=None;vx=vy=None
            if continuous:
                vx=(wrist['x']-old['p'][0]['x'])*width/body_scale/dt;vy=(wrist['y']-old['p'][0]['y'])*height/body_scale/dt
                speed=math.hypot(vx,vy)
                tip_speed=float(np.mean([distance(p[k],old['p'][k],width,height)/scale/dt for k in [4,8,12,16,20]]))
                if old.get('speed') is not None:accel=(speed-old['speed'])/dt
                if old.get('v') and speed>1e-5:
                    ax=(vx-old['v'][0])/dt;ay=(vy-old['v'][1])/dt;curvature=abs(vx*ay-vy*ax)/(speed**3+1e-6)
                jitter.append(abs(accel) if accel is not None else 0)
            else:breaks+=1
            origin=old['origin'] if continuous else wrist
            face_dist=distance(wrist,pose[0],width,height)/body_scale if pose else None
            chest={'x':(pose[11]['x']+pose[12]['x'])/2,'y':(pose[11]['y']+pose[12]['y'])/2} if pose else None
            chest_dist=distance(wrist,chest,width,height)/body_scale if chest else None
            values.extend(static+[wrist['x']*width/max(width,height),wrist['y']*height/max(width,height),rotation,speed,tip_speed,accel,curvature,distance(origin,wrist,width,height)/body_scale,face_dist,chest_dist])
            previous[ident]={'t':t,'p':p,'speed':speed,'v':(vx,vy) if vx is not None else None,'origin':origin}
            hand_info.append((ident,wrist,body_scale,speed))
        separation=approach=sync=None
        if len(hand_info)==2:
            a,b=hand_info;separation=distance(a[1],b[1],width,height)/max(a[2],b[2])
            pair=(a[0],b[0]);dt=(t-last_pair['t'])/1000 if last_pair else 0
            if last_pair and last_pair['pair']==pair and 0<dt<=.35:approach=(separation-last_pair['distance'])/dt
            if a[3] is not None and b[3] is not None:sync=1-abs(a[3]-b[3])/(a[3]+b[3]+1e-6)
            last_pair={'pair':pair,'distance':separation,'t':t}
        observed=sum(v is not None and math.isfinite(v) for v in values)/max(1,len(values))
        values.extend([separation,approach,sync,observed,float(breaks),float(np.mean(jitter)) if jitter else None])
        mask=[v is not None and math.isfinite(v) for v in values]
        values=[float(v) if m else None for v,m in zip(values,mask)]
        rows.append({'timestamp_ms':t,'scene_id':f['scene_id'],'values':values,'mask':mask,'hand_track_ids':[str(h[0][1]) for h in hand_info]})
    return {'version':VERSION+('-fixedslots' if fixed_slots else ''),'names':NAMES,'rows':rows}
