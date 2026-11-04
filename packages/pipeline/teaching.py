"""Conservative 2-D landmark descriptions; never presented as AI semantics."""
import math
from .geometry import angle as geometric_angle,distance as geometric_distance

def dist(a,b):return math.hypot(a['x']-b['x'],a['y']-b['y'])
def angle(a,b,c):
    u=(a['x']-b['x'],a['y']-b['y']);v=(c['x']-b['x'],c['y']-b['y'])
    norm=math.hypot(*u)*math.hypot(*v)
    if norm<1e-8:return 0
    return math.degrees(math.acos(max(-1,min(1,(u[0]*v[0]+u[1]*v[1])/norm))))
def shape(p,width=1,height=1):
    if len(p)!=21:return '手型需核对'
    extended=[(geometric_angle(p[m],p[pip],p[tip],width,height) or 0)>155 and geometric_distance(p[0],p[tip],width,height)>geometric_distance(p[0],p[pip],width,height)*1.1 for m,pip,tip in [(5,6,8),(9,10,12),(13,14,16),(17,18,20)]]
    if all(extended):return '四指展开'
    if not any(extended):return '四指弯曲'
    if extended==[True,False,False,False]:return '食指伸展'
    if extended==[True,True,False,False]:return '食指与中指伸展'
    return '部分手指伸展'

def enrich_action(action,frames):
    samples=[f for f in frames if action['start_ms']<=f['timestamp_ms']<action['end_ms']]
    if not samples:return
    key=min(samples,key=lambda f:abs(f['timestamp_ms']-action['keyframe_ms']))
    size=key.get('image_size',{});width=size.get('width',1);height=size.get('height',1)
    hands=[]
    for track in key['tracks']:
        for side,hand in track['hands'].items():
            label={'left':'人物左手','right':'人物右手'}.get(side,'可见手部')
            wrist=hand['landmarks'][0]
            location=('画面左侧' if wrist['x']<.4 else '画面右侧' if wrist['x']>.6 else '画面中部')
            hands.append(label+'位于'+location+'，参考手型为“'+shape(hand['landmarks'],width,height)+'”')
    paths=[]
    # Stable track+side combinations only. Missing frames split trajectories.
    keys={(t['track_id'],side) for f in samples for t in f['tracks'] for side in t['hands']}
    for ident,side in sorted(keys):
        points=[]
        for f in samples:
            tr=next((t for t in f['tracks'] if t['track_id']==ident),None)
            hand=tr['hands'].get(side) if tr else None
            if not hand:
                if len(points)>1:paths.append({'track_id':ident,'side':side,'points':points})
                points=[];continue
            p=hand['landmarks'][0];points.append({'x':p['x'],'y':p['y'],'timestamp_ms':f['timestamp_ms']})
        if len(points)>1:paths.append({'track_id':ident,'side':side,'points':points})
    meaningful=[p for p in paths if sum(geometric_distance(a,b,width,height) for a,b in zip(p['points'],p['points'][1:]))>.04*max(width,height)]
    movement=[]
    for p in meaningful[:2]:
        a,b=p['points'][0],p['points'][-1];dx=b['x']-a['x'];dy=b['y']-a['y']
        horizontal=('右' if dx>0 else '左') if abs(dx)>.035 else ''
        vertical=('下' if dy>0 else '上') if abs(dy)>.035 else ''
        movement.append('手腕向画面'+horizontal+vertical+'方移动' if horizontal or vertical else '手腕沿曲线路径移动并回到附近位置')
    rotations=[]
    for ident,side in sorted(keys):
        observations=[]
        for f in samples:
            tr=next((tr for tr in f['tracks'] if tr['track_id']==ident),None);hand=tr['hands'].get(side) if tr else None
            if not hand:continue
            p=hand['landmarks'];observations.append((math.atan2((p[9]['y']-p[0]['y'])*height,(p[9]['x']-p[0]['x'])*width),p[0],geometric_distance(p[0],p[9],width,height)))
        if len(observations)<2:continue
        delta=sum((b[0]-a[0]+math.pi)%(2*math.pi)-math.pi for a,b in zip(observations,observations[1:]))
        if abs(delta)>math.radians(20):
            rotations.append({'track_id':ident,'side':side,'delta_radians':delta,'start_angle':observations[0][0],'center':observations[len(observations)//2][1],'radius_pixels':observations[len(observations)//2][2],'interpretation':'腕至中指掌指关节的二维方向变化，不能代表真实翻掌角度'})
    if rotations:movement.append('手掌参考轴的画面方向有旋转变化，需对照原片确认翻掌过程')
    action['rotation_facts']=rotations
    if hands:
        action['title']='手部移动' if movement else '参考手型：'+shape(next(h['landmarks'] for t in key['tracks'] for h in t['hands'].values()))
        action['description']='；'.join(hands+movement)+'。先对照关键帧摆好手型，再跟随原片确认动作顺序。以上为二维关键点规则推测，手指遮挡和掌心朝向需人工核对。'
    action['motion_type']='moving_hands' if movement else 'pose_candidate'
    action['trajectories']=meaningful
    action['description_source']='landmark_rules_v1'

def trajectory_svg(paths,width,height,rotations=None):
    pieces=[]
    for path in paths:
        pts=path['points']
        # Keep endpoints while limiting SVG size.
        selected=pts[::max(1,len(pts)//16)]
        if selected[-1]!=pts[-1]:selected.append(pts[-1])
        coordinates=' '.join(f'{p["x"]*width:.1f},{p["y"]*height:.1f}' for p in selected)
        pieces.append(f'<polyline points="{coordinates}" fill="none" stroke="#df9955" stroke-width="{max(3,width/200):.1f}" stroke-dasharray="8 5" marker-end="url(#direction)"/>')
    for rotation in rotations or []:
        center=rotation['center'];radius=max(8,rotation['radius_pixels']*1.1);delta=max(-2*math.pi,min(2*math.pi,rotation['delta_radians']))
        pts=[]
        for i in range(20):
            a=rotation['start_angle']+delta*i/19;pts.append(f'{center["x"]*width+radius*math.cos(a):.1f},{center["y"]*height+radius*math.sin(a):.1f}')
        pieces.append('<polyline points="'+' '.join(pts)+'" fill="none" stroke="#df9955" stroke-width="3" marker-end="url(#direction)"/>')
    return '<defs><marker id="direction" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#df9955"/></marker></defs>'+''.join(pieces)
