import hashlib,json
from pathlib import Path
import numpy as np
from .annotations import TAXONOMY
from packages.pipeline.geometry import palm_scale,shoulder_scale

def group_splits(annotations,group_by='video'):
    groups={}
    for a in annotations:
        key=a.get('person_id') if group_by=='person' else a['video_id']
        if not key:raise ValueError('按人物分组需要每条标注提供真实 person_id')
        groups[a['id']]=key
    unique=sorted(set(groups.values()),key=lambda k:hashlib.sha256(k.encode()).hexdigest())
    if group_by=='person':
        # Connected components of people sharing a source video avoid frame leakage.
        parent={k:k for k in unique}
        def find(k):
            while parent[k]!=k:k=parent[k]
            return k
        by_video={}
        for a in annotations:
            k=groups[a['id']];other=by_video.setdefault(a['video_id'],k);parent[find(k)]=find(other)
        groups={ident:find(k) for ident,k in groups.items()}
        unique=sorted(set(groups.values()),key=lambda k:hashlib.sha256(k.encode()).hexdigest())
    if len(unique)<3:return {a['id']:'unassigned' for a in annotations}
    n=len(unique);test=max(1,round(n*.15));val=max(1,round(n*.15));mapping={k:('test' if i<test else 'validation' if i<test+val else 'train') for i,k in enumerate(unique)}
    return {a['id']:mapping[groups[a['id']]] for a in annotations}

def skeleton_tensor(frame,width,height):
    coords=np.zeros((75,2),np.float32);mask=np.zeros(75,bool)
    bodies=[t for t in frame['tracks'] if len(t['pose_landmarks'])==33]
    body=max(bodies,key=lambda t:shoulder_scale(t['pose_landmarks'],width,height),default=None)
    if body:
        p=body['pose_landmarks'];origin=p[11];scale=shoulder_scale(p,width,height)
        for i,point in enumerate(p):
            if (point.get('visibility') or 0)<.5:continue
            coords[i]=[(point['x']-origin['x'])*width/scale,(point['y']-origin['y'])*height/scale];mask[i]=True
    hands=[(side,h) for t in frame['tracks'] for side,h in t['hands'].items()]
    hands.sort(key=lambda x:({'left':0,'right':1}.get(x[0],2),x[1]['landmarks'][0]['x']))
    for side,hand in hands:
        if side not in ['left','right']:continue
        if body and hand.get('owner_track_id')!=body['track_id']:continue
        slot=0 if side=='left' else 1
        p=hand['landmarks'];scale=shoulder_scale(body['pose_landmarks'],width,height) if body else palm_scale(p,width,height);origin=body['pose_landmarks'][11] if body else p[0]
        for i,point in enumerate(p):
            if point.get('low_quality'):continue
            coords[33+slot*21+i]=[(point['x']-origin['x'])*width/scale,(point['y']-origin['y'])*height/scale];mask[33+slot*21+i]=True
    return coords,mask

def export_dataset(annotations,load_result,output,length=32,group_by='video'):
    if length not in [32,64]:raise ValueError('序列长度只支持32或64')
    accepted=[a for a in annotations if a.get('reviewed') and a['label']!='unknown']
    if not accepted:raise ValueError('没有人工确认的真实标注，不能导出训练数据')
    splits=group_splits(accepted,group_by);samples=[];cache={}
    for a in accepted:
        r=cache.setdefault(a['video_id'],load_result(a['video_id']));features=r.get('features')
        if not features:raise ValueError('该视频缺少v2特征，请先重新分析')
        rows=features['rows'];frames=r.get('processed_frames',r['frames']);ts=np.array([f['timestamp_ms'] for f in frames]);target=np.linspace(a['start_ms'],a['end_ms']-1,length)
        x=np.zeros((length,len(features['names'])),np.float32);mask=np.zeros_like(x,bool);graph=np.zeros((length,75,2),np.float32);gmask=np.zeros((length,75),bool)
        phase=np.full(length,-100,np.int64);boundary=np.zeros(length,np.float32);bmask=np.zeros(length,bool)
        phases=[p for p in accepted if p['video_id']==a['video_id'] and p['kind']=='phase' and p.get('person_id')==a.get('person_id')]
        for i,t in enumerate(target):
            idx=int(np.argmin(abs(ts-t)))
            if abs(ts[idx]-t)>200:continue
            row=rows[idx];x[i]=[v if m else 0 for v,m in zip(row['values'],row['mask'])];mask[i]=row['mask']
            graph[i],gmask[i]=skeleton_tensor(frames[idx],r['width'],r['height'])
            for p in phases:
                if p['start_ms']<=t<p['end_ms']:phase[i]=TAXONOMY['phase'].index(p['label'])-1;break
            bmask[i]=True;boundary[i]=float(min(abs(t-a['start_ms']),abs(t-a['end_ms']))<=150)
        category=TAXONOMY['dynamic'].index(a['label']) if a['kind']=='dynamic' else -100
        static=TAXONOMY['static'].index(a['label']) if a['kind']=='static' else -100
        samples.append({'x':x,'mask':mask,'graph':graph,'graph_mask':gmask,'phase':phase,'boundary':boundary,'boundary_mask':bmask,'category':category,'static':static,'annotation':a,'split':splits[a['id']]})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:np.stack([s[k] for s in samples]) for k in ['x','mask','graph','graph_mask','phase','boundary','boundary_mask']}
    arrays.update(category=np.array([s['category'] for s in samples]),static=np.array([s['static'] for s in samples]),split=np.array([s['split'] for s in samples]))
    np.savez_compressed(output,**arrays)
    manifest={'schema_version':'dataset-1.0','length':length,'feature_version':features['version'],'feature_names':features['names'],'graph_topology':'mediapipe_33_pose_plus_21_plus_21_v1','group_by':group_by,'taxonomy':TAXONOMY,
              'samples':[{'annotation_id':s['annotation']['id'],'video_id':s['annotation']['video_id'],'person_id':s['annotation'].get('person_id'),'split':s['split'],'label':s['annotation']['label'],'kind':s['annotation']['kind'],'start_ms':s['annotation']['start_ms'],'end_ms':s['annotation']['end_ms'],'action_id':s['annotation'].get('action_id') or None,'boundary_source':s['annotation'].get('boundary_source','action_seed'),'source_result_version':s['annotation'].get('source_result_version')} for s in samples],
              'boundary_provenance':{'action_seed':sum(1 for s in samples if s['annotation'].get('boundary_source','action_seed')=='action_seed'),'manual':sum(1 for s in samples if s['annotation'].get('boundary_source')=='manual'),'note':'action_seed 表示边界取自教学步骤，manual 表示标注者另定了动作实例边界；两者混用会稀释教学与训练的可比性。'},
              'training_ready':all(v!='unassigned' for v in splits.values()),'warning':'少于三个独立组时不划分训练/验证/测试；不自动训练。'}
    output.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');return manifest
