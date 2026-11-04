"""Optional checkpoint inference; rules remain the default when no model exists."""
import os
from pathlib import Path

def predict(features,frames,width,height):
    checkpoint=os.getenv('GESTUREFLOW_MODEL')
    if not checkpoint:return {'available':False,'reason':'未配置人工标注训练的模型，使用规则分段。'}
    if not Path(checkpoint).is_file():return {'available':False,'reason':'模型文件不存在，使用规则分段。'}
    try:
        import torch,numpy as np
        from .models import build_model
        from .export import skeleton_tensor
        bundle=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=bundle['metadata']
        if not meta.get('trained_on_human_labels') or meta['feature_version']!=features['version'] or meta['feature_names']!=features['names']:raise ValueError('模型数据来源或特征版本不匹配')
        rows=features['rows']
        if meta['architecture']=='linear_change':
            # 变化能量特征是逐帧的，不需要滑窗；逐视频标准化必须在整条序列上做，比分块更准。
            from .features_diff import change_features
            values=np.array([[v if m else 0 for v,m in zip(r['values'],r['mask'])] for r in rows],dtype=np.float32)
            masks=np.array([r['mask'] for r in rows],dtype=bool)
            f=change_features(values,masks,features['names'],k=meta.get('k',2))
            model=build_model('linear_change',f.shape[1]);model.load_state_dict(bundle['state_dict']);model.eval()
            with torch.inference_mode():
                out=model(torch.tensor(f[None]),torch.ones(1,len(f),f.shape[1],dtype=torch.bool))['boundary'].sigmoid()[0]
            return {'available':True,'metadata':meta,'predictions':[{'timestamp_ms':r['timestamp_ms'],'boundary_probability':float(p)} for r,p in zip(rows,out)],'requires_review':True,'fusion':'candidate_assistance_only'}
        model=build_model(meta['architecture'],len(features['names']),meta.get('width',64));model.load_state_dict(bundle['state_dict']);model.eval()
        predictions=[];length=64
        with torch.inference_mode():
            for start in range(0,len(rows),length):
                chunk=rows[start:start+length]
                if meta['architecture']=='tcn':
                    x=torch.tensor([[[v if m else 0 for v,m in zip(r['values'],r['mask'])] for r in chunk]],dtype=torch.float32);mask=torch.tensor([[r['mask'] for r in chunk]])
                else:
                    graphs=[skeleton_tensor(f,width,height) for f in frames[start:start+length]];x=torch.tensor(np.stack([g[0] for g in graphs])[None]);mask=torch.tensor(np.stack([g[1] for g in graphs])[None])
                out=model(x,mask);task=meta['task']
                if task=='phase':
                    prob=out['phase'].softmax(-1)[0];predictions.extend({'timestamp_ms':r['timestamp_ms'],'phase_probabilities':p.tolist(),'confidence':float(p.max())} for r,p in zip(chunk,prob))
                elif task=='boundary':predictions.extend({'timestamp_ms':r['timestamp_ms'],'boundary_probability':float(p)} for r,p in zip(chunk,out['boundary'].sigmoid()[0]))
                else:
                    probabilities=out[task].softmax(-1)[0];predictions.append({'start_ms':chunk[0]['timestamp_ms'],'end_ms':chunk[-1]['timestamp_ms'],'category_probabilities':probabilities.tolist(),'confidence':float(probabilities.max())})
        return {'available':True,'metadata':meta,'predictions':predictions,'requires_review':True,'fusion':'candidate_assistance_only'}
    except Exception as exc:return {'available':False,'reason':'模型回退：'+str(exc)}

def fuse_boundaries(actions,assistance,frames):
    """模型可用时按逐帧概率重新切分，每段继承覆盖它的规则步骤属性。

    阈值、最小间隔、合并强度一律从权重自带的 metadata 读。上一版在这里写死 0.8，
    而线性模型的阈值是 0.15——模型挂上去等于没挂。
    """
    if not assistance.get('available'):return actions
    predictions=assistance.get('predictions') or []
    if not predictions or 'boundary_probability' not in predictions[0]:return actions
    from .boundary_post import boundaries_from_probability
    meta=assistance.get('metadata') or {}
    bounds=boundaries_from_probability([p['boundary_probability'] for p in predictions],
                                       [p['timestamp_ms'] for p in predictions],
                                       float(meta.get('threshold',0.5)),int(meta.get('min_gap_ms',500)),
                                       float(meta.get('valley_ratio',1.0)))
    if len(bounds)<2:return actions
    duration=int(frames[-1]['timestamp_ms']) if frames else max(a['end_ms'] for a in actions)
    edges=sorted({int(b) for b in bounds if 0<=b<=duration}|{0})
    output=[]
    for a,b in zip(edges,edges[1:]+[duration]):
        if b-a<200:continue
        ts=[f['timestamp_ms'] for f in frames if a<=f['timestamp_ms']<b]
        if not ts:continue
        source=min(actions,key=lambda x:abs(x['start_ms']-a))
        copy={**source,'start_ms':a,'end_ms':b,'keyframe_ms':min(ts,key=lambda t:abs(t-(a+b)/2)),'start_frame_ms':ts[0],'end_frame_ms':ts[-1],
              'id':f'action_{len(output)+1:04}','segmentation_method':'model_boundary_v1','model_review_required':True,
              'boundary_probability':next((p['boundary_probability'] for p in predictions if int(p['timestamp_ms'])==a),None)}
        output.append(copy)
    return output
