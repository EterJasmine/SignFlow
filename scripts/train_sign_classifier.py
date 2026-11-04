"""Train a small fixed-length temporal classifier; split by source video."""
import sys,json,hashlib,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apps.api.store import all_of
from packages.pipeline.signmatch import usable

def vector(seq):
    a=np.array([[np.nan if v is None else v for v in row] for row in seq['rows']],dtype=np.float32)
    x=np.linspace(0,1,len(a));target=np.linspace(0,1,24);out=[];masks=[]
    for col in a.T:
        good=np.isfinite(col)
        out.append(np.interp(target,x[good],col[good]) if good.any() else np.zeros(24))
        masks.append(np.interp(target,x,good.astype(float)))
    return np.concatenate([np.array(out).T.ravel(),np.array(masks).T.ravel()]).astype(np.float32)

def main():
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    items=[t for t in all_of('sign_template')+all_of('sign_training_sample') if usable(t)]
    signatures=Counter((t['sequence']['version'],tuple(t['sequence']['names'])) for t in items)
    signature=signatures.most_common(1)[0][0]
    items=[t for t in items if (t['sequence']['version'],tuple(t['sequence']['names']))==signature]
    # Exact duplicate sequences are not independent observations.
    unique={}
    for t in items:
        digest=hashlib.sha256(json.dumps(t['sequence'],sort_keys=True).encode()).hexdigest()
        key=(digest,t['meaning'])
        if key not in unique:unique[key]=t
    items=list(unique.values())
    sources={word:{t['video_id'] for t in items if t['meaning']==word} for word in {t['meaning'] for t in items}}
    eligible={word for word,videos in sources.items() if len(videos)>=2}
    items=[t for t in items if t['meaning'] in eligible]
    if len(eligible)<2:raise RuntimeError('至少需要两个有跨视频示范的词')
    classes=sorted(eligible);counts=Counter(t['meaning'] for t in items)
    videos=sorted({t['video_id'] for t in items});choices=[]
    if len(videos)>16:raise RuntimeError('来源太多，请改用分组交叉验证')
    for bits in range(1,2**len(videos)-1):
        held={v for i,v in enumerate(videos) if bits & (1<<i)}
        validation=[t for t in items if t['video_id'] in held]
        training=[t for t in items if t['video_id'] not in held]
        if {t['meaning'] for t in training}!=eligible:continue
        coverage=len({t['meaning'] for t in validation})
        choices.append((coverage,-abs(len(validation)/len(items)-.3),tuple(sorted(held))))
    if not choices:raise RuntimeError('没有保持训练类别完整的按视频划分')
    held=set(max(choices)[2]);test_video=sorted(held)
    train=[t for t in items if t['video_id'] not in held];val=[t for t in items if t['video_id'] in held]
    raw=np.stack([vector(t['sequence']) for t in train]);mean=raw.mean(0);scale=np.maximum(raw.std(0),.1)
    x=torch.tensor((raw-mean)/scale);y=torch.tensor([classes.index(t['meaning']) for t in train])
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,len(classes)))
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    weights=torch.tensor([len(train)/max(1,sum(t['meaning']==c for t in train)) for c in classes]);weights/=weights.mean()
    for epoch in range(250):
        model.train();optim.zero_grad();loss=nn.functional.cross_entropy(model(x),y,weight=weights);loss.backward();optim.step()
    model.eval()
    with torch.inference_mode():
        train_acc=float((model(x).argmax(-1)==y).float().mean())
        details=[]
        for t in val:
            prob=model(torch.tensor(((vector(t['sequence'])-mean)/scale)[None])).softmax(-1)[0]
            details.append({'id':t['id'],'expected':t['meaning'],'predicted':classes[int(prob.argmax())],'max_probability':float(prob.max())})
    folder=ROOT/'training/runs'/('sign-'+time.strftime('%Y%m%d-%H%M%S'));folder.mkdir(parents=True)
    accuracy=sum(d['expected']==d['predicted'] for d in details)/len(details) if details else None
    report={'samples':len(items),'class_counts':dict(counts),'train_samples':len(train),'validation_samples':len(val),'heldout_video':test_video,'training_accuracy':train_acc,'validation_accuracy':accuracy,'validation_details':details,'epochs':250,'architecture':'resample24_mask_mlp64','deployed':False,'limitations':['Small user-labeled dataset; probabilities are uncalibrated.','Validation is one held-out video, not signer-independent.','Single-example classes cannot establish generalization.','Checkpoint is experimental and does not replace live DTW.']}
    torch.save({'state_dict':model.state_dict(),'mean':torch.tensor(mean),'scale':torch.tensor(scale),'metadata':{'classes':classes,'feature_version':signature[0],'feature_names':list(signature[1]),'steps':24,'width':64,'input_dim':x.shape[1],'architecture':report['architecture']}},folder/'classifier.pt')
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'folder':str(folder),'samples':len(items),'classes':len(classes),'train':len(train),'validation':len(val),'training_accuracy':train_acc,'validation_accuracy':accuracy},ensure_ascii=True))
if __name__=='__main__':main()
