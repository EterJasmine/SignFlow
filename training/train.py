"""Explicit CLI only. Refuses unlabeled or ungrouped data; never auto-trains."""
import argparse,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .models import build_model
from evaluation.metrics import macro_f1,confusion_matrix,edit_score

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--model',choices=['tcn','stgcn'],default='tcn');p.add_argument('--task',choices=['phase','category','static','boundary'],default='phase');p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--patience',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda');p.add_argument('--output',default='training/runs');args=p.parse_args()
    path=Path(args.dataset);manifest=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
    if not manifest.get('training_ready'):raise SystemExit('数据未完成分组划分，禁止训练。需要至少三个独立来源组。')
    data=np.load(path,allow_pickle=False);feature_dim=data['x'].shape[-1];target=data[args.task]
    if args.task!='boundary' and not np.any(target>=0):raise SystemExit('所选任务没有人工标签，禁止训练。')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device(args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    model=build_model(args.model,feature_dim).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    xkey,mkey=('x','mask') if args.model=='tcn' else ('graph','graph_mask')
    def loader(split,shuffle=False):
        indices=np.where(data['split']==split)[0]
        if not len(indices):raise ValueError('训练/验证/测试分组不可为空')
        if args.task!='boundary' and not np.any(target[indices]>=0):raise ValueError(f'{split} 没有当前任务标签')
        return DataLoader(TensorDataset(torch.tensor(data[xkey][indices]),torch.tensor(data[mkey][indices]),torch.tensor(target[indices]),torch.tensor(data['boundary_mask'][indices])),batch_size=args.batch_size,shuffle=shuffle,num_workers=0)
    train,validation,test=loader('train',True),loader('validation'),loader('test')
    run=Path(args.output)/(time.strftime('%Y%m%d-%H%M%S')+'-'+args.model);run.mkdir(parents=True,exist_ok=False)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');best=float('inf');bad=0;history=[]
    def epoch(batches,training):
        model.train(training);total=0.;count=0;truth=[];pred=[]
        for x,m,y,bmask in batches:
            x,m,y,bmask=x.to(device),m.to(device),y.to(device),bmask.to(device)
            with torch.set_grad_enabled(training),torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
                output=model(x,m)[args.task]
                if args.task=='boundary':
                    keep=bmask.bool() & m.reshape(m.shape[0],m.shape[1],-1).any(-1);loss=nn.functional.binary_cross_entropy_with_logits(output[keep],y[keep].float()) if keep.any() else None
                else:
                    keep=y>=0
                    if args.task=='phase':keep=keep & m.reshape(m.shape[0],m.shape[1],-1).any(-1)
                    loss=nn.functional.cross_entropy(output[keep],y[keep].long()) if keep.any() else None
                if loss is None:continue
            if training:optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
            total+=float(loss.detach());count+=1
            if args.task!='boundary':truth.extend(y[keep].cpu().tolist());pred.extend(output[keep].argmax(-1).detach().cpu().tolist())
        if count==0:raise ValueError('当前任务没有同时有效的人工标签和输入，禁止保存训练权重')
        return total/count,truth,pred
    metadata={'architecture':args.model,'task':args.task,'input_dim':feature_dim,'feature_version':manifest['feature_version'],'feature_names':manifest['feature_names'],'topology':manifest['graph_topology'],'schema_version':'model-1.0','seed':args.seed,'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'trained_on_human_labels':True,'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,'device':str(device)}
    for i in range(args.epochs):
        train_loss,_,_=epoch(train,True);val_loss,_,_=epoch(validation,False);history.append({'epoch':i+1,'train_loss':train_loss,'val_loss':val_loss});print(history[-1],flush=True)
        (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
        if val_loss<best:
            best=val_loss;bad=0;torch.save({'state_dict':model.state_dict(),'metadata':metadata},run/'best.pt')
        else:bad+=1
        if bad>=args.patience:break
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True)['state_dict']);test_loss,truth,pred=epoch(test,False)
    classes=4 if args.task=='phase' else 9
    metrics={'test_loss':test_loss,'macro_f1':macro_f1(truth,pred,classes) if truth else None,'confusion_matrix':confusion_matrix(truth,pred,classes).tolist() if truth else None,'edit_score':None,'boundary_f1':None,
             'note':'时序Edit/Boundary指标需按原视频恢复时间轴后用evaluation模块计算，不能将不同视频拼接。','peak_cuda_memory_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,'metadata':metadata}
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8');print(metrics)
if __name__=='__main__':main()
