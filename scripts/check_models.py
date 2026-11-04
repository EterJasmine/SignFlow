"""Random tensors test code/VRAM only, not training or recognition accuracy."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training.models import TCN,STGCN,graph_adjacency
from packages.pipeline.features import NAMES
device='cuda' if torch.cuda.is_available() else 'cpu';results=[]
for name,model,x,mask in [('tcn',TCN(len(NAMES)),torch.randn(2,64,len(NAMES)),torch.ones(2,64,len(NAMES),dtype=torch.bool)),('stgcn',STGCN(),torch.randn(2,64,75,2),torch.ones(2,64,75,dtype=torch.bool))]:
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    model=model.to(device);x=x.to(device);mask=mask.to(device)
    start=time.perf_counter()
    out=model(x,mask);loss=sum(v.float().square().mean() for v in out.values());loss.backward()
    assert out['phase'].shape==(2,64,4)
    if device=='cuda':torch.cuda.synchronize()
    results.append({'model':name,'device':device,'parameters':sum(p.numel() for p in model.parameters()),'forward_backward_seconds':time.perf_counter()-start,'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,'purpose':'random_tensor_code_check_not_training'})
    del model,x,mask,out
    if device=='cuda':torch.cuda.empty_cache()
print(json.dumps(results,indent=2));out=Path('data/model-code-check.json');out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(results,indent=2))
