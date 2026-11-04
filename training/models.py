"""Small offline TCN and ST-GCN-style models with MediaPipe-specific topology."""
import torch
from torch import nn

POSE_EDGES=[(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(3,7),(6,8),(9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(12,14),(14,16),(16,18),(16,20),(16,22),(11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]
HAND_EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
def graph_adjacency():
    inward=torch.zeros(75,75)
    edges=POSE_EDGES+[(a+offset,b+offset) for offset in [33,54] for a,b in HAND_EDGES]+[(15,33),(16,54)]
    for a,b in edges:inward[b,a]=1
    outward=inward.T.clone();identity=torch.eye(75)
    matrices=[]
    for a in [identity,inward,outward]:matrices.append(a/a.sum(0,keepdim=True).clamp_min(1))
    return torch.stack(matrices)

class Heads(nn.Module):
    def __init__(self,channels,categories=9):
        super().__init__();self.phase=nn.Conv1d(channels,4,1);self.boundary=nn.Conv1d(channels,1,1);self.category=nn.Linear(channels,categories);self.static=nn.Linear(channels,9)
    def forward(self,x,time_mask):
        pooled=(x*time_mask[:,None,:]).sum(-1)/time_mask.sum(-1,keepdim=True).clamp_min(1)
        return {'phase':self.phase(x).transpose(1,2),'boundary':self.boundary(x).squeeze(1),'category':self.category(pooled),'static':self.static(pooled)}
class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),nn.GroupNorm(8,channels),nn.GELU(),nn.Dropout(.15),nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),nn.GELU())
    def forward(self,x):return self.net(x)+x
class TCN(nn.Module):
    def __init__(self,input_dim,categories=9,width=64):
        super().__init__();self.input=nn.Conv1d(input_dim*2,width,1);self.blocks=nn.Sequential(*[TemporalBlock(width,d) for d in [1,2,4,8]]);self.heads=Heads(width,categories)
        # 特征标准化作为 buffer 存进权重，推理时不必外挂统计量；默认恒等，不影响既有行为。
        self.register_buffer('feat_mean',torch.zeros(input_dim));self.register_buffer('feat_std',torch.ones(input_dim))
    def forward(self,x,mask):
        x=((x-self.feat_mean)/self.feat_std)*mask
        h=torch.cat([x*mask,mask.to(x.dtype)],-1).transpose(1,2);h=self.blocks(self.input(h));return self.heads(h,mask.any(-1).to(x.dtype))
class GraphBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__();self.spatial=nn.Conv2d(cin,cout*3,1);self.temporal=nn.Sequential(nn.GroupNorm(8,cout),nn.GELU(),nn.Conv2d(cout,cout,(3,1),padding=(1,0)),nn.Dropout(.15));self.residual=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,a):
        b,_,t,v=x.shape;z=self.spatial(x).reshape(b,3,-1,t,v);z=torch.einsum('bkctv,kvw->bctw',z,a);return self.temporal(z)+self.residual(x)
class STGCN(nn.Module):
    def __init__(self,input_dim=0,categories=9):
        super().__init__();self.register_buffer('adjacency',graph_adjacency());self.layers=nn.ModuleList([GraphBlock(3,32),GraphBlock(32,64),GraphBlock(64,64)]);self.heads=Heads(64,categories)
    def forward(self,x,mask):
        # x B,T,75,2; mask B,T,75. Mask also enters as a channel.
        z=torch.cat([x*mask[...,None],mask[...,None].to(x.dtype)],-1).permute(0,3,1,2)
        for layer in self.layers:z=layer(z,self.adjacency)
        pooled=(z*mask[:,None,:,:]).sum(-1)/mask.sum(-1)[:,None,:].clamp_min(1)
        return self.heads(pooled,mask.any(-1).to(x.dtype))
def build_model(name,input_dim,width=64):
    # width 只对 TCN 生效；样本量很小时（如 3 条视频）用 width=32 压参数量。
    if name=='tcn':return TCN(input_dim,width=width)
    if name=='stgcn':return STGCN(input_dim)
    if name=='linear_change':
        from .features_diff import LinearBoundary
        return LinearBoundary(input_dim)
    raise ValueError('Unknown model')
