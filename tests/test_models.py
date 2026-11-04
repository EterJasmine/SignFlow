import pytest
torch=pytest.importorskip('torch')
from training.models import TCN,STGCN,graph_adjacency
def test_models_have_correct_outputs_and_gradients():
    for model,x,mask in [(TCN(6),torch.randn(1,32,6),torch.ones(1,32,6,dtype=torch.bool)),(STGCN(),torch.randn(1,32,75,2),torch.ones(1,32,75,dtype=torch.bool))]:
        y=model(x,mask);assert y['phase'].shape==(1,32,4);y['phase'].sum().backward()
        assert any(p.grad is not None for p in model.parameters())
def test_topology_uses_mediapipe_offsets():
    a=graph_adjacency();assert a.shape==(3,75,75);assert a[1,33,15]>0;assert a[1,54,16]>0
