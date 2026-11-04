import math
from packages.pipeline.geometry import angle,observed_point
from packages.pipeline.tracking_v2 import PredictiveHands
from packages.pipeline.trajectory import OneEuro
from types import SimpleNamespace

def test_pixel_angle_is_aspect_invariant():
    pts=[{'x':0,'y':0},{'x':100,'y':100},{'x':200,'y':0}]
    for w,h in [(400,800),(800,400)]:
        a,b,c=[{'x':p['x']/w,'y':p['y']/h} for p in pts]
        assert abs(angle(a,b,c,w,h)-90)<1e-6
def test_unknown_confidence_is_not_one():
    p=observed_point(SimpleNamespace(x=.1,y=.2,z=.3))
    assert p['visibility'] is None and p['confidence'] is None and p['detected']
def hand(x):
    p=[{'x':x+(i%5)*.01,'y':.5+(i//5)*.01,'z':0,'visibility':None} for i in range(21)]
    return {'landmarks':p,'model_handedness':'Left','handedness_score':.9}
def test_predictive_hands_short_occlusion_and_order():
    t=PredictiveHands(640,480);a=t.assign([],[hand(.2),hand(.7)],0)
    t.assign([],[],100);b=t.assign([],[hand(.69),hand(.21)],200)
    assert b[0]['track_id']==a[1]['track_id'];assert b[1]['track_id']==a[0]['track_id']
    assert b[0]['hands']['unknown']['identity_continuous']
    t.reset();assert not t.assign([],[hand(.2)],300)[0]['hands']['unknown']['identity_continuous']
def test_filter_reduces_jitter_and_preserves_fast_change():
    f=OneEuro();raw=[100+(-1)**i for i in range(30)];smoothed=[f.filter(v,i*100) for i,v in enumerate(raw)]
    assert sum(abs(smoothed[i]-smoothed[i-1]) for i in range(1,30))<sum(abs(raw[i]-raw[i-1]) for i in range(1,30))
    jump=f.filter(300,3000);assert jump>260
