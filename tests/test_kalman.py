from packages.pipeline.kalman import WristKalman
from packages.pipeline.tracking_v2 import PredictiveHands
from test_geometry_v2 import hand

def test_forecast_follows_motion_without_mutating_state():
    k=WristKalman(100,100,0)
    for t in range(100,600,100):k.update(100+t*.2,100,t)
    x,y,sigma=k.point(700)
    assert 220<x<260 and abs(y-100)<1 and sigma>0
    assert k.t==500

def test_prediction_roi_moves_and_expires():
    tracker=PredictiveHands(640,480)
    for t in range(0,500,100):tracker.assign([],[hand(.2+t*.0001)],t)
    box=tracker.predicted_region(600)
    assert box and box[0]>=0 and box[2]<=640
    assert tracker.predicted_region(1000) is None

def test_large_jump_does_not_keep_identity():
    tracker=PredictiveHands(640,480)
    a=tracker.assign([],[hand(.2)],0)[0]['track_id']
    b=tracker.assign([],[hand(.9)],100)[0]['track_id']
    assert a!=b


def test_bad_geometry_does_not_update_filter(monkeypatch):
    import packages.pipeline.tracking_v2 as module
    tracker=PredictiveHands(640,480)
    first=tracker.assign([],[hand(.2)],0)[0]['track_id']
    monkeypatch.setattr(module,'assess',lambda *args:{'suspect':True})
    tracker.assign([],[hand(.21)],100)
    assert tracker.state[first]['filter'].t==0
    assert tracker.state[first]['t']==0
