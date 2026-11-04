from copy import deepcopy
from packages.pipeline.tracking_v2 import PredictiveHands
from packages.pipeline.trajectory import process_trajectories

def hand(x=.3):
    return {'landmarks':[{'x':x+(i%5)*.01,'y':.4+(i//5)*.01,'z':0,'detected':True,'visibility':None} for i in range(21)],'identity_status':'tracked'}

def frame(t,h=None):
    return {'timestamp_ms':t,'scene_id':'s','tracks':[] if h is None else [{'track_id':'h','pose_landmarks':[],'hands':{'unknown':h}}]}

def test_one_missing_sample_at_actual_sampling_rate_is_filled():
    source=[frame(0,hand()),frame(133),frame(267,hand(.31))]
    frames,stats=process_trajectories(source,640,480)
    assert stats['interpolated_hand_frames']==1
    h=frames[1]['tracks'][0]['hands']['unknown']
    assert h['interpolated'] and not h['detected']
    assert all(p['interpolated'] and not p['detected'] for p in h['landmarks'])
    assert source[1]['tracks']==[]

def test_no_fill_for_long_or_suspect_gap():
    for end,suspect in [(800,False),(267,True)]:
        h=hand(.31);h['quality']={'suspect':suspect}
        _,stats=process_trajectories([frame(0,hand()),frame(133),frame(end,h)],640,480)
        assert stats['interpolated_hand_frames']==0

def test_identity_survives_brief_overlap_but_expires():
    tracker=PredictiveHands(640,480)
    first=tracker.assign([],[hand()],0)[0]['track_id']
    tracker.assign([],[],267)
    assert tracker.assign([],[hand(.31)],400)[0]['track_id']==first
    assert tracker.assign([],[hand(.31)],1200)[0]['track_id']!=first
