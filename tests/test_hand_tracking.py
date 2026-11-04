from packages.pipeline.tracking import HandAssociator

def hand(x,y=.5):return {'landmarks':[{'x':x,'y':y,'z':0,'visibility':1} for _ in range(21)]}
def body():
    pose=[{'x':.5,'y':.5,'visibility':.9} for _ in range(33)]
    pose[15]['x']=.3;pose[16]['x']=.7
    return {'track_id':'p1','pose_landmarks':pose,'face_landmarks':[],'hands':{}}
def test_two_hands_without_body_regression():
    tracks=HandAssociator().assign([], [hand(.2),hand(.8)])
    assert len(tracks)==2;assert sum(len(t['hands']) for t in tracks)==2
    assert all(t['pose_landmarks']==[] for t in tracks)
def test_unmatched_first_hand_does_not_break_second_hand():
    tracks=HandAssociator().assign([body()], [hand(.02),hand(.7)])
    assert len(tracks)==2;assert 'right' in tracks[0]['hands']
def test_partial_body_record_is_safe():
    tracks=HandAssociator().assign([{'track_id':'partial','pose_landmarks':[],'hands':{}}],[hand(.2),hand(.8)])
    assert len(tracks)==3
def test_identity_survives_detection_order_change():
    a=HandAssociator();first=a.assign([],[hand(.2),hand(.8)])
    second=a.assign([],[hand(.79),hand(.21)])
    assert second[0]['track_id']==first[1]['track_id']
    assert second[1]['track_id']==first[0]['track_id']
def test_gap_and_scene_reset_dont_create_false_continuity():
    a=HandAssociator();first=a.assign([],[hand(.2)])[0]['track_id'];a.assign([],[])
    assert a.assign([],[hand(.2)])[0]['track_id']!=first
    a.reset();assert not a.assign([],[hand(.2)])[0]['hands']['unknown']['identity_continuous']
