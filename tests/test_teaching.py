from packages.pipeline.teaching import enrich_action,trajectory_svg

def frame(t,x):
    p=[{'x':x,'y':.5,'z':0,'visibility':1} for _ in range(21)]
    return {'timestamp_ms':t,'tracks':[{'track_id':'hand_1','hands':{'unknown':{'landmarks':p}},'pose_landmarks':[]}]}
def test_description_and_arrow_use_measured_track():
    a={'start_ms':0,'end_ms':1000,'keyframe_ms':500}
    enrich_action(a,[frame(0,.2),frame(500,.3),frame(900,.4)])
    assert a['description_source']=='landmark_rules_v1'
    assert '画面右' in a['description']
    assert len(a['trajectories'])==1
    assert 'marker-end' in trajectory_svg(a['trajectories'],640,480)
def test_missing_detection_splits_path():
    a={'start_ms':0,'end_ms':1000,'keyframe_ms':0}
    enrich_action(a,[frame(0,.2),{'timestamp_ms':500,'tracks':[]},frame(900,.4)])
    assert a['trajectories']==[]
