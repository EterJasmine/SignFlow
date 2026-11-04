from packages.pipeline.features import extract_features,NAMES
from packages.pipeline.segmentation import segment_actions
def empty(t):return {'timestamp_ms':t,'scene_id':'s','tracks':[]}
def test_missing_values_have_explicit_masks():
    f=extract_features([empty(0),empty(100)],640,480)
    assert len(f['rows'][0]['values'])==len(NAMES)
    assert not f['rows'][0]['mask'][0];assert f['rows'][0]['values'][0] is None
def test_finger_change_without_wrist_motion_is_detected():
    frames=[empty(i*100) for i in range(30)]
    names=['hand_0_finger_0_pip_angle','hand_0_speed','valid_ratio']
    rows=[]
    for i in range(30):
        angle=0.2 if i<12 else .9
        rows.append({'timestamp_ms':i*100,'scene_id':'s','values':[angle,0,1],'mask':[True]*3})
    actions,phases,_=segment_actions(frames,[{'start_ms':0}],3000,{'names':names,'rows':rows})
    assert any(p['phase']=='transition' for p in phases);assert len(actions)>=2
def test_continuous_movement_not_cut_at_speed_peak():
    frames=[empty(i*100) for i in range(40)];rows=[]
    for i in range(40):rows.append({'timestamp_ms':i*100,'scene_id':'s','values':[i*.1,1,1],'mask':[True]*3})
    a,_,_=segment_actions(frames,[{'start_ms':0}],4000,{'names':['rotation','hand_0_speed','valid_ratio'],'rows':rows})
    assert len(a)==1


def test_suspect_hand_is_masked_and_breaks_velocity_history():
    import copy
    hand={'landmarks':[{'x':.2+i*.01,'y':.3+i*.005,'z':0,'visibility':1} for i in range(21)],'identity_status':'known'}
    frames=[]
    for i in range(3):
        h=copy.deepcopy(hand)
        if i==1:h['quality']={'suspect':True}
        frames.append({'timestamp_ms':i*100,'scene_id':'s','tracks':[{'track_id':'p','pose_landmarks':[],'hands':{'left':h}}]})
    rows=extract_features(frames,640,480)['rows']
    assert not any(rows[1]['mask'][:35])
    assert not rows[2]['mask'][NAMES.index('hand_0_speed')]
