import pytest
from packages.pipeline.scene import evidence,summarize_scene
from packages.pipeline.motion import lyric_overlap,motion_steps
from apps.api.main import ActionPatch,Lyric

def pose():return [{'x':.5,'y':.5,'visibility':.95} for _ in range(33)]
def test_framing_and_count_are_independent():
    p=pose();assert evidence([p,p],[])=={'framing_type':'full_body','person_count':2,'visible_parts':['ankles','face','hips','knees','shoulders']}
    p[27]['visibility']=.1;assert evidence([p],[])['framing_type']=='upper_body'
    assert evidence([],[[{}]])['framing_type']=='hands_only'
    assert evidence([],[])['framing_type']=='unknown'
def test_scene_uncertainty_and_no_single_frame_flicker():
    fs=[{'evidence':evidence([pose()],[])}]*9+[{'evidence':evidence([],[])}]
    s=summarize_scene(fs,0,1000,'s');assert s['framing_type']=='full_body';assert s['confidence']==.9
def test_half_open_intervals_and_multiple_lyrics():
    lyrics=[{'id':'a','start_ms':0,'end_ms':1000},{'id':'b','start_ms':1000,'end_ms':2000}]
    assert lyric_overlap({'start_ms':1000,'end_ms':2000},lyrics)==[{'lyric_id':'b','overlap_ms':1000}]
    assert len(lyric_overlap({'start_ms':500,'end_ms':1500},lyrics))==2
def test_no_motion_fabrication_for_missing_tracks():
    frames=[{'timestamp_ms':0,'scene_id':'s','tracks':[]},{'timestamp_ms':1000,'scene_id':'s','tracks':[]}]
    steps=motion_steps(frames,[{'start_ms':0}],2000)
    assert len(steps)==1;assert steps[0]['start_ms']==0;assert steps[0]['end_ms']==2000
def test_edit_time_validation():
    with pytest.raises(ValueError):ActionPatch(start_ms=1000,end_ms=1000,keyframe_ms=1000,title='x',description='x')
    with pytest.raises(ValueError):Lyric(start_ms=1000,end_ms=500,text='x')
