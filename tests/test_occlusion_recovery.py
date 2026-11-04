from packages.pipeline.occlusion import map_crop,OcclusionRecovery

def test_crop_mapping_preserves_original_and_coordinates():
    source=[{'x':.5,'y':.25,'z':.1}]
    out=map_crop(source,(100,50,300,250),1000,500)
    assert out[0]['x']==.2 and out[0]['y']==.2
    assert out[0]['raw_coordinates']['x']==.2
    assert source[0]['x']==.5

def test_recovery_roi_expires_and_resets():
    recovery=OcclusionRecovery()
    recovery.recent=[{'landmarks':[{'x':.5,'y':.5}]}];recovery.last_seen=100
    box=recovery.crop_box(200,640,480)
    assert box and 0<=box[0]<box[2]<=640
    assert recovery.crop_box(800,640,480) is None
    recovery.reset();assert recovery.crop_box(200,640,480) is None


def test_good_roi_replaces_bad_original(monkeypatch):
    import packages.pipeline.occlusion as module
    def sample(bad):
        return {'landmarks':[{'x':.3+(i%5)*.01,'y':.4+(i//5)*.01,'z':0,'bad':bad} for i in range(21)]}
    monkeypatch.setattr(module,'assess',lambda p,w,h:{'suspect':p[0]['bad']})
    original=sample(True);candidate=sample(False)
    output,count=module.merge_recovered([original],[candidate],640,480)
    assert len(output)==1 and output[0] is candidate and count==1
    assert original['landmarks'][0]['bad']


def test_regions_cover_full_frame_and_stay_in_bounds():
    from packages.pipeline.occlusion import search_regions
    boxes=search_regions(640,480)
    assert all(0<=x<r<=640 and 0<=y<b<=480 for x,y,r,b in boxes)
    assert min(y for x,y,r,b in boxes)==0
    assert max(b for x,y,r,b in boxes)==480
