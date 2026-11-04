import pytest
from fastapi import HTTPException
from apps.api import sign_routes as routes
from apps.api import store

def test_delete_removes_only_the_matching_kind(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path)
    store.put('x','sign_template',{'id':'x'})
    store.put('y','video',{'id':'y'})
    assert store.delete('x','video') is False
    assert store.get('x','sign_template') is not None
    assert store.delete('x','sign_template') is True
    assert store.get('x') is None

def test_removing_one_word_keeps_the_shared_video(monkeypatch):
    records={'s1':{'id':'s1','video_id':'v','meaning':'谢谢'},'s2':{'id':'s2','video_id':'v','meaning':'再见'}}
    monkeypatch.setattr(routes,'get',lambda i,k=None: records.get(i))
    monkeypatch.setattr(routes,'all_of',lambda k: list(records.values()))
    monkeypatch.setattr(routes,'delete',lambda i,k=None: bool(records.pop(i,None)))
    result=routes.remove('s1')
    assert result=={'deleted':'s1','video_id':'v','orphan_video':False}
    assert 's2' in records

def test_removing_the_last_word_reports_orphan_video(monkeypatch):
    records={'s1':{'id':'s1','video_id':'v','meaning':'谢谢'}}
    monkeypatch.setattr(routes,'get',lambda i,k=None: records.get(i))
    monkeypatch.setattr(routes,'all_of',lambda k: list(records.values()))
    monkeypatch.setattr(routes,'delete',lambda i,k=None: bool(records.pop(i,None)))
    assert routes.remove('s1')['orphan_video'] is True

def test_removing_unknown_template_is_404(monkeypatch):
    monkeypatch.setattr(routes,'get',lambda i,k=None: None)
    with pytest.raises(HTTPException) as e:routes.remove('missing')
    assert e.value.status_code==404

def test_candidates_flag_ranges_already_labelled(monkeypatch):
    monkeypatch.setattr(routes,'get',lambda i,k=None:{'id':'v'})
    monkeypatch.setattr(routes,'load_result',lambda i:{'duration_ms':5000,'actions':[
        {'start_ms':0,'end_ms':1000,'title':'第一段'},{'start_ms':2000,'end_ms':3000,'title':'第二段'}]})
    monkeypatch.setattr(routes,'all_of',lambda k:[{'video_id':'v','start_ms':100,'end_ms':900}])
    data=routes.candidates('v')
    assert [a['taken'] for a in data['actions']]==[True,False]
    assert data['actions'][0]['label']=='第一段'

def test_unusable_range_reports_actual_palm_size():
    from packages.pipeline.signmatch import sequence
    result={'features':{'names':['finger_0_mcp_angle'],'rows':[
                {'timestamp_ms':t,'values':[1.0],'mask':[False]} for t in range(0,1000,100)]},
            'frames':[{'timestamp_ms':t,'tracks':[{'hands':{'Right':{'quality':{'geometry':{'palm_px':33.4}}}}}]}
                      for t in range(0,1000,100)]}
    with pytest.raises(ValueError) as e:sequence(result,0,1000)
    assert '33' in str(e.value) and '像素' in str(e.value)

def test_candidates_mark_ranges_that_cannot_be_extracted(monkeypatch):
    # 用真实的单手列数（25 个指特征）：列数太少时模型本来就读不出手型。
    names=[f'finger_{i}_mcp_angle' for i in range(5)]+[f'finger_{i}_angle' for i in range(20)]
    rows=[{'timestamp_ms':t,'values':[1.0]*25,'mask':[True]*25} for t in range(0,1000,100)]
    result={'duration_ms':4000,'features':{'version':'v','names':names,'rows':rows},
            'frames':[],'coverage':{'hand_quality':{'suspect_rate':.25}},'scenes':[{'framing_type':'hands_only'}],
            'actions':[{'start_ms':0,'end_ms':1000,'title':'可提取'},{'start_ms':2000,'end_ms':3000,'title':'空段'}]}
    monkeypatch.setattr(routes,'get',lambda i,k=None:{'id':'v'})
    monkeypatch.setattr(routes,'load_result',lambda i:result)
    monkeypatch.setattr(routes,'all_of',lambda k:[])
    data=routes.candidates('v')
    assert data['usable_count']==1
    assert [a['usable'] for a in data['actions']]==[True,False]
    assert '太短' in data['actions'][1]['reason']

    monkeypatch.setattr(routes,'get',lambda i,k=None:{'id':'v'})
    monkeypatch.setattr(routes,'load_result',lambda i:{'duration_ms':5000})
    monkeypatch.setattr(routes,'sequence',lambda r,s,e:{'rows':[[1.0]]})
    # 检索现在放在 signmatch.retrieve 里，距离函数要从那里打桩。
    from packages.pipeline import signmatch
    monkeypatch.setattr(signmatch,'dtw',lambda a,b:.01)
    monkeypatch.setattr(routes,'all_of',lambda k:[
        {'id':'t1','video_id':'v','start_ms':0,'end_ms':1000,'meaning':'谢谢','recognition_ready':True,'sequence':{'rows':[[1.0]]}},
        {'id':'t2','video_id':'v','start_ms':2000,'end_ms':3000,'meaning':'再见','recognition_ready':True,'sequence':{'rows':[[1.0]]}}])
    result=routes.match(routes.Selection(video_id='v',start_ms=0,end_ms=1000))
    assert result['skipped']==1
    assert result['meaning']=='再见'
    assert [c['meaning'] for c in result['candidates']]==['再见']

def test_frames_requires_an_existing_template(monkeypatch):
    monkeypatch.setattr(routes,'get',lambda i,k=None:None)
    with pytest.raises(HTTPException) as e:routes.frames('missing')
    assert e.value.status_code==404

def test_frames_explains_a_missing_source_video(monkeypatch):
    monkeypatch.setattr(routes,'get',lambda i,k=None:{'id':'s','video_id':'v','start_ms':0,'end_ms':500})
    monkeypatch.setattr(routes,'video_file',lambda i,r:None)
    with pytest.raises(HTTPException) as e:routes.frames('s')
    assert e.value.status_code==410 and '源视频' in e.value.detail

def test_frames_returns_png_and_clamps_its_size(monkeypatch):
    seen={}
    monkeypatch.setattr(routes,'get',lambda i,k=None:{'id':'s','video_id':'v','start_ms':100,'end_ms':600})
    monkeypatch.setattr(routes,'video_file',lambda i,r:object())
    monkeypatch.setattr(routes,'load_result',lambda i:{'video_id':'v'})
    monkeypatch.setattr(routes,'preview_strip',lambda r,a,b,root,count,height:(seen.update(count=count,height=height,start=a,end=b) or b'png'))
    response=routes.frames('s',count=1,height=9999)
    assert response.body==b'png' and response.media_type=='image/png'
    assert seen=={'count':1,'height':400,'start':100,'end':600}

def test_preview_strip_stitches_one_image_per_sample(tmp_path,monkeypatch):
    from PIL import Image
    import io
    from packages.pipeline import sheet
    monkeypatch.setattr(sheet,'_sample_video',lambda r,t,a:Image.new('RGB',(320,180),(30,30,30)))
    strip=Image.open(io.BytesIO(sheet.preview_strip({'video_id':'v'},0,1000,tmp_path,count=3,height=180)))
    assert strip.width>strip.height*2
    assert strip.width<strip.height*5

def test_preview_strip_reports_when_no_frame_can_be_read(tmp_path,monkeypatch):
    from packages.pipeline import sheet
    monkeypatch.setattr(sheet,'_sample_video',lambda *a,**k:None)
    with pytest.raises(ValueError) as e:sheet.preview_strip({'video_id':'v'},0,500,tmp_path)
    assert '原视频画面' in str(e.value)
