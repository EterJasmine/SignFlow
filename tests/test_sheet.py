from pathlib import Path
import io
from PIL import Image
from packages.pipeline.sheet import asset_file,long_image,png_bytes,pdf_bytes,sheet_html,steps_of


def build(tmp_path):
    assets=tmp_path/'assets'/'video_demo';assets.mkdir(parents=True)
    Image.new('RGB',(240,400),(120,140,170)).save(assets/'action_0001.jpg')
    Image.new('RGB',(240,400),(140,120,170)).save(assets/'action_0001_start.jpg')
    Image.new('RGB',(240,400),(170,120,140)).save(assets/'action_0001_end.jpg')
    (assets/'action_0001.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>',encoding='utf8')
    result={'video_id':'video_demo','duration_ms':8000,'width':240,'height':400,'sample_fps':8.0,
            'actions':[{'id':'action_0001','start_ms':0,'end_ms':3000,'keyframe_ms':1500,'title':'比心 <script>','description':'双手在胸前靠拢，指尖相触。','review_status':'approved',
                        'transition_frames_ms':[1000,2200],
                        'lyric_ids':['lyric_a'],'assets':{'source':'/assets/video_demo/action_0001.jpg','diagram':'/assets/video_demo/action_0001.svg',
                                                          'start_source':'/assets/video_demo/action_0001_start.jpg','end_source':'/assets/video_demo/action_0001_end.jpg'}},
                       {'id':'action_0002','start_ms':3000,'end_ms':6000,'keyframe_ms':4000,'title':'展开','description':'双臂向两侧打开。','review_status':'pending',
                        'lyric_ids':[],'assets':{'source':None,'diagram':None}}],
            'lyrics':[{'id':'lyric_a','start_ms':0,'end_ms':3000,'text':'把喜欢唱给你'}],
            'processed_frames':[{'timestamp_ms':1500,'tracks':[{'track_id':'person_0001','pose_landmarks':[{'x':.4,'y':.2,'z':0,'visibility':.9}]*33,'hands':{'left':{'landmarks':[{'x':.5,'y':.5,'z':0,'visibility':.9}]*21}}}]}],
            'tutorial_steps':[{'action_id':'action_0001','layout':'hands_closeup'}],
            'limitations':['动作边界为腕部速度峰值候选，需人工校正。'],'warnings':['采样画面中没有检测到人体骨骼。']}
    return result,{'filename':'屏幕录制 2026-11-01.mp4'},tmp_path/'assets'

def test_sheet_html_escapes_text_and_keeps_traceability(tmp_path):
    result,video,root=build(tmp_path)
    page=sheet_html(result,video,root)
    assert '比心 &lt;script&gt;' in page and '<script>' not in page
    assert 'STEP 02' in page and '把喜欢唱给你' in page
    assert '动作边界为腕部速度峰值候选' in page and '采样画面中没有检测到人体骨骼' in page
    assert 'data:image/jpeg;base64,' in page and '<svg xmlns' in page
    assert '屏幕录制 2026-11-01' in page and '.mp4' not in page.split('GESTUREFLOW · 图文教程')[1][:120]

def test_asset_path_rejects_traversal(tmp_path):
    root=tmp_path/'assets';root.mkdir()
    (root/'ok.jpg').write_bytes(b'x')
    assert asset_file('/assets/ok.jpg',root)==(root/'ok.jpg').resolve()
    assert asset_file('/assets/../../secret.jpg',root) is None
    assert asset_file('http://elsewhere/x.jpg',root) is None

def test_long_image_and_pdf_are_renderable(tmp_path):
    result,video,root=build(tmp_path)
    image=long_image(result,video,root)
    assert image.mode=='RGB' and image.width==1240 and image.height>600
    png=png_bytes(result,video,root)
    assert png.startswith(b'\x89PNG') and Image.open(io.BytesIO(png)).size==image.size
    pdf=pdf_bytes(result,video,root)
    assert pdf.startswith(b'%PDF') and len(pdf)>10000

def test_steps_map_lyrics_by_id(tmp_path):
    result,video,root=build(tmp_path)
    steps=steps_of(result)
    assert steps[0]['lyric']=='把喜欢唱给你' and steps[0]['layout']=='hands_closeup'
    assert steps[1]['lyric']=='' and steps[1]['layout']=='source_frame'

def test_step_carries_a_start_key_end_strip_and_a_timeline(tmp_path):
    """一步要能看出是"怎么动的"，所以除了代表姿态还要有起止帧与时间轴。"""
    result,video,root=build(tmp_path)
    page=sheet_html(result,video,root)
    assert page.count('<figcaption><b>起</b>')==1
    assert '<b>代表姿态</b>' in page and '<b>止</b>' in page
    assert page.count('class="strip"')==1
    assert 'class="tl"' in page and page.count('tl-dot transition')==2
    assert '过渡 1.00s · 2.20s' in page and '止 3.00s' in page

def test_strip_is_dropped_when_only_one_frame_is_available(tmp_path):
    """起止帧都取不到时宁可不出这一排，也不要在图里写"该帧未存"。"""
    result,video,root=build(tmp_path)
    assets=result['actions'][0]['assets']
    assets['start_source']=None; assets['end_source']=None
    page=sheet_html(result,video,root)
    assert 'class="strip"' not in page and '该帧未存' not in page
    assert 'class="tl"' in page

def test_hands_crop_zooms_into_the_hand_and_squares_the_box(tmp_path):
    """图解必须放大到手部，否则 300px 宽的栏位里骨架只有几十像素高。"""
    from packages.pipeline.sheet import _crop_svg
    result,video,root=build(tmp_path)
    action=result['actions'][0]
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 400"><g/></svg>'
    full=_crop_svg(svg,result,action)
    hands=_crop_svg(svg,result,action,hands_only=True)
    box=lambda text:[float(v) for v in text.split('viewBox="')[1].split('"')[0].split()]
    wide, tight=box(full), box(hands)
    assert tight[2]<wide[2] and tight[3]<wide[3]          # 比全身裁剪更紧
    assert abs(tight[2]-tight[3])<1.5                     # 方形，避免窄条
    assert 0<=tight[0] and tight[0]+tight[2]<=240.5       # 不越出画面

def test_crop_falls_back_when_no_hand_is_usable(tmp_path):
    from packages.pipeline.sheet import _crop_svg
    result,video,root=build(tmp_path)
    for hand in result['processed_frames'][0]['tracks'][0]['hands'].values():
        for point in hand['landmarks']:
            point['low_quality']=True
    action=result['actions'][0]
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 400"><g/></svg>'
    assert _crop_svg(svg,result,action,hands_only=True)==_crop_svg(svg,result,action)

def test_joints_are_dotted_once_per_endpoint(tmp_path):
    from packages.pipeline.sheet import _add_joints
    svg='<svg viewBox="0 0 10 10"><line x1="1.00" y1="2.00" x2="3.00" y2="4.00" stroke="#39a89b"/>'\
        '<line x1="3.00" y1="4.00" x2="5.00" y2="6.00" stroke="#39a89b"/></svg>'
    out=_add_joints(svg)
    assert out.count('<circle')==3                        # 共享端点只画一次
    assert out.index('<circle')<out.index('</svg>')
