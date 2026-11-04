import pytest
from packages.pipeline.composer import TeachingText,diagram_viewbox
from packages.pipeline.providers import DeepSeekProvider
def test_ai_text_is_validated():
    with pytest.raises(ValueError):TeachingText(title='',description='x')
def test_crop_keeps_source_coordinates():
    svg='<svg viewBox="0 0 640 480"></svg>';frame={'tracks':[{'pose_landmarks':[],'hands':{'unknown':{'landmarks':[{'x':.5,'y':.5}]}}}]}
    out,meta=diagram_viewbox(svg,frame,'hands_only',640,480)
    assert meta['crop_only_no_pose_change'];assert 'viewBox="0 0 640 480"' not in out
def test_provider_does_not_claim_user_vision_access():
    assert DeepSeekProvider().capabilities()['vision_access']=='not_verified_for_user_account'
