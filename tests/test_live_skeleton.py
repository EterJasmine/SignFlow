"""实时预览要叠的骨骼：只带可用于绘制的点，塌缩或补间的手不该画出来。"""
import pytest

from packages.pipeline.live import skeleton


def hand(points=21, suspect=False, interpolated=False, x=0.5):
    return {'landmarks': [{'x': x, 'y': 0.5, 'z': 0.0} for _ in range(points)],
            'quality': {'suspect': suspect}, 'interpolated': interpolated}


def test_normalised_points_are_sent_without_depth():
    found = skeleton([{'hands': {'left': hand(x=0.25)}}])
    assert len(found) == 1
    assert found[0]['side'] == 'left'
    assert len(found[0]['points']) == 21
    # 预览每 100ms 请求一次，多带深度只会白白撑大响应。
    assert found[0]['points'][0] == [0.25, 0.5]
    assert found[0]['suspect'] is False


def test_suspect_hand_is_marked_but_still_drawn():
    found = skeleton([{'hands': {'right': hand(suspect=True)}}])
    assert found[0]['suspect'] is True


def test_interpolated_and_short_hands_are_skipped():
    tracks = [{'hands': {'left': hand(interpolated=True), 'right': hand(points=5)}}]
    assert skeleton(tracks) == []


def test_frame_without_hands_returns_empty_list():
    assert skeleton([{'hands': {}}]) == []
    assert skeleton([]) == []
