"""检索规则的边界：只认就绪的示范、坏帧不参与对齐、拒绝时要说清是哪一种拒绝。"""
import pytest

from packages.pipeline.features import slot_key
from packages.pipeline.signmatch import MIN_COLUMNS, retrieve, sequence, usable


def seq(values, version='v', names=None):
    names = names or ['a_angle'] * len(values)
    return {'version': version, 'names': names, 'rows': [values]}


def item(label, side, track):
    return ({'track_id': track}, side, {'model_handedness': label})


def test_slot_follows_handedness_not_detection_order():
    # 先检测到的是右手，但槽位必须仍然是「左手在前」，否则示范与实时会左右对调。
    left = slot_key(item('Left', 'unknown', 'hand_v2_0009'))
    right = slot_key(item('Right', 'unknown', 'hand_v2_0001'))
    assert left < right


def test_slot_falls_back_to_anatomy_then_track():
    assert slot_key(item(None, 'left', 'hand_v2_0009')) < slot_key(item(None, 'right', 'hand_v2_0001'))
    assert slot_key(item(None, 'unknown', 'hand_v2_0001')) < slot_key(item(None, 'unknown', 'hand_v2_0002'))


def test_draft_templates_never_participate():
    assert usable({'recognition_ready': True, 'sequence': {'rows': [[1.0]]}}) is True
    assert usable({'recognition_ready': False, 'sequence': {'rows': [[1.0]]}}) is False
    assert usable({'recognition_ready': True, 'sequence': {}}) is False


def features(rows):
    names = ['finger_0_angle'] * MIN_COLUMNS + ['hand_distance']
    return {'version': 'v', 'names': names, 'rows': rows}


def test_frames_without_a_readable_hand_shape_are_dropped():
    good = {'timestamp_ms': 0, 'values': [0.1] * MIN_COLUMNS + [0.2], 'mask': [True] * (MIN_COLUMNS + 1)}
    bad = {'timestamp_ms': 10, 'values': [0.1] * MIN_COLUMNS + [0.2],
           'mask': [True] * 3 + [False] * (MIN_COLUMNS - 2)}
    rows = [dict(good, timestamp_ms=i * 10) for i in range(4)] + [dict(bad, timestamp_ms=100)]
    result = sequence({'features': features(rows), 'frames': rows}, 0, 200)
    assert len(result['rows']) == 4


def test_range_with_no_usable_frame_explains_why():
    bad = {'timestamp_ms': 0, 'values': [0.1] * MIN_COLUMNS + [0.2], 'mask': [True] + [False] * MIN_COLUMNS}
    rows = [dict(bad, timestamp_ms=i * 10) for i in range(6)]
    with pytest.raises(ValueError, match='手型'):
        sequence({'features': features(rows), 'frames': rows}, 0, 100)


def test_clear_best_is_accepted():
    templates = [{'id': 'a', 'meaning': '谢谢', 'recognition_ready': True, 'sequence': seq([0.0] * 5)},
                 {'id': 'b', 'meaning': '再见', 'recognition_ready': True, 'sequence': seq([0.05] * 5)}]
    found = retrieve(seq([0.0] * 5), templates)
    assert found['matched'] is True
    assert found['meaning'] == '谢谢'
    assert found['reason'] is None


def test_two_similar_words_are_not_guessed():
    templates = [{'id': 'a', 'meaning': '出现', 'recognition_ready': True, 'sequence': seq([0.0] * 5)},
                 {'id': 'b', 'meaning': '顺利', 'recognition_ready': True, 'sequence': seq([0.02] * 5)}]
    found = retrieve(seq([0.01] * 5), templates)
    assert found['matched'] is False
    assert '太接近' in found['reason']


def test_far_away_query_says_the_template_is_missing():
    templates = [{'id': 'a', 'meaning': '谢谢', 'recognition_ready': True, 'sequence': seq([0.0] * 5)}]
    found = retrieve(seq([0.9] * 5), templates)
    assert found['matched'] is False
    assert '超出可确认范围' in found['reason']
