"""手型换态分段（segmentation-2.1）的行为测试。

真实视频里连续动作没有停顿时，只看"起停"会系统性欠分割；
这里用手型恒定/跳变/来回抖动三类合成序列把新加的检测器钉住。
"""
from packages.pipeline.segmentation import (
    DEFAULT_SHAPE_DELTA, _shape_columns, segment_actions, shape_state_boundaries,
)

KEYS = ('mcp_angle', 'pip_angle', 'dip_angle', 'tip_radius', 'thumb_distance')
NAMES = ['valid_ratio'] + [f'hand_0_finger_{i}_{k}' for i in range(5) for k in KEYS]


def make_rows(shape_at, n=60, fps=7.5):
    """shape_at(秒) -> 25 维手型向量。"""
    rows = []
    for i in range(n):
        t = i / fps
        values = [1.0] + list(shape_at(t))
        rows.append({'timestamp_ms': int(t * 1000), 'scene_id': 'scene_0001',
                     'values': values, 'mask': [True] * len(values)})
    return rows


def base_shape(v=0.2):
    return [v] * 25


def test_shape_columns_counts_both_hands():
    names = NAMES + [f'hand_1_finger_{i}_{k}' for i in range(5) for k in KEYS]
    assert len(_shape_columns(names)) == 50
    assert _shape_columns(['valid_ratio', 'hand_0_speed']) == []


def test_constant_shape_gives_no_boundary():
    rows = make_rows(lambda t: base_shape(0.2))
    assert shape_state_boundaries(rows, NAMES, DEFAULT_SHAPE_DELTA) == []


def test_one_step_change_is_one_boundary_near_transition():
    def shape(t):
        return base_shape(0.2) if t < 2.0 else base_shape(0.8)

    rows = make_rows(shape)
    found = shape_state_boundaries(rows, NAMES, DEFAULT_SHAPE_DELTA)
    assert len(found) == 1
    # 边界应落在跳变附近（中位数窗口会往前挪半窗，容差放到 0.5 秒）
    assert abs(found[0] - 2000) <= 500


def test_alternating_shape_keeps_boundaries_min_gap_apart():
    def shape(t):
        return base_shape(0.2) if int(t * 2) % 2 == 0 else base_shape(0.8)

    rows = make_rows(shape, n=120)
    found = shape_state_boundaries(rows, NAMES, DEFAULT_SHAPE_DELTA, min_step_ms=600)
    assert len(found) >= 2
    assert all(b - a >= 600 for a, b in zip(found, found[1:]))


def test_large_delta_disables_detector():
    def shape(t):
        return base_shape(0.2) if t < 2.0 else base_shape(0.25)

    rows = make_rows(shape)
    assert shape_state_boundaries(rows, NAMES, 0.5) == []


def test_segment_actions_adds_steps_when_shape_changes():
    """连续微动（没有停顿帧）时，起停检测给不出步，手型换态能补上。"""
    import math

    def shape(t):
        variant = 0.2 if t < 2.0 else (0.8 if t < 4.0 else 0.2)
        return [variant + 0.02 * math.sin(30 * t)] * 25

    rows = make_rows(shape, n=40)
    frames = [{'timestamp_ms': r['timestamp_ms'], 'scene_id': r['scene_id'], 'sharpness': 300.0}
              for r in rows]
    features = {'names': NAMES, 'rows': rows}
    duration = rows[-1]['timestamp_ms']
    off, _, info_off = segment_actions(frames, [{'start_ms': 0}], duration, features, shape_delta=0.0)
    on, _, info_on = segment_actions(frames, [{'start_ms': 0}], duration, features, shape_delta=0.1)
    assert len(off) == 1, '连续微动时旧逻辑本就不该切步'
    assert len(on) >= 3
    assert info_on['shape_delta'] == 0.1
    assert all(a['segmentation_method'].endswith('shape_state') for a in on)
    assert info_off['shape_delta'] == 0.0


def test_scene_cut_does_not_create_shape_change():
    rows=make_rows(lambda t: base_shape(.2 if t<2 else .8))
    for r in rows:r['scene_id']='a' if r['timestamp_ms']<2000 else 'b'
    assert shape_state_boundaries(rows,NAMES,DEFAULT_SHAPE_DELTA)==[]
