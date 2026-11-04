"""边界检测训练脚本的可测部分：打点、目标、归一化、窗口切分。

真实数据只有 3 条训练视频，训练本身不可单测；这里把决定成败的四个纯函数钉住。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_boundary as tb  # noqa: E402


def case(times, truth=(), dims=3, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.array(times, np.int64)
    x = rng.normal(size=(len(ts), dims)).astype(np.float32)
    return {'source_id': f's{seed}', 'name': f'v{seed}.mp4', 'ts': ts, 'x': x,
            'mask': np.ones((len(ts), dims), bool), 'truth': list(truth),
            'names': [f'f{i}' for i in range(dims)], 'version': 'features-2.0', 'result': {}}


def test_pick_always_contains_zero_and_respects_threshold():
    ts = np.array([0, 133, 267, 400, 533], np.int64)
    prob = np.array([0.10, 0.95, 0.90, 0.20, 0.05])
    assert tb.pick(prob, ts, 0.5, 300) == [0, 133]      # 267 与 133 只隔 133ms，被间隔挡掉
    assert tb.pick(prob, ts, 0.5, 100) == [0, 133, 267]
    assert tb.pick(prob, ts, 0.99, 300) == [0]          # 一个峰都不够阈值时仍保留首帧


def test_pick_gap_is_measured_in_milliseconds_not_frames():
    ts = np.array([0, 500, 1000], np.int64)
    prob = np.array([0.1, 0.9, 0.8])
    assert tb.pick(prob, ts, 0.5, 600) == [0, 500]


def test_make_target_snaps_to_nearest_frame():
    c = case([0, 133, 267, 400], truth=[0, 120])
    target = tb.make_target(c)
    assert target.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_make_target_ignores_duplicate_boundaries_on_same_frame():
    c = case([0, 133, 267], truth=[130, 140])
    assert tb.make_target(c).sum() == pytest.approx(1.0)


def test_normalise_video_removes_per_video_scale():
    a, b = case([0, 133, 267, 400], seed=1), case([0, 133, 267, 400], seed=2)
    b['x'] = b['x'] * 100 + 5                        # 模拟另一机位的量纲/偏移
    out, mean, std = tb.normalise([a, b], 'video')
    for item in out:
        assert np.allclose(item['x'].mean(0), 0.0, atol=1e-4)
        assert np.allclose(item['x'].std(0), 1.0, atol=1e-4)
    assert np.allclose(mean, 0.0) and np.allclose(std, 1.0)   # 逐视频统计量不进权重


def test_normalise_global_uses_train_stats_and_keeps_zeros_outside_mask():
    a = case([0, 133, 267, 400], seed=3)
    a['mask'][1, 0] = False
    out, mean, std = tb.normalise([a], 'global')
    assert out[0]['x'][1, 0] == 0.0                  # 无效维不做标准化，避免把洞变成 0 之外的数
    assert np.isfinite(std).all() and (std > 0).all()


def test_crop_pads_short_sequence_and_flags_padding():
    c = case(list(range(0, 133 * 4, 133)), truth=[0])
    c['y'] = tb.make_target(c)
    x, mask, y, valid = tb.crop(c, length=8, rng=np.random.default_rng(0), augment=False)
    assert x.shape == (8, 3) and y.shape == (8,) and valid.tolist() == [True] * 4 + [False] * 4
    assert mask.all()


def test_crop_keeps_targets_aligned_with_frames():
    c = case(list(range(0, 133 * 40, 133)), truth=[2660])
    c['y'] = tb.make_target(c)
    x, mask, y, valid = tb.crop(c, length=16, rng=np.random.default_rng(7), augment=False)
    start = int(np.argmin(np.abs(c['x'][:, 0] - x[0, 0])))     # 连续值随机数，匹配唯一
    assert np.allclose(x, c['x'][start:start + 16])
    assert np.allclose(y, c['y'][start:start + 16])            # 目标与帧同步切片
    assert valid.all()


def test_change_features_is_affine_invariant_per_channel():
    """训练与推理必须同口径：整条序列先逐维标准化，所以逐维缩放不该改变特征。"""
    from training.features_diff import change_features
    a = case([0, 133, 267, 400, 533, 667, 800], seed=11)
    scaled = dict(a)
    scaled['x'] = (a['x'] * 7.5 + 3.0).astype(np.float32)
    assert np.allclose(change_features(a['x'], a['mask'], a['names']),
                       change_features(scaled['x'], scaled['mask'], a['names']), atol=1e-4)


def test_linear_boundary_returns_one_probability_per_frame():
    import torch
    from training.models import build_model
    model = build_model('linear_change', 14)
    out = model(torch.zeros(1, 9, 14), torch.ones(1, 9, 14, dtype=torch.bool))['boundary']
    assert out.shape == (1, 9)


def test_model_carries_normalisation_buffers():
    import torch
    from training.models import build_model
    model = build_model('tcn', 76, width=32)
    state = model.state_dict()
    assert state['feat_mean'].shape == (76,) and state['feat_std'].shape == (76,)
    x = torch.ones(1, 6, 76)
    mask = torch.ones(1, 6, 76, dtype=torch.bool)
    before = model(x, mask)['boundary']
    model.feat_mean.fill_(1.0)                       # 中心化后输入全 0，输出应随之改变
    assert not torch.allclose(before, model(x, mask)['boundary'])
