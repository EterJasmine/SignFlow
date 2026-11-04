"""边界后处理：取峰与合并。训练与部署共用这一份代码，两边行为必须一致。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.boundary_post import (  # noqa: E402
    boundaries_from_probability,
    merge_split_steps,
    pick_boundaries,
)


def test_pick_drops_below_threshold_and_keeps_zero():
    ts = np.array([0, 200, 400, 600, 800])
    prob = np.array([0.9, 0.05, 0.8, 0.05, 0.7])
    bounds = pick_boundaries(prob, ts, threshold=0.5, min_gap_ms=0)
    assert bounds == [0, 400, 800]


def test_pick_respects_min_gap():
    ts = np.array([0, 200, 400, 600, 800])
    prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    assert pick_boundaries(prob, ts, 0.5, 400) == [0, 400, 800]


def test_pick_force_zero_can_be_disabled():
    ts = np.array([0, 200, 400])
    prob = np.array([0.1, 0.9, 0.1])
    assert pick_boundaries(prob, ts, 0.5, 0, force_zero=False) == [200]


def test_merge_keeps_real_dips():
    """两侧都有深谷时不能合并——那确实是两个动作。"""
    ts = np.array(list(range(0, 1800, 200)))
    prob = np.array([0.9, 0.9, 0.05, 0.05, 0.9, 0.9, 0.05, 0.05, 0.9])
    bounds = pick_boundaries(prob, ts, 0.2, 400)
    assert bounds == [0, 800, 1600]
    assert merge_split_steps(bounds, prob, ts, 0.6) == [0, 800, 1600]


def test_merge_collapses_flat_plateau():
    """概率一路平坦说明这里没有真正的切换，贪心取峰会沿平台打出一串点，应当合并掉。"""
    ts = np.array([0, 400, 800, 1200, 1600])
    prob = np.array([0.90, 0.88, 0.86, 0.88, 0.90])
    bounds = pick_boundaries(prob, ts, 0.2, 400)
    assert len(bounds) == 5
    merged = merge_split_steps(bounds, prob, ts, 0.6)
    assert len(merged) < 5
    assert merged[0] == 0                      # 首帧不删
    assert merged == sorted(merged)


def test_merge_ratio_one_is_a_no_op():
    ts = np.array([0, 400, 800])
    prob = np.array([0.9, 0.9, 0.9])
    assert merge_split_steps([0, 400, 800], prob, ts, 1.0) == [0, 400, 800]


def test_merge_never_removes_first_frame():
    """真值口径每段都从 0 起，合并只能删另一侧。"""
    ts = np.array([0, 400, 800, 1200])
    prob = np.array([0.2, 0.9, 0.9, 0.9])
    merged = merge_split_steps([0, 400, 800, 1200], prob, ts, 0.5)
    assert merged[0] == 0


def test_boundaries_from_probability_matches_two_steps():
    ts = np.array([0, 400, 800, 1200, 1600])
    prob = np.array([0.90, 0.88, 0.86, 0.88, 0.90])
    picked = pick_boundaries(prob, ts, 0.2, 400)
    assert (boundaries_from_probability(prob, ts, 0.2, 400, 0.6)
            == merge_split_steps(picked, prob, ts, 0.6))
