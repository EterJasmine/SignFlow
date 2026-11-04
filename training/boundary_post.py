"""边界后处理：从逐帧概率取峰，再把被切开的一步合并回去。

训练（`scripts/train_boundary.py`）与部署（`training/inference.py`）必须共用这一份实现，
否则「训练时一套、上线另一套」的老问题会重演——这条踩过坑。
"""
from __future__ import annotations

import numpy as np


def pick_boundaries(prob, ts, threshold: float, min_gap_ms: int,
                    force_zero: bool = True) -> list[int]:
    """按概率从高到低贪心取峰，间隔不够就跳过；首帧固定为边界（与真值同口径）。"""
    prob = np.asarray(prob, dtype=float)
    ts = np.asarray(ts)
    chosen: list[int] = []
    # 同分时按时间先后取（stable），否则同一段平台每次跑出来的位置都不一样。
    for i in np.argsort(-prob, kind="stable"):
        if prob[i] < threshold:
            break
        if all(abs(int(ts[i]) - int(ts[j])) >= min_gap_ms for j in chosen):
            chosen.append(int(i))
    bounds = sorted(int(ts[i]) for i in chosen)
    if force_zero:
        bounds = [0] + [b for b in bounds if b > 0]
    return bounds


def merge_split_steps(bounds, prob, ts, valley_ratio: float,
                      max_gap_ms: int | None = None) -> list[int]:
    """合并「中间没真正停下来」的相邻两刀。

    每对相邻边界，取两刀之间概率的最低点与两侧峰值之比（`depth`）。`depth` 接近 1 说明
    中间几乎没有沉寂，两刀切的是同一个动作，删掉峰值较低的一刀。`valley_ratio=1.0`
    表示永不合并，等于关闭。
    """
    if valley_ratio >= 1.0 or len(bounds) < 3:
        return sorted(bounds)
    ts_arr = np.asarray(ts)
    prob_arr = np.asarray(prob, dtype=float)
    order = np.argsort(ts_arr)
    ts_sorted, prob_sorted = ts_arr[order], prob_arr[order]
    index = {int(t): i for i, t in enumerate(ts_sorted)}

    keep = sorted(int(b) for b in bounds)
    while len(keep) >= 3:
        target = None
        for a, b in zip(keep, keep[1:]):
            ia, ib = index.get(a), index.get(b)
            if ia is None or ib is None or ib <= ia:
                continue
            peak = min(prob_sorted[ia], prob_sorted[ib])
            if peak <= 1e-6:
                continue
            if max_gap_ms is not None and b - a > max_gap_ms:
                continue
            depth = float(prob_sorted[ia:ib + 1].min()) / float(peak)
            if depth >= valley_ratio and (target is None or depth > target[0]):
                target = (depth, a, b, ia, ib)
        if target is None:
            break
        _, a, b, ia, ib = target
        # 首帧的 0 不能删（真值口径每段都从 0 起），只能删另一侧。
        drop = b if a == 0 or prob_sorted[ia] >= prob_sorted[ib] else a
        keep.remove(drop)
    return keep


def boundaries_from_probability(prob, ts, threshold: float, min_gap_ms: int,
                                valley_ratio: float = 1.0,
                                max_gap_ms: int | None = None) -> list[int]:
    """取峰后合并，一步到位的入口。"""
    picked = pick_boundaries(prob, ts, threshold, min_gap_ms)
    return merge_split_steps(picked, prob, ts, valley_ratio, max_gap_ms)
