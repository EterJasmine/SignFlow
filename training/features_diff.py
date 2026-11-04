"""边界模型用的变化能量特征（13 维）。

做法：对每个特征组算「前 k 帧中位数」与「后 k 帧中位数」的逐维差，再取均值和最大值——
这正是手型换态检测器在用的量，但把 76 维原始序列压成 13 维的「变化有多大」。

原因很实际：只有 3 条训练视频时，把原始序列按帧直接喂时序网络，网络会记住这几条视频
（训练集 F1 0.945、留出集 0.446）；压成变化能量再学一个线性校准器，跨视频稳得多
（见 evaluation/results/train-boundary.md 的选型表）。
"""
import numpy as np
import torch
from torch import nn

EXTRA_NAMES = ('hand_0_speed', 'hand_1_speed', 'hand_distance', 'synchrony')


def group_indices(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        if name.startswith('hand_0_finger'):
            key = 'h0_shape'
        elif name.startswith('hand_1_finger'):
            key = 'h1_shape'
        elif name.startswith('hand_0'):
            key = 'h0_motion'
        elif name.startswith('hand_1'):
            key = 'h1_motion'
        else:
            key = 'misc'
        groups.setdefault(key, []).append(i)
    return groups


def local_change(x: np.ndarray, mask: np.ndarray, names: list[str], k: int = 2) -> np.ndarray:
    """x: T×F，mask: T×F。返回 T×(2*组数+额外维)。前后窗口任一维不是全程有效就跳过该维。"""
    groups = group_indices(names)
    keys = list(groups)
    base = len(keys) * 2
    out = np.zeros((len(x), base + len(EXTRA_NAMES)), np.float32)
    n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k)
        if lo == i or hi == i:
            continue
        valid = mask[lo:i].all(0) & mask[i:hi].all(0)
        diff = np.abs(np.median(x[lo:i], 0) - np.median(x[i:hi], 0))
        for gi, key in enumerate(keys):
            idx = np.asarray(groups[key])
            keep = idx[valid[idx]]
            if len(keep):
                out[i, gi * 2] = diff[keep].mean()
                out[i, gi * 2 + 1] = diff[keep].max()
    for j, name in enumerate(EXTRA_NAMES):
        if name in names:
            out[:, base + j] = x[:, names.index(name)]
    return out


def standardise_channels(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐视频逐维标准化：不同机位、不同景别的量纲差在这一步消掉。

    训练与推理都走这个函数，两边才算同一口径；否则"先标准化再求差分"与
    "直接求差分"会给出不同结果——组内平均会被量纲大的维度独占。
    """
    count = mask.sum(0)
    mean = np.where(count > 0, np.where(mask, x, 0.0).sum(0) / np.maximum(count, 1), 0.0)
    dev = np.where(mask, x - mean, 0.0)
    std = np.sqrt((dev * dev).sum(0) / np.maximum(count, 1))
    std = np.where(std < 1e-6, 1.0, std)
    return np.where(mask, (x - mean) / std, 0.0).astype(np.float32)


def robust_standardise(f: np.ndarray) -> np.ndarray:
    """输出列再用中位数/四分位距压一次，让不同视频的差分幅度可比。"""
    mu = np.median(f, 0)
    iqr = np.percentile(f, 75, 0) - np.percentile(f, 25, 0)
    return ((f - mu) / np.where(iqr < 1e-6, 1.0, iqr)).astype(np.float32)


def change_features(x, mask, names, k: int = 2) -> np.ndarray:
    return robust_standardise(local_change(standardise_channels(x, mask), mask, names, k))


class LinearBoundary(nn.Module):
    """逐帧线性边界概率。输入是已经标准化好的变化能量特征。"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer('feat_mean', torch.zeros(input_dim))
        self.register_buffer('feat_std', torch.ones(input_dim))
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        z = ((x - self.feat_mean) / self.feat_std) * mask
        return {'boundary': self.linear(z).squeeze(-1)}
