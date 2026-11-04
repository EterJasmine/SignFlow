"""在已落盘的分析结果上离线重算分段，对真值算边界 F1——不用重跑视觉检测。

用法：
    python scripts/tune_segmentation.py                 # 当前默认参数
    python scripts/tune_segmentation.py --grid          # 扫 shape_delta
    python scripts/tune_segmentation.py --grid --holdout  # 只用 11-01 两条当验证集

只读 result.json 与 truth.json，不写工程数据。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from packages.pipeline.segmentation import segment_actions  # noqa: E402
from metrics import boundary_f1  # noqa: E402


def load_cases() -> list[dict]:
    truth = json.loads((ROOT / "evaluation/truth.json").read_text(encoding="utf-8"))
    cases = []
    for item in truth["videos"]:
        if not item.get("confirmed") or not item.get("boundaries_truth_ms"):
            continue
        path = ROOT / "data/assets" / item["review_video_id"] / "result.json"
        if not path.exists():
            print(f"跳过（无分析结果）：{item['filename']}")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        cases.append({
            "name": item["filename"],
            "truth": sorted(item["boundaries_truth_ms"]),
            "result": result,
        })
    return cases


def run_case(case: dict, shape_delta: float) -> tuple[list[int], dict]:
    r = case["result"]
    actions, _, info = segment_actions(
        r["frames"], r["scenes"], r["duration_ms"], r["features"],
        min_step_ms=600, max_step_ms=6000, shape_delta=shape_delta)
    return sorted(a["start_ms"] for a in actions), info


def evaluate(cases: list[dict], shape_delta: float, verbose: bool) -> float:
    scores = []
    for case in cases:
        predicted, _ = run_case(case, shape_delta)
        f1 = boundary_f1(case["truth"], predicted)
        scores.append(f1)
        if verbose:
            print(f"  {case['name']:<34} 真值{len(case['truth']):>3} 预测{len(predicted):>3} F1={f1:.3f}")
    return sum(scores) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="扫描 shape_delta")
    ap.add_argument("--holdout", action="store_true", help="只用 11-01 两条做验证集")
    args = ap.parse_args()

    cases = load_cases()
    if args.holdout:
        cases = [c for c in cases if "11-01" in c["name"]]
    print(f"{len(cases)} 条已确认视频")

    if not args.grid:
        from packages.pipeline.segmentation import DEFAULT_SHAPE_DELTA
        print(f"当前默认（shape_delta={DEFAULT_SHAPE_DELTA}）:")
        evaluate(cases, DEFAULT_SHAPE_DELTA, True)
        return

    print(f"{'delta':>6} {'F1均值':>8}")
    best = (0.0, -1.0)
    for delta in (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20):
        mean = evaluate(cases, delta, False)
        print(f"{delta:>6.2f} {mean:>8.3f}")
        if mean > best[1]:
            best = (delta, mean)
    print(f"最优 delta={best[0]}  F1={best[1]:.3f}")
    evaluate(cases, best[0], True)


if __name__ == "__main__":
    main()
