"""从已落盘的分析结果里算出"手势变化线索"，用于人工/机器标注边界时定位该看哪里。

只做两件事：
1) 把每一采样帧的左右手腕位置、手型（五指伸展度）抽成一条时间序列；
2) 输出"手型变化 + 运动方向变化"的峰值时刻，作为待人工确认的候选边界。

脚本不读算法给出的 actions / segmentation 结果，也不写任何文件（除非 --json 指定）。

用法：
    python scripts/boundary_cues.py data/assets/video_xxx/result.json --top 16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

WRIST_L, WRIST_R = 15, 16
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _pt(p):
    return float(p["x"]), float(p["y"])


def hand_shape(lms) -> list[float]:
    """五指各自"指尖到腕距离 / 掌宽"，对角度旋转不敏感，能区分张开与握拳。"""
    w = _pt(lms[0])
    mcp = _pt(lms[9])
    palm = math.hypot(mcp[0] - w[0], mcp[1] - w[1]) or 1e-6
    out = []
    for idx in FINGERS.values():
        tip = _pt(lms[idx[3]])
        out.append(math.hypot(tip[0] - w[0], tip[1] - w[1]) / palm)
    return out


def sample(result: dict) -> list[dict]:
    rows = []
    for f in result["frames"]:
        row = {"t": f["timestamp_ms"] / 1000.0, "wl": None, "wr": None,
               "sl": None, "sr": None, "head": None, "hand_n": 0}
        for tr in f["tracks"]:
            pose = tr.get("pose_landmarks") or []
            if len(pose) == 33:
                row["wl"] = _pt(pose[WRIST_L])
                row["wr"] = _pt(pose[WRIST_R])
                row["head"] = _pt(pose[0])
            for side, hand in (tr.get("hands") or {}).items():
                lms = hand.get("landmarks") or []
                if len(lms) != 21:
                    continue
                side = (hand.get("anatomical_side") or side or "").lower()
                shape = hand_shape(lms)
                row["hand_n"] += 1
                if side in ("left",) and row["sl"] is None:
                    row["sl"] = shape
                elif side in ("right",) and row["sr"] is None:
                    row["sr"] = shape
                elif row["sl"] is None:
                    row["sl"] = shape
                elif row["sr"] is None:
                    row["sr"] = shape
        rows.append(row)
    return rows


def dist(a, b):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shape_delta(a, b):
    if a is None or b is None:
        return None
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def build(rows: list[dict]) -> list[dict]:
    """逐帧打分：move = 手腕位移/秒，shape = 手型变化率，turn = 运动方向转角。"""
    out = []
    for i, r in enumerate(rows):
        prev = rows[max(i - 1, 0)]
        nxt = rows[min(i + 1, len(rows) - 1)]
        dt = max(nxt["t"] - prev["t"], 1e-3)
        mv = max(dist(r["wl"], prev["wl"]) or 0, dist(r["wr"], prev["wr"]) or 0) / dt
        sh = max(shape_delta(r["sl"], prev["sl"]) or 0, shape_delta(r["sr"], prev["sr"]) or 0) / dt
        # 方向转角：手腕速度矢量的夹角余弦，1=同向，-1=反向
        turn = 0.0
        for key in ("wl", "wr"):
            a, b, c = prev[key], r[key], nxt[key]
            if None in (a, b, c):
                continue
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1 = math.hypot(*v1)
            n2 = math.hypot(*v2)
            if n1 < 1e-4 or n2 < 1e-4:
                continue
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            turn = max(turn, (1 - cos) / 2)
        out.append({"t": r["t"], "move": mv, "shape": sh, "turn": turn, "hand_n": r["hand_n"]})
    return out


def peaks(scored: list[dict], min_gap: float) -> list[dict]:
    for s in scored:
        s["score"] = 1.6 * min(s["shape"], 12) + 2.4 * min(s["move"], 2.0) + 1.2 * s["turn"]
    chosen: list[dict] = []
    for s in sorted(scored, key=lambda x: -x["score"]):
        if s["t"] < 0.35:
            continue
        if all(abs(s["t"] - c["t"]) >= min_gap for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda x: x["t"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--gap", type=float, default=0.55)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    result = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    rows = sample(result)
    scored = build(rows)
    picked = peaks(scored, args.gap)[: args.top]

    print(f"时长 {result['duration_ms'] / 1000:.2f}s  采样帧 {len(rows)}")
    for s in picked:
        print(f"  {s['t']:6.2f}s  move={s['move']:.2f} shape={s['shape']:.2f} turn={s['turn']:.2f} hands={s['hand_n']}")
    if args.json:
        Path(args.json).write_text(
            json.dumps([round(s["t"], 2) for s in picked], ensure_ascii=False), encoding="utf-8")
        print("候选写入", args.json)


if __name__ == "__main__":
    main()
