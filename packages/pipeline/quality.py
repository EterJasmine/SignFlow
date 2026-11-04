"""关键点骨架的几何可信度检查（手部 + 人体）。

MediaPipe 在双手交叠、袖口遮挡、低对比度素材上会给出「置信度很高但几何不可能」的骨架：
掌心被拉长、五指挤成一团、指节落在另一只手上。这类结果置信度看不出来（常常是 0.98 以上），
只能从几何关系上判断。这里只用与身份无关的比值，不猜左右手，也不改检测结果本身。

判据（均在逐帧尺度无关的比值上）：
- 指链比 = 指根到指尖的折线长 / 掌长（腕 → 中指掌指关节）。成人手伸直约 0.7–1.1，
  握拳约 0.3–0.5；跨越两只手时会明显偏离。
- 掌指关节展开度 = 5/9/13/17 投影到掌轴上的跨度 / 掌长。正常手这四个点几乎在同一平面，
  不到 0.4；骨架错乱时会摊开成一条长链。
- 最短指链比：五指全被压缩在一起时该值趋近 0。

另有一组给人体骨架（pose）用的判据，专门抓「画面里只有手却被检成人」：
MediaPipe 在近景手部、纯色暗背景上会把手指映射成四肢，给出 visibility 接近 1、
但骨架整体缩成一团的结果。置信度和可见性都看不出来，只能比比例。
"""
import math

FINGER_BASES = (5, 9, 13, 17)
FINGER_NAMES = ("index", "middle", "ring", "pinky")
# 阈值取自 6 条真实素材（含暗调、外景、近景）的分布：落在正常手的中位数与握拳下限之间。
FINGERS_MIN, FINGERS_MAX = 0.30, 1.80
SPREAD_MAX = 1.00
MIN_FINGER_MIN = 0.15
# 单条即可判定不可信：这些几何量在真实手上不可能出现，不是"稍微不准"。
DECISIVE = frozenset({'degenerate', 'fingers_stretched', 'fingers_collapsed', 'scale_jump'})


def _points(landmarks, width, height):
    return [(p["x"] * width, p["y"] * height) for p in landmarks]


def hand_geometry(landmarks, width, height):
    """返回尺度无关的几何量；关键点不足或掌长退化时返回 None。"""
    if not landmarks or len(landmarks) < 21:
        return None
    p = _points(landmarks, width, height)
    dist = lambda i, j: math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1])
    palm = dist(0, 9)
    if palm < 1e-3:
        return None
    ux, uy = (p[9][0] - p[0][0]) / palm, (p[9][1] - p[0][1]) / palm
    projection = [((p[i][0] - p[0][0]) * ux + (p[i][1] - p[0][1]) * uy) / palm for i in FINGER_BASES]
    chains = [sum(dist(base + k, base + k + 1) for k in range(3)) / palm for base in FINGER_BASES]
    return {
        "palm_px": palm,
        "finger_ratio_max": max(chains),
        "finger_ratio_min": min(chains),
        "knuckle_spread": max(projection) - min(projection),
        "palm_aspect": dist(5, 17) / palm,
    }


def assess(landmarks, width, height, previous_palm_px=None, max_palm_jump=0.6):
    """给出可信度。previous_palm_px 用于抓「同一身份尺度突变」——手不会一帧内变大 60%。"""
    geo = hand_geometry(landmarks, width, height)
    if geo is None:
        return {"score": 0.0, "suspect": True, "reasons": ["degenerate"], "geometry": None}
    reasons = []
    if geo["finger_ratio_max"] > FINGERS_MAX:
        reasons.append("fingers_stretched")
    if geo["finger_ratio_max"] < FINGERS_MIN:
        reasons.append("fingers_collapsed")
    if geo["finger_ratio_min"] < MIN_FINGER_MIN:
        reasons.append("finger_lost")
    if geo["knuckle_spread"] > SPREAD_MAX:
        reasons.append("palm_twisted")
    if previous_palm_px and abs(geo["palm_px"] / previous_palm_px - 1) > max_palm_jump:
        reasons.append("scale_jump")
    # 硬判据一条就够；软判据（指节展布、单指丢失）要两条同时成立才算，避免把正常握拳误杀。
    decisive = any(r in DECISIVE for r in reasons)
    score = max(0.0, 1.0 - 0.6 * len(reasons))
    return {"score": round(score, 2), "suspect": decisive or len(reasons) >= 2,
            "reasons": reasons, "geometry": {k: round(v, 3) for k, v in geo.items()}}


# ---- 人体骨架可信度 ----------------------------------------------------------
# 阈值取自 7 条真实素材（半身、全身、纯手、暗调）逐轨道人工核对后的分布：
# 真实人体的肩宽占画面宽 0.20-0.62、躯干长/肩宽 1.0-1.7、髋宽/肩宽 0.45-0.75；
# 被当成人的手则整体塌缩，肩宽往往只有画面宽的 0.01-0.12。
SHOULDER_MIN_W_RATIO = 0.12
TORSO_RANGE = (0.85, 1.75)
HIP_RANGE = (0.40, 1.00)
ARM_MAX = 1.35
# 手可以比人更靠近镜头，所以尺度只判极端值：肩宽还不到掌长的 0.7 倍。
# 实测这条对「手被检成人」的召回没有额外贡献（形态判据已经抓到），只作兜底。
SHOULDER_PALM_MIN = 0.70


def pose_geometry(pose, width, height):
    """人体骨架的尺度无关比例；肩宽退化时返回 None。"""
    if not pose or len(pose) < 33:
        return None
    p = _points(pose, width, height)
    dist = lambda i, j: math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1])
    shoulder = dist(11, 12)
    if shoulder < 1e-3:
        return None
    sx = (p[11][0] + p[12][0]) / 2
    sy = (p[11][1] + p[12][1]) / 2
    hx = (p[23][0] + p[24][0]) / 2
    hy = (p[23][1] + p[24][1]) / 2
    xs = [q[0] for q in p]
    ys = [q[1] for q in p]
    return {
        "shoulder_px": shoulder,
        "shoulder_over_width": shoulder / max(1e-6, width),
        "torso_over_shoulder": math.hypot(sx - hx, sy - hy) / shoulder,
        "hip_over_shoulder": dist(23, 24) / shoulder,
        "arm_over_shoulder": (dist(11, 13) + dist(12, 14)) / 2 / shoulder,
        "box_px": max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys))),
    }


def pose_assess(pose, width, height, palm_px=None, hand_box_px=None):
    """判断这条骨架是否可能是真人。palm_px / hand_box_px 是同一帧手部的参照尺度。

    只要一条不成立就判为不可能：这些比例在真人身上不会同时以这种幅度偏离，
    而「手指被当成四肢」时会同时崩掉好几条。
    """
    geo = pose_geometry(pose, width, height)
    if geo is None:
        return {"plausible": False, "reasons": ["degenerate"], "metrics": None}
    reasons = []
    if geo["shoulder_over_width"] < SHOULDER_MIN_W_RATIO:
        reasons.append("shoulder_collapsed")
    if not TORSO_RANGE[0] <= geo["torso_over_shoulder"] <= TORSO_RANGE[1]:
        reasons.append("torso_distorted")
    if not HIP_RANGE[0] <= geo["hip_over_shoulder"] <= HIP_RANGE[1]:
        reasons.append("hip_distorted")
    if geo["arm_over_shoulder"] > ARM_MAX:
        reasons.append("arm_distorted")
    metrics = {k: round(v, 3) for k, v in geo.items()}
    # 尺度比只记录、只判极端值：手贴到镜头前时肩宽可以接近掌长，不能据此误杀真人。
    if palm_px:
        metrics["shoulder_over_palm"] = round(geo["shoulder_px"] / palm_px, 3)
        if geo["shoulder_px"] / palm_px < SHOULDER_PALM_MIN:
            reasons.append("smaller_than_hand")
    if hand_box_px:
        metrics["over_hand_box"] = round(geo["box_px"] / hand_box_px, 3)
    return {"plausible": not reasons, "reasons": reasons, "metrics": metrics}


def hand_reference(detected_hands, width, height):
    """同一帧手部的参照尺度：掌长中位数与最大包围盒面积。

    不做 suspect 过滤——闸门需要的是「手有多大」，不是「手有多准」；
    参照本身偏大只会让闸门更宽松，不会误杀真人。
    """
    palms, areas = [], []
    for hand in detected_hands:
        landmarks = hand.get("landmarks") or []
        geo = hand_geometry(landmarks, width, height)
        if not geo:
            continue
        palms.append(geo["palm_px"])
        xs = [p["x"] * width for p in landmarks]
        ys = [p["y"] * height for p in landmarks]
        areas.append(max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys))))
    palm = sorted(palms)[len(palms) // 2] if palms else None
    return palm, (max(areas) if areas else None)


def usable_for_recognition(hand):
    """Orange is a warning; reserve whole-hand rejection for severe geometry."""
    points=hand.get('landmarks',[])
    if len(points)!=21 or any(not math.isfinite(p.get(k,float('nan'))) for p in points for k in ('x','y')):return False
    q=hand.get('quality',{})
    if not q.get('suspect'):return True
    geo=q.get('geometry')
    if not geo:return False
    return (geo.get('palm_px',0)>1 and .08<geo.get('finger_ratio_max',0)<4
            and geo.get('knuckle_spread',99)<2.5 and 'degenerate' not in q.get('reasons',[]))
