"""人体骨架可信度（「只有手被检成人」）的单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.pipeline.quality import hand_reference, pose_assess  # noqa: E402

W, H = 832, 520


def body(sw_ratio=0.35, torso=1.3, hip=0.6, arm=0.8, cx=0.5, cy=0.5):
    """造一条正面半身骨架：所有点相对肩宽按比例摆放，坐标归一化。"""
    sw = sw_ratio                      # 肩宽（x 归一化）
    unit = sw * W / H                  # 一个肩宽长度换算到 y 归一化方向
    pts = [{'x': cx, 'y': cy, 'z': 0.0} for _ in range(33)]

    def put(i, dx, dy):                # dx 以肩宽（x 归一化）计，dy 以肩宽（y 归一化）计
        pts[i] = {'x': cx + dx, 'y': cy + dy, 'z': 0.0}

    put(11, -sw / 2, 0.0); put(12, sw / 2, 0.0)        # 双肩
    put(23, -sw * hip / 2, unit * torso); put(24, sw * hip / 2, unit * torso)
    put(13, -sw / 2 - sw * .15, unit * arm * .7); put(14, sw / 2 + sw * .15, unit * arm * .7)
    put(15, -sw / 2 - sw * .3, unit * arm * 1.4); put(16, sw / 2 + sw * .3, unit * arm * 1.4)
    put(0, 0.0, -unit * .55)                            # 鼻在肩线上方
    put(1, -sw * .1, -unit * .5); put(2, -sw * .22, -unit * .5)
    put(3, -sw * .2, -unit * .55); put(4, sw * .2, -unit * .55)
    put(5, sw * .22, -unit * .5); put(6, sw * .2, -unit * .55)
    return pts


def hand_at(cx=0.5, cy=0.6, palm_ratio=0.06):
    """造一只掌长为 palm_ratio（画面宽归一化）的手，用于提供参照尺度。"""
    palm = palm_ratio * W
    pts = [{'x': cx, 'y': cy, 'z': 0.0} for _ in range(21)]
    pts[0] = {'x': cx, 'y': cy, 'z': 0.0}
    pts[9] = {'x': cx, 'y': cy - palm / H, 'z': 0.0}
    for base, dx in zip((5, 9, 13, 17), (-.4, 0, .3, .6)):
        pts[base] = {'x': cx + dx * palm / W, 'y': cy - palm / H, 'z': 0.0}
        for k in range(3):
            pts[base + 1 + k] = {'x': pts[base]['x'], 'y': pts[base]['y'] - (palm / H) * (.3 - k * .08), 'z': 0.0}
    return pts


def test_real_half_body_is_kept():
    verdict = pose_assess(body(), W, H)
    assert verdict['plausible'] is True, verdict
    assert 0.30 < verdict['metrics']['shoulder_over_width'] < 0.40


def test_collapsed_skeleton_is_rejected():
    """只有手的画面：整条骨架缩成一团，肩宽不到画面宽的 3%。"""
    verdict = pose_assess(body(sw_ratio=0.03), W, H)
    assert verdict['plausible'] is False
    assert 'shoulder_collapsed' in verdict['reasons']


def test_absurd_torso_is_rejected():
    """手指被映射成胯部时躯干会被拉成好几倍肩宽。"""
    verdict = pose_assess(body(torso=4.0), W, H)
    assert verdict['plausible'] is False
    assert 'torso_distorted' in verdict['reasons']


def test_hip_wider_than_shoulder_is_rejected():
    verdict = pose_assess(body(hip=1.4), W, H)
    assert verdict['plausible'] is False
    assert 'hip_distorted' in verdict['reasons']


def test_shoulder_smaller_than_palm_is_rejected():
    """肩宽还不到手掌长的一半：物理上不可能，无视置信度直接判假。"""
    palm_px, box_px = hand_reference([{'landmarks': hand_at(palm_ratio=0.10)}], W, H)
    assert palm_px > 0 and box_px > 0
    verdict = pose_assess(body(sw_ratio=0.04), W, H, palm_px=palm_px, hand_box_px=box_px)
    assert 'smaller_than_hand' in verdict['reasons']
    assert verdict['metrics']['shoulder_over_palm'] < 0.7


def test_real_body_with_close_hand_is_kept():
    """手伸到镜头前、比人体局部还大时，骨架本身仍应保留——尺度判据不是主判据。"""
    palm_px, box_px = hand_reference([{'landmarks': hand_at(palm_ratio=0.30)}], W, H)
    verdict = pose_assess(body(sw_ratio=0.30), W, H, palm_px=palm_px, hand_box_px=box_px)
    assert verdict['plausible'] is True, verdict


def test_degenerate_shoulder_is_rejected():
    pts = [{'x': .5, 'y': .5, 'z': 0.0} for _ in range(33)]
    verdict = pose_assess(pts, W, H)
    assert verdict['plausible'] is False
    assert verdict['reasons'] == ['degenerate']


def test_short_landmark_list_is_rejected():
    verdict = pose_assess([{'x': .5, 'y': .5}] * 5, W, H)
    assert verdict['plausible'] is False
