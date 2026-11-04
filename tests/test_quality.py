"""手部几何可信度的单元测试。"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.pipeline.quality import assess, hand_geometry  # noqa: E402

BASES = (5, 9, 13, 17)


def hand(tip_scale=1.0, palm=0.10, bones=(0.035, 0.022, 0.022), knuckle_shift=0.0):
    """造一只标准手：掌轴朝 -y，四指在 x 方向排开，指节逐段变短。"""
    pts = [{'x': .5, 'y': .7, 'z': 0.0}] * 21
    pts = [dict(p) for p in pts]
    pts[0] = {'x': .5, 'y': .7, 'z': 0.0}
    pts[9] = {'x': .5, 'y': .7 - palm, 'z': 0.0}
    xs = (.44, .50, .53, .56)   # 中指根落在掌轴上，掌轴才竖直
    for base, x in zip(BASES, xs):
        pts[base] = {'x': x, 'y': .7 - palm + knuckle_shift, 'z': 0.0}
        y = .7 - palm + knuckle_shift
        for k in range(3):
            y -= bones[k] * tip_scale
            pts[base + 1 + k] = {'x': x, 'y': y, 'z': 0.0}
    for k, y in zip((1, 2, 3, 4), (-0.02, -0.04, -0.06, -0.08)):
        pts[k] = {'x': .5 + abs(y) * .8, 'y': .7 + y, 'z': 0.0}
    return pts


def test_normal_hand_is_trusted():
    result = assess(hand(), 1000, 1000)
    assert result['suspect'] is False, result
    assert result['score'] == 1.0
    geo = result['geometry']
    assert 0.5 < geo['finger_ratio_max'] < 1.2
    assert geo['knuckle_spread'] < 0.05


def test_stretched_fingers_are_suspect():
    """骨架横跨两只手时，指链会被拉得远长于掌心。"""
    result = assess(hand(tip_scale=6.0), 1000, 1000)
    assert result['suspect'] is True
    assert 'fingers_stretched' in result['reasons']


def test_collapsed_fingers_are_suspect():
    """五指挤成一团：指链比接近 0，且最短指链同时触线。"""
    result = assess(hand(tip_scale=0.05), 1000, 1000)
    assert result['suspect'] is True
    assert 'fingers_collapsed' in result['reasons']


def test_twisted_palm_is_soft_only():
    """掌指关节被摊开是软判据：单条不足以判死，避免误杀侧视角的手。"""
    pts = hand()
    shift = .18
    for k in (5, 6, 7, 8):                       # 食指整根下移，指链长度不变
        pts[k] = {**pts[k], 'y': pts[k]['y'] + shift}
    result = assess(pts, 1000, 1000)
    assert 'palm_twisted' in result['reasons']
    assert result['suspect'] is False


def test_scale_jump_flags_identity_switch():
    """同一身份掌长一帧内翻倍，比几何形状更能说明换了一只手。"""
    normal = assess(hand(), 1000, 1000)
    jumped = assess(hand(palm=0.10), 1000, 1000, previous_palm_px=normal['geometry']['palm_px'] * 4)
    assert 'scale_jump' in jumped['reasons']


def test_degenerate_landmarks_are_suspect():
    zero = [{'x': .5, 'y': .5, 'z': 0.0}] * 21
    result = assess(zero, 1000, 1000)
    assert result['suspect'] is True
    assert result['reasons'] == ['degenerate']


def test_geometry_uses_true_pixel_aspect():
    """在 1920x1080 上画一只真实比例的手，归一化后交给 assess，比值必须和像素空间一致；
    只用 x/y 的归一化差（不乘宽高）会把 16:9 的手压扁，比值为 1.78 倍。"""
    w, h = 1920, 1080
    pixels = [{'x': p['x'] * w, 'y': p['y'] * h, 'z': 0.0} for p in hand()]
    dist = lambda i, j: math.hypot(pixels[i]['x'] - pixels[j]['x'], pixels[i]['y'] - pixels[j]['y'])
    palm = dist(0, 9)
    expected_aspect = dist(5, 17) / palm        # 像素空间的真值
    normalized = [{'x': p['x'] / w, 'y': p['y'] / h, 'z': 0.0} for p in pixels]
    got = hand_geometry(normalized, w, h)
    assert math.isclose(expected_aspect, got['palm_aspect'], rel_tol=1e-6)
    # 若忘了乘宽高，掌宽会被 16:9 压扁成 1/1.78
    naive = [{'x': p['x'] / w, 'y': p['y'] / h, 'z': 0.0} for p in pixels]
    flat = hand_geometry([{'x': q['x'], 'y': q['y'], 'z': 0.0} for q in naive], 1, 1)
    assert flat['palm_aspect'] < got['palm_aspect'] * .7
