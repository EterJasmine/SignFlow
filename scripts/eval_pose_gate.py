"""评估 pose 可信性闸门。逐轨道标注真伪，统计真人误杀率与幻觉召回率。

判据（全部是与身份无关的比值，不猜左右手）：
  a 肩宽/画宽  —— 幻觉骨架整体塌缩，占画面比例远小于真实半身
  b 躯干长/肩宽 ─ 成人约 1.0-1.7
  c 髋宽/肩宽  ─ 成人约 0.4-0.8
  d 上臂长/肩宽 ─ 成人约 0.6-1.1
  e 肩宽/掌长  ─ 成人约 1.8-2.6（手更近镜头时更大，故只判下界）
"""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GHOST_TRACKS = {  # 人工逐轨道核对：只有手、被检成人的轨道
    'video_52f9b2de269243ff95a50dc88af22896': None,   # None = 全部轨道都是幻觉
    'video_173988cb50e24b3c9d308c1de54ab323': {'person_0001', 'person_0002', 'person_0003', 'person_0004',
                                               'person_0005', 'person_0010', 'person_0011', 'person_0012',
                                               'person_0013', 'person_0014', 'person_0015', 'person_0016',
                                               'person_0017', 'person_0018', 'person_0019', 'person_0021',
                                               'person_0022'},
}


def rows(video_id):
    p = ROOT / 'data' / 'assets' / video_id / 'result.json'
    r = json.loads(p.read_text(encoding='utf8'))
    w, h = r['width'], r['height']
    out = []
    for f in r['frames']:
        palms, boxes = [], []
        for tr in f['tracks']:
            for hand in tr['hands'].values():
                q = hand.get('quality') or {}
                lm = hand.get('landmarks') or []
                if hand.get('interpolated') or len(lm) < 21 or q.get('suspect') or not q.get('geometry'):
                    continue
                palms.append(q['geometry']['palm_px'])
                xs = [q['x'] * w for q in lm]; ys = [q['y'] * h for q in lm]
                boxes.append(max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys))))
        palm = sorted(palms)[len(palms) // 2] if palms else None
        hbox = max(boxes) if boxes else None
        for tr in f['tracks']:
            pose = tr['pose_landmarks']
            if len(pose) != 33:
                continue
            pt = [(q['x'] * w, q['y'] * h) for q in pose]
            d = lambda i, j: math.hypot(pt[i][0] - pt[j][0], pt[i][1] - pt[j][1])
            sw = d(11, 12)
            xs = [q[0] for q in pt]; ys = [q[1] for q in pt]
            pbox = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            m = {
                'sw/w': sw / w,
                't/sw': math.hypot((pt[11][0] + pt[12][0]) / 2 - (pt[23][0] + pt[24][0]) / 2,
                                   (pt[11][1] + pt[12][1]) / 2 - (pt[23][1] + pt[24][1]) / 2) / sw if sw > 1e-6 else 99,
                'h/sw': d(23, 24) / sw if sw > 1e-6 else 0,
                'arm/sw': (d(11, 13) + d(12, 14)) / 2 / sw if sw > 1e-6 else 99,
                'box/sw': (pbox / hbox) if hbox else None,
                'sw/palm': (sw / palm) if palm else None,
            }
            out.append((tr['track_id'], m))
    return out


def judge(m, use=('a', 'b', 'c', 'd', 'e', 'f')):
    bad = []
    if 'a' in use and m['sw/w'] < 0.12: bad.append('a')
    if 'b' in use and not (0.85 <= m['t/sw'] <= 1.75): bad.append('b')
    if 'c' in use and not (0.40 <= m['h/sw'] <= 1.00): bad.append('c')
    if 'd' in use and m['arm/sw'] > 1.35: bad.append('d')
    if 'e' in use and m['sw/palm'] is not None and m['sw/palm'] < 1.40: bad.append('e')
    if 'f' in use and m['box/sw'] is not None and m['box/sw'] < 3.0: bad.append('f')
    return bad


def main():
    data = {}
    for vid in list(GHOST_TRACKS) + ['video_3bafb703f66049ec81923e5aad91d2fa',
                                     'video_9f24bd561bf64339a9cd036460ebd639',
                                     'video_5706a1f457e8409fb9a6a160d103c03a',
                                     'video_7a4a44a556f1461691b5c68ec388948a',
                                     'video_880e11f0f7b84c6fb5116f6bf1e92da5']:
        data[vid] = rows(vid)

    lines = []
    for use, need in [(x, n) for x in [('a', 'b', 'c', 'd', 'e', 'f'), ('a', 'b', 'c', 'd'),
                                       ('a', 'b', 'c', 'e', 'f'), ('a', 'b', 'e', 'f')] for n in (1, 2)]:
        fp = tn = tp = fn = 0
        for vid, rs in data.items():
            ghost = GHOST_TRACKS.get(vid, set())
            for track, m in rs:
                is_ghost = (ghost is None) or (track in ghost)
                n = len(judge(m, use))
                bad = n >= need
                if is_ghost:
                    tp += bad; fn += not bad
                else:
                    fp += bad; tn += not bad
        lines.append('need=%d 判据=%-18s 真人误杀=%5.1f%% (%3d/%3d)  幻觉召回=%5.1f%% (%3d/%3d)' % (
            need, ''.join(use), 100 * fp / max(1, fp + tn), fp, fp + tn,
            100 * tp / max(1, tp + fn), tp, tp + fn))

    lines.append('\n漏网的幻觉帧（判据 a-f、need=1 仍通过）的指标')
    for vid, rs in data.items():
        ghost = GHOST_TRACKS.get(vid, set())
        for track, m in rs:
            is_ghost = (ghost is None) or (track in ghost)
            if is_ghost and not judge(m):
                lines.append('%-20s %-11s sw/w=%.3f t/sw=%.2f h/sw=%.2f arm=%.2f box=%s scale=%s' % (
                    vid[:18], track, m['sw/w'], m['t/sw'], m['h/sw'], m['arm/sw'],
                    '-' if m['box/sw'] is None else '%.1f' % m['box/sw'],
                    '-' if m['sw/palm'] is None else '%.2f' % m['sw/palm']))
    (ROOT / '.tmp' / 'pose_gate_eval.txt').write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    main()
