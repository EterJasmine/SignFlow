"""生成「修复前 / 修复后」骨架对比图：同一帧，旧结果把手指当四肢，新结果只留手。

旧结果取自 result_versions 里的历史 revision，新结果取当前 rerun 版本。
"""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
VIDEO = 'video_52f9b2de269243ff95a50dc88af22896'
OLD_REVISION = 'revision_816b3812c20d492ea4288ea3f91c2cd0'
MOMENTS = (2000, 4500)
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
BODY_COLOR = (178, 152, 216, 255)
HAND_COLOR = (72, 202, 182, 255)
CELL = (420, 262)
FONT = 'C:/Windows/Fonts/msyh.ttc'


def load(video_id, revision=None):
    base = ROOT / 'data' / 'assets' / video_id
    if revision:
        return json.loads((base / 'history' / revision / 'result.json').read_text(encoding='utf8'))
    latest = sorted((base / 'versions').glob('rerun_*'), key=lambda p: p.stat().st_mtime)[-1]
    return json.loads((latest / 'result.json').read_text(encoding='utf8'))


def frame_at(result, timestamp):
    return min(result['frames'], key=lambda f: abs(f['timestamp_ms'] - timestamp))


def overlay_skeleton(frame_data, size):
    W, H = size
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for track in frame_data['tracks']:
        layers = [(track['pose_landmarks'], BODY_EDGES, BODY_COLOR)] + \
                 [(hand['landmarks'], HAND_EDGES, HAND_COLOR) for hand in track['hands'].values()]
        for landmarks, edges, color in layers:
            for a, b in edges:
                if max(a, b) >= len(landmarks):
                    continue
                p, q = landmarks[a], landmarks[b]
                if p.get('visibility') is not None and p['visibility'] < .5:
                    continue
                draw.line([(p['x'] * W, p['y'] * H), (q['x'] * W, q['y'] * H)], fill=color, width=2)
    return layer


def main():
    old, new = load(VIDEO, OLD_REVISION), load(VIDEO)
    cap = cv2.VideoCapture(str(ROOT / 'data' / 'videos' / f'{VIDEO}.normalized.mp4'))
    W, H = CELL
    pad, gap, top = 12, 10, 46
    canvas = Image.new('RGB', (pad * 2 + W * 2 + gap, top + (H + gap) * len(MOMENTS) + pad), (246, 243, 250))
    title = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 14)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), '修复前：明明只有手，却叠出一副人体骨架', font=title, fill=(58, 48, 78))
    draw.text((pad + W + gap, 14), '修复后：骨架被丢弃，只保留手部', font=title, fill=(58, 48, 78))
    for row, t in enumerate(MOMENTS):
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ok, bgr = cap.read()
        if not ok:
            continue
        base = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((W, H), Image.LANCZOS)
        for col, result in enumerate((old, new)):
            frame_data = frame_at(result, t)
            cell = Image.alpha_composite(base.convert('RGBA'), overlay_skeleton(frame_data, CELL)).convert('RGB')
            x, y = pad + col * (W + gap), top + row * (H + gap)
            canvas.paste(cell, (x, y))
            draw.rectangle([x, y, x + W - 1, y + H - 1], outline=(210, 202, 226))
            draw.text((x + 8, y + H - 24), '%.2fs · 人体 %d 条 / 手 %d 只' % (
                t / 1000, sum(1 for tr in frame_data['tracks'] if tr['pose_landmarks']),
                sum(len(tr['hands']) for tr in frame_data['tracks'])), font=small, fill=(255, 255, 255))
    cap.release()
    out = ROOT / 'evaluation' / 'results' / 'pose-gate-compare.jpg'
    canvas.save(out, quality=92)
    print(out)


if __name__ == '__main__':
    main()
