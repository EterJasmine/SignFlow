"""Render a shareable tutorial sheet from an analysis result.

Three outputs from the same data: a printable HTML page (browser saves PDF),
a long PNG for sharing, and a per-step A4 PDF. Everything shown here already
exists in the result: real keyframes, landmark diagrams and rule-based text.
No motion is described that is not in the stored facts.
"""
from __future__ import annotations

import base64
import html
import io
import os
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
BODY_EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
STATUS = {'pending':'待审核','approved':'已确认','rejected':'需修正'}
LAYOUT = {'hands_closeup':'手部特写','upper_body_figure':'上半身图解','full_body_figure':'全身图解','visible_region':'可见区域','source_frame':'原片取景'}
PURPLE = (111, 91, 181)
INK = (42, 38, 53)
MUTED = (124, 117, 137)
LINE = (231, 227, 240)
PAPER = (251, 250, 253)
FONT_PATHS = [('C:/Windows/Fonts/msyhbd.ttc','C:/Windows/Fonts/msyh.ttc'),('/System/Library/Fonts/PingFang.ttc','/System/Library/Fonts/PingFang.ttc'),('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
WIDTH = 1240


def clock(ms: int) -> str:
    return f'{ms/1000:.2f}s'


def _font(size: int, bold: bool = False):
    custom = os.environ.get('GESTUREFLOW_FONT')
    candidates = ([custom, custom] if custom else []) + [pair[0 if bold else 1] for pair in FONT_PATHS]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def asset_file(url: str | None, asset_root: Path) -> Path | None:
    """Map a stored /assets/... URL back to disk, refusing path traversal."""
    if not url or '/assets/' not in url:
        return None
    relative = url.split('/assets/', 1)[1]
    candidate = (asset_root / relative).resolve()
    try:
        candidate.relative_to(Path(asset_root).resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def steps_of(result: dict) -> list[dict]:
    """Actions carry the editable text; tutorial_steps add layout hints."""
    layouts = {s['action_id']: s.get('layout') for s in result.get('tutorial_steps', []) if s.get('action_id')}
    lyrics = {l['id']: l for l in result.get('lyrics', [])}
    steps = []
    for action in result.get('actions', []):
        lines = [lyrics[i]['text'] for i in action.get('lyric_ids', []) if i in lyrics]
        steps.append({'action': action, 'lyric': ' / '.join(l for l in lines if l), 'layout': layouts.get(action['id']) or 'source_frame'})
    return steps


def summary(result: dict, video: dict | None) -> dict:
    actions = result.get('actions', [])
    name = Path((video or {}).get('filename') or '手势舞教程').stem or '手势舞教程'
    return {'title': name,
            'duration': clock(result.get('duration_ms', 0)),
            'count': len(actions),
            'approved': sum(1 for a in actions if a.get('review_status') == 'approved'),
            'created': time.strftime('%Y-%m-%d %H:%M'),
            'resolution': f"{result.get('width', 0)}×{result.get('height', 0)}",
            'sample_fps': round(float(result.get('sample_fps', 0) or 0), 1)}


# ---------------------------------------------------------------- HTML sheet

def _svg_inline(path: Path | None) -> str:
    if not path:
        return ''
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return ''
    return text if text.lstrip().startswith('<svg') else ''


def _crop_svg(svg: str, result: dict, action: dict, pad: float = .16, hands_only: bool = False) -> str:
    """Zoom the full-frame diagram onto the detected landmarks.

    Only the viewBox changes: every drawn coordinate stays exactly where the
    analysis put it, so the diagram still matches the stored keyframe.
    ``hands_only`` narrows the box to the hands and squares it, which is what
    makes a close-up skeleton panel readable in print.
    """
    frame = _nearest_frame(result, action.get('keyframe_ms', action.get('start_ms', 0)))
    if not frame:
        return svg
    width, height = float(result.get('width') or 0), float(result.get('height') or 0)
    if not width or not height:
        return svg
    points, hand_points = [], []
    for track in frame.get('tracks', []):
        body = [p for p in track.get('pose_landmarks') or [] if p.get('visibility') is None or p['visibility'] >= .5]
        hands = [[p for p in hand.get('landmarks') or [] if p.get('visibility') is None or p['visibility'] >= .5]
                 for hand in (track.get('hands') or {}).values()]
        points += body + [p for hand in hands for p in hand]
        hand_points += [p for hand in hands for p in hand]
    usable = lambda group: [p for p in group if 0 <= p['x'] <= 1 and 0 <= p['y'] <= 1 and not p.get('low_quality')]
    if hands_only and len(usable(hand_points)) >= 4:
        points, pad = hand_points, max(pad, .32)
    points = usable(points)
    if len(points) < 2:
        return svg
    left = min(p['x'] for p in points) * width; right = max(p['x'] for p in points) * width
    top = min(p['y'] for p in points) * height; bottom = max(p['y'] for p in points) * height
    margin_x, margin_y = (right - left) * pad + 24, (bottom - top) * pad + 24
    box = [max(0, left - margin_x), max(0, top - margin_y), min(width, right + margin_x), min(height, bottom + margin_y)]
    if hands_only:
        # Square the box so the panel is not a thin strip when two hands sit side by side.
        span = max(box[2] - box[0], box[3] - box[1])
        for index, limit in ((0, width), (1, height)):
            centre = (box[index] + box[index + 2]) / 2
            box[index] = max(0, min(limit - span, centre - span / 2))
            box[index + 2] = box[index] + span
    crop = f'viewBox="{box[0]:.1f} {box[1]:.1f} {max(1, box[2]-box[0]):.1f} {max(1, box[3]-box[1]):.1f}"'
    return re.sub(r'viewBox="[^"]*"', crop, svg, count=1)


def _add_joints(svg: str, radius: float = 3.4) -> str:
    """Dot every skeleton line endpoint so the joints read as points, not crossings."""
    segments = re.findall(
        r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"[^>]*?stroke="(#[0-9a-fA-F]{6})"',
        svg)
    seen, dots = set(), []
    for x1, y1, x2, y2, colour in segments:
        for x, y in ((x1, y1), (x2, y2)):
            key = (x, y, colour)
            if key in seen:
                continue
            seen.add(key)
            dots.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{colour}"/>')
    return svg.replace('</svg>', ''.join(dots) + '</svg>') if dots else svg



def _img_src(path: Path | None) -> str:
    if not path:
        return ''
    try:
        return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return ''


def _sample_video(result: dict, timestamp: int, asset_root: Path) -> Image.Image | None:
    """Grab one frame from the stored source video when no still was saved for it."""
    try:
        import cv2
    except ImportError:
        return None
    ident = result.get('video_id') or ''
    if not ident:
        return None
    folder = Path(asset_root).parent / 'videos'
    for name in (f'{ident}.normalized.mp4', f'{ident}.mp4'):
        path = folder / name
        if not path.is_file():
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            continue
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp))
            ok, frame = capture.read()
        finally:
            capture.release()
        if ok:
            return Image.fromarray(frame[:, :, ::-1]).convert('RGB')
    return None


def _frame_overlay(result: dict, url: str | None, timestamp: int, asset_root: Path, width: int = 640) -> str:
    """One frame of this step with the landmarks recorded at that moment drawn on it."""
    path = asset_file(url, asset_root)
    image = None
    if path:
        try:
            image = Image.open(path).convert('RGB')
        except OSError:
            image = None
    if image is None:
        image = _sample_video(result, timestamp, asset_root)
    if image is None:
        return ''
    image.thumbnail((width, width * 2), Image.LANCZOS)
    image = _skeleton(image, _nearest_frame(result, timestamp))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=84)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def _overlay_src(result: dict, action: dict, asset_root: Path, width: int = 640) -> str:
    """Keyframe with the stored landmarks drawn on it, as a self-contained JPEG."""
    return _frame_overlay(result, (action.get('assets') or {}).get('source'),
                          action.get('keyframe_ms', action.get('start_ms', 0)), asset_root, width)


def _timecode_strip(result: dict, action: dict, asset_root: Path, index: int) -> str:
    """Start / representative / end frames in a row: the step as a movement, not a pose."""
    assets = action.get('assets') or {}
    start = action.get('start_ms', 0)
    end = max(action.get('end_ms', start + 1), start + 1)
    marks = [('起', action.get('start_frame_ms', start), assets.get('start_source')),
             ('代表姿态', action.get('keyframe_ms', start), assets.get('source')),
             ('止', action.get('end_frame_ms', end), assets.get('end_source'))]
    figures = []
    for label, moment, url in marks:
        source = _frame_overlay(result, url, moment, asset_root, width=520)
        if not source:
            continue
        figures.append(f'<figure><img src="{source}" alt="步骤 {index} {label}">'
                       f'<figcaption><b>{label}</b> {clock(moment)}</figcaption></figure>')
    return ''.join(figures) if len(figures) >= 2 else ''


def _timeline(action: dict) -> str:
    """A bar showing where the transitions and the representative pose sit in the step."""
    start = action.get('start_ms', 0)
    end = max(action.get('end_ms', start + 1), start + 1)
    span = end - start
    position = lambda moment: max(0.0, min(100.0, (moment - start) / span * 100))

    def dot(moment: float, kind: str) -> str:
        return f'<i class="tl-dot {kind}" style="left:{position(moment):.1f}%"></i>'

    transitions = action.get('transition_frames_ms') or []
    body = ''.join(dot(t, 'transition') for t in transitions)
    body += dot(action.get('keyframe_ms', start), 'key')
    body += dot(start, 'edge') + dot(end, 'edge')
    middle = ' · '.join(clock(t) for t in transitions) or '无过渡帧'
    legend = f'起 {clock(start)} · 过渡 {middle} · 代表姿态 {clock(action.get("keyframe_ms", start))} · 止 {clock(end)}'
    return f'<div class="tl"><span class="tl-bar"></span>{body}</div><p class="tl-legend">{legend}</p>'


def sheet_html(result: dict, video: dict | None, asset_root: Path) -> str:
    info = summary(result, video)
    cards = []
    for index, step in enumerate(steps_of(result), 1):
        action = step['action']
        assets = action.get('assets') or {}
        frame = _overlay_src(result, action, asset_root) or _img_src(asset_file(assets.get('source'), asset_root))
        diagram = _crop_svg(_add_joints(_svg_inline(asset_file(assets.get('diagram'), asset_root))), result, action, hands_only=True)
        status = STATUS.get(action.get('review_status'), '待审核')
        if diagram:
            visual = f'<figure class="diagram">{diagram}<figcaption>骨骼图解 · 关键点均来自原片检测</figcaption></figure>'
        elif frame:
            visual = f'<figure><img src="{frame}" alt="步骤 {index} 原片关键帧"><figcaption>原片关键帧</figcaption></figure>'
        else:
            visual = '<div class="none">无关键帧</div>'
        strip = _timecode_strip(result, action, asset_root, index)
        strip_block = f'<div class="strip">{strip}</div>' if strip else ''
        timeline = _timeline(action)
        lyric = f'<p class="lyric">♫ {html.escape(step["lyric"])}</p>' if step['lyric'] else '<p class="lyric muted">间奏 · 未关联歌词</p>'
        cards.append(f'''<article class="card">
  <header><b>STEP {index:02d}</b><span>{clock(action.get('start_ms', 0))} – {clock(action.get('end_ms', 0))}</span><em class="chip">{status}</em></header>
  <div class="body">
    <div class="visual">{visual}</div>
    <div class="text">
      <h3>{html.escape(action.get('title') or '未命名动作')}</h3>
      {lyric}
      <p class="desc">{html.escape(action.get('description') or '')}</p>
      <p class="meta">代表姿态 {clock(action.get('keyframe_ms', 0))} · 图解布局 {LAYOUT.get(step['layout'], step['layout'])}</p>
    </div>
  </div>
  {strip_block}
  {timeline}
</article>''')
    warnings = ''.join(f'<li>{html.escape(w)}</li>' for w in result.get('warnings', []))
    limitations = ''.join(f'<li>{html.escape(l)}</li>' for l in result.get('limitations', []))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>{html.escape(info['title'])} · GestureFlow 教程</title>
<style>
@page{{size:A4;margin:12mm}}
*{{box-sizing:border-box}}
body{{margin:0;padding:28px;background:#fbfafd;color:#2a2635;font:14px/1.7 "Microsoft YaHei","Noto Sans SC",sans-serif}}
.sheet{{max-width:900px;margin:0 auto}}
.cover{{background:linear-gradient(140deg,#f4efff,#fdf8f1);border:1px solid #e7e3f0;border-radius:18px;padding:26px 30px;margin-bottom:18px}}
.eyebrow{{font-size:10px;letter-spacing:2.4px;color:#9a8ab4;font-weight:700}}
h1{{font-size:26px;margin:8px 0 6px}}
.cover p{{margin:0;color:#6f6580;font-size:12px}}
.card{{background:#fff;border:1px solid #e7e3f0;border-radius:16px;padding:18px 20px;margin-bottom:14px;break-inside:avoid;page-break-inside:avoid}}
.card header{{display:flex;align-items:center;gap:14px;margin-bottom:12px}}
.card header b{{color:#6f5bb5;letter-spacing:1px;font-size:13px}}
.card header span{{color:#7c7589;font-size:12px}}
.chip{{margin-left:auto;font-style:normal;font-size:11px;color:#6f5bb5;background:#f0ecfb;border-radius:20px;padding:3px 10px}}
.body{{display:flex;gap:18px;align-items:flex-start}}
.visual{{width:300px;flex:0 0 300px}}
.visual figure{{margin:0 0 8px}}
.visual img{{width:100%;max-height:300px;object-fit:contain;border-radius:12px;display:block;border:1px solid #ece8f4;background:#fdfcff}}
.visual .diagram svg{{width:100%;height:248px;border-radius:12px;border:1px solid #ece8f4;background:#f7f4fb}}
.visual figcaption{{font-size:10px;color:#a49caf;margin-top:4px;text-align:center}}
.visual .none{{height:120px;display:flex;align-items:center;justify-content:center;color:#a49caf;background:#f7f5fb;border-radius:12px;font-size:12px}}
.strip{{display:flex;gap:10px;margin-top:14px}}
.strip figure{{margin:0;flex:1;min-width:0}}
.strip img{{width:100%;height:200px;object-fit:contain;border-radius:10px;border:1px solid #ece8f4;background:#f7f5fb;display:block}}
.strip figcaption{{font-size:10px;color:#a49caf;margin-top:4px;text-align:center}}
.strip figcaption b{{color:#6f5bb5;font-weight:500}}
.strip .none.small{{height:200px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#a49caf;background:#f7f5fb;border-radius:10px}}
.tl{{position:relative;height:12px;margin:10px 4px 0}}
.tl-bar{{position:absolute;left:0;right:0;top:5px;height:3px;border-radius:2px;background:#e7e3f0;display:block}}
.tl-dot{{position:absolute;top:1px;width:9px;height:9px;border-radius:50%;transform:translateX(-50%);display:block}}
.tl-dot.edge{{background:#6f5bb5}}
.tl-dot.key{{background:#df9955;width:11px;height:11px;top:0}}
.tl-dot.transition{{background:#b4b2a9;width:7px;height:7px;top:2px}}
.tl-legend{{font-size:10px;color:#a49caf;margin:3px 0 0;text-align:center}}
.text{{flex:1}}
.text h3{{margin:0 0 6px;font-size:18px}}
.lyric{{margin:0 0 8px;color:#6f5bb5;font-size:13px}}
.muted{{color:#a49caf;font-size:12px}}
.desc{{margin:0;font-size:13px;color:#3a3446}}
.meta{{margin:10px 0 0;font-size:11px;color:#a49caf}}
footer{{margin-top:20px;font-size:11px;color:#8c8598;border-top:1px solid #e7e3f0;padding-top:14px}}
footer ul{{margin:6px 0 0 18px;padding:0}}
@media print{{body{{background:#fff;padding:0}}.card,.cover{{box-shadow:none}}}}
</style></head><body><div class="sheet">
<section class="cover"><div class="eyebrow">GESTUREFLOW · 图文教程</div>
<h1>{html.escape(info['title'])}</h1>
<p>时长 {info['duration']} · {info['count']} 个步骤 · 已确认 {info['approved']} 步 · {info['resolution']} · 采样 {info['sample_fps']} fps · 生成于 {info['created']}</p></section>
{''.join(cards)}
<footer>步骤文字由关键点规则生成，取景与动作边界需人工复核；关键帧来自原片，图解为检测骨架。<p>已知限制：</p><ul>{limitations}</ul>{f'<p>本次提示：</p><ul>{warnings}</ul>' if warnings else ''}</footer>
</div></body></html>'''


# ------------------------------------------------------------ raster helpers

def _nearest_frame(result: dict, timestamp_ms: int) -> dict | None:
    frames = result.get('processed_frames') or result.get('frames') or []
    return min(frames, key=lambda f: abs(f.get('timestamp_ms', 0) - timestamp_ms), default=None)


def _skeleton(img: Image.Image, frame: dict | None) -> Image.Image:
    """Draw stored landmarks over the real keyframe; no pose is added or moved."""
    if not frame or not frame.get('tracks'):
        return img
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size
    unit = max(3, width // 150)
    for track in frame['tracks']:
        layers = [(track.get('pose_landmarks') or [], BODY_EDGES, (155, 130, 195, 220))]
        layers += [(hand.get('landmarks') or [], HAND_EDGES, (57, 168, 155, 235)) for hand in (track.get('hands') or {}).values()]
        for points, edges, color in layers:
            if not points:
                continue
            for a, b in edges:
                if max(a, b) >= len(points):
                    continue
                p, q = points[a], points[b]
                if any(x.get('low_quality') or (x.get('visibility') is not None and x['visibility'] < .5) for x in (p, q)):
                    continue
                draw.line([(p['x'] * width, p['y'] * height), (q['x'] * width, q['y'] * height)], fill=color, width=unit)
            for point in points:
                if point.get('low_quality') or (point.get('visibility') is not None and point['visibility'] < .5):
                    continue
                r = unit * 1.2
                draw.ellipse([point['x'] * width - r, point['y'] * height - r, point['x'] * width + r, point['y'] * height + r], fill=color)
    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


def _frame_image(result: dict, action: dict, asset_root: Path, box: tuple[int, int]) -> Image.Image:
    path = asset_file((action.get('assets') or {}).get('source'), asset_root)
    if path:
        try:
            img = Image.open(path).convert('RGB')
        except OSError:
            img = Image.new('RGB', box, (247, 245, 251))
    else:
        img = Image.new('RGB', box, (247, 245, 251))
    img = _skeleton(img, _nearest_frame(result, action.get('keyframe_ms', action.get('start_ms', 0))))
    return ImageOps.contain(img, box, Image.LANCZOS)


def _wrap(draw, text: str, font, width: int) -> list[str]:
    lines, line = [], ''
    for char in text or '':
        if char == '\n':
            lines.append(line); line = ''; continue
        if draw.textlength(line + char, font=font) > width and line:
            lines.append(line); line = char
        else:
            line += char
    if line:
        lines.append(line)
    return lines or ['']


def _text_block(draw, xy, text, font, fill, width, line_height=None):
    x, y = xy
    for line in _wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height or int(font.size * 1.55)
    return y


def long_image(result: dict, video: dict | None, asset_root: Path) -> Image.Image:
    info = summary(result, video)
    margin, pad, image_box = 56, 24, (400, 400)
    title_font, sub_font, step_font = _font(40, True), _font(19), _font(17, True)
    head_font, desc_font, meta_font = _font(30, True), _font(20), _font(16)
    measure = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    desc_line = int(desc_font.size * 1.55)

    cards = []
    for step in steps_of(result):
        action = step['action']
        text_width = WIDTH - margin * 2 - image_box[0] - pad * 2 - 36
        desc_lines = _wrap(measure, action.get('description') or '', desc_font, text_width)
        height = max(image_box[1] + 96, 236 + len(desc_lines) * desc_line)
        cards.append((step, height))
    header_height = 214
    footer_height = 128
    total = header_height + sum(h + 24 for _, h in cards) + footer_height
    canvas = Image.new('RGB', (WIDTH, total), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, WIDTH, 6], fill=PURPLE)
    draw.text((margin, 52), 'GESTUREFLOW · 图文教程', font=sub_font, fill=PURPLE)
    draw.text((margin, 82), info['title'][:42], font=title_font, fill=INK)
    draw.text((margin, 146), f"时长 {info['duration']} · {info['count']} 个步骤 · 已确认 {info['approved']} 步 · {info['resolution']} · 采样 {info['sample_fps']} fps · 生成于 {info['created']}", font=sub_font, fill=MUTED)

    y = header_height
    for index, (step, height) in enumerate(cards, 1):
        action = step['action']
        draw.rounded_rectangle([margin, y, WIDTH - margin, y + height], radius=20, fill=(255, 255, 255), outline=LINE, width=1)
        draw.text((margin + pad, y + 22), f'STEP {index:02d}', font=step_font, fill=PURPLE)
        draw.text((margin + pad + 92, y + 24), f"{clock(action.get('start_ms', 0))} – {clock(action.get('end_ms', 0))}", font=meta_font, fill=MUTED)
        chip = STATUS.get(action.get('review_status'), '待审核')
        chip_width = int(measure.textlength(chip, font=meta_font)) + 26
        draw.rounded_rectangle([WIDTH - margin - pad - chip_width, y + 20, WIDTH - margin - pad, y + 50], radius=15, fill=(240, 236, 251))
        draw.text((WIDTH - margin - pad - chip_width + 13, y + 25), chip, font=meta_font, fill=PURPLE)
        frame = _frame_image(result, action, asset_root, image_box)
        fx, fy = margin + pad, y + 76
        draw.rectangle([fx, fy, fx + image_box[0], fy + image_box[1]], fill=(250, 249, 253), outline=LINE)
        canvas.paste(frame, (fx + (image_box[0] - frame.width) // 2, fy + (image_box[1] - frame.height) // 2))
        tx = fx + image_box[0] + 36
        text_width = WIDTH - margin - pad - tx
        ty = _text_block(draw, (tx, y + 82), action.get('title') or '未命名动作', head_font, INK, text_width)
        ty += 6
        lyric = step['lyric'] or '间奏 · 未关联歌词'
        ty = _text_block(draw, (tx, ty), ('歌词 · ' if step['lyric'] else '') + lyric, sub_font, PURPLE if step['lyric'] else (164, 156, 175), text_width)
        ty += 10
        ty = _text_block(draw, (tx, ty), action.get('description') or '', desc_font, (58, 52, 70), text_width)
        _text_block(draw, (tx, ty + 12), f"代表姿态 {clock(action.get('keyframe_ms', 0))} · 骨架来自检测数据", meta_font, (164, 156, 175), text_width)
        y += height + 24

    y += 8
    draw.line([margin, y, WIDTH - margin, y], fill=LINE)
    _text_block(draw, (margin, y + 20), '说明：步骤文字由关键点规则生成，图取景与动作边界需人工复核；关键帧与骨架均来自原片检测结果，未做美化或补全。', sub_font, MUTED, WIDTH - margin * 2)
    return canvas


def pdf_pages(result: dict, video: dict | None, asset_root: Path) -> list[Image.Image]:
    info = summary(result, video)
    page = (1248, 1754)
    title_font, sub_font, step_font = _font(34, True), _font(19), _font(17, True)
    head_font, desc_font, meta_font = _font(30, True), _font(19), _font(16)
    pages = []
    steps = steps_of(result)
    for index, step in enumerate(steps, 1):
        action = step['action']
        canvas = Image.new('RGB', page, (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, page[0], 6], fill=PURPLE)
        draw.text((80, 64), 'GESTUREFLOW · 图文教程', font=sub_font, fill=PURPLE)
        draw.text((80, 96), info['title'][:40], font=title_font, fill=INK)
        draw.text((80, 148), f"第 {index} / {len(steps)} 步 · 时长 {info['duration']} · {info['created']}", font=sub_font, fill=MUTED)
        draw.text((80, 210), f'STEP {index:02d}', font=step_font, fill=PURPLE)
        draw.text((182, 212), f"{clock(action.get('start_ms', 0))} – {clock(action.get('end_ms', 0))}", font=meta_font, fill=MUTED)
        chip = STATUS.get(action.get('review_status'), '待审核')
        draw.rounded_rectangle([page[0] - 220, 206, page[0] - 80, 240], radius=17, fill=(240, 236, 251))
        draw.text((page[0] - 205, 212), chip, font=meta_font, fill=PURPLE)
        frame = _frame_image(result, action, asset_root, (1088, 900))
        canvas.paste(frame, (80 + (1088 - frame.width) // 2, 264 + (900 - frame.height) // 2))
        draw.rectangle([80, 264, 80 + 1088, 264 + 900], outline=LINE)
        y = _text_block(draw, (80, 1200), action.get('title') or '未命名动作', head_font, INK, 1088)
        y += 8
        y = _text_block(draw, (80, y), ('歌词 · ' if step['lyric'] else '') + (step['lyric'] or '间奏 · 未关联歌词'), sub_font, PURPLE if step['lyric'] else (164, 156, 175), 1088)
        y += 12
        _text_block(draw, (80, y), action.get('description') or '', desc_font, (58, 52, 70), 1088)
        draw.line([80, 1660, page[0] - 80, 1660], fill=LINE)
        draw.text((80, 1676), '文字为关键点规则推测，需对照原片核对；骨架绘制自检测数据。', font=meta_font, fill=MUTED)
        pages.append(canvas)
    return pages


def png_bytes(result: dict, video: dict | None, asset_root: Path) -> bytes:
    buffer = io.BytesIO()
    long_image(result, video, asset_root).save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


def pdf_bytes(result: dict, video: dict | None, asset_root: Path) -> bytes:
    pages = pdf_pages(result, video, asset_root)
    if not pages:
        pages = [Image.new('RGB', (1248, 1754), (255, 255, 255))]
    buffer = io.BytesIO()
    pages[0].save(buffer, format='PDF', save_all=True, append_images=pages[1:], resolution=150.0)
    return buffer.getvalue()


def sheet_name(result: dict, video: dict | None, suffix: str) -> str:
    stem = Path(summary(result, video)['title']).stem or '手势舞教程'
    return f'{stem}_教程.{suffix}'


def _stored_still(result: dict, timestamp: int, asset_root: Path, tolerance: int) -> Image.Image | None:
    """Reuse a keyframe the analysis already saved when it sits close to the wanted time."""
    best = None
    for action in result.get('actions') or []:
        assets = action.get('assets') or {}
        for moment, url in ((action.get('start_frame_ms'), assets.get('start_source')),
                            (action.get('keyframe_ms'), assets.get('source')),
                            (action.get('end_frame_ms'), assets.get('end_source'))):
            if moment is None or not url:
                continue
            gap = abs(int(moment) - timestamp)
            if gap > tolerance or (best and gap >= best[0]):
                continue
            path = asset_file(url, asset_root)
            if path:
                best = (gap, path)
    if not best:
        return None
    try:
        return Image.open(best[1]).convert('RGB')
    except OSError:
        return None


def preview_strip(result: dict, start_ms: int, end_ms: int, asset_root: Path,
                  count: int = 3, height: int = 200) -> bytes:
    """Stills sampled across one demonstration range, with its recorded landmarks drawn on top.

    Used by the sign library so a saved demonstration can be inspected without
    loading the whole video: the frames are read straight from the stored source
    file and the skeleton is the one saved with the analysis, never re-detected.
    """
    count = max(1, min(int(count), 5))
    span = max(1, end_ms - start_ms)
    stamps = [start_ms + round(span * index / max(1, count - 1)) for index in range(count)]
    tolerance = max(120, min(400, span // 4))
    shots = []
    for stamp in stamps:
        moment = min(end_ms - 1, max(start_ms, stamp))
        image = _stored_still(result, moment, asset_root, tolerance) or _sample_video(result, moment, asset_root)
        if image is None:
            continue
        image = _skeleton(image, _nearest_frame(result, stamp))
        image.thumbnail((height * 3, height), Image.LANCZOS)
        shots.append((stamp, image))
    if not shots:
        raise ValueError('取不到原视频画面：视频文件可能已被移动或不在本地')
    label_height = 24 if count > 1 else 0
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    font = _font(15)
    gap, pad = 8, 10
    width = sum(image.width for _, image in shots) + gap * (len(shots) - 1)
    canvas = Image.new('RGB', (width + pad * 2, max(i.height for _, i in shots) + label_height + pad * 2), PAPER)
    pen = ImageDraw.Draw(canvas)
    x = pad
    for stamp, image in shots:
        canvas.paste(image, (x, pad))
        if label_height:
            text = clock(stamp)
            pen.text((x + (image.width - draw.textlength(text, font=font)) / 2, pad + image.height + 4),
                     text, font=font, fill=MUTED)
        x += image.width + gap
    buffer = io.BytesIO()
    canvas.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()
