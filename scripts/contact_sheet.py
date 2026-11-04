"""把视频按固定帧率抽帧拼成带时间戳的接触表（contact sheet），用于人工/机器判读动作边界。

用法：
    python scripts/contact_sheet.py <video> --fps 2 --out .tmp/sheet
    python scripts/contact_sheet.py <video> --fps 10 --start 4.0 --end 7.0 --out .tmp/fine

每张表 5 列 × 4 行 = 20 帧，帧左下角标注该帧在视频中的秒数。
只读视频，不改任何工程数据。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 5, 4
THUMB_W = 360
FONT_PATH = r"C:/Windows/Fonts/msyh.ttc"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def grab(video: Path, fps: float, start: float, end: float) -> list[tuple[float, Image.Image]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{video}")
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / native if total else 0.0
    if end <= 0 or end > duration:
        end = duration

    step = max(int(round(native / fps)), 1)
    first = int(start * native)
    frames: list[tuple[float, Image.Image]] = []
    idx = first
    while idx < int(end * native):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, raw = cap.read()
        if not ok:
            break
        ts = idx / native
        h, w = raw.shape[:2]
        thumb = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        thumb = thumb.resize((THUMB_W, int(THUMB_W * h / w)), Image.LANCZOS)
        frames.append((ts, thumb))
        idx += step
    cap.release()
    return frames


def build_sheets(frames: list[tuple[float, Image.Image]], outdir: Path, tag: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    font = _font(22)
    per = COLS * ROWS
    paths: list[Path] = []
    for page in range(0, len(frames), per):
        chunk = frames[page:page + per]
        tw, th = chunk[0][1].size
        pad = 6
        label_h = 26
        cell_h = th + label_h
        sheet = Image.new("RGB", (COLS * (tw + pad) + pad, ROWS * (cell_h + pad) + pad), (28, 28, 30))
        draw = ImageDraw.Draw(sheet)
        for i, (ts, img) in enumerate(chunk):
            r, c = divmod(i, COLS)
            x = pad + c * (tw + pad)
            y = pad + r * (cell_h + pad)
            sheet.paste(img, (x, y))
            text = f"{ts:6.2f}s"
            draw.rectangle([x, y + th, x + tw, y + cell_h], fill=(16, 16, 18))
            draw.text((x + 6, y + th + 2), text, font=font, fill=(255, 214, 102))
        p = outdir / f"{tag}_p{page // per + 1:02d}.jpg"
        sheet.save(p, quality=88)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0, help="0 = 到片尾")
    ap.add_argument("--out", default=".tmp/contact")
    ap.add_argument("--tag", default="")
    ap.add_argument("--thumb", type=int, default=360, help="每帧宽度（像素）")
    args = ap.parse_args()

    global THUMB_W
    THUMB_W = args.thumb
    video = Path(args.video)
    frames = grab(video, args.fps, args.start, args.end)
    tag = args.tag or video.stem.replace(" ", "_")[:24]
    paths = build_sheets(frames, Path(args.out), tag)
    print(f"{len(frames)} 帧 / {len(paths)} 张表")
    for p in paths:
        print(p.resolve())


if __name__ == "__main__":
    main()
