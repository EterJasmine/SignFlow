"""比较 probe 阶段不同设置下骨架判定的差异：真人素材应保留、纯手素材应全丢。

变量：检测分辨率 / 静态 detect 还是视频时序 detect_for_video / 是否 CLAHE。
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from packages.pipeline.analyze import prepare_frame, points
from packages.pipeline.quality import pose_assess, hand_reference

REAL = 'video_173988cb50e24b3c9d308c1de54ab323'
HAND = 'video_52f9b2de269243ff95a50dc88af22896'
MODELS = ROOT / 'models'


def sample(video_id, size, video_mode, enhance):
    os.environ['GESTUREFLOW_ENHANCE'] = '1' if enhance else '0'
    src = ROOT / 'data' / 'videos' / f'{video_id}.normalized.mp4'
    if not src.exists():
        src = ROOT / 'data' / 'videos' / f'{video_id}.mp4'
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps * 1000)
    keep = drop = none = 0
    pose_kwargs = dict(base_options=python.BaseOptions(model_asset_buffer=(MODELS / 'pose_landmarker.task').read_bytes()),
                       num_poses=4)
    hand_kwargs = dict(base_options=python.BaseOptions(model_asset_buffer=(MODELS / 'hand_landmarker.task').read_bytes()),
                       num_hands=4)
    if video_mode:
        pose_kwargs['running_mode'] = vision.RunningMode.VIDEO
        hand_kwargs['running_mode'] = vision.RunningMode.VIDEO
    with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(**hand_kwargs)) as ph, \
            vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(**pose_kwargs)) as pp:
        for t in range(0, duration, 500):
            cap.set(cv2.CAP_PROP_POS_MSEC, t); ok, bgr = cap.read()
            if not ok:
                continue
            rgb, _ = prepare_frame(bgr, size)
            hh, ww = rgb.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            if video_mode:
                pr = pp.detect_for_video(image, t); hr = ph.detect_for_video(image, t)
            else:
                pr = pp.detect(image); hr = ph.detect(image)
            hands = [points(h) for h in hr.hand_landmarks]
            palm, box = hand_reference([{'landmarks': h} for h in hands], ww, hh)
            if not pr.pose_landmarks:
                none += 1
                continue
            for pose in pr.pose_landmarks:
                v = pose_assess(points(pose), ww, hh, palm, box)
                if v['plausible']:
                    keep += 1
                else:
                    drop += 1
    cap.release()
    return keep, drop, none


def main():
    lines = ['视频  尺寸 模式  CLAHE   keep drop none']
    for vid, label in [(REAL, 'REAL'), (HAND, 'HAND')]:
        for size in (640, 960):
            for video_mode in (False, True):
                for enhance in (True, False):
                    k, d, n = sample(vid, size, video_mode, enhance)
                    lines.append('%-5s %4d %-5s %-5s  %4d %4d %4d' % (
                        label, size, 'video' if video_mode else 'image', 'on' if enhance else 'off', k, d, n))
    (ROOT / '.tmp' / 'probe_matrix.txt').write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    main()
