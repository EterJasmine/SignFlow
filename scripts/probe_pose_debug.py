"""诊断 probe 阶段的 pose 判定：逐 500ms 打印骨架比例与被拒理由。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from packages.pipeline.analyze import prepare_frame, points
from packages.pipeline.quality import pose_assess, hand_reference


def run(video_id, out):
    src = ROOT / 'data' / 'videos' / f'{video_id}.normalized.mp4'
    if not src.exists():
        src = ROOT / 'data' / 'videos' / f'{video_id}.mp4'
    models = ROOT / 'models'
    lines = []
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS); count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = round(count / fps * 1000)
    with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=(models / 'hand_landmarker.task').read_bytes()),
            num_hands=4)) as ph, vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=(models / 'pose_landmarker.task').read_bytes()),
                num_poses=4)) as pp:
        for t in range(0, duration, 500):
            cap.set(cv2.CAP_PROP_POS_MSEC, t); ok, bgr = cap.read()
            if not ok:
                continue
            rgb, _ = prepare_frame(bgr, 640)
            hh, ww = rgb.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            pr = pp.detect(image); hr = ph.detect(image)
            hands = [points(h) for h in hr.hand_landmarks]
            palm, box = hand_reference([{'landmarks': h} for h in hands], ww, hh)
            for pose in pr.pose_landmarks:
                v = pose_assess(points(pose), ww, hh, palm, box)
                m = v['metrics']
                lines.append('t=%5d %-5s sw/w=%.3f torso=%.2f hip=%.2f arm=%.2f swpalm=%s %s' % (
                    t, 'KEEP' if v['plausible'] else 'DROP', m['shoulder_over_width'],
                    m['torso_over_shoulder'], m['hip_over_shoulder'], m['arm_over_shoulder'],
                    m.get('shoulder_over_palm'), v['reasons']))
            if not pr.pose_landmarks:
                lines.append('t=%5d 无骨架  hands=%d' % (t, len(hands)))
    cap.release()
    Path(out).write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
