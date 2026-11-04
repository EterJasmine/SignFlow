"""Real-time capture that follows the offline pipeline's landmark and feature path.

The demonstration library stores features produced by `extract_features` over the
same MediaPipe task models. Live recognition only stays comparable if it runs the
same models, the same hand association and the same feature code, so this module
deliberately imports the offline helpers instead of re-implementing them.
"""
from __future__ import annotations

import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .analyze import Tracker, credible_poses, points, prepare_frame
from .features import extract_features
from .geometry import distance, palm_scale
from .quality import assess,usable_for_recognition
from .tracking_v2 import PredictiveHands

BUFFER_MS = 8000


def skeleton(tracks) -> list[dict]:
    """Keypoints of every detected hand in one frame, for drawing on the camera preview.

    Only normalised x/y are sent: the preview is a live overlay, not a record, so
    depth and confidence would be dead weight on a request that runs at 10 fps.
    """
    hands = []
    for track in tracks or []:
        for side, hand in (track.get('hands') or {}).items():
            landmarks = hand.get('landmarks') or []
            if len(landmarks) != 21 or hand.get('interpolated'):
                continue
            hands.append({'side': side,
                          'suspect': bool((hand.get('quality') or {}).get('suspect')),
                          'points': [[round(point['x'], 4), round(point['y'], 4)] for point in landmarks]})
    return hands


def wrists(frame: dict, width: int, height: int) -> list[tuple[dict, float]]:
    """Wrist position plus palm scale in pixels for every usable hand in one frame."""
    found = []
    for track in frame.get('tracks') or []:
        for hand in (track.get('hands') or {}).values():
            if hand.get('interpolated') or not usable_for_recognition(hand):
                continue
            landmarks = hand.get('landmarks') or []
            if len(landmarks) != 21:
                continue
            found.append((landmarks[0], palm_scale(landmarks, width, height)))
    return found


class LiveCapture:
    """One camera stream held in memory; frames are pushed and never stored to disk."""

    def __init__(self, models: Path, max_side: int = 512, pose_every: int = 4):
        self.max_side = max_side
        # 人体检测只用来判解剖左右，而手部槽位优先用模型给的左右手标签（见 features.slot_key），
        # 所以它在实时链路里是可选信息。实测单帧 18ms、占整条链路六分之一，隔几帧跑一次足够。
        self.pose_every = max(1, pose_every)
        self.pose_seen = 0
        self.pose_points: list = []
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=(models / 'hand_landmarker.task').read_bytes()),
            running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=.5))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=(models / 'pose_landmarker.task').read_bytes()),
            running_mode=vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=.5))
        self.body = Tracker()
        self.hands = None
        self.frames: list[dict] = []
        self.palms: dict = {}
        self.pose_stats = {'observed': 0, 'rejected': 0, 'reasons': {}}
        self.width = self.height = 0
        self.lock = threading.Lock()

    def push(self, bgr, timestamp_ms: int) -> dict:
        with self.lock:
            return self._push(bgr, timestamp_ms)

    def _push(self, bgr, timestamp_ms: int) -> dict:
        rgb, sharpness = prepare_frame(bgr, self.max_side)
        height, width = rgb.shape[:2]
        if (width, height) != (self.width, self.height):
            self.width, self.height = width, height
            self.hands = PredictiveHands(width, height)
            self.palms.clear()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        found = self.hand.detect_for_video(image, timestamp_ms)
        detected = []
        for index, landmarks in enumerate(found.hand_landmarks):
            label = found.handedness[index][0]
            detected.append({'detected': True, 'landmarks': points(landmarks),
                             'model_handedness': label.category_name, 'handedness_score': float(label.score),
                             'detection_source': 'live'})
        poses = []
        if detected and self.pose_seen % self.pose_every == 0:
            result = self.pose.detect_for_video(image, timestamp_ms)
            self.pose_points = [points(p) for p in result.pose_landmarks]
        self.pose_seen += 1
        # 每帧都重新过一遍可信度判定（纯几何，很便宜），但只在刚跑过模型的那帧才有新点位。
        if detected:
            poses = credible_poses([list(p) for p in self.pose_points], width, height,
                                   [item['landmarks'] for item in detected], self.pose_stats)
        tracks = self.hands.assign(self.body.update(poses), detected, timestamp_ms)
        suspect = False
        for track in tracks:
            for side, hand in track['hands'].items():
                if hand.get('interpolated'):
                    continue
                key = (track['track_id'], side)
                quality = assess(hand['landmarks'], width, height, self.palms.get(key))
                hand['quality'] = quality
                if quality['geometry'] and not quality['suspect']:
                    self.palms[key] = quality['geometry']['palm_px']
                suspect = suspect or bool(quality['suspect'])
        frame = {'timestamp_ms': timestamp_ms, 'scene_id': 'live', 'tracks': tracks,
                 'coordinate_system': 'normalized_image',
                 'image_size': {'width': width, 'height': height},
                 'sharpness': sharpness, 'evidence': {}}
        self.frames.append(frame)
        cutoff = timestamp_ms - BUFFER_MS
        while self.frames and self.frames[0]['timestamp_ms'] < cutoff:
            self.frames.pop(0)
        return {'hands': len(detected), 'suspect': suspect, 'sharpness': sharpness,
                'size': [width, height], 'skeleton': skeleton(tracks)}

    def wrist_motion(self, window: int = 4) -> float:
        """Wrist travel in palm-lengths per second over the last few frames.

        Measured as displacement between the oldest and newest frame in the window,
        not as a per-frame difference: MediaPipe jitters a few pixels even on a
        frozen image, and dividing that by a short interval turns noise into
        motion (3 px over 70 ms reads as 0.9 palm/s, which is above the "hand is
        moving" threshold). Over a few hundred milliseconds the jitter mostly
        cancels while real movement accumulates.
        """
        if len(self.frames) < 2:
            return 0.0
        first = self.frames[-min(len(self.frames), window)]
        last = self.frames[-1]
        seconds = (last['timestamp_ms'] - first['timestamp_ms']) / 1000
        if seconds <= 0:
            return 0.0
        def indexed(frame):
            return {(tr['track_id'],side):hand for tr in frame.get('tracks',[]) for side,hand in tr.get('hands',{}).items()
                    if len(hand.get('landmarks',[]))==21 and not hand.get('interpolated') and usable_for_recognition(hand)}
        old=indexed(first);new=indexed(last);speeds=[]
        for ident in old.keys() & new.keys():
            a=old[ident]['landmarks'];b=new[ident]['landmarks']
            scale=min(palm_scale(a,self.width,self.height),palm_scale(b,self.width,self.height))
            if not scale:continue
            wrist=distance(a[0],b[0],self.width,self.height)/scale/seconds
            tips=float(np.median([distance(a[k],b[k],self.width,self.height)/scale/seconds for k in (4,8,12,16,20)]))
            value=max(wrist,tips)
            if value<=8:speeds.append(value)
        if not speeds:
            return 0.0
        speeds.sort()
        return speeds[len(speeds) // 2]

    def window(self, start_ms: int, end_ms: int) -> list[dict]:
        return [f for f in self.frames if start_ms <= f['timestamp_ms'] <= end_ms]

    def features(self, start_ms: int, end_ms: int) -> dict:
        return extract_features(self.window(start_ms, end_ms), self.width, self.height)

    def reset(self):
        with self.lock:
            self.frames.clear()
            self.palms.clear()
            self.pose_points = []
            self.pose_seen = 0
            self.body = Tracker()
            if self.hands:
                self.hands.reset()

    def close(self):
        for model in (self.hand, self.pose):
            try:
                model.close()
            except Exception:
                pass
