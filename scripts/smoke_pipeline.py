"""Real detector smoke test using Google's pose fixture, not an accuracy benchmark."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from packages.pipeline.analyze import analyze_video

root=Path(__file__).resolve().parents[1]
folder=root/'data'/'smoke';folder.mkdir(parents=True,exist_ok=True)
image=cv2.imdecode(np.fromfile(str(root/'tests'/'fixtures'/'pose.jpg'),dtype=np.uint8),cv2.IMREAD_COLOR)
assert image is not None
h,w=image.shape[:2];w=w//2*2;h=h//2*2;image=cv2.resize(image,(w,h))
path=folder/'official-pose.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(w,h))
assert writer.isOpened()
for _ in range(24):writer.write(image)
writer.release()
result=analyze_video(path,folder/'result','smoke',lambda stage,progress:print(stage,progress,flush=True))
assert result['frames']
assert any(t['pose_landmarks'] for f in result['frames'] for t in f['tracks'])
assert result['actions'][0]['assets']['diagram']
print({'scenes':[(s['framing_type'],s['person_count']) for s in result['scenes']],'frames':len(result['frames']),'actions':len(result['actions'])})
