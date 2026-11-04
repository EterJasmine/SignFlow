from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'models'
MODELS = {
    'hand_landmarker.task': 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker.task': 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'face_landmarker.task': 'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
}
if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = ROOT / name
        if dest.exists() and dest.stat().st_size > 100_000:
            print('Already downloaded:', name)
            continue
        tmp = dest.with_suffix('.part')
        print('Downloading:', name, flush=True)
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/' + url, tmp)
        if tmp.stat().st_size < 100_000:
            raise RuntimeError('Invalid model file: ' + name)
        tmp.replace(dest)
        print('Ready:', name, dest.stat().st_size, flush=True)
