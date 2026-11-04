"""在临时目录重跑指定视频，检查 pose 闸门是否把「只有手被检成人」清掉。"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.pipeline.analyze import analyze_video  # noqa: E402

OUT = ROOT / '.tmp' / 'gatecheck'


def main(video_ids):
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for vid in video_ids:
        src = ROOT / 'data' / 'videos' / f'{vid}.normalized.mp4'
        if not src.exists():
            src = ROOT / 'data' / 'videos' / f'{vid}.mp4'
        target = OUT / vid
        result = analyze_video(src, target, vid, lambda stage, pct: None, checkpoint_directory=target)
        cov = result['coverage']
        lines.append('%s  scenes=%s' % (vid[:20], [(s['framing_type'], s['person_count'], s['confidence']) for s in result['scenes']]))
        lines.append('   pose_frames=%s hand_frames=%s' % (cov['pose_frames'], cov['hand_frames']))
        lines.append('   pose_quality=%s' % json.dumps(cov['pose_quality'], ensure_ascii=False))
        lines.append('   warnings=%s' % json.dumps(result['warnings'], ensure_ascii=False))
        (ROOT / '.tmp' / 'gatecheck.txt').write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    main(sys.argv[1:])
