"""重算示范库里的手型序列，并用当前代码重写分析结果中的特征。

特征的定义一旦变化（例如 hand_0/hand_1 的槽位改成按左右手判定），旧模板与新算出
的实时窗口就不再可比：版本号不同会被直接跳过，表现为「一个词都认不出」。这个脚本
按当前代码把这两边重新算到同一版本。

用法：
    python scripts/rebuild_sign_sequences.py          # 只看会改动什么
    python scripts/rebuild_sign_sequences.py --apply  # 备份后写回
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.api.store import DATA, all_of, put
from apps.api.versions import commit_result, load_result
from packages.pipeline.features import extract_features
from packages.pipeline.signmatch import dtw, sequence

REASON = 'features-2.3：橙色警告不再整手排除，严重退化仍拒绝'


def backup():
    target = DATA / ('gestureflow.db.backup-' + time.strftime('%Y%m%d-%H%M%S'))
    source = sqlite3.connect(DATA / 'gestureflow.db')
    copy = sqlite3.connect(target)
    with copy:
        source.backup(copy)
    copy.close()
    source.close()
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='真正写回（默认只预览）')
    args = parser.parse_args()

    templates = all_of('sign_template')
    if not templates:
        print('示范库是空的，没有需要重算的东西。')
        return 0
    if args.apply:
        print('已备份到', backup().name)

    written = cleared = 0
    for video_id in sorted({t['video_id'] for t in templates}):
        result = load_result(video_id)
        features = extract_features(result['processed_frames'], result['width'], result['height'])
        for template in [t for t in templates if t['video_id'] == video_id]:
            try:
                seq = sequence({'features': features, 'frames': result['processed_frames']},
                               template['start_ms'], template['end_ms'])
            except ValueError as error:
                print('%-6s 清空：%s' % (template['meaning'], error))
                template['sequence'] = {}
                template['recognition_ready'] = False
                template['draft_reason'] = str(error)
                if args.apply:
                    put(template['id'], 'sign_template', template)
                cleared += 1
                continue
            before = len((template.get('sequence') or {}).get('rows') or [])
            print('%-6s %3d → %3d 帧  自距离 %.4f' % (template['meaning'], before, len(seq['rows']),
                                                    dtw(seq, seq)))
            template['sequence'] = seq
            template['recognition_ready'] = True
            template['draft_reason'] = None
            if args.apply:
                put(template['id'], 'sign_template', template)
            written += 1
        if args.apply:
            result['features'] = features
            commit_result(video_id, result, REASON)

    print(('已写回 ' if args.apply else '将写回 ') + '%d 条，清空 %d 条。' % (written, cleared))
    return 0


if __name__ == '__main__':
    sys.exit(main())
