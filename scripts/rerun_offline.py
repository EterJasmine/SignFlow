"""离线重跑分析并写入新版本（不改 job 表，等价于一次重新分析）。

用途：改了视觉/几何判据后，让已有结果按新代码刷新，而不必在页面里逐条重传。
资产 URL、歌词、人工编辑保护与 worker.py 保持一致。
"""
import json, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.api.store import DATA, all_of, get  # noqa: E402
from apps.api.versions import load_result, commit_result  # noqa: E402
from packages.pipeline.analyze import analyze_video  # noqa: E402
from packages.pipeline.composer import compose_step  # noqa: E402


def rerun(video, log):
    vid = video['id']
    try:
        previous = load_result(vid)
    except FileNotFoundError:
        previous = None
    run_id = 'rerun_' + time.strftime('%Y%m%d%H%M%S')
    output = DATA / 'assets' / vid / 'versions' / run_id
    started = time.time()
    result = analyze_video(
        video['path'], output, vid, lambda stage, pct: None, lambda: False,
        checkpoint_directory=DATA / 'assets' / vid / 'checkpoints',
        scene_overrides=[s for s in previous.get('scenes', []) if s.get('classification_method') == 'manual'] if previous else None)
    text = json.dumps(result, ensure_ascii=False).replace('/assets/' + vid + '/', '/assets/' + vid + '/versions/' + run_id + '/')
    result = json.loads(text)
    result['analysis_version'] = run_id
    if previous:
        result['lyrics'] = previous.get('lyrics', [])
        protected = (previous.get('manual_tutorial_protected') or previous.get('schema_version') == '1.0'
                     or any(a.get('edited_by_human') or a.get('review_status') == 'approved' for a in previous['actions']))
        if protected:
            result['automatic_actions'] = result['actions']
            result['actions'] = previous['actions']
            result['manual_tutorial_protected'] = True
            result['warnings'].append('检测已更新，人工编辑的教程结构受保护。新自动分段另存 automatic_actions，可显式应用或恢复历史版本。')
    result['tutorial_steps'] = [compose_step(a, result) for a in result['actions']]
    commit_result(vid, result, 'rerun:' + run_id, previous.get('result_revision', 0) if previous else 0)
    cov = result['coverage']
    log.append('%-22s %5.1fs scenes=%-28s pose=%3s hand=%3s 丢弃骨架=%s' % (
        vid[:20], time.time() - started,
        str([(s['framing_type'], s['person_count']) for s in result['scenes']])[:28],
        cov.get('pose_frames'), cov.get('hand_frames'),
        (cov.get('pose_quality') or {}).get('rejected_total')))


def main(targets):
    videos = {v['id']: v for v in all_of('video')}
    if not targets:
        targets = sorted(videos)
    log = []
    for vid in targets:
        video = videos.get(vid)
        if not video:
            log.append('%s 未登记' % vid)
            continue
        try:
            rerun(video, log)
        except Exception:
            log.append('%s 失败\n%s' % (vid, traceback.format_exc()[-500:]))
        (ROOT / '.tmp' / 'rerun_offline.txt').write_text('\n'.join(log), encoding='utf8')
    print('done', len(log))


if __name__ == '__main__':
    main(sys.argv[1:])
