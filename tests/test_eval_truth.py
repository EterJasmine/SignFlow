import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import eval_truth  # noqa: E402


def doc(**overrides):
    item = {'source_id': 's1', 'filename': '甲.mp4', 'duration_ms': 8000,
            'review_video_id': 'v1', 'media_url': '/api/v1/videos/v1/media',
            'duplicate_video_ids': ['v1'],
            'predictions': [{'algo': 'multi_feature', 'video_id': 'v1', 'boundaries_ms': [1000, 3000, 5000]},
                            {'algo': 'wrist_speed', 'video_id': 'v1', 'boundaries_ms': [1000, 4000]}],
            'boundaries_truth_ms': [1000, 3200], 'confirmed': True, 'review_minutes': 6}
    item.update(overrides)
    return {'schema_version': 'truth-2.0', 'ground_truth_source': 'human', 'videos': [item]}


def test_boundaries_reads_start_times_only():
    assert eval_truth._boundaries([{'start_ms': 3000}, {'start_ms': 1000}]) == [1000, 3000]
    assert eval_truth._boundaries(None) == []


def test_check_separates_progress_notes_from_errors():
    outcome = eval_truth.check_truth(doc(confirmed=False))
    assert outcome['ok'] is False and outcome['confirmed'] == 0
    assert any('未确认' in note for note in outcome['notes'])
    assert outcome['errors'] == []


def test_check_requires_minutes_and_nonempty_truth_when_confirmed():
    assert any('review_minutes' in e for e in eval_truth.check_truth(doc(review_minutes=None))['errors'])
    empty = eval_truth.check_truth(doc(boundaries_truth_ms=[]))['errors']
    assert any('为空' in e for e in empty)


def test_check_flags_anchoring_against_either_algorithm():
    anchored = doc(boundaries_truth_ms=[1000, 3000, 5000])
    assert any('multi_feature' in e and '完全一致' in e for e in eval_truth.check_truth(anchored)['errors'])
    anchored_old = doc(boundaries_truth_ms=[1000, 4000])
    assert any('wrist_speed' in e and '完全一致' in e for e in eval_truth.check_truth(anchored_old)['errors'])


def test_check_flags_out_of_order_and_out_of_range():
    assert any('升序' in e for e in eval_truth.check_truth(doc(boundaries_truth_ms=[3200, 1000]))['errors'])
    assert any('超出' in e for e in eval_truth.check_truth(doc(boundaries_truth_ms=[1000, 9000]))['errors'])
    assert any('重复' in e for e in eval_truth.check_truth(doc(boundaries_truth_ms=[1000, 1000]))['errors'])


def test_report_refuses_accuracy_without_confirmed_truth():
    markdown = eval_truth.render_report(doc(confirmed=False))
    assert '不给出准确率' in markdown and '新 F1' not in markdown


def test_report_compares_new_and_old_algorithms():
    markdown = eval_truth.render_report(doc())
    assert '新多特征算法 边界 F1 均值 **0.800**' in markdown
    assert '旧腕速算法 边界 F1 均值 **0.500**' in markdown
    assert '对照结论：新算法 − 旧算法 = **+0.300**' in markdown
    assert '判定为「提升」' in markdown
    assert '6.0 分/条' in markdown


def test_report_marks_missing_minutes_but_keeps_accuracy():
    item_doc = doc(review_minutes=None)
    item_doc['videos'][0]['review_minutes'] = None
    markdown = eval_truth.render_report(item_doc)
    assert '无法回答' in markdown and '0.800' in markdown


def test_report_counts_unconfirmed_remainder():
    assert '还有 1 条源视频未确认' in eval_truth.render_report(doc(confirmed=False, boundaries_truth_ms=[]))


def _wire(monkeypatch, rows):
    """rows: [(record, result, source_name, mtime)]"""
    monkeypatch.setattr(eval_truth, 'all_of', lambda kind: [row[0] for row in rows])
    monkeypatch.setattr(eval_truth, 'video_file', lambda ident, record: Path('/fake') / record['_file'])
    monkeypatch.setattr(eval_truth, 'load_result', lambda ident: next(row[1] for row in rows if row[0]['id'] == ident))
    monkeypatch.setattr(eval_truth, '_source_hash', lambda path: path.name)
    monkeypatch.setattr(eval_truth, '_recency', lambda record: record['_mtime'])


def test_build_merges_duplicate_uploads_of_the_same_source(monkeypatch):
    old = {'id': 'v_old', '_file': 'same.mp4', '_mtime': 1.0, 'filename': '同一段.mp4', 'duration_ms': 5000}
    new = {'id': 'v_new', '_file': 'same.mp4', '_mtime': 9.0, 'filename': '同一段.mp4', 'duration_ms': 5000}
    other = {'id': 'v_x', '_file': 'other.mp4', '_mtime': 5.0, 'filename': '另一段.mp4', 'duration_ms': 3000}
    _wire(monkeypatch, [
        (old, {'duration_ms': 5000, 'actions': [{'start_ms': 0}, {'start_ms': 2000}],
               'baseline_actions': [{'start_ms': 0}]}, 'same', 1.0),
        (new, {'duration_ms': 5000, 'actions': [{'start_ms': 0}, {'start_ms': 2100}],
               'baseline_actions': [{'start_ms': 0}, {'start_ms': 1500}]}, 'same', 9.0),
        (other, {'duration_ms': 3000, 'actions': [{'start_ms': 0}]}, 'other', 5.0),
    ])
    built = eval_truth.build_truth()
    assert len(built['videos']) == 2, '同一源视频重复上传必须合并成一条'
    merged = next(v for v in built['videos'] if v['filename'] == '同一段.mp4')
    assert merged['duplicate_video_ids'] == ['v_new', 'v_old']
    assert merged['review_video_id'] == 'v_new', '用最近一次分析作为播放与当前预测来源'
    assert [p['boundaries_ms'] for p in merged['predictions'] if p['algo'] == 'multi_feature'] == [[0, 2100]]
    assert [p['boundaries_ms'] for p in merged['predictions'] if p['algo'] == 'wrist_speed'] == [[0, 1500]]
    assert merged['boundaries_truth_ms'] == [], '模板必须留空，预填会造成锚定偏差'
    assert merged['confirmed'] is False


def test_build_skips_videos_without_result_or_file(monkeypatch):
    ghost = {'id': 'v_g', '_file': 'gone.mp4', '_mtime': 1.0, 'filename': '缺失.mp4', 'duration_ms': 1000}
    _wire(monkeypatch, [(ghost, {'duration_ms': 1000, 'actions': []}, 'gone', 1.0)])
    monkeypatch.setattr(eval_truth, 'video_file', lambda ident, record: None)
    assert eval_truth.build_truth()['videos'] == []


def test_build_refuses_to_overwrite_confirmed_truth(tmp_path, monkeypatch, capsys):
    target = tmp_path / 'truth.json'
    target.write_text(__import__('json').dumps(doc()), encoding='utf8')
    monkeypatch.setattr(eval_truth, 'build_truth', lambda: {'schema_version': 'truth-2.0', 'videos': []})
    monkeypatch.setattr('sys.argv', ['eval_truth.py', 'build', '--out', str(target)])
    assert eval_truth.main() == 3, '已确认的真值不能被悄悄覆盖'
    assert '拒绝覆盖' in capsys.readouterr().out


def test_build_force_overwrites(tmp_path, monkeypatch):
    target = tmp_path / 'truth.json'
    target.write_text(__import__('json').dumps(doc()), encoding='utf8')
    monkeypatch.setattr(eval_truth, 'build_truth',
                        lambda: {'schema_version': 'truth-2.0', 'videos': []})
    monkeypatch.setattr('sys.argv', ['eval_truth.py', 'build', '--out', str(target), '--force'])
    assert eval_truth.main() == 0


def test_merge_keeps_labeled_boundaries_by_source_hash():
    existing = doc()
    fresh = {'schema_version': 'truth-2.0', 'ground_truth_source': 'human',
             'videos': [
                 {'source_id': 's1', 'filename': '甲.mp4', 'duration_ms': 8000,
                  'review_video_id': 'v1', 'duplicate_video_ids': ['v1'],
                  'predictions': [], 'boundaries_truth_ms': [], 'confirmed': False,
                  'review_minutes': None},
                 {'source_id': 's2', 'filename': '乙.mp4', 'duration_ms': 5000,
                  'review_video_id': 'v2', 'duplicate_video_ids': ['v2'],
                  'predictions': [], 'boundaries_truth_ms': [], 'confirmed': False,
                  'review_minutes': None}]}
    merged, notes = eval_truth.merge_truth(fresh, existing)
    by_id = {item['source_id']: item for item in merged['videos']}
    assert by_id['s1']['boundaries_truth_ms'] == [1000, 3200]
    assert by_id['s1']['confirmed'] is True and by_id['s1']['review_minutes'] == 6
    assert by_id['s2']['boundaries_truth_ms'] == [] and by_id['s2']['confirmed'] is False
    assert any('新增待标 1 条' in note for note in notes)


def test_merge_reports_truth_whose_source_disappeared():
    existing = doc()
    fresh = {'schema_version': 'truth-2.0', 'ground_truth_source': 'human', 'videos': []}
    _, notes = eval_truth.merge_truth(fresh, existing)
    assert any('需手工处理' in note for note in notes)


def test_agent_prelabel_is_not_reported_as_human_accuracy():
    document = doc()
    document['ground_truth_source'] = 'agent-vision'
    document['videos'][0]['review_minutes'] = None
    assert eval_truth.check_truth(document)['ok'] is True, '预标注不该被要求填人工耗时'
    markdown = eval_truth.render_report(document)
    assert '未经人工确认' in markdown and '不能作为对外宣称的分段准确率' in markdown
