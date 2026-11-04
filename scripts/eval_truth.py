"""人工真值的生成、标注、校验与报告。

四个子命令，避免"没有真值就宣称准确率"或"拿预测当标签"：

1. ``build``  从已完成的分析结果生成待标注模板。同一段源视频被重复上传多次时只保留一条，
   并把"新多特征算法"与"旧腕速算法"的边界分别存为两组预测，供后续对照。
   真值字段留空——预填预测会造成锚定偏差（看到 1200 就想调成 1200 附近）。
2. ``label``  生成 ``evaluation/label.html``：本地标注页，空格打点、方向键微调、自动计时。
3. ``check``  校验自相矛盾项（未确认、未升序、越界、与预测完全一致、缺耗时）。
4. ``report`` 只统计已确认项，输出新旧算法的边界 F1 对照表。

边界取每个动作步骤的起始时刻，容差默认 ±300ms。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.api.media import video_file  # noqa: E402
from apps.api.store import DATA, all_of  # noqa: E402
from apps.api.versions import load_result  # noqa: E402
from evaluation.metrics import boundary_f1  # noqa: E402

INSTRUCTIONS = (
    '空格键在当前播放位置打点，方向键微调 10ms。'
    '只标真正的手势动作起始时刻：手型或运动方向发生实质变化的那一下，'
    '不要以"步骤卡好看"为准，也不要把停顿当成边界。'
    '标完点「确认本段」，页面会记下你花的分钟数。'
)
ALGO_NEW = 'multi_feature'
ALGO_OLD = 'wrist_speed'


def _human(document: dict) -> bool:
    """真值是否由人工确认。机器预标注（``agent-vision``）不满足这一条。"""
    return (document or {}).get('ground_truth_source') == 'human'


def _source_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _boundaries(actions) -> list[int]:
    return sorted(int(action['start_ms']) for action in actions or [])


def _recency(record: dict) -> float:
    result = DATA / 'assets' / record['id'] / 'result.json'
    return result.stat().st_mtime if result.exists() else 0.0


def build_truth() -> dict:
    """按源视频内容去重，每组存一份真值槽位与新/旧两组预测。"""
    groups: dict[str, dict] = {}
    for record in all_of('video'):
        path = video_file(record['id'], record)
        if not path:
            continue
        try:
            result = load_result(record['id'])
        except FileNotFoundError:
            continue
        key = _source_hash(path)
        entry = groups.setdefault(key, {'source_id': key, 'filename': record.get('filename'),
                                        'duration_ms': result.get('duration_ms'),
                                        'records': []})
        entry['records'].append((_recency(record), record, result))

    videos = []
    for key, entry in groups.items():
        records = sorted(entry['records'], key=lambda item: item[0], reverse=True)
        primary = records[0][1]
        predictions = []
        # rule_actions 是挂模型之前存下的规则结果：模型接管后 actions 会被覆盖，基线仍要留住。
        for algo, getter in ((ALGO_NEW, lambda r: _boundaries(r.get('rule_actions') or r.get('automatic_actions') or r.get('actions'))),
                             (ALGO_OLD, lambda r: _boundaries(r.get('baseline_actions')))):
            # 同一算法只保留最近一次分析的预测；更早的重传记录是同一算法的旧版本，不与基线混为一谈。
            for _, record, result in records:
                bounds = getter(result)
                if bounds:
                    prediction = {'algo': algo, 'video_id': record['id'], 'boundaries_ms': bounds}
                    if algo == ALGO_NEW and result.get('automatic_actions'):
                        # 该视频的分段被人工改过，新自动结果还没应用；评估取未应用的新结果，避免把人工分段当算法输出
                        prediction['note'] = '自动分段未应用'
                    predictions.append(prediction)
                    break
        videos.append({'source_id': key, 'filename': entry['filename'],
                       'duration_ms': entry['duration_ms'],
                       'review_video_id': primary['id'],
                       'media_url': '/api/v1/videos/' + primary['id'] + '/media',
                       'duplicate_video_ids': [record['id'] for _, record, _ in records],
                       'predictions': predictions,
                       'boundaries_truth_ms': [], 'confirmed': False, 'review_minutes': None})
    videos.sort(key=lambda item: item['filename'] or '')
    return {'schema_version': 'truth-2.0', 'ground_truth_source': 'human',
            'instructions': INSTRUCTIONS, 'videos': videos}


def merge_truth(fresh: dict, existing: dict) -> tuple[dict, list[str]]:
    """把新建的清单与旧清单合并：按 ``source_id``（视频内容哈希）保留已标结果。

    返回 (合并后的清单, 变动说明)。新增的素材以未确认状态加入，被删掉源视频的旧真值
    不会静默丢弃——它们留在返回的第二项里提示，由人决定是否手工恢复。
    """
    previous = {item.get('source_id'): item for item in existing.get('videos') or []}
    notes: list[str] = []
    added, kept = [], 0
    for item in fresh['videos']:
        old = previous.get(item.get('source_id'))
        if old and old.get('boundaries_truth_ms'):
            item['boundaries_truth_ms'] = list(old['boundaries_truth_ms'])
            item['confirmed'] = bool(old.get('confirmed'))
            for field in ('review_minutes', 'confidence', 'annotation_note'):
                if old.get(field) is not None:
                    item[field] = old[field]
            kept += 1
        else:
            added.append(item.get('filename') or item.get('source_id'))
    fresh['videos'].sort(key=lambda entry: entry.get('filename') or '')
    seen = {item.get('source_id') for item in fresh['videos']}
    for source_id, old in previous.items():
        if source_id not in seen and old.get('boundaries_truth_ms'):
            notes.append(f'旧真值不在当前素材中（{old.get("filename")}），需手工处理')
    notes.insert(0, f'保留已标 {kept} 条，新增待标 {len(added)} 条' +
                     (f'：{", ".join(added)}' if added else ''))
    if previous and existing.get('ground_truth_source'):
        fresh['ground_truth_source'] = existing['ground_truth_source']
    for field in ('annotator', 'criterion', 'labeled_at'):
        if existing.get(field):
            fresh[field] = existing[field]
    return fresh, notes


def _prediction_of(item: dict, algo: str) -> list[int]:
    for prediction in item.get('predictions') or []:
        if prediction.get('algo') == algo:
            return sorted(prediction.get('boundaries_ms') or [])
    return []


def check_truth(document: dict) -> dict:
    """返回 {ok, confirmed, total, errors, notes}。

    ``errors`` 是必须修的自相矛盾项（已确认却空、乱序、越界、与预测完全一致、缺耗时）；
    ``notes`` 只是进度提示（还有几条没确认），不影响 ok。
    """
    videos = document.get('videos') or []
    errors, notes, confirmed = [], [], 0
    for item in videos:
        tag = item.get('filename') or item.get('source_id') or '?'
        truth = item.get('boundaries_truth_ms')
        if not item.get('confirmed'):
            notes.append(f'{tag}: 未确认，不进入统计')
            continue
        confirmed += 1
        if not isinstance(truth, list) or not truth:
            errors.append(f'{tag}: 已确认但真值边界为空')
            continue
        if truth != sorted(truth):
            errors.append(f'{tag}: 真值边界未按时间升序')
        duration = item.get('duration_ms')
        if duration and any(value < 0 or value >= duration for value in truth):
            errors.append(f'{tag}: 真值边界超出 0…{duration}ms')
        if len(set(truth)) != len(truth):
            errors.append(f'{tag}: 真值边界有重复值（同一时刻被标了两次）')
        for algo in (ALGO_NEW, ALGO_OLD):
            predicted = _prediction_of(item, algo)
            if predicted and sorted(truth) == predicted:
                errors.append(f'{tag}: 真值与 {algo} 预测完全一致，请确认确实逐条核对过（否则存在锚定偏差）')
        if not item.get('review_minutes') and _human(document):
            errors.append(f'{tag}: 未填 review_minutes，无法统计人工耗时')
    return {'ok': confirmed > 0 and not errors, 'confirmed': confirmed, 'total': len(videos),
            'errors': errors, 'notes': notes}


def render_report(document: dict, tolerance_ms: int = 300) -> str:
    check = check_truth(document)
    rows = []
    new_f1s, old_f1s, diffs, minutes = [], [], [], []
    for item in document.get('videos') or []:
        if not item.get('confirmed'):
            continue
        truth = sorted(item.get('boundaries_truth_ms') or [])
        if not truth:
            continue
        new = _prediction_of(item, ALGO_NEW)
        old = _prediction_of(item, ALGO_OLD)
        new_f1 = boundary_f1(truth, new, tolerance_ms) if new else None
        old_f1 = boundary_f1(truth, old, tolerance_ms) if old else None
        if new_f1 is not None:
            new_f1s.append(new_f1)
            diffs.append(len(new) - len(truth))
        if old_f1 is not None:
            old_f1s.append(old_f1)
        if item.get('review_minutes'):
            minutes.append(float(item['review_minutes']))
        tag = item.get('filename') or item.get('source_id')
        if any(p.get('note') for p in item.get('predictions') or []):
            tag += '（自动分段未应用）'
        rows.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            tag, len(truth),
            len(new) if new else '—', f'{new_f1:.3f}' if new_f1 is not None else '—',
            len(old) if old else '—', f'{old_f1:.3f}' if old_f1 is not None else '—',
            item.get('review_minutes') or '—'))

    lines = ['# 动作分段真值评估报告', '',
             f'- 已确认视频：{check["confirmed"]} / {check["total"]}（只有已确认的才进入统计）',
             f'- 边界容差：±{tolerance_ms}ms，边界取每个动作步骤的起始时刻',
             ('- 真值为人工盲标（模板不预填预测），避免锚定偏差' if _human(document) else
              '- **真值来源：模型预标注（agent-vision），未经人工确认**')]
    if not _human(document):
        lines += ['',
                  '> 本报告的数字只能用作「管线自检 / 预标注质量参考」，**不能作为对外宣称的分段准确率**。'
                  '预标注由视觉模型逐帧判读得到，与算法预测相互独立，但不等同于人工真值；'
                  '要对外引用，请在 `evaluation/label.html` 里逐条核对后把 `ground_truth_source` 改回 `human`。']
    if not rows:
        lines += ['', '**没有任何已确认的真值，本报告不给出准确率。**',
                  '先跑 `build` 生成模板、`label` 打开标注页，标完再回来。']
        if check['total']:
            lines += ['', f'> 还有 {check["total"]} 条源视频未确认，未纳入统计。']
        return '\n'.join(lines) + '\n'

    lines += ['', '## 逐条结果', '',
              '| 源视频 | 真值步数 | 新算法步数 | 新 F1 | 旧算法步数 | 旧 F1 | 耗时(分) |',
              '| --- | --- | --- | --- | --- | --- | --- |', *rows, '', '## 汇总', '']
    if new_f1s:
        lines.append(f'- 新多特征算法 边界 F1 均值 **{statistics.mean(new_f1s):.3f}**，'
                     f'中位数 {statistics.median(new_f1s):.3f}，最低 {min(new_f1s):.3f}（{len(new_f1s)} 条）')
    if old_f1s:
        lines.append(f'- 旧腕速算法 边界 F1 均值 **{statistics.mean(old_f1s):.3f}**，'
                     f'中位数 {statistics.median(old_f1s):.3f}（{len(old_f1s)} 条）')
    if new_f1s and old_f1s:
        delta = statistics.mean(new_f1s) - statistics.mean(old_f1s)
        verdict = '提升' if delta > 0.02 else ('下降' if delta < -0.02 else '基本持平')
        lines.append(f'- 对照结论：新算法 − 旧算法 = **{delta:+.3f}**，判定为「{verdict}」')
    if diffs:
        lines.append(f'- 步数差（新算法 − 真值）均值 {statistics.mean(diffs):+.2f}，'
                     f'偏多 {sum(1 for d in diffs if d > 0)} 条、偏少 {sum(1 for d in diffs if d < 0)} 条、'
                     f'一致 {sum(1 for d in diffs if d == 0)} 条')
    if minutes:
        lines.append(f'- 人工核对耗时：{len(minutes)} 条，均值 {statistics.mean(minutes):.1f} 分/条，'
                     f'合计 {sum(minutes):.0f} 分钟')
    else:
        lines.append('- 人工核对耗时：未填写，无法回答"半自动比手工快多少"')
    if check['errors']:
        lines += ['', '## 需要先修正的问题', '', *[f'- {issue}' for issue in check['errors']]]
    pending = check['total'] - check['confirmed']
    if pending:
        lines += ['', f'> 还有 {pending} 条源视频未确认，未纳入统计。']
    return '\n'.join(lines) + '\n'


LABEL_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>动作边界盲标</title>
<style>
:root{--bg:#fbf9fd;--card:#fff;--ink:#2a2635;--muted:#8b8299;--line:#e9e4f2;--accent:#7b5cd6;--warm:#c96a4a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.65 "Microsoft YaHei","Noto Sans SC",system-ui,sans-serif}
header{padding:16px 22px;border-bottom:1px solid var(--line);background:var(--card);display:flex;gap:18px;align-items:center;flex-wrap:wrap}
h1{margin:0;font-size:16px;letter-spacing:.02em}
.tag{font:11px/1.6 ui-monospace,monospace;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 9px}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:18px;padding:18px 22px 60px;max-width:1500px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
video{width:100%;max-height:56vh;border-radius:10px;background:#000;display:block}
.picks{list-style:none;margin:12px 0 0;padding:0;display:flex;flex-direction:column;gap:6px}
.picks li{display:flex;align-items:center;gap:10px;padding:7px 10px;border:1px solid var(--line);border-radius:9px;cursor:pointer;font:13px/1.5 ui-monospace,monospace}
.picks li.sel{border-color:var(--accent);background:#f4f0ff}
.picks li span{flex:1}
.picks li b{font-weight:600}
.picks button{border:0;background:transparent;color:var(--muted);cursor:pointer;font-size:15px;padding:0 4px}
.picks button:hover{color:var(--warm)}
.row{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}
button.act{background:var(--accent);color:#fff;border:0;border-radius:9px;padding:9px 15px;cursor:pointer;font:inherit}
button.ghost{background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:9px;padding:8px 13px;cursor:pointer;font:inherit}
button:disabled{opacity:.45;cursor:not-allowed}
select,input{font:inherit;border:1px solid var(--line);border-radius:9px;padding:7px 10px;background:#fff;color:var(--ink)}
.clock{font:600 30px/1.1 ui-monospace,monospace;letter-spacing:.01em}
.hint{color:var(--muted);font-size:12.5px;margin:10px 0 0}
.done{color:#2f7a51}
.warn{color:var(--warm)}
.list{max-height:46vh;overflow:auto;margin-top:10px}
.list div{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:8px;font:13px/1.5 ui-monospace,monospace;cursor:pointer}
.list div:hover{background:#f6f2ff}
.list div.sel{background:#efe8ff;outline:1px solid var(--accent)}
.list i{font-style:normal;color:var(--muted);width:22px}
.list button{margin-left:auto;border:0;background:transparent;color:var(--muted);cursor:pointer}
footer{padding:0 22px 40px;color:var(--muted);font-size:12.5px;max-width:1500px}
.guide{margin-top:14px;border:1px solid var(--line);border-radius:10px;padding:9px 12px;background:#fcfbff}
.guide summary{cursor:pointer;font-weight:600;font-size:13px}
.guide ul{margin:8px 0 0;padding-left:20px;font-size:12.5px;line-height:1.75}
.guide li b{color:var(--accent);font-weight:600}
.guide p{margin:6px 0 0}
</style></head><body>
<header><h1>动作边界盲标</h1><span class="tag" id="progress"></span><span class="tag">空格打点 · ←→ 微调 · 0.5x 更准</span></header>
<main>
<section class="panel">
  <video id="v" controls preload="auto"></video>
  <div class="row"><span class="clock" id="clock">0.000</span>
    <select id="rate"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="0.25">0.25x</option></select>
    <button class="ghost" id="play">播放 / 暂停</button></div>
  <ul class="picks" id="picks"></ul>
</section>
<section class="panel">
  <div class="row" style="margin:0"><b id="name">—</b></div>
  <p class="hint" id="meta"></p>
  <div class="row"><button class="act" id="mark">打点（空格）</button><button class="ghost" id="back">−10ms</button><button class="ghost" id="fwd">+10ms</button><button class="ghost" id="undo">撤销</button></div>
  <div class="list" id="list"></div>
  <div class="row"><button class="act" id="confirm">确认本段</button><button class="ghost" id="reset">清空重标</button></div>
  <p class="hint" id="status"></p>
  <details class="guide"><summary>什么算一个边界？</summary>
    <ul>
      <li><b>算</b>：手型变了——掌 → 拳 → 伸食指 → 比心 → 捏。</li>
      <li><b>算</b>：运动方向变了——向上挥突然改向下压。</li>
      <li><b>算</b>：原来动的忽然定住并停住（收势也常是独立动作）。</li>
      <li><b>不算</b>：同一个动作里手指微动、身体轻晃、呼吸起伏。</li>
      <li><b>不算</b>：画面黑屏、遮挡、手出画——这些位置一律不打点。</li>
      <li><b>不算</b>：为了让步骤卡看起来整齐而凑数。</li>
    </ul>
    <p class="hint">拿不准时问自己：<b>两帧可以合并成同一张教学图吗？</b>可以就不是边界。</p>
  </details>
  <div class="row"><button class="ghost" id="save">导出 truth.json</button></div>
  <p class="hint">导出会下载一份完整文件，用它覆盖 <code>evaluation/truth.json</code>，再跑 check / report。</p>
</section>
</main>
<footer>预测边界不在本页显示，避免"照着预测微调"造成锚定偏差。想对比时用 <code>python scripts/eval_truth.py report</code>。</footer>
<script>
const DATA = __DATA__;
const BASE = '__BASE__';
// 优先落到第一条"未确认且够长"的素材：几秒的冒烟测试片没有标注价值。
const startAt = DATA.videos.findIndex(v => !v.confirmed && (v.duration_ms || 0) >= 3000);
let idx = startAt >= 0 ? startAt : Math.max(0, DATA.videos.findIndex(v => !v.confirmed));
let sel = -1;
const clocks = new Map();
const v = document.getElementById('v');
const $ = id => document.getElementById(id);
const mmss = ms => String(Math.floor(ms / 60000)).padStart(2, '0') + ':' + ((ms % 60000) / 1000).toFixed(3).padStart(6, '0');
const cur = () => DATA.videos[idx];
const truth = () => cur().boundaries_truth_ms || (cur().boundaries_truth_ms = []);
function touch() { if (!clocks.has(idx)) clocks.set(idx, Date.now()); }
function elapsed() { const t = clocks.get(idx); return t ? Math.max(1, Math.round((Date.now() - t) / 60000)) : 1; }

function render() {
  const item = cur();
  $('progress').textContent = `已确认 ${DATA.videos.filter(x => x.confirmed).length} / ${DATA.videos.length}`;
  $('picks').innerHTML = DATA.videos.map((x, i) =>
    `<li class="${i === idx ? 'sel' : ''}" data-i="${i}"><b>${i + 1}</b><span>${x.filename || x.source_id}</span><em style="font-style:normal;color:${x.confirmed ? '#2f7a51' : '#8b8299'}">${x.confirmed ? '已确认' : (x.boundaries_truth_ms || []).length + ' 点'}</em></li>`).join('');
  $('name').textContent = item.filename || item.source_id;
  $('meta').textContent = `${(item.duration_ms / 1000).toFixed(2)} 秒`
    + (item.duplicate_video_ids && item.duplicate_video_ids.length > 1 ? ` · 同源记录 ${item.duplicate_video_ids.length} 条已合并` : '');
  item.boundaries_truth_ms = truth().slice().sort((a, b) => a - b);
  const t = item.boundaries_truth_ms;
  $('list').innerHTML = t.length
    ? t.map((ms, i) => `<div class="${i === sel ? 'sel' : ''}" data-i="${i}"><i>${i + 1}</i>${mmss(ms)}<button data-del="${i}" title="删除">×</button></div>`).join('')
    : '<p class="hint">还没有打点。播放时按空格记下每个动作的起点。</p>';
  $('status').innerHTML = item.confirmed
    ? `<span class="done">已确认 · 记录耗时 ${item.review_minutes} 分钟</span>`
    : '<span class="warn">未确认：标完点「确认本段」才会进入统计</span>';
  // 只有换视频才重设 src，否则每次打点都会把播放位置冲回 0。
  const url = BASE + item.media_url;
  if (v.getAttribute('data-src') !== url) { v.setAttribute('data-src', url); v.src = url; }
}

function select(i) { idx = i; sel = -1; v.pause(); render(); }

$('picks').onclick = e => { const li = e.target.closest('li'); if (li) select(+li.dataset.i); };
$('list').onclick = e => {
  const del = e.target.closest('[data-del]');
  if (del) { e.stopPropagation(); touch(); truth().splice(+del.dataset.del, 1); sel = -1; render(); return; }
  const row = e.target.closest('[data-i]');
  if (row) { sel = +row.dataset.i; v.currentTime = truth()[sel] / 1000; render(); }
};
function mark() {
  if (!v.getAttribute('data-src')) return;
  touch();
  const ms = Math.round(v.currentTime * 100) * 10;
  const list = truth();
  if (list.some(x => Math.abs(x - ms) < 120)) { $('status').textContent = '离已有点太近（<120ms），忽略'; return; }
  list.push(ms); list.sort((a, b) => a - b);
  sel = list.indexOf(ms); render();
}
function nudge(delta) {
  if (sel < 0 || !truth().length) return;
  touch();
  const list = truth();
  const next = Math.max(0, list[sel] + delta);
  list[sel] = next; list.sort((a, b) => a - b);
  sel = list.indexOf(next); render();
}
$('mark').onclick = mark;
$('undo').onclick = () => { touch(); const list = truth(); if (list.length) { list.pop(); sel = -1; render(); } };
$('reset').onclick = () => { if (confirm('清空本段的真值？')) { truth().length = 0; cur().confirmed = false; cur().review_minutes = null; clocks.delete(idx); sel = -1; render(); } };
$('play').onclick = () => { touch(); v.paused ? v.play() : v.pause(); };
$('rate').onchange = e => v.playbackRate = +e.target.value;
$('back').onclick = () => nudge(-10);
$('fwd').onclick = () => nudge(10);
$('confirm').onclick = () => {
  const item = cur();
  if (!truth().length) { $('status').textContent = '至少打一个点再确认'; return; }
  const guess = Math.max(elapsed(), item.review_minutes || 0);
  const input = prompt('这段视频你花了多少分钟？（用于回答"半自动比手工快多少"）', guess);
  if (input === null) return;
  item.review_minutes = Number(input) || guess;
  item.confirmed = true;
  clocks.delete(idx); render();
  const next = DATA.videos.findIndex(x => !x.confirmed);
  if (next >= 0) select(next);
};
$('save').onclick = () => {
  const blob = new Blob([JSON.stringify(DATA, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'truth.json'; a.click();
  URL.revokeObjectURL(a.href);
};
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space') { e.preventDefault(); mark(); }
  else if (e.code === 'ArrowLeft') { e.preventDefault(); nudge(-10); }
  else if (e.code === 'ArrowRight') { e.preventDefault(); nudge(10); }
});
const tick = () => { $('clock').textContent = v.currentTime.toFixed(3); requestAnimationFrame(tick); };
tick();
render();
</script></body></html>
"""


def write_label_page(document: dict, target: Path, base: str = 'http://127.0.0.1:8000') -> Path:
    payload = json.dumps(document, ensure_ascii=False).replace('</', '<\\/')
    target.parent.mkdir(parents=True, exist_ok=True)
    html = LABEL_HTML.replace('__DATA__', payload).replace('__BASE__', base.rstrip('/'))
    target.write_text(html, encoding='utf8')
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description='人工真值模板、标注页、校验与报告')
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build', help='生成待标注模板（按源视频去重、盲标）')
    build.add_argument('--out', default='evaluation/truth.json')
    build.add_argument('--force', action='store_true', help='即使已有已确认真值也覆盖')
    build.add_argument('--merge', action='store_true',
                       help='把新素材并入现有清单，保留已确认的边界（按视频内容哈希对齐）')
    build.add_argument('--dry-run', action='store_true', help='只打印将发生的变动，不写文件')
    label = sub.add_parser('label', help='生成标注页 evaluation/label.html')
    label.add_argument('truth', nargs='?', default='evaluation/truth.json')
    label.add_argument('--out', default='evaluation/label.html')
    label.add_argument('--base', default='http://127.0.0.1:8000',
                       help='视觉服务地址，标注页从这里加载视频')
    check = sub.add_parser('check', help='校验模板一致性')
    check.add_argument('truth', nargs='?', default='evaluation/truth.json')
    report = sub.add_parser('report', help='输出 Markdown 报告')
    report.add_argument('truth', nargs='?', default='evaluation/truth.json')
    report.add_argument('--out', default='evaluation/results/report.md')
    report.add_argument('--tolerance', type=int, default=300)
    args = parser.parse_args()

    if args.command == 'build':
        target = Path(args.out)
        existing = json.loads(target.read_text(encoding='utf8')) if target.exists() else None
        if existing and not (args.force or args.merge):
            done = [v for v in existing.get('videos') or [] if v.get('confirmed')]
            if done:
                print(f'拒绝覆盖：{target} 已有 {len(done)} 条已确认真值。')
                print('加 --merge 把新素材并入（保留已标），或确认要丢弃后加 --force 重跑。')
                return 3
        document = build_truth()
        if args.merge and existing:
            document, notes = merge_truth(document, existing)
            for note in notes:
                print('•', note)
        if args.dry_run:
            print(f'（预演）将写入 {target}：{len(document["videos"])} 条源视频，未落盘')
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf8')
        merged = sum(len(v['duplicate_video_ids']) - 1 for v in document['videos'])
        pending = sum(1 for v in document['videos'] if not v.get('confirmed'))
        print(f'已生成 {target}：{len(document["videos"])} 条源视频'
              f'（合并重复上传 {merged} 条），待标 {pending} 条、已确认 '
              f'{len(document["videos"]) - pending} 条。')
        return 0

    source = Path(args.truth)
    if not source.exists():
        print(f'找不到 {source}，请先运行 build。')
        return 2
    document = json.loads(source.read_text(encoding='utf8'))

    if args.command == 'label':
        print(write_label_page(document, Path(args.out), args.base))
        return 0

    if args.command == 'check':
        outcome = check_truth(document)
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 0 if outcome['ok'] else 1

    markdown = render_report(document, args.tolerance)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding='utf8')
    print(markdown)
    print(f'报告已写入 {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
