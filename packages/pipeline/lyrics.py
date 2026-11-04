"""歌词来源统一层：LRC 解析、音轨提取、云端转写。

三条来源（LRC 导入 / 自动转写 / 前端逐句编辑）都收敛成同一份
``{start_ms, end_ms, text}`` 列表，再由 :func:`apply_lyrics` 落到动作步骤上，
下游的教程导出因此不必关心歌词是从哪来的。
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx

_STAMP = re.compile(r'\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]')
_META = re.compile(r'^\[(ti|ar|al|by|offset|re|ve|length|kana|language):(.*)\]$', re.I)
_MIN_MS = 300


def parse_lrc(text: str) -> list[dict]:
    """按起始秒数排序的 ``{start_s, text}`` 列表；忽略标签行，支持一行多时间戳。"""
    items: list[dict] = []
    offset = 0.0
    for raw in (text or '').splitlines():
        line = raw.strip().lstrip('\ufeff')
        if not line:
            continue
        meta = _META.match(line)
        if meta:
            if meta.group(1).lower() == 'offset':
                try:
                    offset = float(meta.group(2).strip()) / 1000
                except ValueError:
                    pass
            continue
        stamps: list[float] = []
        cursor = 0
        while True:
            found = _STAMP.match(line, cursor)
            if not found:
                break
            stamps.append(int(found.group(1)) * 60 + float(found.group(2).replace(':', '.')))
            cursor = found.end()
        content = line[cursor:].strip()
        if not stamps or not content:
            continue
        items.extend({'start_s': stamp + offset, 'text': content} for stamp in stamps)
    items.sort(key=lambda item: item['start_s'])
    return items


def to_lyrics(items: list[dict], duration_ms: int, min_ms: int = _MIN_MS) -> list[dict]:
    """把起始时刻列表补成区间：每句持续到下一句开始，越界裁剪，重复行合并。"""
    rows: list[dict] = []
    kept = []
    for item in items:
        text = (item.get('text') or '').strip()
        if text and (not kept or kept[-1]['text'] != text):
            kept.append({'start_s': float(item['start_s']), 'text': text})
    for index, item in enumerate(kept):
        start = max(0, int(round(item['start_s'] * 1000)))
        if start >= duration_ms:
            continue
        nxt = kept[index + 1]['start_s'] if index + 1 < len(kept) else duration_ms / 1000
        end = min(duration_ms, int(round(nxt * 1000)))
        if end - start < min_ms:
            end = min(duration_ms, start + min_ms)
        if end <= start:
            continue
        rows.append({'id': 'lyric_' + uuid4().hex, 'start_ms': start, 'end_ms': end, 'text': item['text'][:1000]})
    return rows


def apply_lyrics(result: dict, lyrics: list[dict]) -> list[dict]:
    """写入歌词并重算每个动作的歌词归属，返回排好序的歌词。"""
    from .motion import lyric_overlap

    result['lyrics'] = sorted(lyrics, key=lambda row: row['start_ms'])
    for action in result.get('actions', []):
        action['lyric_ids'] = [match['lyric_id'] for match in lyric_overlap(action, result['lyrics'])]
    return result['lyrics']


def audio_track(video: Path, target: Path, timeout: int = 300) -> Path:
    """抽 16kHz 单声道 WAV：转写只需语音，比原视频小一个量级。"""
    import imageio_ffmpeg

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(video),
               '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    try:
        done = subprocess.run(command, capture_output=True, timeout=timeout, creationflags=flags)
    except subprocess.TimeoutExpired:
        raise ValueError('音轨提取超时，请缩短视频或改用 LRC 导入') from None
    if done.returncode != 0 or not target.exists() or target.stat().st_size < 2048:
        detail = done.stderr.decode('utf8', 'ignore').strip().splitlines()
        raise ValueError('这段视频没有可用音轨，无法自动识别歌词；请改用 LRC 导入' + (f'（{detail[-1][:120]}）' if detail else ''))
    return target


def asr_config() -> dict:
    return {'base': os.getenv('AI_BASE_URL', '').rstrip('/'),
            'key': os.getenv('AI_API_KEY', ''),
            'model': os.getenv('ASR_MODEL', 'whisper-1')}


def asr_status() -> dict:
    config = asr_config()
    ready = bool(config['base'] and config['key'])
    return {'configured': ready, 'model': config['model'] if ready else None,
            'endpoint': (config['base'] + '/audio/transcriptions') if ready else None,
            'note': '任意兼容 OpenAI /audio/transcriptions 的服务；未配置时仍可导入 LRC。'}


async def transcribe(wav: Path, config: dict | None = None, language: str | None = 'zh') -> list[dict]:
    """返回带秒级起始时间的 ``{start_s, text}`` 列表；段落缺失时明确报错而不是猜。"""
    config = config or asr_config()
    if not (config['base'] and config['key']):
        raise ValueError('未配置 AI_BASE_URL / AI_API_KEY，无法自动识别歌词')
    form = {'model': config['model'], 'response_format': 'verbose_json', 'timestamp_granularities[]': 'segment'}
    if language:
        form['language'] = language
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with wav.open('rb') as handle:
                response = await client.post(config['base'] + '/audio/transcriptions',
                                             headers={'Authorization': 'Bearer ' + config['key']},
                                             data=form, files={'file': (wav.name, handle, 'audio/wav')})
    except httpx.TimeoutException:
        raise ValueError('语音识别超时，请缩短视频或改用 LRC 导入') from None
    if response.status_code != 200:
        raise ValueError(f'语音识别接口返回 {response.status_code}，请检查服务商地址、模型名与余额')
    payload = response.json()
    items = []
    for segment in payload.get('segments') or []:
        text = str(segment.get('text') or '').strip()
        if not text or 'start' not in segment:
            continue
        items.append({'start_s': float(segment['start']), 'text': text})
    if not items:
        raise ValueError('接口没有返回带时间戳的段落，请在服务商侧改用 verbose_json，或导入 LRC 字幕')
    items.sort(key=lambda item: item['start_s'])
    return items
