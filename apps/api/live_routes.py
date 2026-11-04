"""实时手语识别：摄像头帧进，切出的词出。

同一台机器只有一个摄像头，所以进程内保留一个引擎，前端串行推帧。
所有识别都走 `signmatch.sequence` 与 `retrieve`，与手动选区间用的是同一套特征与阈值。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
from uuid import uuid4
from pydantic import BaseModel,Field
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from packages.pipeline.live import LiveCapture,skeleton
from packages.pipeline.signmatch import retrieve, sequence, usable as usable_template

from .store import all_of,put,get,delete

ROOT = Path(__file__).resolve().parents[2]
router = APIRouter(prefix='/api/v1/sign/live')

# 阈值以「掌长/秒」为单位，与人的远近无关。速度已做时间中位数平滑，所以可以贴地一些。
ACTIVE_SPEED = 0.5
QUIET_SPEED = 0.32
QUIET_HOLD_MS = 420
MIN_WORD_MS = 320
MAX_WORD_MS = 4000
LOOK_BACK_MS = 600  # 动作起点最多回溯这么久，避免把上一句的尾巴吃进来
MISSING_FRAMES = 2  # 手短暂丢失一帧不算放下，避免把词切断

_engine: LiveCapture | None = None
_lock = threading.Lock()
_state = {'phase': 'idle', 'start_ms': None, 'quiet_since': None, 'last_ts': 0, 'raw_ts': None,
          'last_quiet': None, 'heard': [], 'missing': 0,'session_id':uuid4().hex}

# 模板在内存里留一份：原来每帧都去 SQLite 读全表并把每条示范的 sequence 解一次 JSON，
# 实测 34ms，比手部检测还贵。示范库只在用户标注/删除时变化，短 TTL 足够跟上。
_template_cache = {'at': -1e9, 'rows': [], 'ok': False}
TEMPLATE_TTL = 2.0


def _usable_templates(force: bool = False):
    now = time.monotonic()
    if not force and _template_cache['ok'] and now - _template_cache['at'] < TEMPLATE_TTL:
        return _template_cache['rows']
    rows = [t for t in all_of('sign_template')+all_of('sign_training_sample') if usable_template(t)]
    _template_cache.update({'at': now, 'rows': rows, 'ok': True})
    return rows


def _window_score(engine: LiveCapture, head: int, tail: int, templates):
    """Score one candidate window; returns (result, why_it_failed)."""
    if tail - head < MIN_WORD_MS:
        return None, None
    frames = engine.window(head, tail)
    if len(frames) < 4:
        return None, '这段只收到 %d 帧，动作太快或推帧被拖慢，请放慢一点再来一次。' % len(frames)
    try:
        seq = sequence({'features': engine.features(head, tail), 'frames': frames}, head, tail)
    except ValueError as error:
        return None, str(error)
    return retrieve(seq, templates), None


def _recognise(engine: LiveCapture, start_ms: int, end_ms: int) -> dict | None:
    if end_ms - start_ms < MIN_WORD_MS:
        return None
    templates = _usable_templates()
    if not templates:
        return {'matched': False, 'meaning': None, 'distance': None, 'candidates': [],
                'reason': '示范库里还没有可检索的示范：先把动作标成示范（识别就绪）再来比。'}
    best = None
    skipped = None
    # 速度阈值切出来的边界总有几十到一百多毫秒的偏差。先试切分出来的那个窗口：
    # 多数情况下它就是对的，而每多试一次都要对全部示范做一遍对齐，慢了用户就等在那儿。
    found, why = _window_score(engine, start_ms, end_ms, templates)
    if found is not None:
        best = found
        if found['matched']:
            return found
    skipped = why or skipped
    for shift_start in (-120, 120):
        for shift_end in (-120, 120):
            head = max(0, start_ms + shift_start)
            tail = max(head + 1, end_ms + shift_end)
            found, why = _window_score(engine, head, tail, templates)
            if found is None:
                skipped = why or skipped
                continue
            if best is None or (found.get('distance') is not None and (best.get('distance') is None or found['distance'] < best['distance'])):
                best = found
    if best is None:
        return {'matched': False, 'meaning': None, 'distance': None, 'candidates': [],
                'reason': skipped or '没有可用的动作窗口，请把手抬到画面中间再试一次。'}
    return best


_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sign-match')
_future=None
_completed=None

def _match_snapshot(engine,start,end,trigger,session):
    global _completed
    began=time.perf_counter()
    try:
        word=_recognise(engine,start,end)
        if word is None:return
        ident='livecheck_'+uuid4().hex
        put(ident,'sign_live_check',{'id':ident,'created_at':time.time(),'start_ms':start,'end_ms':end,
            'session_id':session,'trigger':trigger,'recognition_ms':round((time.perf_counter()-began)*1000,1),
            'frame_count':len(engine.window(start,end)),'preview_frames':[{'t':f['timestamp_ms']-start,'hands':skeleton(f['tracks'])} for f in engine.window(start,end)],
            'features':engine.features(start,end),'result':word.copy(),'expected_meaning':None})
        for record in sorted(all_of('sign_live_check'),key=lambda x:x['created_at'])[:-50]:delete(record['id'],'sign_live_check')
        word['diagnostic_id']=ident
        with _lock:
            if session==_state.get('session_id'):_completed=word
    except Exception:logging.exception('Background sign matching failed')


def _process(jpeg: bytes, ts: int) -> dict:
    global _engine,_future,_completed
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, '画面解码失败，请重试')
    with _lock:
        if _engine is None:
            _engine = LiveCapture(ROOT / 'models')
        # 用客户端时间差推进服务端时间轴：既要单调（MediaPipe 的要求），也要保留真实间隔，
        # 否则 100ms 推一帧会被压成 33ms，动作时长与静默判定全部失真。
        raw = int(ts)
        previous_raw = _state.get('raw_ts')
        delta = 33 if previous_raw is None else raw - previous_raw
        _state['raw_ts'] = raw
        if delta <= 0 or delta > 1000:
            delta = 33
        now = _state['last_ts'] + max(20, min(300, delta))
        _state['last_ts'] = now
        info = _engine.push(image, now)
        speed = _engine.wrist_motion()
        _state['missing'] = 0 if info['hands'] else _state['missing'] + 1
        word=_completed;_completed=None
        if word and word.get('matched'):
            _state['heard'].insert(0,{'meaning':word['meaning'],'distance':round(word['distance'],3),'at':int(time.time()*1000)})
            del _state['heard'][8:]
        if _state['phase'] == 'idle':
            if speed < QUIET_SPEED:
                _state['last_quiet'] = now
            if info['hands'] and speed >= ACTIVE_SPEED:
                # 从「最后一次静止」开始收，速度要等手真的动起来才过阈值，直接取当前时刻会吃掉起手。
                anchor = _state['last_quiet'] if _state['last_quiet'] is not None else now
                _state['phase'] = 'recording'
                _state['start_ms'] = max(0, max(anchor, now - LOOK_BACK_MS))
                _state['quiet_since'] = None
        elif _state['phase']=='recording':
            if _state['missing'] >= MISSING_FRAMES:
                speed = 0.0
            if speed < QUIET_SPEED:
                _state['quiet_since'] = _state['quiet_since'] or now
            else:
                _state['quiet_since'] = None
            span = now - _state['start_ms']
            quiet = _state['quiet_since'] is not None and now - _state['quiet_since'] >= QUIET_HOLD_MS
            if span >= MAX_WORD_MS or (quiet and span >= MIN_WORD_MS):
                end = _state['quiet_since'] if quiet else now
                if _future is None or _future.done():
                    snapshot=object.__new__(LiveCapture)
                    snapshot.frames=deepcopy(_engine.frames)
                    snapshot.width,snapshot.height=_engine.width,_engine.height
                    _future=_pool.submit(_match_snapshot,snapshot,_state['start_ms'],end,'quiet' if quiet else 'max_duration',_state.get('session_id'))
                _state['phase'] = 'idle'
                _state['start_ms'] = None
                _state['quiet_since'] = None
        return {'phase': _state['phase'], 'hands': info['hands'], 'suspect': info['suspect'],
                'speed': round(speed, 2), 'word': word, 'heard': list(_state['heard']),
                'matching':bool(_future and not _future.done()),'skeleton': info['skeleton'], 'ready': bool(_usable_templates())}


@router.post('/frame')
async def frame(request: Request, ts: int = 0):
    data = await request.body()
    if not data:
        raise HTTPException(422, '没有收到画面')
    return await run_in_threadpool(_process, data, ts)


@router.post('/reset')
def reset():
    """清空当前一词的进度，但不回退时间轴：MediaPipe 的视频模式要求时间戳单调递增。"""
    global _completed
    with _lock:
        _completed=None
        _state.update({'phase': 'idle', 'start_ms': None, 'quiet_since': None, 'last_quiet': None,
                       'heard': [], 'missing': 0,'session_id':uuid4().hex})
        if _engine is not None:
            _engine.reset()
    # 开始之前多半刚标注过示范，别让缓存里的旧模板拖到 TTL 过期。
    _usable_templates(force=True)
    return {'ok': True}


@router.delete('')
def close():
    """关闭摄像头时释放模型实例，把内存还给系统。"""
    global _engine,_completed
    with _lock:
        _completed=None
        if _engine is not None:
            _engine.close()
            _engine = None
        _state.update({'phase': 'idle', 'start_ms': None, 'quiet_since': None, 'last_ts': 0,
                       'raw_ts': None, 'last_quiet': None, 'heard': [], 'missing': 0,'session_id':uuid4().hex})
    return {'ok': True}


class Correction(BaseModel):
    meaning:str=Field(min_length=1,max_length=100)
    include_training:bool=True

@router.get('/diagnostics')
def diagnostics():
    return [{k:v for k,v in r.items() if k not in ('features','preview_frames')} for r in sorted(all_of('sign_live_check'),key=lambda x:x['created_at'],reverse=True)]

@router.post('/diagnostics/{ident}/label')
def label_diagnostic(ident:str,body:Correction):
    record=get(ident,'sign_live_check')
    if not record:raise HTTPException(404,'诊断记录不存在或已过期')
    if not body.meaning.strip():raise HTTPException(422,'请填写实际含义')
    if body.include_training:
        try:seq=sequence({'features':record['features']},record['start_ms'],record['end_ms'])
        except (ValueError,KeyError) as e:raise HTTPException(422,'该片段暂不能训练：'+str(e))
        sample_id='correction_'+ident
        put(sample_id,'sign_training_sample',{'id':sample_id,'meaning':body.meaning.strip(),'sequence':seq,
            'video_id':'live_session_'+record.get('session_id','legacy'),'source_diagnostic_id':ident,
            'recognition_ready':True,'provenance':'user_confirmed_live_correction','created_at':time.time()})
        record['training_sample_id']=sample_id
    elif record.get('training_sample_id'):
        sample=get(record['training_sample_id'],'sign_training_sample')
        if sample:sample['meaning']=body.meaning.strip();put(sample['id'],'sign_training_sample',sample)
    record['expected_meaning']=body.meaning.strip();put(ident,'sign_live_check',record)
    if record.get('training_sample_id'):
        _template_cache['ok']=False
        from .sign_retraining import request_training
        revision=request_training()
        return {'ok':True,'revision':revision,'message':'纠正已热更新到识别库，下一次匹配即可使用；GRU训练已在后台排队。'}
    return {'ok':True,'message':'已记录纠正，用于误差分析；不会自动当作训练示范。'}


@router.post('/manual/start')
def manual_start():
    with _lock:
        if _engine is None or not _engine.frames:raise HTTPException(409,'请先开启实时识别并等待摄像头画面')
        _state.update(phase='manual',start_ms=_state['last_ts'],quiet_since=None)
    return {'ok':True}

@router.post('/manual/end')
def manual_end():
    with _lock:
        if _engine is None or _state['phase']!='manual':raise HTTPException(409,'请先标记动作开始')
        start=_state['start_ms'];end=_state['last_ts']
        _state.update(phase='manual_idle',start_ms=None,quiet_since=None)
        if end-start>6500:raise HTTPException(422,'动作超过6.5秒，请缩短后重新标记，避免缓存截断')
        if end-start<MIN_WORD_MS:raise HTTPException(422,'动作区间太短，请重新标记完整动作')
        began=time.perf_counter();word,why=_window_score(_engine,start,end,_usable_templates())
        if word is None:raise HTTPException(422,why or '没有可识别动作')
        ident='livecheck_'+uuid4().hex
        put(ident,'sign_live_check',{'id':ident,'created_at':time.time(),'start_ms':start,'end_ms':end,'session_id':_state.get('session_id','legacy'),'trigger':'manual',
            'recognition_ms':round((time.perf_counter()-began)*1000,1),'frame_count':len(_engine.window(start,end)),
            'preview_frames':[{'t':f['timestamp_ms']-start,'hands':skeleton(f['tracks'])} for f in _engine.window(start,end)],
            'features':_engine.features(start,end),'result':word.copy(),'expected_meaning':None})
        checks=sorted(all_of('sign_live_check'),key=lambda x:x['created_at'])
        for record in checks[:-50]:delete(record['id'],'sign_live_check')
        word['diagnostic_id']=ident
        return word


@router.get('/diagnostics/{ident}/preview')
def preview_diagnostic(ident:str):
    record=get(ident,'sign_live_check')
    if not record:raise HTTPException(404,'诊断记录不存在或已过期')
    return {'id':ident,'duration_ms':record['end_ms']-record['start_ms'],'frames':record.get('preview_frames',[]),
            'result':record['result'],'expected_meaning':record.get('expected_meaning')}


@router.get('/training/status')
def training_status():
    from .sign_retraining import status
    return status()


@router.post('/training/retrain')
def retrain_now():
    from .sign_retraining import request_training
    _template_cache['ok']=False
    revision=request_training()
    return {'ok':True,'revision':revision,'message':'已使用当前示范库和纠错样本提交后台训练；已删除的示范不会从历史模型补回。'}
