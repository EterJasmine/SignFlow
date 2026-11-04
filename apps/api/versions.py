"""Append-only JSON revisions on existing SQLite; legacy result.json stays readable."""
from pathlib import Path
from uuid import uuid4
from collections import OrderedDict
import json,time,hashlib,threading
from .store import DATA,connection

# One analysed result is ~10MB of frames and features; re-parsing it per request
# starved the video stream while the user was only tagging a word. Keep the last
# few results in memory and re-read only when the file on disk changes.
_RESULT_CACHE: OrderedDict = OrderedDict()
_RESULT_LOCK = threading.Lock()
_RESULT_CACHE_LIMIT = 2

def invalidate(video_id):
    with _RESULT_LOCK:
        _RESULT_CACHE.pop(str(video_id), None)

def migrate():
    with connection() as db:
        db.execute('CREATE TABLE IF NOT EXISTS result_versions (id TEXT PRIMARY KEY,video_id TEXT,revision INTEGER,path TEXT,created REAL,reason TEXT,sha256 TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS result_heads (video_id TEXT PRIMARY KEY,revision INTEGER,version_id TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS task_leases (resource TEXT PRIMARY KEY,owner TEXT,expires REAL)')
        db.execute('PRAGMA user_version=2')
def load_result(video_id):
    migrate()
    with connection() as db:row=db.execute('SELECT v.path FROM result_heads h JOIN result_versions v ON v.id=h.version_id WHERE h.video_id=?',(video_id,)).fetchone()
    path=Path(row[0]) if row else DATA/'assets'/video_id/'result.json'
    if not path.exists():raise FileNotFoundError('分析尚未完成')
    try:stamp=path.stat().st_mtime_ns
    except OSError:stamp=None
    key=str(video_id)
    # Parsing holds the lock on purpose: concurrent preview/summary requests would
    # otherwise each parse the same 10MB file and fight over the GIL.
    with _RESULT_LOCK:
        hit=_RESULT_CACHE.get(key)
        if hit and stamp is not None and hit[0]==stamp:
            _RESULT_CACHE.move_to_end(key)
            return hit[1]
        data=json.loads(path.read_text(encoding='utf8'))
        _RESULT_CACHE[key]=(stamp,data);_RESULT_CACHE.move_to_end(key)
        while len(_RESULT_CACHE)>_RESULT_CACHE_LIMIT:_RESULT_CACHE.popitem(last=False)
        return data
def commit_result(video_id,result,reason,expected_revision=None):
    migrate();ident='revision_'+uuid4().hex
    with connection() as db:
        db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT revision FROM result_heads WHERE video_id=?',(video_id,)).fetchone();revision=row[0] if row else 0
        if expected_revision is not None and revision!=expected_revision:raise ValueError('结果已被其他操作更新，请刷新后重试')
        legacy=DATA/'assets'/video_id/'result.json'
        if row is None and legacy.exists():
            legacy_id='revision_'+uuid4().hex;legacy_path=legacy.parent/'history'/legacy_id/'result.json';legacy_path.parent.mkdir(parents=True,exist_ok=True);legacy_content=legacy.read_text(encoding='utf8');legacy_path.write_text(legacy_content,encoding='utf8')
            db.execute('INSERT INTO result_versions VALUES (?,?,?,?,?,?,?)',(legacy_id,video_id,0,str(legacy_path),time.time(),'legacy_baseline',hashlib.sha256(legacy_content.encode()).hexdigest()))
        copy={**result,'result_revision':revision+1,'result_version_id':ident,'schema_version':'1.1'}
        path=DATA/'assets'/video_id/'history'/ident/'result.json';path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(copy,ensure_ascii=False,allow_nan=False);path.write_text(content,encoding='utf8')
        db.execute('INSERT INTO result_versions VALUES (?,?,?,?,?,?,?)',(ident,video_id,revision+1,str(path),time.time(),reason,hashlib.sha256(content.encode()).hexdigest()))
        db.execute('INSERT OR REPLACE INTO result_heads VALUES (?,?,?)',(video_id,revision+1,ident))
    # Compatibility mirror only; database head references the immutable authoritative JSON.
    mirror=DATA/'assets'/video_id/'result.json';tmp=mirror.with_name('result-'+uuid4().hex+'.tmp');tmp.write_text(content,encoding='utf8');tmp.replace(mirror)
    invalidate(video_id)
    return copy
def versions(video_id):
    migrate()
    with connection() as db:rows=db.execute('SELECT id,revision,created,reason,sha256 FROM result_versions WHERE video_id=? ORDER BY revision DESC',(video_id,)).fetchall()
    return [dict(zip(['id','revision','created','reason','sha256'],r)) for r in rows]
def restore(video_id,version_id):
    migrate()
    with connection() as db:row=db.execute('SELECT path FROM result_versions WHERE video_id=? AND id=?',(video_id,version_id)).fetchone()
    if not row:raise ValueError('历史版本不存在')
    current=load_result(video_id);return commit_result(video_id,json.loads(Path(row[0]).read_text(encoding='utf8')),'restore:'+version_id,current.get('result_revision',0))
