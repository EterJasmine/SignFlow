import sqlite3, json, os
from pathlib import Path
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get('GESTUREFLOW_DATA',ROOT/'data'))
DATA.mkdir(parents=True,exist_ok=True)
@contextmanager
def connection():
    db=sqlite3.connect(DATA/'gestureflow.db',timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL)')
    try:yield db;db.commit()
    finally:db.close()

def put(ident,kind,body):
    with connection() as db:db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)',(ident,kind,json.dumps(body,ensure_ascii=False)))
    return body
def get(ident,kind=None):
    with connection() as db:
        row=db.execute('SELECT kind,body FROM records WHERE id=?',(ident,)).fetchone()
    return json.loads(row[1]) if row and (kind is None or row[0]==kind) else None
def all_of(kind):
    with connection() as db:rows=db.execute('SELECT body FROM records WHERE kind=?',(kind,)).fetchall()
    return [json.loads(row[0]) for row in rows]
def delete(ident,kind=None):
    """Remove one record. Returns True when a row was actually removed."""
    with connection() as db:
        if kind is None:
            cur=db.execute('DELETE FROM records WHERE id=?',(ident,))
        else:
            cur=db.execute('DELETE FROM records WHERE id=? AND kind=?',(ident,kind))
    return cur.rowcount>0
def update_job(ident,**values):
    # One SQLite write transaction prevents cancellation from being overwritten.
    with connection() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT body FROM records WHERE id=?',(ident,)).fetchone()
        if not row:raise KeyError(ident)
        body=json.loads(row[0]);body.update(values)
        db.execute('UPDATE records SET body=? WHERE id=?',(json.dumps(body),ident))
    return body
