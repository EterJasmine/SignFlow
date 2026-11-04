import json,time,os
from .store import connection
from .versions import migrate

def reserve_job(job):
    migrate()
    with connection() as db:
        db.execute('BEGIN IMMEDIATE');now=time.time()
        # State and lease are checked and written under the same SQLite lock.
        rows=db.execute("SELECT body FROM records WHERE kind='job'").fetchall()
        for row in rows:
            existing=json.loads(row[0])
            if existing['video_id']==job['video_id'] and (existing['status'] in ['queued','processing'] or job.get('key') and existing.get('key')==job['key']):return existing,False
        lease=db.execute('SELECT owner,expires FROM task_leases WHERE resource=?',('analysis',)).fetchone()
        if lease and lease[1]>now:raise ValueError('已有分析任务运行，请等待或取消')
        db.execute('INSERT OR REPLACE INTO task_leases VALUES (?,?,?)',('analysis',job['id'],now+3600))
        db.execute('INSERT INTO records VALUES (?,?,?)',(job['id'],'job',json.dumps(job)))
    return job,True
def release_job(ident):
    migrate()
    with connection() as db:db.execute('DELETE FROM task_leases WHERE owner=?',(ident,))

def pid_alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p;kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        code=ctypes.c_ulong()
        try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value==259)
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except OSError:return False
