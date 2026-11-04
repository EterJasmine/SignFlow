"""Debounced, serial experimental retraining; serving templates update independently."""
import json,subprocess,sys,threading,time
from .store import ROOT,DATA,put,get
_lock=threading.Lock();_running=False;_revision=0

def request_training():
    global _running,_revision
    with _lock:
        previous=get('sign_training_status','sign_training_status') or {}
        _revision=max(_revision,previous.get('revision',0))+1
        put('sign_training_status','sign_training_status',{'id':'sign_training_status','revision':_revision,'state':'queued','updated_at':time.time(),'serving':'template_hot_update'})
        if not _running:
            _running=True;threading.Thread(target=_worker,daemon=True,name='sign-retraining').start()
    return _revision

def _worker():
    global _running
    while True:
        time.sleep(2)
        with _lock:revision=_revision
        status={'id':'sign_training_status','revision':revision,'state':'running','updated_at':time.time(),'serving':'template_hot_update'}
        try:
            put(status['id'],'sign_training_status',status)
            log=DATA/'sign-retraining.log'
            with log.open('w',encoding='utf-8') as output:
                process=subprocess.run([sys.executable,str(ROOT/'scripts/train_sign_temporal.py')],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            text=log.read_text(encoding='utf-8',errors='replace')
            if process.returncode:status.update(state='failed',message=text[-2000:])
            else:
                result=json.loads(text.strip().splitlines()[-1]);status.update(state='complete',result=result,message='实验模型已训练；线上识别已通过纠错示范热更新，未自动切换未经验证的分类器。')
            status['updated_at']=time.time();put(status['id'],'sign_training_status',status)
        except Exception as error:
            status.update(state='failed',message=str(error));put(status['id'],'sign_training_status',status)
        with _lock:
            if revision==_revision:_running=False;return

def status():return get('sign_training_status','sign_training_status') or {'state':'idle','serving':'template_hot_update'}
