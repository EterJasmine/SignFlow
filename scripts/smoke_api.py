from pathlib import Path
import sys,time,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
root=Path(__file__).resolve().parents[1]
os.environ['GESTUREFLOW_DATA']=str(root/'data'/'api-smoke')
from fastapi.testclient import TestClient
from apps.api.main import app
with TestClient(app) as client:
    p=client.post('/api/v1/projects',json={'name':'官方测试素材验收'});assert p.status_code==201
    with (root/'data'/'smoke'/'official-pose.mp4').open('rb') as f:
        v=client.post('/api/v1/projects/'+p.json()['id']+'/videos',files={'file':('pose.mp4',f,'video/mp4')})
    assert v.status_code==201,v.text
    video_id=v.json()['id']
    j=client.post('/api/v1/videos/'+video_id+'/analysis-jobs',headers={'Idempotency-Key':'smoke-1'});assert j.status_code==202,j.text
    same=client.post('/api/v1/videos/'+video_id+'/analysis-jobs',headers={'Idempotency-Key':'smoke-1'});assert same.json()['id']==j.json()['id']
    for _ in range(240):
        job=client.get('/api/v1/jobs/'+j.json()['id']).json()
        if job['status'] in ['completed','failed','cancelled']:break
        time.sleep(.5)
    assert job['status']=='completed',job
    result=client.get('/api/v1/videos/'+video_id+'/result');assert result.status_code==200,result.text
    r=result.json();assert r['frames'];assert r['scenes'][0]['framing_type']=='full_body'
    for kind in ['source','diagram']:
        asset=client.get(r['actions'][0]['assets'][kind]);assert asset.status_code==200
    lyrics=[{'id':'lyric-1','start_ms':0,'end_ms':1000,'text':'测试歌词'}]
    assert client.put('/api/v1/videos/'+video_id+'/lyrics',json=lyrics).status_code==200
    updated=client.get('/api/v1/videos/'+video_id+'/result').json();assert updated['actions'][0]['lyric_ids']==['lyric-1']
    assert client.get('/api/v1/videos/'+video_id+'/export').status_code==200
    sheet=client.get('/api/v1/videos/'+video_id+'/tutorial.html');assert sheet.status_code==200 and 'STEP 01' in sheet.text and '测试歌词' in sheet.text,sheet.status_code
    long_png=client.get('/api/v1/videos/'+video_id+'/tutorial.png')
    assert long_png.status_code==200 and long_png.content[:4]==b'\x89PNG',(long_png.status_code,long_png.content[:160])
    pdf=client.get('/api/v1/videos/'+video_id+'/tutorial.pdf')
    assert pdf.status_code==200 and pdf.content[:4]==b'%PDF',(pdf.status_code,pdf.content[:160])
    assert client.put('/api/v1/videos/'+video_id+'/lyrics',json=[{'start_ms':2000,'end_ms':1000,'text':'错误'}]).status_code==422
    imported=client.post('/api/v1/videos/'+video_id+'/lyrics/import',json={'lrc':'[ti:烟雾]\n[00:00.10]第一句\n[00:00.80]第二句'})
    assert imported.status_code==200 and imported.json()['source']=='lrc' and imported.json()['count']==2,imported.text[:200]
    assert client.get('/api/v1/ai/asr').status_code==200
    assert client.post('/api/v1/videos/'+video_id+'/lyrics/import',json={'lrc':'没有时间戳'}).status_code==422
    print({'status':'passed','job':job['status'],'frames':len(r['frames']),'scene':r['scenes'][0]['framing_type'],'assets':'source+diagram','lyrics':'overlap mapped + lrc import','tutorial':'html+png+pdf'})
