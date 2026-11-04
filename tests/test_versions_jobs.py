from pathlib import Path
from uuid import uuid4
import pytest
from apps.api import store,versions,jobs

@pytest.fixture
def isolated(monkeypatch):
    path=Path('data/test-isolation')/uuid4().hex;path.mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(store,'DATA',path);monkeypatch.setattr(versions,'DATA',path)
    return path
def test_result_versions_and_optimistic_lock(isolated):
    r=versions.commit_result('v',{'actions':[]},'test',0)
    assert r['result_revision']==1
    with pytest.raises(ValueError):versions.commit_result('v',{'actions':[]},'stale',0)
    r2=versions.commit_result('v',{'actions':[{'title':'manual'}]},'edit',1)
    assert r2['result_revision']==2
    restored=versions.restore('v',r['result_version_id']);assert restored['result_revision']==3 and restored['actions']==[]
    assert len(versions.versions('v'))==3
def test_legacy_json_is_preserved(isolated):
    root=isolated/'assets'/'v';root.mkdir(parents=True);(root/'result.json').write_text('{"old":true}')
    versions.commit_result('v',{'old':False},'upgrade',0)
    assert any(v['reason']=='legacy_baseline' for v in versions.versions('v'))
def test_atomic_job_reservation_and_idempotency(isolated):
    one={'id':'j1','video_id':'v1','status':'queued','key':'key1'}
    assert jobs.reserve_job(one)[1]
    assert not jobs.reserve_job({**one,'id':'j2'})[1]
    with pytest.raises(ValueError):jobs.reserve_job({'id':'j3','video_id':'v2','status':'queued','key':None})
    store.update_job('j1',status='completed');jobs.release_job('j1')
    assert jobs.reserve_job({'id':'j3','video_id':'v2','status':'queued','key':None})[1]
