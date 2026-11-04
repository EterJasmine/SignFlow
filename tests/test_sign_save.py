import pytest
from fastapi import HTTPException
from apps.api import sign_routes as routes

def test_short_sequence_can_be_saved_as_draft(monkeypatch):
    monkeypatch.setattr(routes,'get',lambda *a: {'id':'v'})
    monkeypatch.setattr(routes,'load_result',lambda *a:{'duration_ms':2000})
    body=routes.Selection(video_id='v',start_ms=0,end_ms=1000)
    assert 'draft_reason' in routes.extract(body,allow_draft=True)
    with pytest.raises(HTTPException):routes.extract(body)

def test_invalid_range_cannot_become_draft(monkeypatch):
    monkeypatch.setattr(routes,'get',lambda *a:{'id':'v'})
    monkeypatch.setattr(routes,'load_result',lambda *a:{'duration_ms':1000})
    with pytest.raises(HTTPException):routes.extract(routes.Selection(video_id='v',end_ms=2000),allow_draft=True)
