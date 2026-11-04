from apps.api.coach_routes import decide

def test_low_quality_does_not_certify_a_close_match():
 assert decide(.1,.01,None)['action']=='adjust_capture'
def test_unavailable_comparison_requests_demo_review():
 assert decide(.9,None,None)['action']=='review_demo'
def test_close_match_requires_independent_repeat():
 d=decide(.8,.1,.3)
 assert d['action']=='repeat_to_confirm' and abs(d['change']-.2)<1e-6


def test_repeated_attempt_does_not_duplicate_history(monkeypatch):
 from apps.api import coach_routes as routes
 session={'id':'s','created_at':1,'attempts':[{'diagnostic_id':'d'}]}
 record={'created_at':2}
 monkeypatch.setattr(routes,'get',lambda ident,kind:session if kind=='sign_coach_session' else record)
 result=routes.evaluate('s',routes.Attempt(diagnostic_id='d'))
 assert len(result['attempts'])==1

def test_old_capture_cannot_count_as_new_practice(monkeypatch):
 import pytest
 from fastapi import HTTPException
 from apps.api import coach_routes as routes
 monkeypatch.setattr(routes,'get',lambda ident,kind:{'created_at':5,'attempts':[]} if kind=='sign_coach_session' else {'created_at':2})
 with pytest.raises(HTTPException) as error:routes.evaluate('s',routes.Attempt(diagnostic_id='d'))
 assert error.value.status_code==422
