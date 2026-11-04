from apps.api import live_routes as routes

def test_reset_clears_undelivered_result_and_rotates_session(monkeypatch):
    monkeypatch.setattr(routes,'_engine',None)
    monkeypatch.setattr(routes,'_completed',{'meaning':'old'})
    monkeypatch.setattr(routes,'_state',dict(routes._state))
    monkeypatch.setattr(routes,'_usable_templates',lambda **kwargs:[])
    session=routes._state['session_id']
    routes.reset()
    assert routes._completed is None
    assert routes._state['session_id']!=session
