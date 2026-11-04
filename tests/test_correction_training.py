from apps.api import live_routes as routes

def test_confirmed_correction_is_idempotent_and_separate(monkeypatch):
    from apps.api import sign_retraining
    monkeypatch.setattr(sign_retraining,'request_training',lambda:1)
    names=[f'hand_0_finger_{i}_angle' for i in range(12)]
    features={'version':'test','names':names,'rows':[{'timestamp_ms':i*100,'values':[.2]*12,'mask':[True]*12} for i in range(5)]}
    records={'d':{'id':'d','start_ms':0,'end_ms':500,'features':features,'session_id':'session'}}
    monkeypatch.setattr(routes,'get',lambda ident,kind=None:records.get(ident))
    monkeypatch.setattr(routes,'put',lambda ident,kind,body:records.update({ident:body}) or body)
    routes.label_diagnostic('d',routes.Correction(meaning='谢谢',include_training=True))
    routes.label_diagnostic('d',routes.Correction(meaning='你好',include_training=True))
    assert len(records)==2
    assert records['correction_d']['meaning']=='你好'
    assert records['correction_d']['video_id']=='live_session_session'
    assert records['d']['training_sample_id']=='correction_d'
