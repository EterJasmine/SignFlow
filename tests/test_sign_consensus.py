from packages.pipeline import signmatch

def test_single_accidental_close_template_does_not_dominate(monkeypatch):
    templates=[{'id':str(i),'meaning':meaning,'sequence':{'rows':[[distance]],'distance':distance}} for i,(meaning,distance) in enumerate([('a',.01),('a',.9),('a',.9),('b',.12),('b',.13),('b',.14)])]
    monkeypatch.setattr(signmatch,'dtw',lambda a,b:b['distance'])
    result=signmatch.retrieve({},templates)
    assert result['meaning']=='b'
    assert len(result['candidates'])==2
    assert result['candidates'][0]['support_count']==3

def test_single_example_word_is_still_available(monkeypatch):
    monkeypatch.setattr(signmatch,'dtw',lambda a,b:.1)
    result=signmatch.retrieve({},[{'id':'a','meaning':'a','sequence':{'rows':[[0]]}}])
    assert result['matched'] and result['candidates'][0]['support_count']==1
