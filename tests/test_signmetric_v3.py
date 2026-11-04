from packages.pipeline.signmatch import dtw

def test_experimental_missing_penalty_preserves_exact_self_match():
    a={'version':'test-motion3','names':['a','b','c','d'],'rows':[[0,0,0,None],[1,1,1,None]]}
    assert dtw(a,a)==0

def test_missing_hand_features_raise_experimental_distance():
    a={'version':'test-motion3','names':['a','b','c','d'],'rows':[[0,0,0,0]]}
    b={**a,'rows':[[0,0,0,None]]}
    assert dtw(a,b)>0
    assert dtw({**a,'version':'test'},{**b,'version':'test'})==0
