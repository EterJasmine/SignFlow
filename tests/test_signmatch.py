import pytest
from packages.pipeline.signmatch import dtw,sequence

def seq(rows):return {'version':'test','names':['a','b','c'],'rows':rows}

def test_dtw_tolerates_repeated_frames():
    a=seq([[0,0,0],[1,1,1]])
    assert dtw(a,seq([[0,0,0],[0,0,0],[1,1,1]]))==0
    assert dtw(a,seq([[2,2,2],[2,2,2]]))>0

def test_missing_features_are_not_a_perfect_match():
    assert dtw(seq([[None]*3]),seq([[None]*3]))==2

def test_feature_version_mismatch_is_rejected():
    a=seq([[0]*3]);b=seq([[0]*3]);b['version']='other'
    with pytest.raises(ValueError):dtw(a,b)

def test_empty_sequence_rejected():
    with pytest.raises(ValueError):sequence({},0,1000)


def test_long_alignment_retains_endpoints_and_bounds_work():
    from packages.pipeline.signmatch import _as_array
    a,mask=_as_array(seq([[i,i,i] for i in range(160)]))
    assert len(a)==48 and a[0,0]==0 and a[-1,0]==159
    assert mask.all()
