from evaluation.metrics import macro_f1,boundary_f1,edit_score
def test_perfect_metrics():
    assert macro_f1([0,1],[0,1],2)==1
    assert boundary_f1([1000,2000],[1100,1900])==1
    assert edit_score([0,0,1],[0,1,1])==100
def test_boundary_matching_is_one_to_one():
    assert boundary_f1([1000],[990,1010])==2/3
