import pytest
from training.annotations import Annotation
from training.export import group_splits,export_dataset
def test_labels_require_known_taxonomy():
    with pytest.raises(ValueError):Annotation(kind='dynamic',label='fake',start_ms=0,end_ms=100)
def test_splits_never_split_one_video():
    a=[{'id':str(i),'video_id':f'v{i//3}'} for i in range(12)]
    split=group_splits(a)
    for video in {x['video_id'] for x in a}:assert len({split[x['id']] for x in a if x['video_id']==video})==1
def test_no_labels_cannot_export():
    with pytest.raises(ValueError,match='人工'):export_dataset([],lambda _:None,'unused.npz')
def test_small_dataset_is_not_marked_training_ready():
    assert set(group_splits([{'id':'a','video_id':'v1'}]).values())=={'unassigned'}
def test_annotation_records_boundary_provenance():
    seeded=Annotation(kind='dynamic',label='wave',start_ms=0,end_ms=500,action_id='action_1')
    assert seeded.boundary_source=='action_seed' and seeded.action_id=='action_1'
    manual=Annotation(kind='dynamic',label='wave',start_ms=0,end_ms=500,boundary_source='manual',action_id='')
    assert manual.boundary_source=='manual' and manual.action_id==''
    with pytest.raises(ValueError):Annotation(kind='dynamic',label='wave',start_ms=0,end_ms=500,boundary_source='guess')
