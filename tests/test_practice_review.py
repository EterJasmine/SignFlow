from packages.pipeline.practice_review import build_review

def test_empty_history_has_no_invented_progress():
 r=build_review([]);assert r['total_attempts']==0 and r['words']==[]
def test_same_diagnostic_counted_once():
 a={'diagnostic_id':'x','coverage':.8,'distance':.1,'decision':{'action':'repeat_to_confirm'}}
 r=build_review([{'meaning':'我','created_at':1,'attempts':[a,a]}])
 assert r['total_attempts']==1 and '足够' in r['words'][0]['trend']
def test_progress_only_within_same_session():
 r=build_review([{'meaning':'我','created_at':1,'attempts':[{'diagnostic_id':'a','coverage':.8,'distance':.4}]},{'meaning':'我','created_at':2,'attempts':[{'diagnostic_id':'b','coverage':.8,'distance':.1}]}])
 assert '足够' in r['words'][0]['trend']
def test_repeated_practice_trend_is_measured():
 r=build_review([{'meaning':'我','attempts':[{'diagnostic_id':'a','coverage':.8,'distance':.4},{'diagnostic_id':'b','coverage':.8,'distance':.1}]}])
 assert '更接近' in r['words'][0]['trend']
