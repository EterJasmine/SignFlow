from packages.pipeline import coach_llm

def test_missing_key_preserves_rule_feedback(monkeypatch):
 monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
 assert coach_llm.explain({},'原反馈')['feedback']=='原反馈'

def test_network_failure_does_not_break_coach(monkeypatch):
 monkeypatch.setenv('DEEPSEEK_API_KEY','test');monkeypatch.setenv('DEEPSEEK_MODEL','test')
 def broken(*args,**kwargs):raise RuntimeError('network')
 monkeypatch.setattr(coach_llm.httpx,'Client',broken)
 out=coach_llm.explain({},'原反馈')
 assert out['feedback']=='原反馈' and out['source']=='rules'
