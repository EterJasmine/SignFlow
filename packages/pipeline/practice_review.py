"""Summarize measured practice history without inferring semantic mastery."""
from collections import defaultdict

def build_review(sessions):
    groups=defaultdict(list);seen=set();progress={}
    for session in sorted(sessions,key=lambda s:s.get('created_at',0)):
        reliable=[]
        for attempt in session.get('attempts',[]):
            ident=attempt.get('diagnostic_id')
            if not ident or ident in seen:continue
            seen.add(ident);groups[session['meaning']].append(attempt)
            if attempt.get('coverage',0)>=.35 and attempt.get('distance') is not None:reliable.append(attempt)
        if len(reliable)>=2:
            delta=reliable[0]['distance']-reliable[-1]['distance']
            progress[session['meaning']]='最近一次多轮练习中更接近示范' if delta>.01 else '最近一次多轮练习中差异变大' if delta<-.01 else '最近一次多轮练习变化不大'
    words=[]
    for word,attempts in groups.items():
        last=attempts[-1];quality=sum(a.get('coverage',0)<.35 for a in attempts)
        reliable=[a for a in attempts if a.get('coverage',0)>=.35 and a.get('distance') is not None]
        action=last.get('decision',{}).get('action')
        advice='先调整光线和取景，再重练' if last.get('coverage',0)<.35 else '先慢放重看示范，再做一次' if action=='review_demo' else '再独立做一次，确认表现稳定'
        words.append({'meaning':word,'attempts':len(attempts),'comparable_attempts':len(reliable),'unclear_attempts':quality,
                      'trend':progress.get(word,'还没有足够的同次练习记录判断变化'),'suggestion':advice})
    words.sort(key=lambda w:(-w['unclear_attempts'],-w['attempts'],w['meaning']))
    total=sum(w['attempts'] for w in words)
    if not words:summary='还没有完成的练习记录。先选一个词，完成一次动作分析，小助手就能帮你回顾。'
    else:
        named='、'.join(w['meaning'] for w in words[:8]);summary=f'你已经练习了{named}，共完成{total}次动作分析。'
        improved=[w['meaning'] for w in words if '更接近' in w['trend']]
        if improved:summary+='其中，'+ '、'.join(improved[:4])+'在最近一次多轮练习中更接近示范。'
        summary+='下次建议先练「'+words[0]['meaning']+'」：'+words[0]['suggestion']+'。'
    return {'total_attempts':total,'word_count':len(words),'words':words,'summary':summary,'source':'rules',
            'scope':'仅基于已保存的小助手动作分析记录；相似度变化不代表已掌握手语。'}
