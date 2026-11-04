"""Offline mirror comparison for reflection-invariant template features only."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apps.api.store import all_of
from packages.pipeline.signmatch import dtw,usable

def mirror(seq):
    names=seq['names'];columns=[]
    # These features are lengths and unsigned angles; reflection swaps hand slots.
    allowed=('angle','tip_radius','thumb_distance','face_distance','chest_distance','hand_distance')
    if any(not any(k in n for k in allowed) for n in names):raise ValueError('Directional features require explicit transformation')
    for n in names:
        other=n.replace('hand_0_','hand_1_',1) if n.startswith('hand_0_') else n.replace('hand_1_','hand_0_',1) if n.startswith('hand_1_') else n
        columns.append(names.index(other))
    return {**seq,'rows':[[r[i] for i in columns] for r in seq['rows']]}

def main():
    items=[t for t in all_of('sign_template') if usable(t)];details=[]
    for query in items:
        candidates=[t for t in items if t['video_id']!=query['video_id']]
        if not any(t['meaning']==query['meaning'] for t in candidates):continue
        plain=[];mirrored=[];qmirror=mirror(query['sequence'])
        for t in candidates:
            try:
                d=dtw(query['sequence'],t['sequence']);m=dtw(qmirror,t['sequence'])
                plain.append((d,t['meaning']));mirrored.append((min(d,m),t['meaning']))
            except ValueError:continue
        if plain:details.append({'id':query['id'],'video_id':query['video_id'],'expected':query['meaning'],'original_nearest':min(plain)[1],'mirror_nearest':min(mirrored)[1]})
    report={'eligible':len(details),'original_top1':sum(d['original_nearest']==d['expected'] for d in details)/len(details) if details else None,'mirror_top1':sum(d['mirror_nearest']==d['expected'] for d in details)/len(details) if details else None,'details':details,'note':'Leave-source-video-out nearest-template retrieval, not calibrated recognition accuracy. Only labels represented in other videos are evaluated. No deployment.'}
    target=ROOT/'evaluation/results/sign-mirror-check.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print({k:v for k,v in report.items() if k!='details'})
if __name__=='__main__':main()
