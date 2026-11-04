"""Evaluate paired human labels and predictions; no labels means no accuracy claim."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .metrics import macro_f1,boundary_f1,edit_score

def identity_f1(matches,ground_truth_detections,predicted_detections):
    if ground_truth_detections+predicted_detections==0:return None
    gt=sorted({a for a,b in matches});pred=sorted({b for a,b in matches});matrix=np.zeros((len(gt),len(pred)))
    for a,b in matches:matrix[gt.index(a),pred.index(b)]+=1
    tp=0
    if gt and pred:
        rows,cols=linear_sum_assignment(-matrix);tp=matrix[rows,cols].sum()
    return float(2*tp/(ground_truth_detections+predicted_detections))
def evaluate(document):
    if document.get('ground_truth_source')!='human':raise ValueError('需要人工真值，不能用规则预测充当标签')
    output={'schema_version':'evaluation-1.0','videos':[]}
    for video in document.get('videos',[]):
        item={'video_id':video['video_id']}
        if 'scene_truth' in video:item['scene_macro_f1']=macro_f1(video['scene_truth'],video['scene_pred'],video['scene_classes'])
        if 'action_truth' in video:item['action_macro_f1']=macro_f1(video['action_truth'],video['action_pred'],video['action_classes'])
        if 'boundaries_truth_ms' in video:item['boundary_f1_300ms']=boundary_f1(video['boundaries_truth_ms'],video.get('boundaries_pred_ms',[]),300)
        if 'phase_truth' in video:item['phase_edit_score']=edit_score(video['phase_truth'],video['phase_pred'])
        if 'identity_matches' in video:item['idf1']=identity_f1(video['identity_matches'],video['ground_truth_detections'],video['predicted_detections'])
        if 'sampled_frames' in video:item['detection_coverage']=video['detected_frames']/max(1,video['sampled_frames'])
        if 'diagram_correct' in video:item['human_diagram_agreement']=sum(video['diagram_correct'])/max(1,len(video['diagram_correct']))
        output['videos'].append(item)
    return output
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',default='evaluation/results/report.json');args=p.parse_args()
    report=evaluate(json.loads(Path(args.input).read_text(encoding='utf8')));out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
