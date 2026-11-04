"""Scene framing is separate from person count. Missing points are never filled."""
from collections import Counter

def visible(point, threshold=.6):
    return ((point.get('visibility') or 0) >= threshold
            and 0.015 <= point['x'] <= .985 and 0.015 <= point['y'] <= .985)

def evidence(poses, hands):
    parts = set()
    for pose in poses:
        for name, indices in {'face':[0,2,5], 'shoulders':[11,12], 'hips':[23,24],
                              'knees':[25,26], 'ankles':[27,28]}.items():
            if all(visible(pose[i]) for i in indices):
                parts.add(name)
    if hands:
        parts.add('hands')
    # Require evidence from one body rather than combining different people.
    full = any(all(visible(p[i]) for i in [11,12,23,24,25,26,27,28]) for p in poses)
    upper = any(all(visible(p[i]) for i in [11,12]) for p in poses)
    kind = ('full_body' if full else 'upper_body' if upper else 'hands_only' if hands
            else 'partial_body' if poses else 'unknown')
    return {'framing_type':kind, 'person_count':len(poses), 'visible_parts':sorted(parts)}

def pipeline_for(kind, people=1):
    return ['hands'] + (['pose'] if kind != 'hands_only' else []) + (['face'] if kind == 'upper_body' else [])

def summarize_scene(frames, start, end, scene_id):
    # A shot-level majority is used rather than flickering per-frame labels.
    counts = Counter(f['evidence']['framing_type'] for f in frames)
    kind, votes = counts.most_common(1)[0] if counts else ('unknown', 0)
    confidence = votes / max(1, len(frames))
    if confidence < .55:
        kind = 'unknown'
    people = max((f['evidence']['person_count'] for f in frames), default=0)
    return {'id':scene_id, 'start_ms':start, 'end_ms':end, 'framing_type':kind,
            'person_count':people, 'visible_parts':sorted({p for f in frames for p in f['evidence']['visible_parts']}),
            'pipeline':pipeline_for(kind, people), 'confidence':round(confidence, 3),
            'review_status':'pending', 'classification_method':'shot_majority_visibility_v1'}
