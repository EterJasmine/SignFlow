"""Experimental motion-aware features. Not enabled for live recognition."""
import math
from .signmatch import sequence,MIN_COLUMNS

def motion_sequence(result,start,end):
    base=sequence(result,start,end);features=result['features'];names=features['names']
    selected=[names.index(n) for n in base['names']]
    rows=[r for r in features['rows'] if start<=r['timestamp_ms']<end and sum(bool(r['mask'][i]) for i in selected)>=MIN_COLUMNS]
    rows=rows[::max(1,int(math.ceil(len(rows)/160)))];origins={};output=[]
    extra=[f'hand_{slot}_{name}' for slot in range(2) for name in ('travel_x','travel_y','rotation_sin','rotation_cos')]
    for row,values in zip(rows,base['rows']):
        values=list(values)
        for slot in range(2):
            def read(key):
                n=f'hand_{slot}_{key}'
                if n not in names:return None
                i=names.index(n);return row['values'][i] if row['mask'][i] else None
            x,y,rotation=read('wrist_x'),read('wrist_y'),read('rotation')
            if x is not None and y is not None:
                origin=origins.setdefault(slot,(x,y));values.extend([(x-origin[0])*4,(y-origin[1])*4])
            else:values.extend([None,None])
            values.extend([math.sin(rotation)*.5,math.cos(rotation)*.5] if rotation is not None else [None,None])
        output.append(values)
    return {'version':str(base['version'])+'-motion3','names':base['names']+extra,'rows':output}
