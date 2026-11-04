"""Adaptive sampling and source-resolution ROI recovery, without invented landmarks."""
from copy import deepcopy
from .geometry import distance,palm_scale
from .quality import assess

class OcclusionRecovery:
    def __init__(self):self.recent=[];self.last_seen=-10000;self.dense_until=-1;self.attempts=0;self.recovered=0
    def reset(self):self.recent=[];self.last_seen=-10000;self.dense_until=-1
    def observe(self,hands,t,w,h):
        good=[x for x in hands if not assess(x['landmarks'],w,h)['suspect']]
        if len(good)<len(self.recent) and t-self.last_seen<600:self.dense_until=t+500
        if len(good)>=2:
            for i,a in enumerate(good):
                for b in good[i+1:]:
                    if distance(a['landmarks'][0],b['landmarks'][0],w,h)<4*max(palm_scale(a['landmarks'],w,h),palm_scale(b['landmarks'],w,h)):self.dense_until=t+500
        if good and (len(good)>=len(self.recent) or t-self.last_seen>600):self.recent=deepcopy(good);self.last_seen=t
    def crop_box(self,t,w,h):
        if not self.recent or t-self.last_seen>600:return None
        pts=[p for hand in self.recent for p in hand['landmarks']]
        xs=[p['x']*w for p in pts];ys=[p['y']*h for p in pts]
        pad=max(40,(max(xs)-min(xs))*.6,(max(ys)-min(ys))*.6)
        return (max(0,int(min(xs)-pad)),max(0,int(min(ys)-pad)),min(w,int(max(xs)+pad)),min(h,int(max(ys)+pad)))

def map_crop(points,box,w,h):
    x,y,right,bottom=box;out=deepcopy(points)
    for p in out:
        p['x']=(x+p['x']*(right-x))/w;p['y']=(y+p['y']*(bottom-y))/h
        p['z']=p.get('z',0)*(right-x)/w
        p['raw_coordinates']={k:p[k] for k in ['x','y','z']}
    return out

def merge_recovered(original,candidates,w,h):
    result=list(original);added=0
    for hand in candidates:
        if assess(hand['landmarks'],w,h)['suspect']:continue
        duplicates=[i for i,old in enumerate(result) if distance(hand['landmarks'][0],old['landmarks'][0],w,h)<.7*max(palm_scale(hand['landmarks'],w,h),palm_scale(old['landmarks'],w,h))]
        if duplicates:
            i=min(duplicates,key=lambda i:distance(hand['landmarks'][0],result[i]['landmarks'][0],w,h))
            if assess(result[i]['landmarks'],w,h)['suspect']:
                result[i]=hand;added+=1
            continue
        result.append(hand);added+=1
    return result,added


def search_regions(w,h):
    """Overlapping horizontal bands find hands outside an existing track's ROI."""
    return [(int(w*.1),int(h*.55),int(w*.85),int(h*.9))]+[(int(w*x),int(h*y),int(w*r),min(h,int(h*(y+.45)))) for y in (0.,.275,.55) for x,r in ((0.,1.),(.1,.85))]
