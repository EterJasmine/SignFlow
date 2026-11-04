import math
from .kalman import WristKalman
from .quality import assess
from scipy.optimize import linear_sum_assignment
from .geometry import distance,palm_scale,shoulder_scale

class PredictiveHands:
    """Global hand identity persists independently of body assignment."""
    def __init__(self,width,height):self.w=width;self.h=height;self.state={};self.counter=0
    def reset(self):self.state={}
    def predicted_region(self,t):
        points=[]
        for old in self.state.values():
            if not 0<=t-old['t']<=400:continue
            x,y,sigma=old['filter'].point(t)
            pad=2*old['scale']+min(2*sigma,2*old['scale'])
            points.extend([(x-pad,y-pad),(x+pad,y+pad)])
        if not points:return None
        box=(max(0,int(min(p[0] for p in points))),max(0,int(min(p[1] for p in points))),min(self.w,int(max(p[0] for p in points))),min(self.h,int(max(p[1] for p in points))))
        return box if box[2]>box[0] and box[3]>box[1] else None
    def assign(self,body_tracks,hands,timestamp_ms):
        previous={k:v for k,v in self.state.items() if timestamp_ms-v['t']<=600}
        ids=list(previous);cost=[]
        for ident in ids:
            old=previous[ident]
            px,py,uncertainty=old['filter'].point(timestamp_ms)
            predicted={'x':px/self.w,'y':py/self.h}
            row=[]
            for hand in hands:
                scale=max(old['scale'],palm_scale(hand['landmarks'],self.w,self.h))
                mismatch=(old.get('label')!=hand.get('model_handedness')) and hand.get('handedness_score',0)>.8
                d=distance(predicted,hand['landmarks'][0],self.w,self.h)
                ratio=palm_scale(hand['landmarks'],self.w,self.h)/old['scale']
                invalid=ratio<.4 or ratio>2.5 or d>max(3*scale,min(3*uncertainty,5*scale))
                row.append(100000. if invalid else d/scale+(.4 if mismatch else 0)+.25*abs(math.log(ratio)))
            cost.append(row)
        matched={};ambiguity={}
        if ids and hands:
            rows,cols=linear_sum_assignment(cost)
            for r,c in zip(rows,cols):
                if cost[r][c]<2.5:
                    matched[int(c)]=ids[r]
                    alternatives=sorted(cost[r])
                    column=sorted(row[c] for row in cost)
                    ambiguity[int(c)]=(len(alternatives)>1 and alternatives[1]-alternatives[0]<.15) or (len(column)>1 and column[1]-column[0]<.15)
        # Assign body ownership separately; it never changes stable hand identity.
        owners={};candidates=[]
        for j,h in enumerate(hands):
            for tr in body_tracks:
                p=tr.get('pose_landmarks',[])
                for side,index in [('left',15),('right',16)]:
                    if len(p)<=index or (p[index].get('visibility') or 0)<.5:continue
                    scale=shoulder_scale(p,self.w,self.h)
                    d=distance(p[index],h['landmarks'][0],self.w,self.h)/max(scale,1)
                    if d<.7:candidates.append((d,j,tr['track_id'],side))
        used=set()
        for d,j,person,side in sorted(candidates):
            if j not in owners and (person,side) not in used:owners[j]=(person,side);used.add((person,side))
        tracks=[{**t,'hands':{}} for t in body_tracks]
        for j,hand in enumerate(hands):
            ident=matched.get(j)
            if ident is None:self.counter+=1;ident=f'hand_v2_{self.counter:04}'
            wrist=hand['landmarks'][0];old=previous.get(ident);dt=(timestamp_ms-old['t'])/1000 if old else 0
            velocity=((wrist['x']-old['p']['x'])/dt,(wrist['y']-old['p']['y'])/dt) if old and dt>0 else (0,0)
            kf=old['filter'] if old else WristKalman(wrist['x']*self.w,wrist['y']*self.h,timestamp_ms)
            reliable=not assess(hand['landmarks'],self.w,self.h)['suspect'] and not ambiguity.get(j,False)
            if old and reliable:kf.update(wrist['x']*self.w,wrist['y']*self.h,timestamp_ms)
            if reliable:previous[ident]={'filter':kf,'landmarks':hand['landmarks'],'t':timestamp_ms,'p':wrist,'v':velocity,'scale':palm_scale(hand['landmarks'],self.w,self.h),'label':hand.get('model_handedness')}
            owner,side=owners.get(j,(None,'unknown'));uncertain=ambiguity.get(j,False)
            value={**hand,'assignment':'kalman_hungarian','predicted_wrist':{'x':kf.point(timestamp_ms)[0]/self.w,'y':kf.point(timestamp_ms)[1]/self.h},'identity_continuous':old is not None and not uncertain,
                   'identity_status':'unknown' if uncertain else 'tracked','anatomical_side':side,'owner_track_id':owner}
            tracks.append({'track_id':ident,'person_track_id':owner,'pose_landmarks':[],'face_landmarks':[],'hands':{side:value}})
        self.state=previous
        return tracks
