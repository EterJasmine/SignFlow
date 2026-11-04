import copy,math
from .geometry import palm_scale,distance

class OneEuro:
    def __init__(self,min_cutoff=1.5,beta=.08):self.x=None;self.t=None;self.derivative=0.;self.min_cutoff=min_cutoff;self.beta=beta
    def filter(self,value,t):
        if self.x is None or t<=self.t:self.x=value;self.t=t;return value
        dt=(t-self.t)/1000;dx=(value-self.x)/dt
        a=1/(1+1/(2*math.pi*dt));self.derivative=a*dx+(1-a)*self.derivative
        cutoff=self.min_cutoff+self.beta*abs(self.derivative)
        alpha=1/(1+1/(2*math.pi*cutoff*dt));self.x=alpha*value+(1-alpha)*self.x;self.t=t
        return self.x

def process_trajectories(frames,width,height,max_gap_ms=400):
    result=copy.deepcopy(frames);series={}
    for i,f in enumerate(result):
        for tr in f['tracks']:
            for side,hand in tr['hands'].items():series.setdefault((f['scene_id'],tr['track_id'],side),[]).append((i,tr,hand))
    interpolated=0;outliers=0;observed=0
    for (scene,ident,side),items in series.items():
        # Whole-hand gaps are filled only between observed endpoints of one identity.
        for (ia,ta,ha),(ib,tb,hb) in zip(items,items[1:]):
            start=result[ia]['timestamp_ms'];end=result[ib]['timestamp_ms']
            if ib<=ia+1 or end-start>max_gap_ms or hb.get('identity_status')=='unknown':continue
            if ha.get('quality',{}).get('suspect') or hb.get('quality',{}).get('suspect'):continue
            if len(ha['landmarks'])!=21 or len(hb['landmarks'])!=21:continue
            if distance(ha['landmarks'][0],hb['landmarks'][0],width,height)>3*max(palm_scale(ha['landmarks'],width,height),palm_scale(hb['landmarks'],width,height)):continue
            for i in range(ia+1,ib):
                if result[i]['scene_id']!=scene:continue
                if any(tr['track_id']==ident and side in tr['hands'] for tr in result[i]['tracks']):continue
                ratio=(result[i]['timestamp_ms']-start)/(end-start);hand=copy.deepcopy(ha)
                hand.update(detected=False,interpolated=True,quality={'suspect':False,'score':.5,'reasons':['short_occlusion_interpolation']},identity_status='interpolated')
                for p,a,b in zip(hand['landmarks'],ha['landmarks'],hb['landmarks']):
                    p.update({k:a[k]+ratio*(b[k]-a[k]) for k in ['x','y','z']});p.update(detected=False,interpolated=True,confidence=None,visibility=None,raw_coordinates=None)
                result[i]['tracks'].append({'track_id':ident,'pose_landmarks':[],'face_landmarks':[],'hands':{side:hand},'person_track_id':ta.get('person_track_id')});interpolated+=1
        expanded=[]
        for f in result:
            if f['scene_id']!=scene:continue
            for tr in f['tracks']:
                if tr['track_id']==ident and side in tr['hands']:expanded.append((f['timestamp_ms'],tr['hands'][side]))
        filters=[(OneEuro(),OneEuro()) for _ in range(21)]
        for n,(t,hand) in enumerate(expanded):
            scale=palm_scale(hand['landmarks'],width,height);quality=[]
            for j,p in enumerate(hand['landmarks']):
                observed+=1
                if 0<n<len(expanded)-1:
                    before=expanded[n-1][1]['landmarks'][j];after=expanded[n+1][1]['landmarks'][j]
                    if distance(p,before,width,height)>3*scale and distance(before,after,width,height)<.5*scale:p['low_quality']=True;outliers+=1
                fx,fy=filters[j]
                if p.get('low_quality'):quality.append(0);continue
                # Filtering is in pixels, with adaptive cutoff to retain fast motion.
                p['x']=fx.filter(p['x']*width,t)/width;p['y']=fy.filter(p['y']*height,t)/height
                p['processing']='one_euro_v1';quality.append(.5 if p.get('interpolated') else 1.)
            hand['trajectory_quality']=sum(quality)/max(1,len(quality))
    return result,{'interpolated_hand_frames':interpolated,'outlier_points':outliers,'processed_points':observed,'filter':'one_euro_v1','max_gap_ms':max_gap_ms}
