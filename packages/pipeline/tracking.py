"""Hand assignment remains valid when body landmarks are absent or occluded."""
import math

def distance(a,b):return math.hypot(a['x']-b['x'],a['y']-b['y'])

class HandAssociator:
    def __init__(self):self.previous={};self.counter=0

    def reset(self):self.previous={}

    def assign(self,body_tracks,hands):
        # Never append hand-only records to the iterable of body candidates.
        tracks=list(body_tracks)
        candidates=[]
        for hi,hand in enumerate(hands):
            for ti,track in enumerate(body_tracks):
                pose=track.get('pose_landmarks',[])
                for side,index in [('left',15),('right',16)]:
                    if len(pose)<=index or pose[index].get('visibility',0)<.5:continue
                    d=distance(hand['landmarks'][0],pose[index])
                    if d<.15:candidates.append((d,hi,ti,side))
        used_hands=set();used_wrists=set()
        for d,hi,ti,side in sorted(candidates):
            if hi in used_hands or (ti,side) in used_wrists:continue
            value={**hands[hi],'assignment':'pose_wrist','association_distance':round(d,4)}
            tracks[ti]['hands'][side]=value
            used_hands.add(hi);used_wrists.add((ti,side))
        remaining=set(self.previous);next_state={}
        for hi,hand in enumerate(hands):
            if hi in used_hands:continue
            wrist=hand['landmarks'][0]
            matches=sorted((distance(wrist,old),ident) for ident,old in self.previous.items() if ident in remaining)
            # Reject ambiguous crossings instead of pretending identity is certain.
            confident=bool(matches and matches[0][0]<.18 and (len(matches)==1 or matches[1][0]-matches[0][0]>.03))
            if confident:
                ident=matches[0][1];remaining.remove(ident)
            else:self.counter+=1;ident=f'hand_{self.counter:04}'
            next_state[ident]=wrist
            tracks.append({'track_id':ident,'pose_landmarks':[],'face_landmarks':[],
                           'hands':{'unknown':{**hand,'assignment':'spatial_hand_track',
                                               'identity_continuous':confident,
                                               'anatomical_side':'unconfirmed'}}})
        self.previous=next_state
        return tracks
