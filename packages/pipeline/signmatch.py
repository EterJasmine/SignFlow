"""Masked template retrieval; distances are not semantic confidence."""
import numpy as np

# 一帧里至少要读出这么多手部列，才拿它参与对齐。只手 25 列指特征，低于这个数
# 说明多数手指关键点塌缩，对齐时只能给出「完全不匹配」的上限惩罚。
MIN_COLUMNS = 12

def palm_stats(result,start,end):
    """Median palm length (px) inside a window, used to explain why a range is unusable."""
    palms=[]
    for frame in result.get('frames') or []:
        if not start<=frame['timestamp_ms']<end:continue
        for track in frame.get('tracks') or []:
            for hand in (track.get('hands') or {}).values():
                palm=((hand.get('quality') or {}).get('geometry') or {}).get('palm_px')
                if palm:palms.append(palm)
    if not palms:return None
    palms.sort()
    return {'median':palms[len(palms)//2],'frames':len(palms)}

def sequence(result,start,end):
    features=result.get('features',{});names=features.get('names',[])
    cols=[i for i,n in enumerate(names) if any(k in n for k in ('angle','tip_radius','thumb_distance','face_distance','chest_distance','hand_distance')) and not n.endswith('approach_speed')]
    rows=[r for r in features.get('rows',[]) if start<=r['timestamp_ms']<end]
    if not cols:raise ValueError('这次分析没有记录手部特征，请重新分析该视频后再标注。')
    if len(rows)<4:raise ValueError('动作区间太短：本段只有 %d 个采样帧，至少要 4 个。请把起止拉长到 0.5 秒以上。'%len(rows))
    # 坏帧直接丢弃：这类帧在 DTW 里会被按上限惩罚，一条就能把整段距离顶上去，
    # 连「自己匹配自己」都不为 0（现有模板里就有 0.67 的），等于这个示范永远认不出。
    rows=[r for r in rows if sum(1 for i in cols if r['mask'][i])>=MIN_COLUMNS]
    if len(rows)<4:
        palms=palm_stats(result,start,end)
        detail='，本段手掌中位仅 %.0f 像素'%palms['median'] if palms else ''
        raise ValueError('这段区间里只有 %d 帧能读出完整手型，其余帧手指被遮挡或手掌塌缩%s。请让手完整入镜、手指不要互相遮挡后重拍。'%(len(rows),detail))
    rows=rows[::max(1,int(np.ceil(len(rows)/160)))];out=[]
    for r in rows:
        out.append([r['values'][i] if r['mask'][i] else None for i in cols])
    coverage=sum(v is not None for row in out for v in row)/max(1,len(out)*len(cols))
    if coverage<.2:
        palms=palm_stats(result,start,end)
        detail='，本段手掌中位仅 %.0f 像素'%palms['median'] if palms else '，本段几乎没有检出完整手掌'
        raise ValueError('有效手部特征只占 %d%%%s。这通常是手占画面太小、录屏小窗放大、或手指被遮挡导致关键点塌缩；几何上已经不是一只新手。请让手占画面更大后重拍，或改用原始拍摄而不是录屏。'%(round(coverage*100),detail))
    return {'version':features.get('version'),'names':[names[i] for i in cols],'rows':out}


def usable(template):
    """A template may only be retrieved when it is marked ready and really has a sequence.

    Drafts keep whatever they had before; treating them as templates is how a
    half-labelled range quietly becomes an always-wrong answer.
    """
    return bool(template.get('recognition_ready')) and 'rows' in (template.get('sequence') or {})

def retrieve(seq,templates,accept=.18,margin=.04):
    """Score one captured sequence against every template. Shared by the manual
    range check and the live window so both apply the same rule."""
    scores=[]
    for t in templates:
        if 'rows' not in (t.get('sequence') or {}):continue
        try:scores.append({'template_id':t['id'],'meaning':t['meaning'],'distance':dtw(seq,t['sequence'])})
        except ValueError:continue
    grouped={}
    for score in scores:grouped.setdefault(score['meaning'],[]).append(score)
    scores=[]
    for meaning,examples in grouped.items():
        examples.sort(key=lambda x:x['distance']);support=examples[:3]
        scores.append({**examples[0],'distance':float(np.mean([s['distance'] for s in support])),
                       'support_count':len(support),'nearest_distance':examples[0]['distance']})
    scores.sort(key=lambda x:x['distance'])
    best=scores[0] if scores else None
    different=next((s for s in scores if best and s['meaning']!=best['meaning']),None)
    accepted=bool(best and best['distance']<accept and (different is None or different['distance']-best['distance']>margin))
    # 拒绝也要说清理由：是「没有足够接近的示范」，还是「两个词太像分不开」。
    # 前者多加示范有用，后者加示范也不一定有用，用户据此决定要不要重录。
    if best is None:reason='示范库里还没有可检索的示范。'
    elif accepted:reason=None
    elif best['distance']>=accept:
        reason='最接近的「%s」距离 %.2f，超出可确认范围 %.2f。这个词还没有足够接近的示范。'%(best['meaning'],best['distance'],accept)
    else:
        reason='「%s」与「%s」太接近（%.2f / %.2f），分不开所以不猜。给这两个词各多录一条示范最有效。'%(best['meaning'],different['meaning'],best['distance'],different['distance'])
    return {'matched':accepted,'meaning':best['meaning'] if accepted else None,
            'distance':best['distance'] if best else None,'candidates':scores[:5],'reason':reason}

def _as_array(seq):
    """Turn a sequence into a dense array plus a validity mask (NaN marks a masked column)."""
    rows=seq['rows']
    # Bound alignment work while retaining the whole action and both endpoints.
    if len(rows)>48:
        rows=[rows[i] for i in np.linspace(0,len(rows)-1,48).round().astype(int)]
    arr=np.array([[np.nan if v is None else float(v) for v in row] for row in rows],dtype=float)
    return arr,~np.isnan(arr)


def dtw(a,b):
    """Average masked distance of the best alignment between two sequences.

    The per-cell cost used to be computed in a Python loop with a numpy mean call
    inside it, which cost ~5 ms per pair and therefore ~0.4 s for the nine window
    shifts tried on every recognised word. Computing the whole cost matrix at once
    keeps the arithmetic identical while the loop only walks the alignment itself.
    """
    if a['version']!=b['version'] or a['names']!=b['names']:raise ValueError('特征版本不同，请重新分析示范')
    x,x_mask=_as_array(a);y,y_mask=_as_array(b)
    if not len(x) or not len(y):raise ValueError('示范里没有可对齐的手型帧')
    # 逐列差值的截断绝对差；只在两边都有效的列上取平均，两列都不足时给上限惩罚。
    diff=np.abs(x[:,None,:]-y[None,:,:])
    np.minimum(diff,2.,out=diff)
    both=x_mask[:,None,:]&y_mask[None,:,:]
    count=both.sum(-1)
    total=np.where(both,diff,0.).sum(-1)
    need=max(3,x.shape[1]//5)
    cost=np.where(count>=need,total/np.maximum(count,1),2.)
    if str(a['version']).endswith('-motion3'):
        union=(x_mask[:,None,:]|y_mask[None,:,:]).sum(-1)
        cost+=.25*(union-count)/np.maximum(union,1)
    rows,cols=cost.shape
    dp=np.full((rows+1,cols+1),np.inf);steps=np.zeros((rows+1,cols+1));dp[0,0]=0.
    for i in range(1,rows+1):
        for j in range(1,cols+1):
            diagonal=dp[i-1,j-1]
            if diagonal<=dp[i-1,j] and diagonal<=dp[i,j-1]:best,pi,pj=diagonal,i-1,j-1
            elif dp[i-1,j]<=dp[i,j-1]:best,pi,pj=dp[i-1,j],i-1,j
            else:best,pi,pj=dp[i,j-1],i,j-1
            dp[i,j]=best+cost[i-1,j-1];steps[i,j]=steps[pi,pj]+1
    return float(dp[rows,cols]/max(1.,steps[rows,cols]))
