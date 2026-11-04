import numpy as np
from scipy.optimize import linear_sum_assignment

def confusion_matrix(truth,pred,classes):
    matrix=np.zeros((classes,classes),dtype=int)
    for a,b in zip(np.asarray(truth).ravel(),np.asarray(pred).ravel()):
        if 0<=a<classes and 0<=b<classes:matrix[a,b]+=1
    return matrix
def macro_f1(truth,pred,classes):
    m=confusion_matrix(truth,pred,classes);den=m.sum(0)+m.sum(1);valid=den>0
    return float(np.mean(2*np.diag(m)[valid]/den[valid])) if valid.any() else None
def boundary_f1(truth_ms,pred_ms,tolerance_ms=300):
    truth_ms=list(truth_ms);pred_ms=list(pred_ms)
    if not truth_ms and not pred_ms:return 1.
    if not truth_ms or not pred_ms:return 0.
    costs=np.abs(np.array(truth_ms)[:,None]-np.array(pred_ms)[None,:]);r,c=linear_sum_assignment(np.where(costs<=tolerance_ms,costs,1e9))
    tp=sum(costs[i,j]<=tolerance_ms for i,j in zip(r,c));return float(2*tp/(len(truth_ms)+len(pred_ms)))
def edit_score(truth,pred):
    def compress(seq):return [x for i,x in enumerate(seq) if i==0 or x!=seq[i-1]]
    a=compress(list(truth));b=compress(list(pred));d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        previous=d;d=[i]+[0]*len(b)
        for j,y in enumerate(b,1):d[j]=min(d[j-1]+1,previous[j]+1,previous[j-1]+(x!=y))
    return 100*(1-d[-1]/max(1,len(a),len(b)))
