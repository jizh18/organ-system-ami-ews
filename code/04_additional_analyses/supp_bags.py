"""S3 stability: per-bag decomposition of the held-out organ-system model."""
exec(open("full_common.py").read())
from scipy.stats import spearmanr
def decompose(m,Xt):
    n=len(Xt); out={}
    for pr in m._predictors:
        nd=pr[0].nodes; node=np.zeros(n,dtype=np.int64); mask=np.zeros(n,dtype=np.int64)
        for _ in range(64):
            leaf=nd["is_leaf"][node].astype(bool)
            if leaf.all(): break
            f=nd["feature_idx"][node]; thr=nd["num_threshold"][node]; ml=nd["missing_go_to_left"][node].astype(bool)
            x=Xt[np.arange(n),f]; left=np.where(np.isnan(x),ml,x<=thr)
            nxt=np.where(left,nd["left"][node],nd["right"][node])
            mask=np.where(leaf,mask,mask|(1<<FA[f])); node=np.where(leaf,node,nxt)
        val=nd["value"][node]
        for mk in np.unique(mask):
            sel=mask==mk; lab="×".join(AXL[i] for i in range(len(AXL)) if mk>>i&1) or "intercept"
            out.setdefault(lab,np.zeros(n))[sel]+=val[sel]
    return out
P0=dict(learning_rate=0.03,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0)
z_,th,models=bag(CST["AXIS"],DEV,LATE,777,P0,keep_models=True)
IMP=[]
for mm,A_,B_ in models:
    out=decompose(mm,X[LATE]); IMP.append({k:float(np.std(A_*v)) for k,v in out.items() if k!="intercept"})
keys=sorted(set().union(*IMP)); M=np.array([[d.get(k,0) for k in keys] for d in IMP])
rho=[spearmanr(M[i],M[j]).correlation for i in range(5) for j in range(i+1,5)]
top=[sorted(keys,key=lambda k:-d.get(k,0))[:5] for d in IMP]
json.dump({"spearman_min":float(min(rho)),"spearman_median":float(np.median(rho)),"top5_per_bag":top},open("SUPP_BAGS.json","w"),indent=1)
print(min(rho),np.median(rho)); print(top)
