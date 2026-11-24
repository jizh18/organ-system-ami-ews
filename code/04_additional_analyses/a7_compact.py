"""Amendment 7, Part A: explanation compactness in the held-out MIMIC-IV period (post hoc, descriptive)."""
import pickle
exec(open("full_common.py").read())
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
M=pickle.load(open("final_models.pkl","rb")); assert M["names"]==names
CON_ID=np.array([sorted(set(concept(n) for n in names)).index(concept(n)) for n in names]); NCON=CON_ID.max()+1
def leaf_terms(m,Xt,key):
    """exact attribution of every leaf value to a key of its path; key='sys' (set of systems, bitmask) or 'con' (measurement id; additive trees use one measurement)"""
    n=len(Xt); out={}
    for pr in m._predictors:
        nd=pr[0].nodes; node=np.zeros(n,dtype=np.int64); mask=np.zeros(n,dtype=np.int64); con=np.full(n,-1,dtype=np.int64)
        for _ in range(64):
            leaf=nd["is_leaf"][node].astype(bool)
            if leaf.all(): break
            f=nd["feature_idx"][node]; thr=nd["num_threshold"][node]; ml=nd["missing_go_to_left"][node].astype(bool)
            x=Xt[np.arange(n),f]; left=np.where(np.isnan(x),ml,x<=thr); nxt=np.where(left,nd["left"][node],nd["right"][node])
            mask=np.where(leaf,mask,mask|(1<<FA[f]))
            if key=="con":
                c=CON_ID[f]; assert np.all((con[~leaf]==-1)|(con[~leaf]==c[~leaf])); con=np.where(leaf,con,c)
            node=np.where(leaf,node,nxt)
        val=nd["value"][node]; k_=mask if key=="sys" else con
        for kk in np.unique(k_):
            sel=k_==kk; out.setdefault(int(kk),np.zeros(n))[sel]+=val[sel]
    return out
def terms(models,Xt,key):
    C={}
    for m,A_,B_ in models:
        o=leaf_terms(m,Xt,key)
        for g,v in o.items():
            if g in (0,-1): continue          # root-only trees: intercept
            C.setdefault(g,np.zeros(len(Xt))); C[g]+=A_*v/len(models)
    return C
def Kq(T,q):
    A=np.sort(np.abs(T),1)[:,::-1]; cs=np.cumsum(A,1); tot=cs[:,-1:]; return (cs<q*tot-1e-12).sum(1)+1
def summ(v): return dict(median=float(np.median(v)),q1=float(np.percentile(v,25)),q3=float(np.percentile(v,75)),mean=float(v.mean()))
R={}
# additive model: refit on development with the frozen procedure and selected hyperparameters
z_add,thr_add,mod_add=bag(CST["ADD"],DEV,LATE,777,dict(learning_rate=0.03,max_leaf_nodes=4,min_samples_leaf=40,l2_regularization=1.0),keep_models=True)
R["additive_refit"]=dict(AUROC=float(AUC(y[LATE],z_add)),AP=float(AP(y[LATE],z_add))); log("ADD refit",R["additive_refit"])
ARM={"Organ-system":(M["dev_AXIS"]["models"],M["dev_AXIS"]["thr"],"sys"),"Unconstrained":(M["dev_FULL"]["models"],M["dev_FULL"]["thr"],"sys"),"Additive":(mod_add,thr_add,"con")}
for name,(models,thr,key) in ARM.items():
    Cd=terms(models,X[DEV],key); Ch=terms(models,X[LATE],key); keys=sorted(set(Cd)|set(Ch))
    T=np.column_stack([Ch.get(k,np.zeros(len(LATE)))-Cd.get(k,np.zeros(len(DEV))).mean() for k in keys])
    zh=np.mean([A_*m.decision_function(X[LATE])+B_ for m,A_,B_ in models],0); al=zh>=thr
    var=T.var(0); nsys=np.array([bin(k).count("1") for k in keys]) if key=="sys" else np.ones(len(keys),int)
    d=dict(n_terms=int(len(keys)),alert_rate=float(al.mean()),AUROC=float(AUC(y[LATE],zh)),
           K80_all=summ(Kq(T,.8)),K90_all=summ(Kq(T,.9)),K80_alert=summ(Kq(T[al],.8)),K90_alert=summ(Kq(T[al],.9)),
           share_le3_terms_80=float((Kq(T,.8)<=3).mean()),share_le5_terms_80=float((Kq(T,.8)<=5).mean()))
    if key=="sys":
        d["terms_by_n_systems"]={int(s):int((nsys==s).sum()) for s in np.unique(nsys)}
        d["variance_share_by_n_systems"]={int(s):float(var[nsys==s].sum()/var.sum()) for s in np.unique(nsys)}
        d["variance_share_3plus"]=float(var[nsys>=3].sum()/var.sum())
    R[name]=d; log(name,json.dumps(d))
json.dump(R,open("A7_COMPACT.json","w"),indent=1)
