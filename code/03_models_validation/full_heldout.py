"""FULL01 held-out evaluation: fit every arm on all development patients, score the 2017-2022 patients once.
Also: exact decomposition of the held-out AXIS predictions (bag-averaged, Platt-scaled) and LR system contributions."""
import os
exec(open("full_common.py").read())
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
Z={}; TH={}; PAR={}; CON={}
for a in ["AXIS","WITHIN","FULL","ADD"]:
    g=tune(CST[a],DEV,777); z_,th,models=bag(CST[a],DEV,LATE,777,g,keep_models=(a=="AXIS")); p=np.full(len(y),np.nan); p[LATE]=z_; Z[a]=p; TH[a]=th; PAR[a]=str(g)
    if a=="AXIS":
        for m,A_,B_ in models:
            out=decompose(m,X[LATE]); base=float(m._baseline_prediction.ravel()[0])
            for gname,v in out.items():
                if gname=="intercept": continue
                CON.setdefault(gname,np.zeros(len(LATE))); CON[gname]+=A_*v/NBAG
            CON.setdefault("intercept",np.zeros(len(LATE))); CON["intercept"]+=(A_*base+B_)/NBAG
        log("recon err",float(np.max(np.abs(sum(CON.values())-z_))))
    log("heldout",a,g)
z_,th,C,S=lr_fit(DEV,LATE,777); p=np.full(len(y),np.nan); p[LATE]=z_; Z["LR"]=p; TH["LR"]=th; PAR["LR"]=C
np.savez_compressed("heldout_scores.npz",**Z,LR_sys=S,**{"thr_"+a:np.array(v) for a,v in TH.items()},**{"C_"+k.replace("×","_x_"):v for k,v in CON.items()})
json.dump(PAR,open("heldout_params.json","w")); log("HELDOUT DONE")
