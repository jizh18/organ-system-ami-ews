"""REV1: reviewer-driven analyses.
A final AXIS/FULL refit with EP3 chosen params, logit-averaged bagging (exactly decomposable), per-rep OOF,
  exact decomposition of the FINAL AXIS model, training-derived alert thresholds (rep 0).
B WITHIN arm (within-system interactions only, no cross-system pairs), same grid/budget as EP3.
C ECG availability delay 0 / 60 min (AXIS, fixed chosen params).
"""
import numpy as np, json, time, sys, pickle, os
src=open("ep3.py").read().split("RES={}")[0]; exec(src)
t0=time.time(); E3=json.load(open("EP3_RESULTS.json")); CH=E3["chosen_params"]
AXN=np.array(axis)[allp]; AXL=sorted(set(AXN)); aid={a:i for i,a in enumerate(AXL)}; FA=np.array([aid[a] for a in AXN])
def log(*a): print(*a,round(time.time()-t0),flush=True)
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
def bag_fixed(Xa,cst,tr,te,seed,params,decomp=False):
    zs=[]; C={}; thr=[]
    for k in range(NBAG):
        a,b=split_val(tr,seed*1000+10+k)
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed*10+k,**params).fit(Xa[a],y[a],X_val=Xa[b],y_val=y[b])
        cal=LogisticRegression(C=1e6).fit(m.decision_function(Xa[b])[:,None],y[b]); A=float(cal.coef_[0,0]); B=float(cal.intercept_[0])
        zs.append(A*m.decision_function(Xa[te])+B); thr.append(np.quantile(A*m.decision_function(Xa[b])+B,0.9))
        if decomp:
            out=decompose(m,Xa[te]); base=float(m._baseline_prediction.ravel()[0])
            for g,v in out.items():
                if g=="intercept": continue
                C.setdefault(g,np.zeros(len(te))); C[g]+=A*v/NBAG
            C.setdefault("intercept",np.zeros(len(te))); C["intercept"]+=(A*base+B)/NBAG
    return np.mean(zs,0),C,float(np.mean(thr))
sig=lambda z:1/(1+np.exp(-z))
up=np.unique(pat); FOLDS=[]
for rep in range(3):
    r=np.random.default_rng(1000+rep); s=up.copy(); r.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}; FOLDS.append(np.array([fm[q] for q in pat]))
OUT={}
# ---------- A
if not os.path.exists("rev1_A.npz"):
    ZR={a:np.zeros((3,len(y))) for a in ["AXIS","FULL"]}; THR=np.zeros(len(y)); CON={}; recon=[]
    ARM2={"AXIS":design(allp,"AXIS",False),"FULL":design(allp,None,False)}
    for rep in range(3):
        f=FOLDS[rep]
        for k in range(5):
            tr,te=np.where(f!=k)[0],np.where(f==k)[0]; i=rep*5+k
            for a,(Xa,cst) in ARM2.items():
                z,C,th=bag_fixed(Xa,cst,tr,te,rep*10+k,CH[a][i],decomp=(a=="AXIS"))
                ZR[a][rep,te]=z
                if a=="AXIS":
                    if rep==0: THR[te]=th
                    recon.append(float(np.max(np.abs(sum(C.values())-z))))
                    for g,v in C.items(): CON.setdefault(g,np.zeros((3,len(y))))[rep,te]=v
            log("A rep",rep,"fold",k)
    np.savez_compressed("rev1_A.npz",ZA=ZR["AXIS"],ZF=ZR["FULL"],THR=THR,recon=np.array(recon),**{"C_"+g.replace("×","_x_"):v for g,v in CON.items()})
log("A done")
# ---------- B WITHIN (tuned, same grid/budget as EP3)
if not os.path.exists("rev1_B.npz"):
    Xw,cw=design(allp,"WITHIN",False); OW=np.zeros(len(y)); chW=[]
    for rep in range(3):
        f=FOLDS[rep]
        for k in range(5):
            tr,te=np.where(f!=k)[0],np.where(f==k)[0]
            p,g=tuned_bagged(Xw,cw,tr,te,rep*10+k); OW[te]+=p/3; chW.append(g)
            log("B rep",rep,"fold",k)
    np.savez_compressed("rev1_B.npz",WITHIN=OW); json.dump(chW,open("rev1_B_chosen.json","w"))
log("B done")
# ---------- C ECG delay
if not os.path.exists("rev1_C.npz"):
    import pandas as pd
    U="/mnt/user-data/uploads/KG-v1/outputs/experiments/ep2_20261009/"
    adm=pd.read_csv(U+"admissions.csv.gz",parse_dates=["admittime"]).set_index("hadm_id")
    hh=np.load("ep2_features.npz")["h"]
    Lt=adm.loc[hadm,"admittime"].values+pd.to_timedelta(hh,unit="h").values
    mm=pd.read_csv(U+"ecg_mm.csv.gz",low_memory=False,parse_dates=["ecg_time"])
    rc=[c for c in mm.columns if c.startswith("report_")]; txt=mm[rc].fillna("").astype(str).agg(" | ".join,axis=1).str.lower()
    mm["ste"]=txt.str.contains(r"st elev|injury|acute mi|stemi"); mm["ant"]=txt.str.contains("anterior")&txt.str.contains(r"infarct|injury|st elev")
    mm["af"]=txt.str.contains(r"atrial fib|atrial flut"); num=lambda s:pd.to_numeric(s,errors="coerce")
    rr=num(mm.rr_interval).where(lambda x:(x>200)&(x<3000)); mm["hr"]=60000/rr
    mm["qrs"]=(num(mm.qrs_end)-num(mm.qrs_onset)).where(lambda x:(x>40)&(x<250))
    mm["qtc"]=((num(mm.t_end)-num(mm.qrs_onset))/np.sqrt(rr/1000)).where(lambda x:(x>300)&(x<700))
    sub=adm.loc[np.unique(hadm)].reset_index()[["hadm_id","subject_id","admittime"]]
    mm=mm.merge(sub,on="subject_id"); mm=mm[mm.ecg_time>=mm.admittime-pd.Timedelta(hours=6)]
    KEYS=["ecg_adm_ste","ecg_adm_anterior","ecg_adm_wide_qrs","ecg_adm_af","ecg_last_hr","ecg_last_qrs","ecg_last_qtc","ecg_last_af","ecg_st_resolved","ecg_new_af"]
    Xa0,cst0=design(allp,"AXIS",False); col={n_:j for j,n_ in enumerate(np.array(names)[allp])}
    RC={}
    for d in [0,60]:
        m2=mm.copy(); m2["avail"]=m2.ecg_time+pd.Timedelta(minutes=d); m2=m2.sort_values(["hadm_id","avail"]); G=dict(tuple(m2.groupby("hadm_id")))
        E={k:np.full(len(y),np.nan) for k in KEYS}
        for i in range(len(y)):
            e=G.get(hadm[i])
            if e is None: continue
            e=e[e.avail<=Lt[i]]
            if not len(e): continue
            f0,fl=e.iloc[0],e.iloc[-1]
            E["ecg_adm_ste"][i]=f0.ste; E["ecg_adm_anterior"][i]=f0.ant; E["ecg_adm_wide_qrs"][i]=float(f0.qrs>=120) if pd.notna(f0.qrs) else np.nan; E["ecg_adm_af"][i]=f0.af
            E["ecg_last_hr"][i]=fl.hr; E["ecg_last_qrs"][i]=fl.qrs; E["ecg_last_qtc"][i]=fl.qtc; E["ecg_last_af"][i]=fl.af
            E["ecg_st_resolved"][i]=float(f0.ste and not fl.ste) if len(e)>1 else np.nan
            E["ecg_new_af"][i]=float(fl.af and not e.af.iloc[:-1].any()) if len(e)>1 else 0.0
        Xd=Xa0.copy()
        for k_ in KEYS:
            if k_ in col: Xd[:,col[k_]]=E[k_]
        # sanity: reproduce 30-min features?
        Z=np.zeros(len(y))
        for rep in range(3):
            f=FOLDS[rep]
            for k in range(5):
                tr,te=np.where(f!=k)[0],np.where(f==k)[0]
                z,_,_=bag_fixed(Xd,cst0,tr,te,rep*10+k,CH["AXIS"][rep*5+k]); Z[te]+=sig(z)/3
        RC[f"delay{d}"]=Z; log("C delay",d,AUC(y,Z))
    np.savez_compressed("rev1_C.npz",**RC)
log("ALL DONE")
