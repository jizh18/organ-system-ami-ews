"""Shared code for FULL01 (enlarged cohort). Imported by full_cv.py, full_heldout.py, full_interp.py."""
import numpy as np, pandas as pd, json, itertools, warnings, time
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss
warnings.filterwarnings("ignore"); t0=time.time()
def log(*a): print(*a,round(time.time()-t0),flush=True)
U="/mnt/user-data/uploads/KG-v1/outputs/experiments/full_v2_20261009/"
z=np.load("full_features_v2.npz"); meta=json.load(open("full_feature_meta_v2.json"))
X0,XP,y,pat,hadm,lead,hh=z["X"],z["XP"],z["y"],z["patient"],z["hadm"],z["lead"],z["h"]
names0,axis0=meta["names"],np.array(meta["axis"])
co=pd.read_csv(U+"cohort.csv").set_index("hadm_id"); period=co.loc[hadm,"period"].values; ecg6=co.loc[hadm,"ecg_6h"].values.astype(bool)
DEV=np.where(period=="development")[0]; LATE=np.where(period=="later")[0]
keep=[j for j in range(X0.shape[1]) if np.isfinite(X0[DEV,j]).sum()>=200 and len(np.unique(X0[DEV][np.isfinite(X0[DEV,j]),j]))>=2]
X=X0[:,keep]; names=[names0[j] for j in keep]; axis=axis0[keep]; N=len(names)
concept=lambda n:n.split(".")[0] if "." in n else n
AXL=sorted(set(axis)); aid={a:i for i,a in enumerate(AXL)}; FA=np.array([aid[a] for a in axis])
PAIRS=[("INJURY","HEMO"),("HEMO","PERFUSION"),("PERFUSION","RESP"),("HEMO","RESP"),("HEMO","RENAL"),("PERFUSION","RENAL"),("ELECTRICAL","HEMO"),("NEURO","PERFUSION")]
def groups(mode):
    idx=list(range(N))
    if mode=="ADD":
        cs={}
        for i in idx: cs.setdefault(concept(names[i]),[]).append(i)
        return list(cs.values())
    G={a:[i for i in idx if axis[i]==a] for a in AXL}; out=[G[a] for a in AXL]
    if mode=="AXIS": out+=[G[a]+G[b] for a,b in PAIRS if a in G and b in G]
    return out
CST={"AXIS":groups("AXIS"),"WITHIN":groups("WITHIN"),"ADD":groups("ADD"),"FULL":None}
GRID=[dict(learning_rate=0.03,max_leaf_nodes=lv,min_samples_leaf=ms,l2_regularization=l2) for lv,ms,l2 in itertools.product([4,8,16],[40,120],[1.,10.])]
NBAG=5
def split_val(idx,seed):
    up_=np.unique(pat[idx]); r=np.random.default_rng(seed); v=set(r.choice(up_,max(1,int(.15*len(up_))),replace=False))
    isv=np.array([q in v for q in pat[idx]]); return idx[~isv],idx[isv]
def ll(yv,zz): return float(np.mean(np.logaddexp(0,zz)-yv*zz))
sig=lambda z_:1/(1+np.exp(-z_))
def tune(cst,tr,seed):
    a,b=split_val(tr,seed*1000+1); best=None
    for g in GRID:
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed,**g).fit(X[a],y[a],X_val=X[b],y_val=y[b]); s=ll(y[b],m.decision_function(X[b]))
        if best is None or s<best[0]: best=(s,g)
    return best[1]
def bag(cst,tr,te,seed,params,keep_models=False):
    zs=[];thr=[];models=[]
    for k in range(NBAG):
        a,b=split_val(tr,seed*1000+10+k)
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed*10+k,**params).fit(X[a],y[a],X_val=X[b],y_val=y[b])
        cal=LogisticRegression(C=1e6).fit(m.decision_function(X[b])[:,None],y[b]); A_,B_=float(cal.coef_[0,0]),float(cal.intercept_[0])
        zs.append(A_*m.decision_function(X[te])+B_); thr.append(np.quantile(A_*m.decision_function(X[b])+B_,0.9))
        if keep_models: models.append((m,A_,B_))
    return np.mean(zs,0),float(np.mean(thr)),models
def lr_fit(tr,te,seed):
    a,b=split_val(tr,seed*1000+1); med=np.nanmedian(X[a],0); med=np.where(np.isfinite(med),med,0); mc=np.where(np.isnan(X[a]).any(0))[0]
    F=lambda r:np.column_stack([np.where(np.isnan(X[r]),med,X[r]),np.isnan(X[r][:,mc]).astype(np.float32)]); Za=F(a); mu=Za.mean(0); sd=Za.std(0)+1e-6; T=lambda r:np.clip((F(r)-mu)/sd,-8,8)
    best=None
    for C in [1e-4,3e-4,1e-3,3e-3,1e-2]:
        m=LogisticRegression(C=C,max_iter=3000).fit(T(a),y[a]); s=ll(y[b],m.decision_function(T(b)))
        if best is None or s<best[0]: best=(s,m,C)
    m=best[1]; cal=LogisticRegression(C=1e6).fit(m.decision_function(T(b))[:,None],y[b]); A_,B_=cal.coef_[0,0],cal.intercept_[0]
    zt=A_*m.decision_function(T(te))+B_; thr=float(np.quantile(A_*m.decision_function(T(b))+B_,0.9))
    colsys=np.concatenate([FA,FA[mc]]); contrib=T(te)*m.coef_[0]*A_
    S=np.stack([contrib[:,colsys==i].sum(1) for i in range(len(AXL))],1)
    return zt,thr,best[2],S
# ---- clinical scores
def L_(n): return X0[:,names0.index(n)]
fio=XP[:,meta["proc_names"].index("fio2_est")]
def news2():
    rr,sp,t,sbp,hr,gcs=L_("resp_rate.last"),L_("spo2.last"),L_("temperature.last"),L_("sbp.last"),L_("heart_rate.last"),L_("gcs_total"); sc=np.zeros(len(y))
    sc+=np.select([rr<=8,rr<=11,rr<=20,rr<=24,rr>24],[3,1,0,2,3],0)*np.isfinite(rr); sc+=np.select([sp<=91,sp<=93,sp<=95,sp>95],[3,2,1,0],0)*np.isfinite(sp)
    sc+=2*(fio>0.21); sc+=np.select([t<=35,t<=36,t<=38,t<=39,t>39],[3,1,0,1,2],0)*np.isfinite(t)
    sc+=np.select([sbp<=90,sbp<=100,sbp<=110,sbp<=219,sbp>219],[3,2,1,0,3],0)*np.isfinite(sbp)
    sc+=np.select([hr<=40,hr<=50,hr<=90,hr<=110,hr<=130,hr>130],[3,1,0,1,2,3],0)*np.isfinite(hr); sc+=3*(np.nan_to_num(gcs,nan=15)<15)
    return sc+np.random.default_rng(0).random(len(y))*1e-6
def grace():
    age=L_("age"); hr=np.where(np.isfinite(L_("heart_rate.last")),L_("heart_rate.last"),L_("ecg_last_hr")); sbp=L_("sbp.last"); cr=L_("creatinine.last"); sf=L_("sf_ratio")
    sc=np.select([age<30,age<40,age<50,age<60,age<70,age<80,age<90,age>=90],[0,8,25,41,58,75,91,100],0)
    sc=sc+np.where(np.isfinite(hr),np.select([hr<50,hr<70,hr<90,hr<110,hr<150,hr<200,hr>=200],[0,3,9,15,24,38,46],9),9)
    sc=sc+np.where(np.isfinite(sbp),np.select([sbp<80,sbp<100,sbp<120,sbp<140,sbp<160,sbp<200,sbp>=200],[58,53,43,34,24,10,0],34),34)
    sc=sc+np.where(np.isfinite(cr),np.select([cr<.4,cr<.8,cr<1.2,cr<1.6,cr<2,cr<4,cr>=4],[1,4,7,10,13,21,28],7),7)
    sc=sc+np.where(np.nan_to_num(sf,nan=999)<315,39,0)
    sc=sc+28*(np.nan_to_num(L_("ecg_adm_ste"))>0)+14*(np.nan_to_num(L_("troponin_t.last"))>0.01)
    return sc+np.random.default_rng(1).random(len(y))*1e-6
SC={"NEWS2":news2(),"GRACE":grace()}
ICU=np.isfinite(L_("heart_rate.last"))
def metrics(p,m,prob=True):
    yy,pp,hd,ld=y[m],p[m],hadm[m],lead[m]; out=dict(n=int(m.sum()),pos=int(yy.sum()),events=int(len(np.unique(hd[yy==1]))),AUROC=float(AUC(yy,pp)),AP=float(AP(yy,pp)))
    ev=np.unique(hd[yy==1])
    for q in (0.05,0.10,0.20):
        al=pp>=np.quantile(pp,1-q); det=[];leads=[]
        for h_ in ev:
            mk=(hd==h_)&(yy==1)
            if (al&mk).any(): det.append(1); leads.append(np.nanmax(ld[mk&al]))
            else: det.append(0)
        out[f"s{int(q*100)}"]=float(np.mean(det)); out[f"ppv{int(q*100)}"]=float(yy[al].mean()); out[f"lead{int(q*100)}"]=float(np.median(leads)) if leads else None
    if prob:
        pc=np.clip(pp,1e-6,1-1e-6); lg=np.log(pc/(1-pc)); out["Brier"]=float(brier_score_loss(yy,pc)); out["cal_slope"]=float(LogisticRegression(C=1e6).fit(lg[:,None],yy).coef_[0,0]); out["mean_pred"]=float(pc.mean()); out["prev"]=float(yy.mean())
    return out
def boot(pa,pb,m,B=1000,seed=7):
    ii=np.where(m)[0]; pp=pat[ii]; up_=np.unique(pp); idx={q:ii[pp==q] for q in up_}; r=np.random.default_rng(seed); dA=[];dU=[]
    for _ in range(B):
        s=np.concatenate([idx[q] for q in r.choice(up_,len(up_))])
        if y[s].sum()==0: continue
        dA.append(AP(y[s],pb[s])-AP(y[s],pa[s])); dU.append(AUC(y[s],pb[s])-AUC(y[s],pa[s]))
    q=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    return dict(dAUROC=float(AUC(y[m],pb[m])-AUC(y[m],pa[m])),dAUROC_ci=q(dU),dAP=float(AP(y[m],pb[m])-AP(y[m],pa[m])),dAP_ci=q(dA))
lgt=lambda p:np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
mix=lambda a,b:sig((lgt(a)+lgt(b))/2)
