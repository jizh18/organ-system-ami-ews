"""Addendum 5d: graph-term recalibration (TR) vs intercept+slope recalibration vs local retraining, by local sample size."""
import numpy as np, pandas as pd, json, pickle, ast, time, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
warnings.filterwarnings("ignore"); t0=time.time(); log=lambda *a:print(*a,round(time.time()-t0),flush=True)
S=np.load("eicu_scores.npz"); z=np.load("eicu_features.npz"); M=pickle.load(open("../full/final_models.pkl","rb"))
meta=json.load(open("../full/full_feature_meta_v2.json")); names0=meta["names"]
y=z["y"]; pat=z["hadm"]; hosp=z["hospital"]; n=len(y)
Xm=z["X"][:,[names0.index(c) for c in M["names"]]]
TK=sorted(k for k in S.files if k.startswith("Cf_") and k!="Cf_intercept"); T=np.column_stack([S[k] for k in TK]); zA=S["AXIS"]; zF=S["FULL"]
sig=lambda v:1/(1+np.exp(-v))
uh=np.unique(hosp); r=np.random.default_rng(2026); half=set(r.choice(uh,len(uh)//2,replace=False)); h1=np.array([h in half for h in hosp])
HALVES=[(np.where(h1)[0],np.where(~h1)[0]),(np.where(~h1)[0],np.where(h1)[0])]
P=json.load(open("../full/heldout_params.json")); AXL=M["AXL"]; axis=np.array(M["axis"])
G={a:[i for i in range(len(axis)) if axis[i]==a] for a in AXL}; CST=[G[a] for a in AXL]+[G[a]+G[b] for a,b in M["PAIRS"]]; PAR=ast.literal_eval(P["AXIS"])
def fit_tr(idx):
    # penalised logistic regression with offset, solved by L-BFGS (bug fix 12 Oct: undamped Newton diverged)
    from scipy.optimize import minimize
    X_=np.column_stack([np.ones(len(idx)),T[idx]]); yy=y[idx]; off=zA[idx]; lam=np.r_[0,np.ones(T.shape[1])]
    def f(b):
        e=off+X_@b; p=sig(e); l=np.sum(np.logaddexp(0,e)-yy*e)+0.5*np.sum(lam*b*b); g=X_.T@(p-yy)+lam*b; return l,g
    r=minimize(f,np.zeros(X_.shape[1]),jac=True,method="L-BFGS-B",options=dict(maxiter=5000,gtol=1e-8)); return r.x
def ap_tr(b,idx): return zA[idx]+np.column_stack([np.ones(len(idx)),T[idx]])@b
def fit_is(zz,idx): m=LogisticRegression(C=1e6).fit(zz[idx][:,None],y[idx]); return m
def local_refit(idx,seed):
    up=np.unique(pat[idx]); rg=np.random.default_rng(seed); v=set(rg.choice(up,max(1,int(.15*len(up))),replace=False)); isv=np.array([q in v for q in pat[idx]])
    a,b=idx[~isv],idx[isv]
    if y[a].sum()==0 or y[b].sum()==0: return None
    X=Xm.copy(); an=np.all(np.isnan(X[a]),0); X[:,an]=0.0
    m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=CST,random_state=seed,**PAR).fit(X[a],y[a],X_val=X[b],y_val=y[b])
    cal=LogisticRegression(C=1e6).fit(m.decision_function(X[b])[:,None],y[b]); return lambda te:cal.coef_[0,0]*m.decision_function(X[te])+cal.intercept_[0]
def calib(zz):
    yy=y; a=0.0
    for _ in range(50): p=sig(zz+a); a+=np.sum(yy-p)/np.sum(p*(1-p))
    return dict(citl=float(a),slope=float(LogisticRegression(C=1e6).fit(zz[:,None],yy).coef_[0,0]),mean_pred=float(sig(zz).mean()))
SIZES=[50,100,200,500,1000,"all"]; RES={}
RES["as_is"]=dict(AUROC=float(AUC(y,zA)),AP=float(AP(y,zA)),**calib(zA)); RES["FULL_as_is"]=dict(AUROC=float(AUC(y,zF)),AP=float(AP(y,zF)))
for nsz in SIZES:
    reps=1 if nsz=="all" else 20; out={k:[] for k in ["TR","IS_AXIS","IS_FULL","LOCAL"]}
    for rep in range(reps):
        oof={k:np.full(n,np.nan) for k in out}; ok=True
        for d,(tr,te) in enumerate(HALVES):
            up=np.unique(pat[tr]); rg=np.random.default_rng(10000*d+rep+(0 if nsz=="all" else nsz))
            for _ in range(100):
                sel=up if nsz=="all" else rg.choice(up,min(nsz,len(up)),replace=False); idx=tr[np.isin(pat[tr],sel)]
                if y[idx].sum()>0: break
            b=fit_tr(idx); oof["TR"][te]=ap_tr(b,te)
            if nsz=="all": RES.setdefault("TR_weights_by_half",[]).append({k[3:]:float(1+v) for k,v in zip(TK,b[1:])}|{"delta_intercept":float(b[0])})
            m=fit_is(zA,idx); oof["IS_AXIS"][te]=m.decision_function(zA[te][:,None]); m=fit_is(zF,idx); oof["IS_FULL"][te]=m.decision_function(zF[te][:,None])
            if rep<10:
                f=local_refit(idx,777+rep+d*100)
                if f is None: ok=False
                else: oof["LOCAL"][te]=f(te)
        for k in out:
            if k=="LOCAL" and (rep>=10 or not ok): continue
            v=oof[k]; out[k].append(dict(AUROC=float(AUC(y,v)),AP=float(AP(y,v)),**calib(v)))
    RES[str(nsz)]={k:dict(n_rep=len(v),AUROC_mean=float(np.mean([x["AUROC"] for x in v])),AUROC_p2_5=float(np.percentile([x["AUROC"] for x in v],2.5)),AUROC_p97_5=float(np.percentile([x["AUROC"] for x in v],97.5)),
        AP_mean=float(np.mean([x["AP"] for x in v])),slope_mean=float(np.mean([x["slope"] for x in v])),citl_mean=float(np.mean([x["citl"] for x in v]))) for k,v in out.items() if v}
    log(nsz,{k:round(v["AUROC_mean"],3) for k,v in RES[str(nsz)].items()})
RES["terms"]=[k[3:] for k in TK]; RES["n_events_per_patient_rate"]=float(len(np.unique(pat[y==1]))/len(np.unique(pat)))
json.dump(RES,open("EICU_TR.json","w"),indent=1); log("DONE")
