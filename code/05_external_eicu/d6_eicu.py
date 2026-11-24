"""Amendment 5 analyses: score frozen MIMIC-IV models once on eICU and run the prespecified analyses."""
import numpy as np, pandas as pd, json, pickle, warnings, time, itertools
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); t0=time.time(); log=lambda *a:print(*a,round(time.time()-t0),flush=True)
M=pickle.load(open("../full/final_models.pkl","rb")); meta=json.load(open("../full/full_feature_meta_v2.json"))
names0=meta["names"]; names=M["names"]; AXL=M["AXL"]; FA=np.array(M["FA"]); axis=np.array(M["axis"])
sig=lambda z:1/(1+np.exp(-z)); lgt=lambda p:np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
def load(suf=""):
    z=np.load(f"eicu_features{suf}.npz",allow_pickle=True); D={k:z[k] for k in z.files}
    D["Xm"]=D["X"][:,[names0.index(n) for n in names]]; return D
D=load(); y=D["y"]; pat=D["hadm"]; hosp=D["hospital"]; lead=D["lead"]; X0=D["X"]; XP=D["XP"]; Xm=D["Xm"]; n=len(y)
def score_bag(models,X): return np.mean([A_*m.decision_function(X)+B_ for m,A_,B_ in models],0)
def score_lr(d,X):
    F=np.column_stack([np.where(np.isnan(X),d["med"],X),np.isnan(X[:,d["mc"]]).astype(np.float32)]); T=np.clip((F-d["mu"])/d["sd"],-8,8)
    return d["A"]*d["model"].decision_function(T)+d["B"]
def L_(n_,X=X0): return X[:,names0.index(n_)]
def news2(X,XP_):
    rr,sp,t,sbp,hr,gcs=[L_(c,X) for c in ["resp_rate.last","spo2.last","temperature.last","sbp.last","heart_rate.last","gcs_total"]]; fio=XP_[:,meta["proc_names"].index("fio2_est")]; sc=np.zeros(len(X))
    sc+=np.select([rr<=8,rr<=11,rr<=20,rr<=24,rr>24],[3,1,0,2,3],0)*np.isfinite(rr); sc+=np.select([sp<=91,sp<=93,sp<=95,sp>95],[3,2,1,0],0)*np.isfinite(sp)
    sc+=2*(fio>0.21); sc+=np.select([t<=35,t<=36,t<=38,t<=39,t>39],[3,1,0,1,2],0)*np.isfinite(t)
    sc+=np.select([sbp<=90,sbp<=100,sbp<=110,sbp<=219,sbp>219],[3,2,1,0,3],0)*np.isfinite(sbp)
    sc+=np.select([hr<=40,hr<=50,hr<=90,hr<=110,hr<=130,hr>130],[3,1,0,1,2,3],0)*np.isfinite(hr); sc+=3*(np.nan_to_num(gcs,nan=15)<15)
    return sc+np.random.default_rng(0).random(len(X))*1e-6
def grace(X,stemi,tni):
    age=L_("age",X); hr=L_("heart_rate.last",X); sbp=L_("sbp.last",X); cr=L_("creatinine.last",X); sf=L_("sf_ratio",X)
    sc=np.select([age<30,age<40,age<50,age<60,age<70,age<80,age<90,age>=90],[0,8,25,41,58,75,91,100],0)
    sc=sc+np.where(np.isfinite(hr),np.select([hr<50,hr<70,hr<90,hr<110,hr<150,hr<200,hr>=200],[0,3,9,15,24,38,46],9),9)
    sc=sc+np.where(np.isfinite(sbp),np.select([sbp<80,sbp<100,sbp<120,sbp<140,sbp<160,sbp<200,sbp>=200],[58,53,43,34,24,10,0],34),34)
    sc=sc+np.where(np.isfinite(cr),np.select([cr<.4,cr<.8,cr<1.2,cr<1.6,cr<2,cr<4,cr>=4],[1,4,7,10,13,21,28],7),7)
    sc=sc+np.where(np.nan_to_num(sf,nan=999)<315,39,0)
    enz=(np.nan_to_num(L_("troponin_t.last",X))>0.01)|(np.nan_to_num(tni)>0.04)
    sc=sc+28*stemi.astype(float)+14*enz
    return sc+np.random.default_rng(1).random(len(X))*1e-6
def all_scores(D):
    S={"AXIS":score_bag(M["final_AXIS"]["models"],D["Xm"]),"FULL":score_bag(M["final_FULL"]["models"],D["Xm"]),"LR":score_lr(M["final_LR"],D["Xm"]),
       "AXIS_dev":score_bag(M["dev_AXIS"]["models"],D["Xm"]),"GRACE":grace(D["X"],D["stemi"],D["tni"]),"NEWS2":news2(D["X"],D["XP"])}
    return S
S=all_scores(D); log("scored")
R={"n_assess":int(n),"n_pos":int(y.sum()),"n_patients":int(len(np.unique(pat))),"n_episodes":int(len(np.unique(pat[y==1]))),"n_hospitals":int(len(np.unique(hosp))),
   "prevalence":float(y.mean()),"y_death":int(D["y_death"].sum()),"y_circ":int(D["y_circ"].sum()),"y_resp":int(D["y_resp"].sum())}
def det(pp,m,al,yy=None):
    yy=y[m] if yy is None else yy[m]; hd=pat[m]; ld=lead[m]; ev=np.unique(hd[yy==1]); d=[];ls=[]
    for h_ in ev:
        mk=(hd==h_)&(yy==1)
        if (al&mk).any(): d.append(1); ls.append(np.nanmax(ld[mk&al]))
        else: d.append(0)
    return float(np.mean(d)),(float(np.median(ls)) if ls else None)
def metrics(p,m=None,yv=None):
    m=np.ones(n,bool) if m is None else m; yy=(y if yv is None else yv)[m]; pp=p[m]
    o=dict(n=int(m.sum()),pos=int(yy.sum()),AUROC=float(AUC(yy,pp)),AP=float(AP(yy,pp)))
    for q in (0.05,0.10,0.20):
        al=pp>=np.quantile(pp,1-q); s_,l_=det(pp,m,al,yv); o[f"s{int(q*100)}"]=s_; o[f"lead{int(q*100)}"]=l_; o[f"ppv{int(q*100)}"]=float(yy[al].mean())
    return o
def boot_ci(f,m=None,B=1000,seed=7,cluster=None):
    m=np.ones(n,bool) if m is None else m; cl=pat if cluster is None else cluster
    ii=np.where(m)[0]; cc=cl[ii]; uc=np.unique(cc); idx={q:ii[cc==q] for q in uc}; r=np.random.default_rng(seed); out=[]
    for _ in range(B):
        s=np.concatenate([idx[q] for q in r.choice(uc,len(uc))])
        if y[s].sum()==0 or y[s].sum()==len(s): continue
        out.append(f(s))
    out=np.array(out,dtype=float)
    if out.ndim==1: return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]
    return np.percentile(out,[2.5,97.5],axis=0).T.tolist()
def ci1(p,m=None,cluster=None,B=1000):
    v=boot_ci(lambda s:AUC(y[s],p[s]),m,B,cluster=cluster); return v
R=json.load(open("EICU_RESULTS.json"))
uh=np.unique(hosp); r=np.random.default_rng(2026); half=set(r.choice(uh,len(uh)//2,replace=False)); h1=np.array([h in half for h in hosp])
def decompose(m,Xt):
    nn=len(Xt); out={}
    for pr in m._predictors:
        nd=pr[0].nodes; node=np.zeros(nn,dtype=np.int64); mask=np.zeros(nn,dtype=np.int64)
        for _ in range(64):
            leaf=nd["is_leaf"][node].astype(bool)
            if leaf.all(): break
            f=nd["feature_idx"][node]; thr=nd["num_threshold"][node]; ml=nd["missing_go_to_left"][node].astype(bool)
            x=Xt[np.arange(nn),f]; left=np.where(np.isnan(x),ml,x<=thr)
            nxt=np.where(left,nd["left"][node],nd["right"][node]); mask=np.where(leaf,mask,mask|(1<<FA[f])); node=np.where(leaf,node,nxt)
        val=nd["value"][node]
        for mk in np.unique(mask):
            sel=mask==mk; lab="×".join(AXL[i] for i in range(len(AXL)) if mk>>i&1) or "intercept"
            out.setdefault(lab,np.zeros(nn))[sel]+=val[sel]
    return out
def terms(models,Xt):
    C={}
    for m,A_,B_ in models:
        o=decompose(m,Xt); base=float(m._baseline_prediction.ravel()[0])
        for g,v in o.items():
            if g=="intercept": continue
            C.setdefault(g,np.zeros(len(Xt))); C[g]+=A_*v/len(models)
        C.setdefault("intercept",np.zeros(len(Xt))); C["intercept"]+=(A_*base+B_)/len(models)
    return C

import ast
P=json.load(open("../full/heldout_params.json"))
G={a:[i for i in range(len(names)) if axis[i]==a] for a in AXL}
ARMS={"AXIS":([G[a] for a in AXL]+[G[a]+G[b] for a,b in M["PAIRS"]],ast.literal_eval(P["AXIS"])),"FULL":(None,ast.literal_eval(P["FULL"])),
      "OPEN":([G[a] for a in AXL]+[G[a]+G[b] for a,b in itertools.combinations(AXL,2)],dict(learning_rate=0.03,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0))}
XM0=Xm.copy(); OOF={k:np.full(n,np.nan) for k in ARMS}
for tr_,te_ in [(np.where(h1)[0],np.where(~h1)[0]),(np.where(~h1)[0],np.where(h1)[0])]:
    for arm,(cst,par) in ARMS.items():
        zs=[]
        for k in range(5):
            Xm=XM0.copy(); up=np.unique(pat[tr_]); rg=np.random.default_rng(777*1000+10+k); v=set(rg.choice(up,max(1,int(.15*len(up))),replace=False)); isv=np.array([q in v for q in pat[tr_]])
            a,b=tr_[~isv],tr_[isv]; an=np.all(np.isnan(Xm[a]),0); Xm[:,an]=0.0
            m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=7770+k,**par).fit(Xm[a],y[a],X_val=Xm[b],y_val=y[b])
            cal=LogisticRegression(C=1e6).fit(m.decision_function(Xm[b])[:,None],y[b]); zs.append(cal.coef_[0,0]*m.decision_function(Xm[te_])+cal.intercept_[0])
        OOF[arm][te_]=np.mean(zs,0); log(arm,"half",float(AUC(y[te_],OOF[arm][te_])))
OOF["NEWS2"]=news2(D["X"],D["XP"])
out={k:dict(AUROC=float(AUC(y,v)),AP=float(AP(y,v)),AUROC_ci=boot_ci(lambda s,v=v:AUC(y[s],v[s]))) for k,v in OOF.items()}
for b in ["FULL","OPEN","NEWS2"]:
    ci=boot_ci(lambda s,b=b:[AUC(y[s],OOF["AXIS"][s])-AUC(y[s],OOF[b][s]),AP(y[s],OOF["AXIS"][s])-AP(y[s],OOF[b][s])])
    out[f"AXIS-{b}"]=dict(dAUROC=float(AUC(y,OOF["AXIS"])-AUC(y,OOF[b])),dAUROC_ci=ci[0],dAP=float(AP(y,OOF["AXIS"])-AP(y,OOF[b])),dAP_ci=ci[1])
pz=sig(OOF["AXIS"]); out["AXIS_calibration"]=dict(mean_pred=float(pz.mean()),observed=float(y.mean()),slope=float(LogisticRegression(C=1e6).fit(OOF["AXIS"][:,None],y).coef_[0,0]))
out["AXIS_death_only_AUROC"]=float(AUC(D["y_death"],OOF["AXIS"]))
json.dump(out,open("EICU_D6.json","w"),indent=1); np.savez_compressed("eicu_d6_oof.npz",**OOF); log(json.dumps(out,indent=1))
