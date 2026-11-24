"""FULL01 evaluation (protocol FULL01_PROTOCOL.md). Usage:
  python full_eval.py cv <rep>        -> development CV, one repeat (5 folds), all arms; writes full_cv_rep<rep>.npz
  python full_eval.py heldout         -> fit on all development, score later-period patients once; writes full_heldout.npz
Keep-rule and every tuning step use development data only."""
import numpy as np, pandas as pd, json, sys, os, time, itertools, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
t0=time.time(); MODE=sys.argv[1]
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); U=os.path.join(HERE,"..","..","outputs","experiments","full_v2_20261009")+os.sep
z=np.load("full_features.npz"); meta=json.load(open("full_feature_meta.json"))
X,y,pat,hadm=z["X"],z["y"],z["patient"],z["hadm"]; names,axis=meta["names"],np.array(meta["axis"])
co=pd.read_csv(U+"cohort.csv").set_index("hadm_id"); period=co.loc[hadm,"period"].values
DEV=np.where(period=="development")[0]; LATE=np.where(period=="later")[0]
keep=[j for j in range(X.shape[1]) if np.isfinite(X[DEV,j]).sum()>=200 and len(np.unique(X[DEV][np.isfinite(X[DEV,j]),j]))>=2]
X=X[:,keep]; names=[names[j] for j in keep]; axis=axis[keep]; N=len(names)
concept=lambda n:n.split(".")[0] if "." in n else n
PAIRS=[("INJURY","HEMO"),("HEMO","PERFUSION"),("PERFUSION","RESP"),("HEMO","RESP"),("HEMO","RENAL"),("PERFUSION","RENAL"),("ELECTRICAL","HEMO"),("NEURO","PERFUSION")]
def groups(mode):
    idx=list(range(N)); ax=sorted(set(axis))
    if mode=="ADD":
        cs={}
        for i in idx: cs.setdefault(concept(names[i]),[]).append(i)
        return list(cs.values())
    G={a:[i for i in idx if axis[i]==a] for a in ax}; out=[G[a] for a in ax]
    if mode=="AXIS": out+=[G[a]+G[b] for a,b in PAIRS if a in G and b in G]
    return out
CST={"AXIS":groups("AXIS"),"WITHIN":groups("WITHIN"),"ADD":groups("ADD"),"FULL":None}
GRID=[dict(learning_rate=lr,max_leaf_nodes=lv,min_samples_leaf=ms,l2_regularization=l2) for lr,lv,ms,l2 in itertools.product([.03,.06],[4,8,16],[40,120],[1.,10.])]
def split_val(idx,seed):
    up_=np.unique(pat[idx]); r=np.random.default_rng(seed); v=set(r.choice(up_,max(1,int(.15*len(up_))),replace=False))
    isv=np.array([q in v for q in pat[idx]]); return idx[~isv],idx[isv]
def ll(yv,zz): return float(np.mean(np.logaddexp(0,zz)-yv*zz))
def tree(cst,tr,te,seed):
    a,b=split_val(tr,seed*1000+1); best=None
    for g in GRID:
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed,**g).fit(X[a],y[a],X_val=X[b],y_val=y[b]); s=ll(y[b],m.decision_function(X[b]))
        if best is None or s<best[0]: best=(s,g)
    zs=[];thr=[]
    for k in range(5):
        a,b=split_val(tr,seed*1000+10+k)
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed*10+k,**best[1]).fit(X[a],y[a],X_val=X[b],y_val=y[b])
        cal=LogisticRegression(C=1e6).fit(m.decision_function(X[b])[:,None],y[b]); A,B=cal.coef_[0,0],cal.intercept_[0]
        zs.append(A*m.decision_function(X[te])+B); thr.append(np.quantile(A*m.decision_function(X[b])+B,0.9))
    return 1/(1+np.exp(-np.mean(zs,0))),best[1],float(1/(1+np.exp(-np.mean(thr))))
def lr(tr,te,seed):
    a,b=split_val(tr,seed*1000+1); med=np.nanmedian(X[a],0); med=np.where(np.isfinite(med),med,0); mc=np.where(np.isnan(X[a]).any(0))[0]
    F=lambda r:np.column_stack([np.where(np.isnan(X[r]),med,X[r]),np.isnan(X[r][:,mc]).astype(np.float32)]); Za=F(a); mu=Za.mean(0); sd=Za.std(0)+1e-6; T=lambda r:np.clip((F(r)-mu)/sd,-8,8)
    best=None
    for C in [1e-4,3e-4,1e-3,3e-3,1e-2]:
        m=LogisticRegression(C=C,max_iter=3000).fit(T(a),y[a]); s=ll(y[b],m.decision_function(T(b)))
        if best is None or s<best[0]: best=(s,m,C)
    m=best[1]; cal=LogisticRegression(C=1e6).fit(m.decision_function(T(b))[:,None],y[b])
    return cal.predict_proba(m.decision_function(T(te))[:,None])[:,1],best[2]
ARMS=["AXIS","WITHIN","FULL","ADD"]
if MODE=="cv":
    rep=int(sys.argv[2]); up=np.unique(pat[DEV]); r=np.random.default_rng(1000+rep); s=up.copy(); r.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}
    f=np.full(len(y),-1); f[DEV]=[fm[q] for q in pat[DEV]]
    CK=f"full_cv_rep{rep}_ckpt.npz"; P={a:np.full(len(y),np.nan) for a in ARMS+["LR"]}; THR={a:np.full(len(y),np.nan) for a in ARMS}; ch=[]; done=set()
    if os.path.exists(CK): c=np.load(CK,allow_pickle=True); P=c["P"].item(); THR=c["THR"].item(); ch=list(c["ch"]); done=set(c["done"].tolist())
    for k in range(5):
        if k in done: continue
        tr=np.where(f>=0)[0][f[f>=0]!=k]; te=np.where(f==k)[0]
        for a in ARMS: P[a][te],g,th=tree(CST[a],tr,te,rep*10+k); THR[a][te]=th; ch.append((a,str(g)))
        P["LR"][te],C=lr(tr,te,rep*10+k); ch.append(("LR",C))
        done.add(k); np.savez(CK,P=np.array(P,dtype=object),THR=np.array(THR,dtype=object),ch=np.array(ch,dtype=object),done=np.array(sorted(done)))
        print("rep",rep,"fold",k,round(time.time()-t0),flush=True)
    np.savez_compressed(f"full_cv_rep{rep}.npz",**P,**{"thr_"+a:v for a,v in THR.items()})
elif MODE=="heldout":
    P={}; THR={}; ch={}
    for a in ARMS: p=np.full(len(y),np.nan); p[LATE],ch[a],THR[a]=tree(CST[a],DEV,LATE,777); P[a]=p; print("heldout",a,round(time.time()-t0),flush=True)
    p=np.full(len(y),np.nan); p[LATE],ch["LR"]=lr(DEV,LATE,777); P["LR"]=p
    np.savez_compressed("full_heldout.npz",**P,**{"thr_"+a:np.array(v) for a,v in THR.items()}); json.dump({k:str(v) for k,v in ch.items()},open("full_heldout_params.json","w"))
print("DONE",round(time.time()-t0))
