"""REV3-REL (prespecified 9 Oct 2026 17:40, before running).
Question: where does 'relational' gain come from, and does it depend on the endpoint?
Same features (v1, 323 columns) on two endpoints: primary (unplanned) and all-support (planned starts counted).
One 5-fold partition (seed 1000), reduced grid (6 configs: lr .03, leaves 4/8/16, min leaf 40/120, l2 1), 3 bags, FULL arm dropped (run time; amended 19:05 before any result was seen). Arms, all tuned with the EP3 grid/budget and 5-bag logit-averaged ensembles:
 ADD (within-concept only), WITHIN (within-system), AXIS (within-system + 8 knowledge pairs), FULL (unconstrained); plus penalised LR.
Contrasts (patient bootstrap 95% CI, AUROC and AP): within-system relations = WITHIN-ADD; cross-system pairs = AXIS-WITHIN;
remaining black-box interactions = FULL-AXIS. Hypothesis: relational gains are larger on the all-support endpoint."""
import numpy as np, json, os, sys, time, itertools, warnings
SUFX=sys.argv[1]
src=open("eval_ep2.py").read().split("up=np.unique(pat); OOF=")[0].replace('SUF=sys.argv[1] if len(sys.argv)>1 else ""',f'SUF="{SUFX}"')
exec(src)
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
GRID=[dict(learning_rate=lr,max_leaf_nodes=lv,min_samples_leaf=ms,l2_regularization=l2) for lr,lv,ms,l2 in itertools.product([.03],[4,8,16],[40,120],[1.])]
def split_val(idx,seed):
    up_=np.unique(pat[idx]); r=np.random.default_rng(seed); v=set(r.choice(up_,max(1,int(.15*len(up_))),replace=False))
    isv=np.array([q in v for q in pat[idx]]); return idx[~isv],idx[isv]
def ll(yv,z): return float(np.mean(np.logaddexp(0,z)-yv*z))
def tuned(Xa,cst,tr,te,seed):
    a,b=split_val(tr,seed*1000+1); best=None
    for g in GRID:
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed,**g).fit(Xa[a],y[a],X_val=Xa[b],y_val=y[b]); s=ll(y[b],m.decision_function(Xa[b]))
        if best is None or s<best[0]: best=(s,g)
    zs=[]
    for k in range(3):
        a,b=split_val(tr,seed*1000+10+k)
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed*10+k,**best[1]).fit(Xa[a],y[a],X_val=Xa[b],y_val=y[b])
        cal=LogisticRegression(C=1e6).fit(m.decision_function(Xa[b])[:,None],y[b]); zs.append(cal.coef_[0,0]*m.decision_function(Xa[te])+cal.intercept_[0])
    return 1/(1+np.exp(-np.mean(zs,0)))
def lr(tr,te,seed):
    a,b=split_val(tr,seed*1000+1); med=np.nanmedian(X[a],0); med=np.where(np.isfinite(med),med,0); mc=np.where(np.isnan(X[a]).any(0))[0]
    F=lambda r:np.column_stack([np.where(np.isnan(X[r]),med,X[r]),np.isnan(X[r][:,mc]).astype(np.float32)]); Za=F(a); mu=Za.mean(0); sd=Za.std(0)+1e-6; T=lambda r:np.clip((F(r)-mu)/sd,-8,8)
    best=None
    for C in [1e-4,3e-4,1e-3,3e-3,1e-2]:
        m=LogisticRegression(C=C,max_iter=2000).fit(T(a),y[a]); s=ll(y[b],m.decision_function(T(b)))
        if best is None or s<best[0]: best=(s,m)
    m=best[1]; cal=LogisticRegression(C=1e6).fit(m.decision_function(T(b))[:,None],y[b]); return cal.predict_proba(m.decision_function(T(te))[:,None])[:,1]
allp=list(range(N)); ARMS={"ADD":design(allp,"ADD",False),"WITHIN":design(allp,"WITHIN",False),"AXIS":design(allp,"AXIS",False)}
CK=f"rev3_rel{SUFX}_ckpt.npz"; up=np.unique(pat)
O={a:np.zeros(len(y)) for a in list(ARMS)+["LR"]}; done=set()
if os.path.exists(CK): c=np.load(CK,allow_pickle=True); O=c["O"].item(); done=set(c["done"].tolist())
for rep in range(1):
    r=np.random.default_rng(1000+rep); s=up.copy(); r.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}; f=np.array([fm[q] for q in pat])
    for k in range(5):
        if rep*5+k in done: continue
        tr,te=np.where(f!=k)[0],np.where(f==k)[0]
        for a,(Xa,cst) in ARMS.items(): O[a][te]+=tuned(Xa,cst,tr,te,rep*10+k)
        O["LR"][te]+=lr(tr,te,rep*10+k)
        done.add(rep*5+k); np.savez(CK,O=np.array(O,dtype=object),done=np.array(sorted(done))); print("rep",rep,"fold",k,round(time.time()-t0),flush=True)
def boot(pa,pb,B=1000,seed=7):
    idx={q:np.where(pat==q)[0] for q in up}; rr=np.random.default_rng(seed); dA=[];dU=[]
    for _ in range(B):
        s=np.concatenate([idx[q] for q in rr.choice(up,len(up))]); dA.append(AP(y[s],pb[s])-AP(y[s],pa[s])); dU.append(AUC(y[s],pb[s])-AUC(y[s],pa[s]))
    q=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    return dict(dAP=float(AP(y,pb)-AP(y,pa)),dAP_ci=q(dA),dAUROC=float(AUC(y,pb)-AUC(y,pa)),dAUROC_ci=q(dU))
R={"n":int(len(y)),"pos":int(y.sum()),"event_patients":int(len(np.unique(pat[y==1]))),"arms":{a:dict(AUROC=float(AUC(y,p)),AP=float(AP(y,p))) for a,p in O.items()},
   "contrasts":{"WITHIN-ADD":boot(O["ADD"],O["WITHIN"]),"AXIS-WITHIN":boot(O["WITHIN"],O["AXIS"]),"AXIS-ADD":boot(O["ADD"],O["AXIS"]),"AXIS-LR":boot(O["LR"],O["AXIS"])}}
json.dump(R,open(f"REV3_REL{SUFX}.json","w"),indent=1); np.savez_compressed(f"rev3_rel{SUFX}_oof.npz",**O); print(json.dumps(R,indent=1))
