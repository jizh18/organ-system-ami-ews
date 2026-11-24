"""Amendment 7, Part B: local updating with few local patients in eICU-CRD (post hoc).
usage: python3 a7_update.py fit [max_seconds] | analyse"""
import pickle,sys,os,time as _t
exec(open("full_common.py").read())
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
from scipy.stats import rankdata
Xmim,ymim=X,y; NM=len(ymim)
M=pickle.load(open("final_models.pkl","rb")); assert M["names"]==names
z=np.load("../eicu/eicu_features.npz",allow_pickle=True); n0=json.load(open("full_feature_meta_v2.json"))["names"]
Xe=z["X"][:,[n0.index(n) for n in names]].astype(np.float32); ye=z["y"].astype(int); pe=z["hadm"]; hosp=z["hospital"]; n=len(ye)
uh=np.unique(hosp); r=np.random.default_rng(2026); half=set(r.choice(uh,len(uh)//2,replace=False)); h1=np.array([h in half for h in hosp])
PAR={"AXIS":dict(learning_rate=0.03,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0),"FULL":dict(learning_rate=0.03,max_leaf_nodes=16,min_samples_leaf=40,l2_regularization=10.0)}
SIZES=[250,500,1000,2000]; REPS=3; DIRS=[(0,np.where(h1)[0],np.where(~h1)[0]),(1,np.where(~h1)[0],np.where(h1)[0])]
os.makedirs("a7_oof",exist_ok=True)
def sample(tr,rep,d,nn):
    up=np.unique(pe[tr]); rg=np.random.default_rng(9000+10*rep+d); order=rg.permutation(up); keep=set(order[:nn]); return tr[np.array([q in keep for q in pe[tr]])]
def split(idx,seed):
    up=np.unique(pe[idx]); rg=np.random.default_rng(seed); v=set(rg.choice(up,max(1,int(.15*len(up))),replace=False)); isv=np.array([q in v for q in pe[idx]]); return idx[~isv],idx[isv]
def fit_one(arm,strat,loc,te):
    zs=[]
    for k in range(5):
        a,b=split(loc,777*1000+10+k); Xl=Xe
        if strat=="local":
            Xl=Xe.copy(); an=np.all(np.isnan(Xl[a]),0); Xl[:,an]=0.0
            m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=CST[arm],random_state=7770+k,**PAR[arm]).fit(Xl[a],ye[a],X_val=Xl[b],y_val=ye[b])
        else:
            XA=np.vstack([Xmim,Xe[a]]); yA=np.concatenate([ymim,ye[a]]); w=np.concatenate([np.ones(NM),np.full(len(a),NM/len(a))])
            m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=CST[arm],random_state=7770+k,**PAR[arm]).fit(XA,yA,sample_weight=w,X_val=Xe[b],y_val=ye[b],sample_weight_val=np.ones(len(b)))
        cal=LogisticRegression(C=1e6).fit(m.decision_function(Xl[b])[:,None],ye[b]); zs.append(cal.coef_[0,0]*m.decision_function(Xl[te])+cal.intercept_[0])
    return np.mean(zs,0)
JOBS=[(strat,arm,nn,rep,d) for strat in ["local","pooled"] for nn in SIZES for rep in range(REPS) for arm in ["AXIS","FULL"] for d in (0,1)]
if sys.argv[1]=="fit":
    lim=float(sys.argv[2]) if len(sys.argv)>2 else 1e9; T0=_t.time()
    for strat,arm,nn,rep,d in JOBS:
        f=f"a7_oof/{strat}_{arm}_{nn}_{rep}_{d}.npy"
        if os.path.exists(f): continue
        if _t.time()-T0>lim: print("TIME LIMIT, resume later",flush=True); sys.exit(0)
        _,tr,te=DIRS[d]; loc=sample(tr,rep,d,nn); t1=_t.time(); zz=fit_one(arm,strat,loc,te); np.save(f,zz)
        print(strat,arm,nn,rep,d,"patients",len(np.unique(pe[loc])),"pos",int(ye[loc].sum()),"AUROC",round(float(AUC(ye[te],zz)),4),"sec",round(_t.time()-t1),flush=True)
    print("ALL FITS DONE",flush=True)
else:
    def oof(strat,arm,nn,rep):
        v=np.full(n,np.nan)
        for d,tr,te in DIRS: v[te]=np.load(f"a7_oof/{strat}_{arm}_{nn}_{rep}_{d}.npy")
        return v
    O={(s,a,nn):[oof(s,a,nn,rep) for rep in range(REPS)] for s in ["local","pooled"] for a in ["AXIS","FULL"] for nn in SIZES}
    S=np.load("../eicu/eicu_scores.npz"); D6=np.load("../eicu/eicu_d6_oof.npz"); sig=lambda q:1/(1+np.exp(-q))
    def fauc(yy,s):
        rk=rankdata(s); n1=yy.sum(); n0_=len(yy)-n1; return (rk[yy==1].sum()-n1*(n1+1)/2)/(n1*n0_)
    up=np.unique(pe); idx={p:np.where(pe==p)[0] for p in up}
    def boot(fn,B=1000,seed=2026):
        rg=np.random.default_rng(seed); out=[]
        for _ in range(B):
            s=np.concatenate([idx[p] for p in rg.choice(up,len(up))]); out.append(fn(s))
        return np.percentile(out,[2.5,97.5]).tolist()
    mean_auc=lambda L,s=None: float(np.mean([fauc(ye if s is None else ye[s],v if s is None else v[s]) for v in L]))
    R={"reference":{"MIMIC unchanged, organ-system":float(AUC(ye,S["AXIS"])),"MIMIC unchanged, unconstrained":float(AUC(ye,S["FULL"])),"full-half refit, organ-system":float(AUC(ye,D6["AXIS"])),"full-half refit, unconstrained":float(AUC(ye,D6["FULL"]))},"table":{},"comparisons":{}}
    for (s,a,nn),L in O.items():
        R["table"][f"{s}|{a}|{nn}"]=dict(AUROC=mean_auc(L),AUROC_reps=[float(AUC(ye,v)) for v in L],AP=float(np.mean([AP(ye,v) for v in L])),mean_pred=float(np.mean([sig(v).mean() for v in L])),observed=float(ye.mean()))
    for nn in SIZES:
        for lab,A_,B_ in [("pooled − local, organ-system",O[("pooled","AXIS",nn)],O[("local","AXIS",nn)]),("pooled − local, unconstrained",O[("pooled","FULL",nn)],O[("local","FULL",nn)]),
                          ("pooled organ-system − pooled unconstrained",O[("pooled","AXIS",nn)],O[("pooled","FULL",nn)]),("local organ-system − local unconstrained",O[("local","AXIS",nn)],O[("local","FULL",nn)]),
                          ("pooled organ-system − MIMIC unchanged",O[("pooled","AXIS",nn)],[S["AXIS"]])]:
            R["comparisons"][f"{lab}|{nn}"]=dict(diff=mean_auc(A_)-mean_auc(B_),ci=boot(lambda s,A_=A_,B_=B_:mean_auc(A_,s)-mean_auc(B_,s)))
            print(lab,nn,R["comparisons"][f"{lab}|{nn}"],flush=True)
    json.dump(R,open("A7_UPDATE.json","w"),indent=1,ensure_ascii=False); print(json.dumps(R["reference"]),json.dumps(R["table"],indent=0))
