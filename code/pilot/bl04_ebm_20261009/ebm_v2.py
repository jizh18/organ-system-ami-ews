"""BL04: Explainable Boosting Machine (GA2M) baseline. Same patient folds (3 reps x 5), 15% inner validation patients
for Platt recalibration, temporal split. Input: ../bl03_sequence_20261009/bl03_input.npz. Output: BL04_RESULTS.json, bl04_oof.npz."""
import numpy as np, json, time, os, interpret

from interpret.glassbox import ExplainableBoostingClassifier as EBM
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss
from sklearn.linear_model import LogisticRegression
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
LOG=open("bl04v2_log.txt","a")
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
log("interpret",interpret.__version__)
D=np.load(os.path.join("..","bl03_sequence_20261009","bl03_input.npz")); X=D["X"]; y=D["y"].astype(int); pat=D["pat"]; hadm=D["hadm"]; folds=D["folds"]; tt=D["temporal_test"]; t0=time.time()
def split_val(idx,seed):
    up_=np.unique(pat[idx]); r=np.random.default_rng(seed); v=set(r.choice(up_,max(1,int(.15*len(up_))),replace=False))
    isv=np.array([q in v for q in pat[idx]]); return idx[~isv],idx[isv]
def fit_predict(tr,te,seed):
    a,b=split_val(tr,seed*1000+1)
    # patient-grouped internal bags: each outer bag holds out 15% of training patients for early stopping
    pa=pat[a]; up_=np.unique(pa); rr=np.random.default_rng(seed); NB=8; bags=np.ones((NB,len(a)),dtype=np.int8)
    for k_ in range(NB):
        hold=set(rr.choice(up_,max(1,int(.15*len(up_))),replace=False)); bags[k_,np.array([q in hold for q in pa])]=-1
    m=EBM(interactions=10,outer_bags=NB,max_rounds=5000,early_stopping_rounds=100,random_state=seed,n_jobs=-1).fit(X[a],y[a],bags=bags.T)
    zb=m.decision_function(X[b]); cal=LogisticRegression(C=1e6).fit(zb[:,None],y[b])
    return cal.predict_proba(m.decision_function(X[te])[:,None])[:,1]
def metrics(p,m):
    yy,pp,hd=y[m],p[m],hadm[m]; out=dict(n=int(m.sum()),pos=int(yy.sum()),AUROC=float(AUC(yy,pp)),AP=float(AP(yy,pp)))
    al=pp>=np.quantile(pp,0.9); ev=np.unique(hd[yy==1]); out["alert10_event_sens"]=float(np.mean([(al&(hd==q)&(yy==1)).any() for q in ev])); out["alert10_PPV"]=float(yy[al].mean())
    pc=np.clip(pp,1e-6,1-1e-6); lg=np.log(pc/(1-pc)); out["Brier"]=float(brier_score_loss(yy,pc)); out["cal_slope"]=float(LogisticRegression(C=1e6).fit(lg[:,None],yy).coef_[0,0]); return out
def boot(pa,pb,m,B=1000,seed=7):
    ii=np.where(m)[0]; pp=pat[ii]; up_=np.unique(pp); idx={q:ii[pp==q] for q in up_}; r=np.random.default_rng(seed); dA=[]
    for _ in range(B):
        s=np.concatenate([idx[q] for q in r.choice(up_,len(up_))])
        if y[s].sum()==0: continue
        dA.append(AUC(y[s],pb[s])-AUC(y[s],pa[s]))
    return dict(dAUROC=float(AUC(y[m],pb[m])-AUC(y[m],pa[m])),dAUROC_ci=[float(np.percentile(dA,2.5)),float(np.percentile(dA,97.5))])
CK="bl04v2_ckpt.npz"; OOF=np.zeros(len(y)); done=set()
if os.path.exists(CK): c=np.load(CK); OOF=c["OOF"]; done=set(c["done"].tolist())
for rep in range(3):
    for k in range(5):
        key=rep*5+k
        if key in done: continue
        tr=np.where(folds[rep]!=k)[0]; te=np.where(folds[rep]==k)[0]
        p=fit_predict(tr,te,rep*10+k)
        OOF[te]+=p/3; done.add(key); np.savez(CK,OOF=OOF,done=np.array(sorted(done)))
        log("rep",rep,"fold",k,round(time.time()-t0))
allm=np.ones(len(y),bool)
RES={"cv":{"EBM":metrics(OOF,allm)},"cv_cmp":{"AXIS_vs_EBM":boot(OOF,D["axis_oof"],allm)}}
tr=np.where(~tt)[0]; te=np.where(tt)[0]; pt=np.zeros(len(y)); pt[te]=fit_predict(tr,te,777); RES["temporal"]={"EBM":metrics(pt,tt)}
np.savez_compressed("bl04v2_oof.npz",EBM=OOF,temporal_EBM=pt); json.dump(RES,open("BL04v2_RESULTS.json","w"),indent=1)
log("cv EBM",round(RES["cv"]["EBM"]["AUROC"],3),round(RES["cv"]["EBM"]["AP"],3),"| temporal",round(RES["temporal"]["EBM"]["AUROC"],3)); log(RES["cv_cmp"]); log("DONE",round(time.time()-t0))
