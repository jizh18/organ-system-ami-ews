"""REV2b: same stability metrics for penalised LR (system attribution = sum of coef*standardised value per system; missingness indicators assigned to their feature's system)."""
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
src=open("rev1.py").read().split("OUT={}")[0]; exec(src)
from scipy.stats import spearmanr
Xall=X[:,allp]; NS=len(AXL)
SYS=np.zeros((3,len(y),NS))
for rep in range(3):
    f=FOLDS[rep]
    for k in range(5):
        tr,te=np.where(f!=k)[0],np.where(f==k)[0]; a,b=split_val(tr,(rep*10+k)*1000+1)
        med=np.nanmedian(Xall[a],0); med=np.where(np.isfinite(med),med,0); mc=np.where(np.isnan(Xall[a]).any(0))[0]
        F=lambda rows:np.column_stack([np.where(np.isnan(Xall[rows]),med,Xall[rows]),np.isnan(Xall[rows][:,mc]).astype(np.float32)])
        Za=F(a); mu=Za.mean(0); sd=Za.std(0)+1e-6; T=lambda rows:np.clip((F(rows)-mu)/sd,-8,8)
        best=None
        for C in [0.0001,0.0003,0.001,0.003,0.01]:
            m=LogisticRegression(C=C,max_iter=2000).fit(T(a),y[a]); s=ll(y[b],m.decision_function(T(b)))
            if best is None or s<best[0]: best=(s,m)
        m=best[1]; A=LogisticRegression(C=1e6).fit(m.decision_function(T(b))[:,None],y[b]).coef_[0,0]
        colsys=np.concatenate([FA,FA[mc]]); Zt=T(te)*m.coef_[0]*A
        for i in range(NS): SYS[rep,te,i]=Zt[:,colsys==i].sum(1)
        log("LR rep",rep,"fold",k)
np.save("rev2b_lr_sys.npy",SYS); log("done")
