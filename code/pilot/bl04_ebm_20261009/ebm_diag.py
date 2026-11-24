"""Quick EBM diagnostic on one fold (rep 0, fold 0): raw test AUROC of several EBM settings vs logistic regression."""
import numpy as np, time, os
from interpret.glassbox import ExplainableBoostingClassifier as EBM
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
os.chdir(os.path.dirname(os.path.abspath(__file__)))
D=np.load(os.path.join("..","bl03_sequence_20261009","bl03_input.npz")); X=D["X"].astype(np.float64); y=D["y"].astype(int); f=D["folds"][0]
tr=np.where(f!=0)[0]; te=np.where(f==0)[0]
out=open("ebm_diag.txt","w")
def rep(name,p,t0): s=f"{name}: test AUROC {AUC(y[te],p):.3f}  ({time.time()-t0:.0f}s)"; print(s,flush=True); out.write(s+"\n"); out.flush()
med=np.nanmedian(X[tr],0); med=np.where(np.isfinite(med),med,0); Xi=np.where(np.isnan(X),med,X)
t=time.time(); m=LogisticRegression(C=1e-3,max_iter=3000).fit((Xi[tr]-Xi[tr].mean(0))/(Xi[tr].std(0)+1e-6),y[tr]); rep("LR (median-imputed)",m.decision_function((Xi[te]-Xi[tr].mean(0))/(Xi[tr].std(0)+1e-6)),t)
t=time.time(); m=EBM(interactions=0,random_state=0,n_jobs=-1).fit(X[tr],y[tr]); rep("EBM main effects, NaN native",m.decision_function(X[te]),t)
t=time.time(); m=EBM(interactions=0,random_state=0,n_jobs=-1).fit(Xi[tr],y[tr]); rep("EBM main effects, median-imputed",m.decision_function(Xi[te]),t)
t=time.time(); m=EBM(interactions=10,random_state=0,n_jobs=-1).fit(X[tr],y[tr]); rep("EBM +10 interactions, NaN native",m.decision_function(X[te]),t)
print("done")
