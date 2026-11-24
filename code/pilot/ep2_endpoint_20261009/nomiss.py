"""稳健性：缺失值一律填中位数（模型看不到“做没做这项检查”），比较AXIS性能"""
import numpy as np, json, warnings, sys
warnings.filterwarnings("ignore")
src=open("eval_ep2.py").read().split("up=np.unique(pat); OOF=")[0].replace('SUF=sys.argv[1] if len(sys.argv)>1 else ""','SUF=""')
exec(src)
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
med=np.nanmedian(X,axis=0); Xi=np.where(np.isfinite(X),X,med)
up=np.unique(pat); res={}
for nmode,XX in [("AXIS_medianimputed",Xi)]:
    oof=np.zeros(len(y))
    for rep in range(3):
        rng=np.random.default_rng(1000+rep); s=up.copy(); rng.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}; f=np.array([fm[q] for q in pat])
        Xa=XX[:,allp]; cst=[[allp.index(i) for i in g] for g in groups(allp,"AXIS")]
        for k in range(5): oof[f==k]+=fit_pred(Xa,cst,f!=k,f==k,rep)/3
    res[nmode]=oof
o=np.load("ep2_oof.npz")["AXIS"]
up_=np.unique(pat); idx={q:np.where(pat==q)[0] for q in up_}; rng=np.random.default_rng(7); dA=[];dU=[]
p=res["AXIS_medianimputed"]
for _ in range(1000):
    r=np.concatenate([idx[q] for q in rng.choice(up_,len(up_))])
    dA.append(AP(y[r],p[r])-AP(y[r],o[r])); dU.append(AUC(y[r],p[r])-AUC(y[r],o[r]))
out=dict(AXIS=dict(AP=float(AP(y,o)),AUROC=float(AUC(y,o))),AXIS_medianimputed=dict(AP=float(AP(y,p)),AUROC=float(AUC(y,p))),
  dAP=float(AP(y,p)-AP(y,o)),dAP_ci=[float(np.percentile(dA,2.5)),float(np.percentile(dA,97.5))],dAUROC=float(AUC(y,p)-AUC(y,o)),dAUROC_ci=[float(np.percentile(dU,2.5)),float(np.percentile(dU,97.5))])
np.save("oof_nomiss.npy",p); json.dump(out,open("NOMISS.json","w"),indent=1); print(out)
