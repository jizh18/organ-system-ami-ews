import numpy as np, json
src=open("rev1.py").read().split("OUT={}")[0]; exec(src)   # ep3 header + bag_fixed, decompose
from scipy.optimize import brentq
F=np.load("rev1_final_oof.npz"); pA,pF=F["AXIS"],F["FULL"]
BL=np.load("bl_oof.npz"); BL2=np.load("bl2_oof.npz"); G3=np.load("bl03_oof.npz"); W=np.load("rev1_B.npz")["WITHIN"]
R={"cv":{"AXIS":metrics(pA),"FULL":metrics(pF)}}
R["cmp"]={n:boot(p,pA) for n,p in [("FULL",pF),("LR",BL2["LR"]),("MLP",BL["MLP"]),("GRU",G3["GRU"]),("GRACE",SC["GRACE"]),("NEWS2",SC["NEWS2"]),("WITHIN",W)]}
up_=np.unique(pat); idx={q:np.where(pat==q)[0] for q in up_}; r=np.random.default_rng(11); S=[np.concatenate([idx[q] for q in r.choice(up_,len(up_))]) for _ in range(1000)]
ci=lambda f:[float(np.percentile([f(s) for s in S],2.5)),float(np.percentile([f(s) for s in S],97.5))]
R["ci"]={"AXIS_AUROC":ci(lambda s:AUC(y[s],pA[s])),"AXIS_AP":ci(lambda s:AP(y[s],pA[s])),"FULL_AUROC":ci(lambda s:AUC(y[s],pF[s])),"WITHIN_AUROC":ci(lambda s:AUC(y[s],W[s]))}
lg=np.log(np.clip(pA,1e-6,1-1e-6)/(1-np.clip(pA,1e-6,1-1e-6)))
R["cal"]={"slope_ci":ci(lambda s:LogisticRegression(C=1e6).fit(lg[s][:,None],y[s]).coef_[0,0]),"intercept":float(brentq(lambda a:sig(lg+a).mean()-y.mean(),-5,5))}
R["cal"]["intercept_ci"]=ci(lambda s:brentq(lambda a:sig(lg[s]+a).mean()-y[s].mean(),-5,5))
ev_h=np.unique(hadm[y==1]); row={}
for q in (0.05,0.10,0.20):
    al=pA>=np.quantile(pA,1-q); det=[];leads=[]
    for h_ in ev_h:
        m=(hadm==h_)&(y==1)
        if (al&m).any(): det.append(1); leads.append(np.nanmax(lead[m&al]))
        else: det.append(0)
    row[f"s{int(q*100)}"]=float(np.mean(det)); row[f"ppv{int(q*100)}"]=float(y[al].mean()); row[f"lead{int(q*100)}"]=float(np.median(leads))
R["alert_AXIS"]=row
stemi=np.nan_to_num(L_("ecg_adm_ste"))>0; icu=np.isfinite(L_("heart_rate.last"))
R["sub"]={nm:{"AXIS":float(AUC(y[mk],pA[mk])),"FULL":float(AUC(y[mk],pF[mk])),"GRACE":float(AUC(y[mk],SC["GRACE"][mk])),"NEWS2":float(AUC(y[mk],SC["NEWS2"][mk]))} for nm,mk in [("STE",stemi),("nonSTE",~stemi),("ICU",icu),("ward",~icu)]}
R["sub_cmp_ICU"]={"AXIS_vs_GRACE":boot(SC["GRACE"],pA,icu),"AXIS_vs_NEWS2":boot(SC["NEWS2"],pA,icu)}
# temporal (logit-averaged, tuned within the earlier groups)
import pandas as pd
ptab=pd.read_csv("/mnt/user-data/uploads/KG-v1/outputs/experiments/ep2_20261009/patients.csv.gz").set_index("subject_id")
grp=pd.Series(pat).map(ptab.anchor_year_group).values; tr=np.where(grp!="2014 - 2016")[0]; te=np.where(grp=="2014 - 2016")[0]; mk=np.zeros(len(y),bool); mk[te]=True
TP={}
for a,(Xa,cst) in {"AXIS":design(allp,"AXIS",False),"FULL":design(allp,None,False)}.items():
    aa,bb=split_val(tr,777*1000+1); best=None
    for g in GRID:
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=777,**g).fit(Xa[aa],y[aa],X_val=Xa[bb],y_val=y[bb]); s_=ll(y[bb],m.decision_function(Xa[bb]))
        if best is None or s_<best[0]: best=(s_,g)
    z,_,_=bag_fixed(Xa,cst,tr,te,777,best[1]); p=np.zeros(len(y)); p[te]=sig(z); TP[a]=p
R["temporal"]={a:metrics(p,mk) for a,p in TP.items()}
R["temporal_cmp"]={"AXIS_vs_FULL":boot(TP["FULL"],TP["AXIS"],mk),"AXIS_vs_GRACE":boot(SC["GRACE"],TP["AXIS"],mk),"AXIS_vs_NEWS2":boot(SC["NEWS2"],TP["AXIS"],mk),
   "AXIS_vs_LR":boot(BL2["temporal_LR"],TP["AXIS"],mk),"AXIS_vs_GRU":boot(G3["temporal_GRU"],TP["AXIS"],mk)}
np.savez_compressed("rev1_temporal.npz",**TP)
json.dump(R,open("REV1_FINAL.json","w"),indent=1)
for k in ["cv","ci","cal","alert_AXIS","sub","temporal"]: print(k,json.dumps(R[k])[:700])
print({k:(round(v["dAUROC"],3),[round(x,3) for x in v["dAUROC_ci"]],round(v["dAP"],3),[round(x,3) for x in v["dAP_ci"]]) for k,v in {**R["cmp"],**{"T_"+a:b for a,b in R["temporal_cmp"].items()},**{"ICU_"+a:b for a,b in R["sub_cmp_ICU"].items()}}.items()})
