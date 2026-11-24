import numpy as np, json, pandas as pd
src=open("ep3.py").read().split("RES={}")[0]; exec(src)
sig=lambda z:1/(1+np.exp(-z))
A=np.load("rev1_A.npz"); B=np.load("rev1_B.npz"); C=np.load("rev1_C.npz"); O3=np.load("ep3_oof.npz")
pA=sig(A["ZA"]).mean(0); pF=sig(A["ZF"]).mean(0)   # average of rep-level calibrated probs (each rep: logit-avg over bags)
R={}
R["final"]={"AXIS_logitavg":metrics(pA),"FULL_logitavg":metrics(pF),"AXIS_ep3_probavg":metrics(O3["AXIS"])}
R["recon_max_err"]=float(A["recon"].max())
R["per_rep"]={a:[float(AUC(y,sig(Z[r]))) for r in range(3)] for a,Z in [("AXIS",A["ZA"]),("FULL",A["ZF"])]}
R["per_rep_AP"]={a:[float(AP(y,sig(Z[r]))) for r in range(3)] for a,Z in [("AXIS",A["ZA"]),("FULL",A["ZF"])]}
R["cmp"]={"AXIS_vs_FULL":boot(pF,pA),"AXIS_vs_WITHIN":boot(B["WITHIN"],pA),"WITHIN_vs_FULL":boot(pF,B["WITHIN"])}
R["WITHIN"]=metrics(B["WITHIN"])
O2=np.load("ep2_oof.npz");
for d in ["delay0","delay60"]:
    R[d]=metrics(C[d]); R["cmp"][f"{d}_vs_delay30"]=boot(pA,C[d])
# ---- window sensitivity: final AXIS predictions vs relabelled outcomes on shared landmarks
hh=np.load("ep2_features.npz")["h"]; key=pd.DataFrame({"hadm_id":hadm,"h":hh,"i":np.arange(len(y))})
R["window"]={}
for s in ["0.5","1.0","1.5"]:
    w=pd.read_csv(f"winsens_{s}.csv.gz").merge(key,on=["hadm_id","h"]); yy=w.y.values; ii=w.i.values
    R["window"][s]=dict(n_landmarks_total=int(len(pd.read_csv(f"winsens_{s}.csv.gz"))),n_shared=int(len(w)),pos=int(yy.sum()),event_patients=int(w[w.y==1].hadm_id.nunique()),AUROC=float(AUC(yy,pA[ii])),AP=float(AP(yy,pA[ii])))
# ---- training-derived thresholds and patient-level burden (rep 0, each landmark predicted once)
z0=A["ZA"][0]; thr=A["THR"]; al=z0>=thr; ev_h=np.unique(hadm[y==1])
def burden(alert):
    out=dict(alert_rate=float(alert.mean()),event_sens=float(np.mean([(alert&(hadm==q)&(y==1)).any() for q in ev_h])),PPV=float(y[alert].mean()))
    out["patients_ever_alerted_pct"]=float(np.mean([alert[hadm==q].any() for q in np.unique(hadm)])*100)
    out["alerts_per_100_patient_days"]=float(alert.sum()/(len(y)/4)*100)
    return out
R["train_threshold"]=burden(al)
# 24-h silencing: after an alert, further alerts for the same admission are suppressed for 24 h
order=np.lexsort((hh,hadm)); sil=np.zeros(len(y),bool); last={}
for i in order:
    if al[i] and (hadm[i] not in last or hh[i]-last[hadm[i]]>=24): sil[i]=True; last[hadm[i]]=hh[i]
R["train_threshold_silenced"]=burden(sil)
# ---- importance and trajectories from FINAL model decomposition
G=[k for k in A.files if k.startswith("C_") and k!="C_intercept"]
name=lambda k:k[2:].replace("_x_","×")
imp={name(k):dict(std=float(np.mean([np.std(A[k][r]) for r in range(3)])),std_min=float(min(np.std(A[k][r]) for r in range(3))),std_max=float(max(np.std(A[k][r]) for r in range(3))),n_axes=len(name(k).split("×"))) for k in G}
rank=[sorted(G,key=lambda k:-np.std(A[k][r]))[:10] for r in range(3)]
R["importance"]=imp; R["top10_per_rep"]=[[name(k) for k in rr] for rr in rank]
# trajectories: primary event patients only, landmarks within 72 h before their (first) primary event
evpat=set(np.unique(hadm[y==1])); evp=np.array([h_ in evpat for h_ in hadm])&np.isfinite(lead)&(lead>0)&(lead<=72)
icu=np.isfinite(L_("heart_rate.last")); day=np.minimum(hh//24,7); non=~np.array([h_ in evpat for h_ in hadm])
traj={}; bins=np.arange(0,73,6)
for k in G:
    c=A[k].mean(0); ref=np.zeros(len(y))
    for a_ in [True,False]:
        for d_ in range(8):
            m=non&(icu==a_)&(day==d_); ref[(icu==a_)&(day==d_)]=c[m].mean() if m.sum()>0 else c[non].mean()
    rows=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        mk=evp&(lead>lo)&(lead<=hi)
        if mk.sum()>=10: v=c[mk]-ref[mk]; rows.append([float(lo),float(hi),float(v.mean()),float(v.std()/np.sqrt(len(v))),int(mk.sum()),int(len(np.unique(pat[mk])))])
    traj[name(k)]=rows
R["trajectory"]=traj; R["traj_patients"]=int(len(np.unique(pat[evp]))); R["traj_landmarks"]=int(evp.sum())
json.dump(R,open("REV1_RESULTS.json","w"),indent=1,ensure_ascii=False)
np.savez_compressed("rev1_final_oof.npz",AXIS=pA,FULL=pF)
for k in ["final","WITHIN","delay0","delay60","window","train_threshold","train_threshold_silenced","per_rep","per_rep_AP","recon_max_err","traj_patients","traj_landmarks","top10_per_rep"]:
    print(k,json.dumps(R[k])[:900])
print({k:(round(v["dAUROC"],3),[round(x,3) for x in v["dAUROC_ci"]]) for k,v in R["cmp"].items()})
print(sorted([(k,round(v["std"],3),round(v["std_min"],3),round(v["std_max"],3)) for k,v in imp.items()],key=lambda t:-t[1])[:16])
