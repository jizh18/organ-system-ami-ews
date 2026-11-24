"""Held-out (2017-2022) analysis. Run only after FULL_DEV_V2.json is written and the freeze note exists."""
import os
assert os.path.exists("FREEZE_NOTE.md"), "freeze note missing"
exec(open("full_common.py").read())
H=np.load("heldout_scores.npz"); m=np.zeros(len(y),bool); m[LATE]=True
P={a:sig(H[a]) for a in ["AXIS","WITHIN","FULL","ADD","LR"]}
for a in ["AXIS","WITHIN","FULL","ADD"]: P["LR+"+a]=mix(P["LR"],P[a])
for k,v in SC.items(): P[k]=v
R={"n":int(m.sum()),"pos":int(y[m].sum()),"events":int(len(np.unique(hadm[m][y[m]==1]))),"patients":int(len(np.unique(hadm[m])))}
R["metrics"]={a:metrics(P[a],m,prob=a not in SC) for a in P}
C={"H1 AXIS-FULL":("FULL","AXIS"),"H2 AXIS-LR":("LR","AXIS"),"H3 AXIS-WITHIN":("WITHIN","AXIS"),"WITHIN-ADD":("ADD","WITHIN"),"AXIS-ADD":("ADD","AXIS"),
   "H5 HYBRID(LR+AXIS)-LR":("LR","LR+AXIS"),"H6 LR+AXIS - LR+ADD":("LR+ADD","LR+AXIS"),"H7 LR+AXIS - LR+FULL":("LR+FULL","LR+AXIS"),"LR+WITHIN - LR+ADD":("LR+ADD","LR+WITHIN"),
   "HYBRID-AXIS":("AXIS","LR+AXIS"),"H4 AXIS-GRACE":("GRACE","AXIS"),"H4 AXIS-NEWS2":("NEWS2","AXIS"),"HYBRID-GRACE":("GRACE","LR+AXIS"),"HYBRID-NEWS2":("NEWS2","LR+AXIS"),"FULL-LR":("LR","FULL")}
R["cmp"]={k:boot(P[a],P[b],m) for k,(a,b) in C.items()}
px=pd.read_csv(U+"procedures_icd.csv.gz",dtype={"icd_code":str}); c=px.icd_code.str.replace(".","",regex=False)
surg_h=set(px[((px.icd_version==10)&(c.str.match(r"^021|^02R|^02Q|^02U")))|((px.icd_version==9)&(c.str.match(r"^361|^35[0-2]")))].hadm_id); surg=np.array([h_ in surg_h for h_ in hadm])
R["sub"]={}
for nm,mk in [("ICU",m&ICU),("ward",m&~ICU),("nonsurgical",m&~surg),("ecg6h",m&ecg6)]:
    R["sub"][nm]={"n":int(mk.sum()),"pos":int(y[mk].sum()),"AUROC":{a:float(AUC(y[mk],P[a][mk])) for a in ["AXIS","LR","LR+AXIS","FULL","GRACE","NEWS2"]},"AP":{a:float(AP(y[mk],P[a][mk])) for a in ["AXIS","LR","LR+AXIS","GRACE","NEWS2"]}}
R["sub_cmp"]={"ICU AXIS-GRACE":boot(P["GRACE"],P["AXIS"],m&ICU),"ICU AXIS-NEWS2":boot(P["NEWS2"],P["AXIS"],m&ICU),"ICU HYBRID-GRACE":boot(P["GRACE"],P["LR+AXIS"],m&ICU),"ICU HYBRID-NEWS2":boot(P["NEWS2"],P["LR+AXIS"],m&ICU),
              "nonsurg AXIS-GRACE":boot(P["GRACE"],P["AXIS"],m&~surg),"nonsurg HYBRID-LR":boot(P["LR"],P["LR+AXIS"],m&~surg)}
# calibration
from scipy.optimize import brentq
R["cal"]={}
for a in ["AXIS","LR","LR+AXIS","FULL"]:
    lg=lgt(P[a][m]); yy=y[m]
    R["cal"][a]={"slope":float(LogisticRegression(C=1e6).fit(lg[:,None],yy).coef_[0,0]),"intercept":float(brentq(lambda d:sig(lg+d).mean()-yy.mean(),-6,6)),"mean_pred":float(P[a][m].mean()),"observed":float(yy.mean()),"Brier":float(brier_score_loss(yy,P[a][m]))}
# alerting with development-fixed thresholds
dev=[np.load(f"cv_rep{r}.npz") for r in range(3) if os.path.exists(f"cv_rep{r}.npz")]
devP={a:np.mean([sig(d[a]) for d in dev],0) for a in ["AXIS","LR"]}; devP["LR+AXIS"]=mix(devP["LR"],devP["AXIS"])
thr={a:float(np.quantile(devP[a][DEV],0.9)) for a in devP}
ev_h=np.unique(hadm[m][y[m]==1]); order=np.lexsort((hh,hadm))
def burden(alert):
    out=dict(alert_rate=float(alert[m].mean()),event_sens=float(np.mean([(alert&(hadm==q)&(y==1)).any() for q in ev_h])),PPV=float(y[m&alert].mean()) if (m&alert).any() else None)
    out["patients_ever_alerted_pct"]=float(np.mean([alert[hadm==q].any() for q in np.unique(hadm[m])])*100); out["alerts_per_100_patient_days"]=float(alert[m].sum()/(m.sum()/4)*100); return out
R["alerting"]={}
for a in thr:
    al=(P[a]>=thr[a])&m; sil=np.zeros(len(y),bool); last={}
    for i in order:
        if al[i] and (hadm[i] not in last or hh[i]-last[hadm[i]]>=24): sil[i]=True; last[hadm[i]]=hh[i]
    R["alerting"][a]={"threshold":thr[a],"fixed":burden(al),"silenced24h":burden(sil)}
# interpretation of held-out AXIS predictions
G=[k for k in H.files if k.startswith("C_") and k!="C_intercept"]; name=lambda k:k[2:].replace("_x_","×")
li=np.where(m)[0]
imp={name(k):dict(std=float(np.std(H[k])),n_axes=len(name(k).split("×"))) for k in G}
R["importance"]=dict(sorted(imp.items(),key=lambda t:-t[1]["std"]))
evpat=set(ev_h); evp=np.array([h_ in evpat for h_ in hadm[li]])&np.isfinite(lead[li])&(lead[li]>0)&(lead[li]<=72)
non=~np.array([h_ in evpat for h_ in hadm[li]]); icu_l=ICU[li]; day=np.minimum(hh[li]//24,7); traj={}
for k in G:
    cc=H[k]; ref=np.zeros(len(li))
    for a_ in [True,False]:
        for d_ in range(8):
            sel=non&(icu_l==a_)&(day==d_); ref[(icu_l==a_)&(day==d_)]=cc[sel].mean() if sel.sum()>0 else cc[non].mean()
    rows=[]
    for lo,hi in zip(range(0,72,6),range(6,73,6)):
        mk=evp&(lead[li]>lo)&(lead[li]<=hi)
        if mk.sum()>=10: v=cc[mk]-ref[mk]; rows.append([lo,hi,float(v.mean()),float(v.std()/np.sqrt(len(v))),int(mk.sum()),int(len(np.unique(pat[li][mk])))])
    traj[name(k)]=rows
R["trajectory"]=traj; R["traj_patients"]=int(len(np.unique(pat[li][evp]))); R["traj_assess"]=int(evp.sum())
S=H["LR_sys"]; R["hybrid_lr_system_std"]={AXL[i]:float(np.std(S[:,i])) for i in range(len(AXL))}
json.dump(R,open("FULL_HELDOUT.json","w"),indent=1)
print(R["n"],R["pos"],R["events"],R["patients"])
for a,v in R["metrics"].items(): print(a,round(v["AUROC"],3),round(v["AP"],3),round(v["s10"],2))
for k,v in R["cmp"].items(): print(k,"dAUROC",round(v["dAUROC"],3),[round(x,3) for x in v["dAUROC_ci"]],"dAP",round(v["dAP"],3),[round(x,3) for x in v["dAP_ci"]])
print(json.dumps(R["sub"])[:1500]); print({k:(round(v["dAUROC"],3),[round(x,3) for x in v["dAUROC_ci"]]) for k,v in R["sub_cmp"].items()})
print(json.dumps(R["cal"])); print(json.dumps(R["alerting"])[:1500]); print(list(R["importance"].items())[:10])
