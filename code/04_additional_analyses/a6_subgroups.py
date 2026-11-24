"""Amendment 6: subgroup performance (post hoc, descriptive)."""
import json,numpy as np,pandas as pd
exec(open("full_common.py").read())
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
H=np.load("heldout_scores.npz")
sig=lambda z:1/(1+np.exp(-z))
adm=pd.read_csv(U+"admissions.csv.gz").set_index("hadm_id")
race=adm.loc[hadm,"race"].fillna("").str.upper().values
def rgrp(r): return "White" if "WHITE" in r else ("Black" if "BLACK" in r else "Other or unknown")
RACE=np.array([rgrp(r) for r in race])
age=L_("age"); fem=L_("female")
def agegrp(a): return np.where(a<65,"<65",np.where(a<80,"65–79","≥80"))
rng0=2026
def boot_auc(yy,ss,pid,B=1000,seed=rng0):
    up=np.unique(pid); idx={p:np.where(pid==p)[0] for p in up}; r=np.random.default_rng(seed); out=[]
    for _ in range(B):
        s=r.choice(up,len(up)); ii=np.concatenate([idx[p] for p in s]); 
        if yy[ii].min()==yy[ii].max(): continue
        out.append(AUC(yy[ii],ss[ii]))
    return np.percentile(out,[2.5,97.5]).tolist()
def diff_boot(yy,ss,pid,g,a,b,B=1000,seed=rng0):
    up=np.unique(pid); idx={p:np.where(pid==p)[0] for p in up}; r=np.random.default_rng(seed); out=[]
    pt=0
    for _ in range(B):
        s=r.choice(up,len(up)); ii=np.concatenate([idx[p] for p in s]); ma=g[ii]==a; mb=g[ii]==b
        ya,yb=yy[ii][ma],yy[ii][mb]
        if ya.min()==ya.max() or yb.min()==yb.max(): continue
        out.append(AUC(ya,ss[ii][ma])-AUC(yb,ss[ii][mb]))
    pa=g==a; pb=g==b
    return float(AUC(yy[pa],ss[pa])-AUC(yy[pb],ss[pb])),np.percentile(out,[2.5,97.5]).tolist()
def table(yy,pid,scores,groups,prob_key):
    res={}
    for gname,g in groups.items():
        res[gname]={}
        for lev in sorted(set(g)):
            m=g==lev; d=dict(n=int(m.sum()),pos=int(yy[m].sum()),event_patients=int(len(np.unique(pid[m][yy[m]==1]))),patients=int(len(np.unique(pid[m]))))
            d["imprecise"]=d["event_patients"]<20
            for k,s in scores.items():
                if yy[m].min()==yy[m].max(): continue
                d[k]=dict(AUROC=float(AUC(yy[m],s[m])),AP=float(AP(yy[m],s[m])))
                if k==prob_key: d[k]["AUROC_ci"]=boot_auc(yy[m],s[m],pid[m]); d[k]["mean_pred"]=float(sig(s[m]).mean()); d[k]["observed"]=float(yy[m].mean())
            res[gname][lev]=d
    return res
OUT={}
# ---- MIMIC held-out
m=LATE; yy=y[m]; pid=hadm[m]
sc={"Organ-system":H["AXIS"][m],"Unconstrained":H["FULL"][m],"Logistic regression":H["LR"][m],"GRACE":SC["GRACE"][m],"NEWS2":SC["NEWS2"][m]}
G={"Sex":np.where(fem[m]>0,"Female","Male"),"Age":agegrp(age[m]),"Race":RACE[m]}
OUT["mimic_heldout"]=table(yy,pid,sc,G,"Organ-system")
OUT["mimic_diffs"]={"Female − Male":diff_boot(yy,sc["Organ-system"],pid,G["Sex"],"Female","Male"),"Black − White":diff_boot(yy,sc["Organ-system"],pid,G["Race"],"Black","White"),"≥80 − <65":diff_boot(yy,sc["Organ-system"],pid,G["Age"],"≥80","<65")}
# ---- eICU
E=np.load("/mnt/user-data/outputs/eicu/eicu_scores.npz"); el=pd.read_csv("/mnt/user-data/outputs/eicu/eicu_landmarks.csv.gz")
P=pd.read_csv("/mnt/user-data/uploads/Eicu/physionet.org/files/eicu-crd/2.0/patient.csv.gz",usecols=["patienthealthsystemstayid","gender","age","ethnicity"])
P=P.groupby("patienthealthsystemstayid").first()
ag=P["age"].astype(str).str.replace("> 89","90"); P["age_n"]=pd.to_numeric(ag,errors="coerce")
pe=P.loc[el.hadm]; ey=el.y.values; epid=el.hadm.values
eth=pe.ethnicity.fillna("").values; EG=np.where(eth=="Caucasian","Caucasian",np.where(eth=="African American","African American","Other or unknown"))
EGs={"Sex":np.where(pe.gender.values=="Female","Female",np.where(pe.gender.values=="Male","Male","Unknown")),"Age":agegrp(pe.age_n.values),"Race":EG}
esc={"Organ-system":E["AXIS"],"Unconstrained":E["FULL"],"Logistic regression":E["LR"],"GRACE":E["GRACE"],"NEWS2":E["NEWS2"]}
OUT["eicu"]=table(ey,epid,esc,EGs,"Organ-system")
OUT["eicu_diffs"]={"Female − Male":diff_boot(ey,esc["Organ-system"],epid,EGs["Sex"],"Female","Male"),"African American − Caucasian":diff_boot(ey,esc["Organ-system"],epid,EGs["Race"],"African American","Caucasian"),"≥80 − <65":diff_boot(ey,esc["Organ-system"],epid,EGs["Age"],"≥80","<65")}
json.dump(OUT,open("A6_SUBGROUPS.json","w"),indent=1,ensure_ascii=False)
for k in ["mimic_heldout","eicu"]:
    print("==",k)
    for g,d in OUT[k].items():
        for lev,v in d.items():
            o=v.get("Organ-system",{}); print(f"{g:5s} {lev:18s} pts {v['patients']:5d} evpts {v['event_patients']:4d} AUROC {o.get('AUROC',float('nan')):.3f} {o.get('AUROC_ci')} pred {o.get('mean_pred',0):.4f} obs {o.get('observed',0):.4f} | GRACE {v.get('GRACE',{}).get('AUROC',float('nan')):.3f} NEWS2 {v.get('NEWS2',{}).get('AUROC',float('nan')):.3f} FULL {v.get('Unconstrained',{}).get('AUROC',float('nan')):.3f}")
print(OUT["mimic_diffs"]); print(OUT["eicu_diffs"])
