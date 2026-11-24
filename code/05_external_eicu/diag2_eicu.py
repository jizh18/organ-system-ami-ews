"""Addendum 5b: D4 and D5 (procedure-aware endpoint)."""
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
E="/mnt/user-data/uploads/Eicu/physionet.org/files/eicu-crd/2.0/"; W="/mnt/user-data/uploads/Eicu/work/"
lm=pd.read_csv("eicu_landmarks.csv.gz"); sup=pd.read_csv("eicu_supports.csv.gz"); S=np.load("eicu_scores.npz")
p=pd.read_csv(E+"patient.csv.gz"); p=p[p.patienthealthsystemstayid.isin(lm.hadm)].copy()
p["uin"]=-p.hospitaladmitoffset; p["uout"]=p.unitdischargeoffset-p.hospitaladmitoffset; p["hdis"]=p.hospitaldischargeoffset-p.hospitaladmitoffset
SH=dict(zip(p.patientunitstayid,p.uin)); HS=dict(zip(p.patientunitstayid,p.patienthealthsystemstayid))
tr=pd.read_csv(W+"treatment.f.csv.gz"); tr=tr[tr.patientunitstayid.isin(SH)]; tr["t"]=tr.treatmentoffset+tr.patientunitstayid.map(SH); tr["hadm"]=tr.patientunitstayid.map(HS)
ts=tr.treatmentstring.str.lower(); PR=tr[ts.str.contains(r"cardiac angiography|\|cabg|cardiac surgery")][["hadm","t"]]
GP=dict(tuple(PR.groupby("hadm")))
def after_proc(h,t):
    g=GP.get(h); return g is not None and bool(((t-g.t>=0)&(t-g.t<=720)).any())
u=sup[sup.cls=="unplanned"].copy(); u["after_proc"]=[after_proc(h,t) for h,t in zip(u.hadm,u.start)]
first=u.sort_values("start").groupby("hadm").head(1)
pos_h=set(lm[lm.y==1].hadm); fe=first[first.hadm.isin(pos_h)]
R={"D4_share_first_unplanned_after_procedure":float(fe.after_proc.mean()),"D4_by_kind":fe.groupby("kind").after_proc.mean().to_dict(),"D4_n_first":int(len(fe))}
sup2=sup.copy(); sup2.loc[(sup2.cls=="unplanned")&sup2.index.isin(u[u.after_proc].index),"cls"]="peri_procedure"
A=p.groupby("patienthealthsystemstayid").agg(hdis=("hdis","max"),st=("hospitaldischargestatus",lambda x:(x=="Expired").any()))
death=dict(zip(A.index,np.where(A.st,A.hdis,np.nan)))
G=dict(tuple(sup2.groupby("hadm")))
y2=[];keep=[]
for r in lm.itertuples():
    s_=G.get(r.hadm); L=r.L; d=death[r.hadm]
    if s_ is None: s_=sup2.iloc[:0]
    ev=s_[s_.cls=="unplanned"]; f_=ev.start.min() if len(ev) else np.nan
    if np.isfinite(f_) and f_<=L: keep.append(False); y2.append(0); continue
    if ((s_.start<=L)&(s_.end>L)).any(): keep.append(False); y2.append(0); continue
    inw=lambda t:np.isfinite(t) and L<t<=L+1440
    keep.append(True); y2.append(int(inw(d) or inw(f_)))
keep=np.array(keep); y2=np.array(y2)
R["D5_n"]=int(keep.sum()); R["D5_pos"]=int(y2[keep].sum()); R["D5_episodes"]=int(lm[keep&(y2==1)].hadm.nunique())
R["D5"]={k:dict(AUROC=float(AUC(y2[keep],S[k][keep])),AP=float(AP(y2[keep],S[k][keep]))) for k in ["AXIS","FULL","LR","AXIS_dev","GRACE","NEWS2"]}
json.dump(R,open("EICU_DIAG2.json","w"),indent=1); print(json.dumps(R,indent=1))
