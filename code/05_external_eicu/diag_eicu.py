"""Addendum 5a post hoc diagnostics D1-D3."""
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
E="/mnt/user-data/uploads/Eicu/physionet.org/files/eicu-crd/2.0/"; W="/mnt/user-data/uploads/Eicu/work/"
lm=pd.read_csv("eicu_landmarks.csv.gz"); sup=pd.read_csv("eicu_supports.csv.gz"); S=np.load("eicu_scores.npz"); z=np.load("eicu_features.npz")
assert (z["hadm"]==lm.hadm.values).all()
p=pd.read_csv(E+"patient.csv.gz",usecols=["patientunitstayid","patienthealthsystemstayid","hospitaladmitoffset"]); p=p[p.patienthealthsystemstayid.isin(lm.hadm)]
SH=dict(zip(p.patientunitstayid,-p.hospitaladmitoffset)); HS=dict(zip(p.patientunitstayid,p.patienthealthsystemstayid))
tr=pd.read_csv(W+"treatment.f.csv.gz"); tr=tr[tr.patientunitstayid.isin(SH)]; tr["t"]=tr.treatmentoffset+tr.patientunitstayid.map(SH); tr["hadm"]=tr.patientunitstayid.map(HS); ts=tr.treatmentstring.str.lower()
TV=tr[ts.str.contains("vasopressors|inotropic agent")]; TM=tr[ts.str.contains("mechanical ventilation")&~ts.str.contains("non-invasive")]
rc=pd.read_csv(W+"respiratoryCharting.f.csv.gz",low_memory=False); rc=rc[rc.patientunitstayid.isin(SH)]
VL={"Vent Rate","Tidal Volume (set)","TV/kg IBW","Set Vt (Servo,LTV)","Adult Con Setting Set Vt","Adult Con Setting Set RR","Pressure Control","Exhaled TV (machine)"}
rc=rc[rc.respchartvaluelabel.isin(VL)]; RA=pd.DataFrame({"hadm":rc.patientunitstayid.map(HS),"t":rc.respchartoffset+rc.patientunitstayid.map(SH)})
rca=pd.read_csv(W+"respiratoryCare.f.csv.gz",low_memory=False); rca=rca[rca.patientunitstayid.isin(SH)&rca.airwaytype.isin(["Oral ETT","Nasal ETT","Tracheostomy"])&(rca.ventstartoffset!=0)]
RA=pd.concat([RA,pd.DataFrame({"hadm":rca.patientunitstayid.map(HS),"t":rca.ventstartoffset+rca.patientunitstayid.map(SH)})])
def near(df,h,t,w=720): d=df[df.hadm==h]; return bool(((d.t-t).abs()<=w).any())
u=sup[sup.cls=="unplanned"].copy()
def corr(r):
    if r.kind=="mcs": return True
    if r.kind in ("vasopressor","inotrope"): return near(TV,r.hadm,r.start)
    return near(RA,r.hadm,r.start) and near(TM,r.hadm,r.start)
u["corr"]=u.apply(corr,axis=1)
R={"D3_corroborated_share_of_unplanned_starts":u.groupby("kind")["corr"].mean().to_dict()}
G=dict(tuple(u[u["corr"]].groupby("hadm")))
yc=[]
for r in lm.itertuples():
    g=G.get(r.hadm); ev=r.y_death==1
    if not ev and g is not None: ev=bool(((g.start>r.L)&(g.start<=r.L+1440)).any())
    yc.append(int(ev))
yc=np.array(yc); R["D3_n_pos"]=int(yc.sum()); R["D3_episodes"]=int(lm[yc==1].hadm.nunique())
mods=["AXIS","FULL","LR","AXIS_dev","GRACE","NEWS2"]
R["D3"]={k:dict(AUROC=float(AUC(yc,S[k])),AP=float(AP(yc,S[k]))) for k in mods}
R["D1"]={o:{k:dict(AUROC=float(AUC(z[o],S[k])),AP=float(AP(z[o],S[k]))) for k in mods} for o in ["y_circ","y_resp","y_death"]}
# D2: NEWS2 in MIMIC-IV ICU assessments (same NEWS2 code)
import os; os.chdir("../full"); exec(open("full_common.py").read().split("# ---- clinical scores")[0]); exec("# ---- clinical scores"+open("full_common.py").read().split("# ---- clinical scores")[1].split("def metrics")[0])
for nm,mm in [("MIMIC_ICU_all",ICU),("MIMIC_ICU_heldout",ICU&np.isin(np.arange(len(y)),LATE)),("MIMIC_ICU_dev",ICU&np.isin(np.arange(len(y)),DEV))]:
    R.setdefault("D2",{})[nm]=dict(n=int(mm.sum()),prev=float(y[mm].mean()),NEWS2_AUROC=float(AUC(y[mm],SC["NEWS2"][mm])),GRACE_AUROC=float(AUC(y[mm],SC["GRACE"][mm])))
os.chdir("../eicu"); R["D2"]["eICU"]=dict(prev=float(lm.y.mean()),NEWS2_AUROC=float(AUC(lm.y,S["NEWS2"])),GRACE_AUROC=float(AUC(lm.y,S["GRACE"])))
json.dump(R,open("EICU_DIAG.json","w"),indent=1); print(json.dumps(R,indent=1))
