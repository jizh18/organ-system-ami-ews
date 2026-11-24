import pandas as pd, numpy as np, json, sys
U="/mnt/user-data/uploads/KG-v1/outputs/experiments/full_v2_20261009/"
co=pd.read_csv(U+"cohort.csv",parse_dates=["admittime","dischtime"]); co=co[co.period.isin(["development","later"])]
L=pd.read_csv("full_landmarks.csv.gz"); L=L[L.definition=="v2"]
inrisk=L.groupby("hadm_id").size(); ev=L.groupby("hadm_id").y.max()
co=co[co.hadm_id.isin(inrisk.index)].copy(); co["nlm"]=co.hadm_id.map(inrisk); co["ev"]=co.hadm_id.map(ev)
p=pd.read_csv(U+"patients.csv.gz").set_index("subject_id"); a=pd.read_csv(U+"admissions.csv.gz").set_index("hadm_id")
co["age"]=p.loc[co.subject_id,"anchor_age"].values+(co.admittime.dt.year.values-p.loc[co.subject_id,"anchor_year"].values)
co["female"]=p.loc[co.subject_id,"gender"].values=="F"; co["los"]=(co.dischtime-co.admittime).dt.total_seconds()/86400
co["death"]=a.loc[co.hadm_id,"hospital_expire_flag"].values==1
icu=pd.read_csv(U+"icustays.csv.gz"); co["icu"]=co.hadm_id.isin(icu.hadm_id)
px=pd.read_csv(U+"procedures_icd.csv.gz",dtype={"icd_code":str}); c=px.icd_code.str.replace(".","",regex=False)
co["pci"]=co.hadm_id.isin(px[((px.icd_version==10)&c.str.match(r"^027"))|((px.icd_version==9)&c.str.match(r"^0066|^3606|^3607"))].hadm_id)
co["cabg"]=co.hadm_id.isin(px[((px.icd_version==10)&c.str.match(r"^021|^02R|^02Q|^02U"))|((px.icd_version==9)&c.str.match(r"^361|^35[0-2]"))].hadm_id)
d=pd.read_csv(U+"diagnoses.csv.gz",dtype={"icd_code":str}); d=d[(d.seq_num==1)&d.hadm_id.isin(co.hadm_id)].drop_duplicates("hadm_id").set_index("hadm_id")
cc=d.icd_code; st=(((d.icd_version==10)&cc.str.match(r"^I21[0-3]"))|((d.icd_version==9)&cc.str.match(r"^410[0-689]")))
co["stemi"]=co.hadm_id.map(st).fillna(False).astype(bool)
f=lambda x:f"{int(x.sum()):,} ({x.mean()*100:.1f})"; q=lambda x:f"{x.median():.1f} ({x.quantile(.25):.1f}–{x.quantile(.75):.1f})"
show_outcome=len(sys.argv)>1
rows={}
for nm,g in [("Development",co[co.period=="development"]),("Later period",co[co.period=="later"])]:
    r={"Patients":f"{len(g):,}","Age":q(g.age),"Female":f(g.female),"STEMI code":f(g.stemi),"ECG within 6 h":f(g.ecg_6h),"PCI":f(g.pci),"Cardiac surgery":f(g.cabg),"ICU stay":f(g.icu),"LOS":q(g.los),"Assessments per patient":q(g.nlm),"Total assessments":f"{int(g.nlm.sum()):,}"}
    if show_outcome or nm=="Development":
        r["In-hospital death"]=f(g.death); r["Patients with event"]=f(g.ev==1)
    rows[nm]=r
print(json.dumps(rows,indent=1,ensure_ascii=False)); json.dump(rows,open("table1.json","w"),indent=1,ensure_ascii=False)
