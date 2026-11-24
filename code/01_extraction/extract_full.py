"""FULL01 extraction (run on the local machine with the Windows Conda Python).
Rebuilds the AMI cohort from raw MIMIC-IV 3.1 (F01-F06 as in outputs/audits/m0_raw_cohort_ecg/cohort_flow.csv),
assigns period by estimated real admission year (development 2008-2016, later 2017-2022, boundary excluded),
and extracts the same files as outputs/experiments/ep2_20261009 for ALL eligible patients.
Output: outputs/experiments/full_20261009/ . Re-running skips files that already exist.
"""
import pandas as pd, numpy as np, os, time, gzip
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
M=os.path.join(ROOT,"data","raw","mimic-iv-3.1","mimic-iv-3.1","mimic-iv-3.1")
E=os.path.join(ROOT,"data","raw","mimic-iv-ecg-1.0","mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0","mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0")
OUT=os.path.join(ROOT,"outputs","experiments","full_v2_20261009"); os.makedirs(OUT,exist_ok=True)
OLD=os.path.join(ROOT,"outputs","experiments","ep2_20261009")
t0=time.time(); LOG=open(os.path.join(OUT,"extract_log.txt"),"a")
def log(*a):
    s=" ".join(str(x) for x in a)+f"  [{round(time.time()-t0)}s]"; print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
P=lambda *x:os.path.join(M,*x)
# ---------------- cohort
if not os.path.exists(os.path.join(OUT,"cohort.csv")):
    adm=pd.read_csv(P("hosp","admissions.csv.gz"),parse_dates=["admittime","dischtime","deathtime"])
    pts=pd.read_csv(P("hosp","patients.csv.gz"))
    dx=pd.read_csv(P("hosp","diagnoses_icd.csv.gz"),dtype={"icd_code":str})
    c=dx.icd_code.str.strip()
    anyami=((dx.icd_version==9)&c.str.startswith("410"))|((dx.icd_version==10)&(c.str.startswith("I21")|c.str.startswith("I22")))
    strict=((dx.icd_version==9)&c.str.startswith("410")&(c.str.len()>=5)&c.str[-1].isin(["0","1"]))|((dx.icd_version==10)&(c.str.match(r"^I21[0-4]")|(c=="I219")))
    f1=set(dx.hadm_id[anyami]); f2=set(dx.hadm_id[anyami&(dx.seq_num==1)]); f3=set(dx.hadm_id[strict&(dx.seq_num==1)])
    a=adm.merge(pts,on="subject_id"); a["age"]=a.anchor_age+(a.admittime.dt.year-a.anchor_year)
    a=a[a.hadm_id.isin(f3)]; f4=a[(a.age>=18)&a.admittime.notna()&a.dischtime.notna()&(a.dischtime>=a.admittime)]
    f5=f4.sort_values(["subject_id","admittime","hadm_id"]).groupby("subject_id").head(1)
    mm=pd.read_csv(os.path.join(E,"machine_measurements.csv"),usecols=["subject_id","ecg_time"],parse_dates=["ecg_time"])
    mm=mm.merge(f5[["subject_id","hadm_id","admittime"]],on="subject_id")
    ok=mm[(mm.ecg_time>=mm.admittime-pd.Timedelta(hours=6))&(mm.ecg_time<=mm.admittime+pd.Timedelta(hours=6))].hadm_id.unique()
    f6=f5.copy(); f6["ecg_6h"]=f6.hadm_id.isin(ok)   # v2: no ECG requirement (matches the previous cohort, which was not ECG-restricted)
    s=f6.anchor_year_group.str.slice(0,4).astype(int); e=f6.anchor_year_group.str.slice(7,11).astype(int); off=f6.admittime.dt.year-f6.anchor_year
    f6["year_start"]=s+off; f6["year_end"]=e+off
    f6["period"]=np.where((f6.year_start>=2008)&(f6.year_end<=2016),"development",np.where((f6.year_start>=2017)&(f6.year_end<=2022),"later","boundary"))
    old=set(int(l.strip().strip(",")) for l in open(os.path.join(OLD,"ami_hadm_pat.txt")) if l.strip())
    f6["in_previous_1688"]=f6.hadm_id.isin(old)
    f6[["subject_id","hadm_id","admittime","dischtime","anchor_year_group","year_start","year_end","period","in_previous_1688","ecg_6h"]].to_csv(os.path.join(OUT,"cohort.csv"),index=False)
    flow=[("F01 any AMI code",len(f1)),("F02 principal AMI",len(f2)),("F03 principal type1/unspecified",len(f3)),("F04 adult valid times",len(f4)),("F05 first per patient",len(f5)),("F06(v2) all first admissions; with ECG within 6h",int(f6.ecg_6h.sum()))]
    for k,v in f6.period.value_counts().items(): flow.append((f"period={k}",int(v)))
    flow.append(("previous 1688 found in F06",int(f6.in_previous_1688.sum())))
    flow.append(("previous 1688 in development",int((f6.in_previous_1688&(f6.period=="development")).sum())))
    pd.DataFrame(flow,columns=["step","n"]).to_csv(os.path.join(OUT,"cohort_flow.csv"),index=False); log(flow)
co=pd.read_csv(os.path.join(OUT,"cohort.csv")); co=co[co.period.isin(["development","later"])]
SUB=set(co.subject_id); HADM=set(co.hadm_id); log("cohort",len(co),co.period.value_counts().to_dict())
with open(os.path.join(OUT,"ami_hadm_pat.txt"),"w") as f:
    for h in sorted(HADM): f.write(f",{h},\n")
def small(src,dst,key="subject_id",cols=None,filt=None):
    dst=os.path.join(OUT,dst)
    if os.path.exists(dst): return
    d=pd.read_csv(src,low_memory=False)
    d=d[d[key].isin(SUB if key=="subject_id" else HADM)]
    if filt is not None: d=filt(d)
    if cols: d=d[cols]
    d.to_csv(dst,index=False,compression="gzip"); log(dst,len(d))
small(P("hosp","admissions.csv.gz"),"admissions.csv.gz")
small(P("hosp","patients.csv.gz"),"patients.csv.gz")
small(P("hosp","diagnoses_icd.csv.gz"),"diagnoses.csv.gz")
small(P("hosp","procedures_icd.csv.gz"),"procedures_icd.csv.gz")
small(P("hosp","transfers.csv.gz"),"transfers.csv.gz")
small(P("icu","icustays.csv.gz"),"icustays.csv.gz")
small(os.path.join(E,"machine_measurements.csv"),"ecg_mm.csv.gz")
VASO={221906,221289,229617,222315,221749,229630,229631,229632,221662,229709,229764,221653,221986}
small(P("icu","inputevents.csv.gz"),"inputs.csv.gz",key="hadm_id",cols=["hadm_id","stay_id","starttime","endtime","itemid","amount","rate","ordercategorydescription","statusdescription"],filt=lambda d:d[d.itemid.isin(VASO)])
small(P("icu","procedureevents.csv.gz"),"procs.csv.gz",key="hadm_id",cols=["hadm_id","stay_id","starttime","endtime","itemid","value","statusdescription"],filt=lambda d:d[d.itemid.isin({225792,224272,228169,229529,229530})])
URI={226559,226560,226561,226563,226564,226565,226567,226584,227488,227489}
small(P("icu","outputevents.csv.gz"),"urine.csv.gz",key="hadm_id",cols=["hadm_id","stay_id","charttime","storetime","itemid","value"],filt=lambda d:d[d.itemid.isin(URI)])
import shutil; shutil.copy(os.path.join(OLD,"lab_itemmap.csv"),os.path.join(OUT,"lab_itemmap.csv"))
CHI=set(int(l.strip().strip(",")) for l in open(os.path.join(OLD,"chart_item_pat.txt")) if l.strip().strip(","))
def big(src,stem,itemfilter=None):
    a,b=os.path.join(OUT,stem+"_a.csv.gz"),os.path.join(OUT,stem+"_b.csv.gz")
    if os.path.exists(b): return
    fa,fb=gzip.open(a+".tmp","wt"),gzip.open(b+".tmp","wt"); n=0
    for ch in pd.read_csv(src,chunksize=1_000_000,low_memory=False,dtype=str):
        ch=ch[ch.subject_id.astype(int).isin(SUB)]
        if itemfilter is not None: ch=ch[ch.itemid.astype(int).isin(itemfilter)]
        odd=ch.subject_id.astype(int)%2==1
        ch[odd].to_csv(fa,header=False,index=False); ch[~odd].to_csv(fb,header=False,index=False); n+=len(ch)
        log(stem,"rows kept so far",n)
    fa.close(); fb.close(); os.replace(a+".tmp",a); os.replace(b+".tmp",b); log(stem,"done",n)
import traceback
try:
    big(P("hosp","labevents.csv.gz"),"lab")
    big(P("icu","chartevents.csv.gz"),"chart",CHI)
except Exception:
    log("ERROR",traceback.format_exc()); raise
log("ALL DONE")
