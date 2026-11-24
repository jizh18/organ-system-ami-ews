"""Amendment 5: eICU cohort, endpoint (supports + planned rules), landmarks and predictors.
Port of endpoint_full.py (v2) and features_v2fix.py. Times are minutes from hospital admission."""
import pandas as pd, numpy as np, json, csv, re, sys, time
t0=time.time(); log=lambda *a:print(*a,round(time.time()-t0),flush=True)
E="/mnt/user-data/uploads/Eicu/physionet.org/files/eicu-crd/2.0/"; W="/mnt/user-data/uploads/Eicu/work/"
POA=int(sys.argv[1]) if len(sys.argv)>1 else 60      # present-on-arrival window (min); 0 = sensitivity without the rule
SUF="" if POA==60 else f"_poa{POA}"
# ---------------- E1 cohort
p=pd.read_csv(E+"patient.csv.gz")
dx=pd.read_csv(E+"diagnosis.csv.gz",usecols=["patientunitstayid","diagnosisstring","diagnosispriority"])
ami=dx.diagnosisstring.str.contains("acute myocardial infarction",case=False)
idx=set(p[p.apacheadmissiondx.isin(["Infarction, acute myocardial (MI)","MI admitted > 24 hrs after onset of ischemia"])].patientunitstayid)|set(dx[ami&(dx.diagnosispriority=="Primary")].patientunitstayid)
s=p[p.patientunitstayid.isin(idx)]
hs=s.groupby("patienthealthsystemstayid").agg(uniquepid=("uniquepid","first"),yr=("hospitaldischargeyear","first")).reset_index()
first=hs.sort_values(["uniquepid","yr","patienthealthsystemstayid"]).groupby("uniquepid").head(1)
FLOW=[("F1 AMI unit stays",len(idx)),("F2 first hospital stay per patient",len(first))]
u=p[p.patienthealthsystemstayid.isin(first.patienthealthsystemstayid)].copy()
u["age_n"]=pd.to_numeric(u.age.replace("> 89","90"),errors="coerce")
adult=u.groupby("patienthealthsystemstayid").age_n.max()>=18; u=u[u.patienthealthsystemstayid.isin(adult[adult].index)]
FLOW.append(("F3 adult",u.patienthealthsystemstayid.nunique()))
allp=p[["patientunitstayid","hospitalid"]]; Hn=allp.groupby("hospitalid").size()
docs={}
for t in ["infusionDrug","respiratoryCharting","treatment"]:
    ids=set(pd.read_csv(f"u_{t}.txt",header=None)[0]); docs[t]=allp[allp.patientunitstayid.isin(ids)].groupby("hospitalid").size().reindex(Hn.index,fill_value=0)/Hn
okh=set(Hn.index[(docs["infusionDrug"]>=.1)&(docs["respiratoryCharting"]>=.1)&(docs["treatment"]>=.1)])
hosp_of=u.groupby("patienthealthsystemstayid").hospitalid.first()
u=u[u.patienthealthsystemstayid.map(hosp_of).isin(okh)]
FLOW.append(("F4 hospitals meeting documentation rule",u.patienthealthsystemstayid.nunique())); FLOW.append(("   hospitals",u.hospitalid.nunique()))
u["shift"]=-u.hospitaladmitoffset                     # unit admit in hospital minutes
u["uin"]=u["shift"]; u["uout"]=u.unitdischargeoffset+u["shift"]; u["hdis"]=u.hospitaldischargeoffset+u["shift"]
SH=dict(zip(u.patientunitstayid,u["shift"])); HS=dict(zip(u.patientunitstayid,u.patienthealthsystemstayid))
A=u.groupby("patienthealthsystemstayid").agg(uniquepid=("uniquepid","first"),hospitalid=("hospitalid","first"),hdis=("hdis","max"),
    status=("hospitaldischargestatus",lambda x:"Expired" if (x=="Expired").any() else x.dropna().iloc[0] if x.notna().any() else np.nan),
    age=("age_n","max"),female=("gender",lambda x:float((x=="Female").any())),weight=("admissionweight","first"),unittype=("unittype","first"))
A["death"]=np.where(A.status=="Expired",A.hdis,np.nan)
H=sorted(A.index); hix={h:i for i,h in enumerate(H)}
uu=u.patientunitstayid.values; log("cohort",len(H))
def toh(df,col="patientunitstayid"):
    df=df[df[col].isin(SH)].copy(); df["sh"]=df[col].map(SH); df["hadm"]=df[col].map(HS); return df
# ---------------- E3 supports
inf=toh(pd.read_csv(W+"infusionDrug.f.csv.gz",low_memory=False))
dn=inf.drugname.str.lower()
kind=np.select([dn.str.contains("isoproterenol"),dn.str.contains("norepi|levophed"),dn.str.contains("epinephrine|adrenalin"),dn.str.contains("dopamine"),
    dn.str.contains("phenylephrine|neo-syn|neosyn"),dn.str.contains("vasopressin"),dn.str.contains("dobutamine|milrinone|primacor")],
    ["","vasopressor","vasopressor","vasopressor","vasopressor","vasopressor","inotrope"],"")
inf["kind"]=kind; rate=pd.to_numeric(inf.drugrate,errors="coerce"); rate=rate.where(rate.notna(),pd.to_numeric(inf.infusionrate,errors="coerce"))
inf=inf[(inf.kind!="")&(rate>0)]; inf["t"]=inf.infusionoffset+inf.sh
def episodes(df,gap,tail,key=("hadm","kind")):
    out=[]
    for k,g in df.sort_values("t").groupby(list(key)):
        t=g.t.values; br=np.r_[0,np.flatnonzero(np.diff(t)>gap)+1,len(t)]
        for a,b in zip(br[:-1],br[1:]): out.append((k[0],t[a],t[b-1]+tail,k[1]))
    return pd.DataFrame(out,columns=["hadm","start","end","kind"])
sup=[episodes(inf,240,60)]
tr=toh(pd.read_csv(W+"treatment.f.csv.gz")); tr["t"]=tr.treatmentoffset+tr.sh; ts=tr.treatmentstring.str.lower()
ia=tr[ts.str.contains("intraaortic balloon pump")&~ts.str.contains("removal")].assign(kind="mcs")
rem=tr[ts.str.contains("intraaortic balloon pump removal")]
e=episodes(ia,1440,1440)
for i,r in e.iterrows():
    rr=rem[(rem.hadm==r.hadm)&(rem.t>=r.start)&(rem.t<r.end)]
    if len(rr): e.loc[i,"end"]=max(rr.t.min(),r.start+1)
sup.append(e)
rc=toh(pd.read_csv(W+"respiratoryCharting.f.csv.gz",low_memory=False)); rc["t"]=rc.respchartoffset+rc.sh
VL={"Vent Rate","Tidal Volume (set)","TV/kg IBW","Set Vt (Servo,LTV)","Adult Con Setting Set Vt","Adult Con Setting Set RR","Pressure Control","Exhaled TV (machine)"}
vpts=[rc[rc.respchartvaluelabel.isin(VL)][["hadm","t"]],tr[ts.str.contains("mechanical ventilation")&~ts.str.contains("non-invasive")][["hadm","t"]]]
vp=pd.concat(vpts).assign(kind="imv"); sup.append(episodes(vp,480,60))
rcare=toh(pd.read_csv(W+"respiratoryCare.f.csv.gz",low_memory=False))
rcare=rcare[rcare.airwaytype.isin(["Oral ETT","Nasal ETT","Tracheostomy"])&(rcare.ventstartoffset!=0)&(rcare.ventendoffset>rcare.ventstartoffset)]
sup.append(pd.DataFrame({"hadm":rcare.hadm,"start":rcare.ventstartoffset+rcare.sh,"end":rcare.ventendoffset+rcare.sh,"kind":"imv"}))
sup=pd.concat(sup,ignore_index=True); sup=sup[sup.hadm.isin(hix)]
log("supports",sup.kind.value_counts().to_dict())
# planned rules
SURG=re.compile(r"cabg|valve|coronary artery bypass",re.I)
u["surg"]=u.apacheadmissiondx.fillna("").str.contains(SURG)
ts_s=tr[ts.str.contains(r"cardiac surgery\|cabg")]
early_cs=ts_s.merge(u[["patientunitstayid"]],on="patientunitstayid"); early_cs=early_cs[early_cs.treatmentoffset<=360]
u.loc[u.patientunitstayid.isin(early_cs.patientunitstayid),"surg"]=True
SE=u[u.surg][["patienthealthsystemstayid","uin","uout"]].rename(columns={"patienthealthsystemstayid":"hadm"}); SE["uout"]=SE.uout+720
apr=pd.read_csv(E+"apachePatientResult.csv.gz",usecols=["patientunitstayid","ptcawithin24h"]); apr=apr[apr.patientunitstayid.isin(SH)]
pci_h=set(apr[apr.ptcawithin24h==1].patientunitstayid.map(HS))
pt=tr[ts.str.contains(r"cardiac angiography\|with stent placement|cardiac angiography\|with angioplasty")&(tr.t<=1440)]
pci_h|=set(pt.hadm)
UST=u[["patienthealthsystemstayid","uin","uout"]].rename(columns={"patienthealthsystemstayid":"hadm"})
def cls(r):
    se=SE[SE.hadm==r.hadm]
    if ((r.start>=se.uin)&(r.start<=se.uout)).any(): return "periop_surgery"
    us=UST[(UST.hadm==r.hadm)&(UST.uin<=r.start)&(UST.uout>=r.start)]
    if POA>0 and ((r.start-us.uin)<=POA).any(): return "on_arrival"
    if r.hadm in pci_h and (((r.start-us.uin)<=360)&(us.uin<=1800)).any(): return "peri_pci"
    return "unplanned"
sup["cls"]=sup.apply(cls,axis=1); sup.loc[sup.cls=="on_arrival","start"]=sup.loc[sup.cls=="on_arrival"].apply(lambda r:UST[(UST.hadm==r.hadm)&(UST.uin<=r.start)&(UST.uout>=r.start)].uin.max(),axis=1)
log("support classes\n",pd.crosstab(sup.kind,sup.cls))
# ---------------- E2 landmarks
rows=[]; G=dict(tuple(sup.groupby("hadm"))); GS=dict(tuple(SE.groupby("hadm"))); GU=dict(tuple(UST.groupby("hadm")))
for h in H:
    a=A.loc[h]; s_=G.get(h,pd.DataFrame(columns=sup.columns)); ev=s_[s_.cls=="unplanned"]
    first_=ev.start.min() if len(ev) else np.nan
    fk={k:(ev[ev.kind==k].start.min() if (ev.kind==k).any() else np.nan) for k in ["vasopressor","inotrope","mcs","imv"]}
    us=GU[h]; se=GS.get(h)
    for hh in range(6,337,6):
        L=hh*60
        if L>=a.hdis or (np.isfinite(a.death) and a.death<=L): continue
        if not ((us.uin+60<=L)&(us.uout>L)).any(): continue
        if np.isfinite(first_) and first_<=L: continue
        if len(s_) and ((s_.start<=L)&(s_.end>L)).any(): continue
        if se is not None and ((se.uin<=L)&(se.uout>=L)).any(): continue
        inw=lambda t:np.isfinite(t) and L<t<=L+1440
        death=inw(a.death); circ=any(inw(fk[k]) for k in ["vasopressor","inotrope","mcs"]); resp=inw(fk["imv"])
        evt=np.nanmin([a.death,first_]) if (np.isfinite(a.death) or np.isfinite(first_)) else np.nan
        rows.append(dict(hadm=h,patient=a.uniquepid,hospital=a.hospitalid,h=hh,L=L,event_time=evt,y=int(death or circ or resp),y_death=int(death),y_circ=int(circ),y_resp=int(resp)))
lm=pd.DataFrame(rows); log("landmarks",len(lm),"patients",lm.hadm.nunique())
FLOW.append(("F5 with >=1 eligible ICU assessment",lm.hadm.nunique())); FLOW.append(("   assessments",len(lm)))
lm.to_csv(f"eicu_landmarks{SUF}.csv.gz",index=False); sup.to_csv(f"eicu_supports{SUF}.csv.gz",index=False)
# ---------------- E4 predictors
EV=[]
lab=toh(pd.read_csv(W+"lab.f.csv.gz",usecols=["patientunitstayid","labresultoffset","labresultrevisedoffset","labname","labresult"]))
LM={"creatinine":"creatinine","BUN":"urea_nitrogen","potassium":"potassium","sodium":"sodium","chloride":"chloride","magnesium":"magnesium","phosphate":"phosphate","calcium":"calcium_total","ionized calcium":"free_calcium",
"Hct":"hematocrit","Hgb":"hemoglobin","RBC":"red_blood_cells","platelets x 1000":"platelet_count","PT":"pt","PTT":"ptt","WBC x 1000":"white_blood_cells","-polys":"neutrophils","-lymphs":"lymphocytes","-monos":"monocytes","-eos":"eosinophils","-basos":"basophils",
"ALT (SGPT)":"alanine_aminotransferase_alt","AST (SGOT)":"asparate_aminotransferase_ast","alkaline phos.":"alkaline_phosphatase","total bilirubin":"bilirubin_total","albumin":"albumin","amylase":"amylase","lipase":"lipase","glucose":"glucose",
"HDL":"cholesterol_hdl","total cholesterol":"cholesterol_total","triglycerides":"triglycerides","TSH":"thyroid_stimulating_hormone","Ferritin":"ferritin","Fe":"iron","TIBC":"iron_binding_capacity_total","transferrin":"transferrin",
"CPK":"creatine_kinase_ck","CPK-MB":"creatine_kinase_mb_isoenzyme","troponin - T":"troponin_t","LDH":"lactate_dehydrogenase_ld","lactate":"lactate","bicarbonate":"bicarbonate","pH":"ph","paCO2":"pco2","paO2":"po2","O2 Sat (%)":"oxygen_saturation",
"FiO2":"fio2","troponin - I":"troponin_i"}
lab=lab[lab.labname.isin(LM)]; lab["c"]=lab.labname.map(LM); lab["v"]=pd.to_numeric(lab.labresult,errors="coerce")
k=(lab.c=="free_calcium")&(lab.v>2.5); lab.loc[k,"v"]=lab.loc[k,"v"]/4.008
k=lab.c=="fio2"; lab.loc[k&(lab.v>1),"v"]=lab.loc[k&(lab.v>1),"v"]/100
lab["tc"]=lab.labresultoffset+lab.sh; lab=lab[lab.tc>=-1440]
lab["t"]=np.fmax(lab.labresultoffset,lab.labresultrevisedoffset.fillna(-1e9))+lab.sh
EV.append(lab[["hadm","t","c","v"]])
vp=toh(pd.read_csv(W+"vitalPeriodic.h.csv.gz")); vp["c"]=vp["var"].map({"heartrate":"heart_rate","sao2":"spo2","respiration":"resp_rate","temperature":"temperature","systemicsystolic":"sbp","systemicdiastolic":"dbp","systemicmean":"map"})
vp["t"]=vp.offset+vp.sh; vp["v"]=pd.to_numeric(vp.value,errors="coerce"); EV.append(vp[vp.c.notna()][["hadm","t","c","v"]])
va=toh(pd.read_csv(W+"vitalAperiodic.f.csv.gz",usecols=["patientunitstayid","observationoffset","noninvasivesystolic","noninvasivediastolic","noninvasivemean"]))
for col,c in [("noninvasivesystolic","sbp"),("noninvasivediastolic","dbp"),("noninvasivemean","map")]:
    EV.append(pd.DataFrame({"hadm":va.hadm,"t":va.observationoffset+va.sh,"c":c,"v":va[col]}))
nc=toh(pd.read_csv(W+"nc.s.csv.gz",quoting=csv.QUOTE_NONE,on_bad_lines="skip",low_memory=False))
nc["v"]=pd.to_numeric(nc.value,errors="coerce"); nc["t"]=np.fmax(nc.offset,pd.to_numeric(nc.entryoffset,errors="coerce").fillna(-1e9))+nc.sh
NM={"Heart Rate":"heart_rate","Respiratory Rate":"resp_rate","O2 Saturation":"spo2","Non-Invasive BP Systolic":"sbp","Non-Invasive BP Diastolic":"dbp","Non-Invasive BP Mean":"map",
    "Invasive BP Systolic":"sbp","Invasive BP Diastolic":"dbp","Invasive BP Mean":"map","Temperature (C)":"temperature","Eyes":"gcs_e","Verbal":"gcs_v","Motor":"gcs_m","GCS Total":"gcs_tot"}
n1=nc[nc.name.isin(NM)].assign(c=lambda d:d.name.map(NM)); EV.append(n1[["hadm","t","c","v"]])
o=nc[nc.name=="O2 L/%"]; EV.append(o[o.v<=15].assign(c="o2flow")[["hadm","t","c","v"]]); EV.append(o[(o.v>=21)&(o.v<=100)].assign(c="fio2",v=lambda d:d.v/100)[["hadm","t","c","v"]])
rc["v"]=pd.to_numeric(rc.respchartvalue.astype(str).str.replace("%",""),errors="coerce"); rc["t"]=np.fmax(rc.respchartoffset,rc.respchartentryoffset)+rc.sh
f=rc[rc.respchartvaluelabel.isin(["FiO2","FIO2 (%)","O2 Percentage"])].copy(); f.loc[f.v>1,"v"]=f.loc[f.v>1,"v"]/100; EV.append(f.assign(c="fio2")[["hadm","t","c","v"]])
EV.append(rc[rc.respchartvaluelabel.isin(["LPM O2","Oxygen Flow Rate"])].assign(c="o2flow")[["hadm","t","c","v"]])
EV.append(pd.DataFrame({"hadm":u.patienthealthsystemstayid,"t":u.uin,"c":"weight","v":pd.to_numeric(u.admissionweight,errors="coerce")}))
io=toh(pd.read_csv(W+"intakeOutput.f.csv.gz",usecols=["patientunitstayid","intakeoutputoffset","intakeoutputentryoffset","cellpath","celllabel","cellvaluenumeric"]))
io=io[io.cellpath.str.contains("Output",case=False,na=False)&io.celllabel.str.contains("urin|foley|urethral|void|catheter",case=False,na=False)&~io.celllabel.str.contains("count|occurrence",case=False,na=False)]
EV.append(pd.DataFrame({"hadm":io.hadm,"t":np.fmax(io.intakeoutputoffset,io.intakeoutputentryoffset)+io.sh,"c":"urine","v":io.cellvaluenumeric}))
ev=pd.concat(EV,ignore_index=True); ev=ev[np.isfinite(ev.v)&np.isfinite(ev.t)&ev.hadm.isin(hix)]
ev=ev[ev.t<=ev.hadm.map(A.hdis)]
RANGE={"heart_rate":(1,300),"sbp":(20,300),"dbp":(5,250),"map":(10,250),"resp_rate":(1,70),"spo2":(30,100),"temperature":(25,45),"fio2":(0.21,1),"o2flow":(0,80),"weight":(25,300)}
for c,(lo,hi) in RANGE.items():
    k=ev.c==c; ev=ev[~k|((ev.v>=lo)&(ev.v<=hi))]
ev["hi"]=ev.hadm.map(hix).astype(np.int64); log("events",len(ev))
# ---- identical feature code (features_v2fix.py), keyed on hospital minutes
LH=lm.hadm.map(hix).values.astype(np.int64); LT=lm.L.values.astype(float); TMIN=min(ev.t.min(),LT.min())-1e4
lkey=LH*(1<<34)+(LT-TMIN).astype(np.int64)
def series(c):
    e=ev[ev.c==c]; k=(e.hi.values*(1<<34)+(e.t.values-TMIN).astype(np.int64)); o=np.argsort(k,kind="stable"); return k[o],e.v.values[o].astype(float)
def at(k,v,win_min=None):
    i=np.searchsorted(k,lkey,side="right")-1; ok=(i>=0)&((k[np.maximum(i,0)]>>34)==LH)
    last=np.where(ok,v[np.maximum(i,0)],np.nan); age=np.where(ok,(lkey-k[np.maximum(i,0)])/60,np.nan)
    if win_min is not None: last=np.where(age*60<=win_min,last,np.nan)
    return i,ok,last,age
FEAT={}; PROC={}
def value_feats(name,k,v,axis):
    i,ok,last,age=at(k,v)
    j=i-1; ok2=ok&(j>=0)&((k[np.maximum(j,0)]>>34)==LH); prev=np.where(ok2,v[np.maximum(j,0)],np.nan)
    first_i=np.searchsorted(k,LH*(1<<34),side="left"); okb=ok&(first_i<=i); base=np.where(okb,v[np.minimum(first_i,len(v)-1)],np.nan)
    s=np.searchsorted(k,lkey-24*60,side="right"); s=np.maximum(s,first_i); e=i+1; has=ok&(e>s)
    vv=np.append(v,np.nan); ii=np.empty(2*len(s),dtype=np.int64); ii[0::2]=np.clip(s,0,len(v)); ii[1::2]=np.clip(e,0,len(v))
    ii=np.minimum(ii,len(v))
    mn=np.fmin.reduceat(vv,ii)[0::2]; mx=np.fmax.reduceat(vv,ii)[0::2]
    mn=np.where(has,mn,np.nan); mx=np.where(has,mx,np.nan)
    tt=(k & ((1<<34)-1)).astype(np.float64); hi_=k>>34; st_=np.flatnonzero(np.r_[True,hi_[1:]!=hi_[:-1]]) if len(k) else np.array([],int)
    base_=np.repeat(tt[st_],np.diff(np.r_[st_,len(k)])) if len(k) else tt
    th=(tt-base_)/60.0
    c1=np.concatenate([[0],np.cumsum(np.ones(len(v)))]); ct=np.concatenate([[0],np.cumsum(th)]); cv=np.concatenate([[0],np.cumsum(v)])
    ctt=np.concatenate([[0],np.cumsum(th*th)]); ctv=np.concatenate([[0],np.cumsum(th*v)])
    s2=np.clip(s,0,len(v)); e2=np.clip(e,0,len(v)); n=c1[e2]-c1[s2]
    St,Sv,Stt,Stv=ct[e2]-ct[s2],cv[e2]-cv[s2],ctt[e2]-ctt[s2],ctv[e2]-ctv[s2]
    den=n*Stt-St*St; slope=np.where(has&(n>=2)&(den>1e-9),(n*Stv-St*Sv)/np.where(den>1e-9,den,1),np.nan)
    for suf,arr in [("last",last),("d_prev",last-prev),("d_base",last-base),("min24",mn),("max24",mx),("slope24",slope)]: FEAT[f"{name}.{suf}"]=(arr,axis)
    PROC[f"{name}.n24"]=np.where(ok,e-s,0); PROC[f"{name}.hrs_since"]=np.minimum(age,168)
axes={"INJURY":["creatine_kinase_ck","creatine_kinase_mb_isoenzyme","troponin_t","lactate_dehydrogenase_ld"],
 "HEMO":["heart_rate","sbp","dbp","map"],"PERFUSION":["lactate","bicarbonate","ph"],"RESP":["resp_rate","spo2","pco2","po2","oxygen_saturation"],
 "RENAL":["creatinine","urea_nitrogen","potassium","sodium","chloride","magnesium","phosphate","calcium_total","free_calcium"],
 "HEME":["hematocrit","hemoglobin","red_blood_cells","platelet_count","pt","ptt"],
 "INFLAM":["white_blood_cells","temperature","neutrophils","lymphocytes","monocytes","eosinophils","basophils","absolute_basophil_count","absolute_eosinophil_count","absolute_lymphocyte_count","absolute_monocyte_count","absolute_neutrophil_count"],
 "META":["alanine_aminotransferase_alt","asparate_aminotransferase_ast","alkaline_phosphatase","bilirubin_total","albumin","amylase","lipase","glucose","hemoglobin_a1c","cholesterol_hdl","cholesterol_total","triglycerides","thyroid_stimulating_hormone","ferritin","iron","iron_binding_capacity_total","transferrin"]}
c2a={c:a for a,cs in axes.items() for c in cs}
for c in sorted(set(ev.c)-{"fio2","o2flow","gcs_e","gcs_v","gcs_m","gcs_tot","weight","urine","troponin_i"}):
    k,v=series(c); value_feats(c,k,v,c2a[c])
L_=lambda n:FEAT[n+".last"][0] if n+".last" in FEAT else np.full(len(lm),np.nan)
FEAT["shock_index"]=(L_("heart_rate")/L_("sbp"),"HEMO")
FEAT["anion_gap"]=(L_("sodium")-L_("chloride")-L_("bicarbonate"),"PERFUSION")
_,_,fio2,_=at(*series("fio2"),win_min=6*60); _,_,flow,_=at(*series("o2flow"),win_min=6*60)
F=np.where(np.isfinite(fio2),fio2,np.where(np.isfinite(flow),np.minimum(0.21+0.04*flow,0.6),0.21))
FEAT["sf_ratio"]=(L_("spo2")/F,"RESP"); PROC["fio2_est"]=F
g=[at(*series(c),win_min=8*60)[2] for c in ["gcs_e","gcs_v","gcs_m"]]; gt=at(*series("gcs_tot"),win_min=8*60)[2]
gs=g[0]+g[1]+g[2]; FEAT["gcs_total"]=(np.where(np.isfinite(gs),gs,gt),"NEURO")
k,v=series("urine"); cs=np.concatenate([[0],np.cumsum(v)])
_,_,wt,_=at(*series("weight"))
for hrs in (6,24):
    e=np.searchsorted(k,lkey,side="right"); s=np.searchsorted(k,lkey-hrs*60,side="right")
    s=np.maximum(s,np.searchsorted(k,LH*(1<<34),side="left")); n=e-s
    tot=cs[e]-cs[s]; FEAT[f"urine_mlkgh_{hrs}h"]=(np.where(n>0,tot/np.where(np.isfinite(wt),wt,80)/hrs,np.nan),"RENAL")
for kk in ["ecg_adm_ste","ecg_adm_anterior","ecg_adm_wide_qrs","ecg_adm_af","ecg_last_hr","ecg_last_qrs","ecg_last_qtc","ecg_last_af","ecg_st_resolved","ecg_new_af"]: FEAT[kk]=(np.full(len(lm),np.nan),"ELECTRICAL")
a=A.loc[lm.hadm]; FEAT["age"]=(a.age.values.astype(float),"BASELINE"); FEAT["female"]=(a.female.values.astype(float),"BASELINE")
ph=toh(pd.read_csv(E+"pastHistory.csv.gz",usecols=["patientunitstayid","pasthistorypath"])); pp=ph.pasthistorypath
pats={"prior_hf":"Congestive Heart Failure","prior_ckd":"Renal Insufficiency|Renal Failure","prior_dm":"Insulin Dependent Diabetes|Non-Insulin Dependent Diabetes","prior_mi":"/Myocardial Infarction/"}
for kk,pt_ in pats.items():
    s_=set(ph[pp.str.contains(pt_)].hadm); FEAT[kk]=(lm.hadm.isin(s_).astype(float).values,"BASELINE")
FEAT["hours_since_admit"]=(lm.h.values.astype(float),"BASELINE")
meta=json.load(open("../full/full_feature_meta_v2.json")); names=meta["names"]
miss=[n for n in names if n not in FEAT]; log("names missing in eICU (set NaN):",len(miss),miss[:40])
for n in miss: FEAT[n]=(np.full(len(lm),np.nan),meta["axis"][names.index(n)])
X=np.column_stack([FEAT[n][0] for n in names]).astype(np.float32)
assert [FEAT[n][1] for n in names]==meta["axis"]
XP=np.column_stack([PROC.get(n,np.full(len(lm),np.nan)) for n in meta["proc_names"]]).astype(np.float32)
stemi=set(toh(dx[dx.diagnosisstring.str.contains("with ST elevation",case=False)]).hadm)
_,_,tni,_=at(*series("troponin_i"))
np.savez_compressed(f"eicu_features{SUF}.npz",X=X,XP=XP,y=lm.y.values,y_death=lm.y_death.values,y_circ=lm.y_circ.values,y_resp=lm.y_resp.values,
    patient=lm.hadm.values,hadm=lm.hadm.values,hospital=lm.hospital.values,h=lm.h.values,lead=((lm.event_time-lm.L)/60).values,
    stemi=lm.hadm.isin(stemi).values,tni=tni,pci=lm.hadm.isin(pci_h).values,surg=lm.hadm.isin(set(SE.hadm)).values,
    cardiac_icu=A.loc[lm.hadm].unittype.isin(["CCU-CTICU","Cardiac ICU","CSICU","CTICU"]).values)
pd.DataFrame(FLOW,columns=["step","n"]).to_csv(f"eicu_flow{SUF}.csv",index=False)
cov=pd.Series(np.isfinite(X).mean(0),index=names)
log("X",X.shape); print(cov[[n for n in names if n.endswith(".last") or "." not in n]].round(2).to_string())
