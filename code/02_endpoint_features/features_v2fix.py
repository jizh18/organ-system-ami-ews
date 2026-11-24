"""EP2 特征：只用生理数值（主分析）；诊疗过程特征单独保存（补充分析）"""
import pandas as pd, numpy as np, time, json, sys
DEFN=sys.argv[1] if len(sys.argv)>1 else "primary"; SUF="" if DEFN=="primary" else "_"+DEFN
U="/mnt/user-data/uploads/KG-v1/outputs/experiments/full_v2_20261009/"; t0=time.time()
lm=pd.read_csv("full_landmarks.csv.gz",parse_dates=["L","event_time"]); lm=lm[lm.definition==DEFN].reset_index(drop=True)
adm=pd.read_csv(U+"admissions.csv.gz",parse_dates=["admittime","dischtime"]).set_index("hadm_id")
H=sorted(lm.hadm_id.unique()); hix={h:i for i,h in enumerate(H)}
T0=pd.Timestamp("2100-01-01"); mins=lambda s:((pd.to_datetime(s,errors="coerce")-T0).dt.total_seconds()/60)
# ---- 事件表 (hadm, avail分钟, concept, value)
EV=[]
# 化验
lmap=pd.read_csv(U+"lab_itemmap.csv"); lmap=dict(zip(lmap.itemid,lmap.code))
cols=["labevent_id","subject_id","hadm_id","specimen_id","itemid","order_provider_id","charttime","storetime","value","valuenum","valueuom","ref_range_lower","ref_range_upper","flag","priority","comments"]
lab=pd.concat([pd.read_csv(U+f,header=None,names=cols,usecols=["subject_id","itemid","charttime","storetime","valuenum"],on_bad_lines="skip",encoding_errors="replace",low_memory=False) for f in ["lab_a.csv.gz","lab_b.csv.gz"]])
lab["itemid"]=pd.to_numeric(lab.itemid,errors="coerce"); lab=lab[lab.itemid.isin(lmap)]
idx=adm.loc[H,["subject_id","admittime","dischtime"]].reset_index()
lab=lab.merge(idx,on="subject_id"); lab["ct"]=pd.to_datetime(lab.charttime,errors="coerce")
lab=lab[(lab.ct>=lab.admittime-pd.Timedelta(hours=24))&(lab.ct<=lab.dischtime)]
lab["t"]=np.fmax(mins(lab.charttime),mins(lab.storetime)); lab["c"]=lab.itemid.map(lmap); lab["v"]=pd.to_numeric(lab.valuenum,errors="coerce")
EV.append(lab[["hadm_id","t","c","v"]])
# 生命体征/床旁
ch=pd.concat([pd.read_csv(U+f,header=None,names=["subject_id","hadm_id","stay_id","caregiver_id","charttime","storetime","itemid","value","valuenum","valueuom","warning"],on_bad_lines="skip",encoding_errors="replace") for f in ["chart_a.csv.gz","chart_b.csv.gz"]])
ch=ch[ch.hadm_id.isin(hix)]; ch["v"]=pd.to_numeric(ch.valuenum,errors="coerce"); ch["t"]=np.fmax(mins(ch.charttime),mins(ch.storetime))
m={220045:"heart_rate",220179:"sbp",220050:"sbp",220180:"dbp",220051:"dbp",220181:"map",220052:"map",220210:"resp_rate",220277:"spo2",
   223762:"temperature",223761:"temperature",223835:"fio2",223834:"o2flow",220739:"gcs_e",223900:"gcs_v",223901:"gcs_m",226512:"weight",224639:"weight"}
ch["c"]=ch.itemid.map(m); ch=ch[ch.c.notna()]
f=ch.itemid==223761; ch.loc[f,"v"]=(ch.loc[f,"v"]-32)/1.8
f=ch.c=="fio2"; ch.loc[f&(ch.v>1),"v"]=ch.loc[f&(ch.v>1),"v"]/100
EV.append(ch[["hadm_id","t","c","v"]])
# 尿量
uo=pd.read_csv(U+"urine.csv.gz"); uo["t"]=np.fmax(mins(uo.charttime),mins(uo.storetime)); uo["v"]=np.where(uo.itemid==227488,-1,1)*pd.to_numeric(uo.value,errors="coerce")
uo["c"]="urine"; EV.append(uo[uo.hadm_id.isin(hix)][["hadm_id","t","c","v"]])
ev=pd.concat(EV,ignore_index=True); ev=ev[np.isfinite(ev.v)&np.isfinite(ev.t)]
RANGE={"heart_rate":(1,300),"sbp":(20,300),"dbp":(5,250),"map":(10,250),"resp_rate":(1,70),"spo2":(30,100),"temperature":(25,45),"fio2":(0.21,1),"o2flow":(0,80),"weight":(25,300)}
for c,(lo,hi) in RANGE.items():
    k=ev.c==c; ev=ev[~k|((ev.v>=lo)&(ev.v<=hi))]
ev["hi"]=ev.hadm_id.map(hix).astype(np.int64)
print("events",len(ev),round(time.time()-t0),flush=True)
LH=lm.hadm_id.map(hix).values.astype(np.int64); LT=mins(lm.L).values; TMIN=min(ev.t.min(),LT.min())-1e4
lkey=LH*(1<<34)+(LT-TMIN).astype(np.int64)
def series(c):
    e=ev[ev.c==c]; k=(e.hi.values*(1<<34)+(e.t.values-TMIN).astype(np.int64)); o=np.argsort(k,kind="stable"); return k[o],e.v.values[o]
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
    ii=np.minimum(ii,len(v)); 
    mn=np.fmin.reduceat(vv,ii)[0::2]; mx=np.fmax.reduceat(vv,ii)[0::2]
    mn=np.where(has,mn,np.nan); mx=np.where(has,mx,np.nan)
    # 24小时线性趋势（斜率，单位/小时），窗口内至少2个值
    tt=(k & ((1<<34)-1)).astype(np.float64); hi_=k>>34; st_=np.flatnonzero(np.r_[True,hi_[1:]!=hi_[:-1]]) if len(k) else np.array([],int)
    base_=np.repeat(tt[st_],np.diff(np.r_[st_,len(k)])) if len(k) else tt
    th=(tt-base_)/60.0   # FIX 9 Oct: times relative to each admission's first value (global offsets caused cancellation in cumulative sums)
    c1=np.concatenate([[0],np.cumsum(np.ones(len(v)))]); ct=np.concatenate([[0],np.cumsum(th)]); cv=np.concatenate([[0],np.cumsum(v)])
    ctt=np.concatenate([[0],np.cumsum(th*th)]); ctv=np.concatenate([[0],np.cumsum(th*v)])
    s2=np.clip(s,0,len(v)); e2=np.clip(e,0,len(v)); n=c1[e2]-c1[s2]
    St,Sv,Stt,Stv=ct[e2]-ct[s2],cv[e2]-cv[s2],ctt[e2]-ctt[s2],ctv[e2]-ctv[s2]
    den=n*Stt-St*St; slope=np.where(has&(n>=2)&(den>1e-9),(n*Stv-St*Sv)/np.where(den>1e-9,den,1),np.nan)
    for suf,arr in [("last",last),("d_prev",last-prev),("d_base",last-base),("min24",mn),("max24",mx),("slope24",slope)]: FEAT[f"{name}.{suf}"]=(arr,axis)
    PROC[f"{name}.n24"]=np.where(ok,e-s,0); PROC[f"{name}.hrs_since"]=np.minimum(age,168)
AX={}
axes={"INJURY":["creatine_kinase_ck","creatine_kinase_mb_isoenzyme","troponin_t","lactate_dehydrogenase_ld"],
 "HEMO":["heart_rate","sbp","dbp","map"],
 "PERFUSION":["lactate","bicarbonate","ph"],
 "RESP":["resp_rate","spo2","pco2","po2","oxygen_saturation"],
 "RENAL":["creatinine","urea_nitrogen","potassium","sodium","chloride","magnesium","phosphate","calcium_total","free_calcium"],
 "HEME":["hematocrit","hemoglobin","red_blood_cells","platelet_count","pt","ptt"],
 "INFLAM":["white_blood_cells","temperature","neutrophils","lymphocytes","monocytes","eosinophils","basophils","absolute_basophil_count","absolute_eosinophil_count","absolute_lymphocyte_count","absolute_monocyte_count","absolute_neutrophil_count"],
 "META":["alanine_aminotransferase_alt","asparate_aminotransferase_ast","alkaline_phosphatase","bilirubin_total","albumin","amylase","lipase","glucose","hemoglobin_a1c","cholesterol_hdl","cholesterol_total","triglycerides","thyroid_stimulating_hormone","ferritin","iron","iron_binding_capacity_total","transferrin"]}
c2a={c:a for a,cs in axes.items() for c in cs}
for c in sorted(set(ev.c)-{"fio2","o2flow","gcs_e","gcs_v","gcs_m","weight","urine"}):
    if c not in c2a: print("unassigned",c); continue
    k,v=series(c); value_feats(c,k,v,c2a[c])
print("base feats",len(FEAT),round(time.time()-t0),flush=True)
# ---- 派生生理量
L_=lambda n:FEAT[n+".last"][0]
FEAT["shock_index"]=(L_("heart_rate")/L_("sbp"),"HEMO")
FEAT["anion_gap"]=(L_("sodium")-L_("chloride")-L_("bicarbonate"),"PERFUSION")
_,_,fio2,_=at(*series("fio2"),win_min=6*60); _,_,flow,_=at(*series("o2flow"),win_min=6*60)
F=np.where(np.isfinite(fio2),fio2,np.where(np.isfinite(flow),np.minimum(0.21+0.04*flow,0.6),0.21))
FEAT["sf_ratio"]=(L_("spo2")/F,"RESP"); PROC["fio2_est"]=F
g=[at(*series(c),win_min=8*60)[2] for c in ["gcs_e","gcs_v","gcs_m"]]
FEAT["gcs_total"]=(g[0]+g[1]+g[2],"NEURO")
# 尿量 mL/kg/h，近6h与24h（仅在有尿量记录的时段）
k,v=series("urine"); cs=np.concatenate([[0],np.cumsum(v)])
_,_,wt,_=at(*series("weight")); wt=np.where(np.isfinite(wt),wt,np.nan)
for hrs in (6,24):
    e=np.searchsorted(k,lkey,side="right"); s=np.searchsorted(k,lkey-hrs*60,side="right")
    s=np.maximum(s,np.searchsorted(k,LH*(1<<34),side="left")); n=e-s
    tot=cs[e]-cs[s]; FEAT[f"urine_mlkgh_{hrs}h"]=(np.where(n>0,tot/np.where(np.isfinite(wt),wt,80)/hrs,np.nan),"RENAL")
print("derived done",round(time.time()-t0),flush=True)
# ---- 心电（精选）
mm=pd.read_csv(U+"ecg_mm.csv.gz",low_memory=False,parse_dates=["ecg_time"])
rc=[c for c in mm.columns if c.startswith("report_")]; txt=mm[rc].fillna("").astype(str).agg(" | ".join,axis=1).str.lower()
mm["ste"]=txt.str.contains(r"st elev|injury|acute mi|stemi"); mm["ant"]=txt.str.contains("anterior")&txt.str.contains(r"infarct|injury|st elev")
mm["af"]=txt.str.contains(r"atrial fib|atrial flut")
num=lambda s:pd.to_numeric(s,errors="coerce")
rr=num(mm.rr_interval).where(lambda x:(x>200)&(x<3000)); mm["hr"]=60000/rr
mm["qrs"]=(num(mm.qrs_end)-num(mm.qrs_onset)).where(lambda x:(x>40)&(x<250))
mm["qtc"]=((num(mm.t_end)-num(mm.qrs_onset))/np.sqrt(rr/1000)).where(lambda x:(x>300)&(x<700))
mm["avail"]=mm.ecg_time+pd.Timedelta(minutes=30)
sub2h=adm.loc[H].reset_index()[["hadm_id","subject_id","admittime"]]
mm=mm.merge(sub2h,on="subject_id"); mm=mm[mm.ecg_time>=mm.admittime-pd.Timedelta(hours=6)].sort_values(["hadm_id","avail"])
G=dict(tuple(mm.groupby("hadm_id"))); E={k:np.full(len(lm),np.nan) for k in ["ecg_adm_ste","ecg_adm_anterior","ecg_adm_wide_qrs","ecg_adm_af","ecg_last_hr","ecg_last_qrs","ecg_last_qtc","ecg_last_af","ecg_st_resolved","ecg_new_af"]}
for i,r in enumerate(lm.itertuples()):
    e=G.get(r.hadm_id)
    if e is None: continue
    e=e[e.avail<=r.L]
    if not len(e): continue
    f0,fl=e.iloc[0],e.iloc[-1]
    E["ecg_adm_ste"][i]=f0.ste; E["ecg_adm_anterior"][i]=f0.ant; E["ecg_adm_wide_qrs"][i]=float(f0.qrs>=120) if pd.notna(f0.qrs) else np.nan; E["ecg_adm_af"][i]=f0.af
    E["ecg_last_hr"][i]=fl.hr; E["ecg_last_qrs"][i]=fl.qrs; E["ecg_last_qtc"][i]=fl.qtc; E["ecg_last_af"][i]=fl.af
    E["ecg_st_resolved"][i]=float(f0.ste and not fl.ste) if len(e)>1 else np.nan
    E["ecg_new_af"][i]=float(fl.af and not e.af.iloc[:-1].any()) if len(e)>1 else 0.0
for k,v in E.items(): FEAT[k]=(v,"ELECTRICAL")
print("ecg done",round(time.time()-t0),flush=True)
# ---- 基线
pt=pd.read_csv(U+"patients.csv.gz").set_index("subject_id")
a=adm.loc[lm.hadm_id]; yr=a.admittime.dt.year.values
FEAT["age"]=(pt.loc[lm.subject_id,"anchor_age"].values+(yr-pt.loc[lm.subject_id,"anchor_year"].values),"BASELINE")
FEAT["female"]=((pt.loc[lm.subject_id,"gender"].values=="F").astype(float),"BASELINE")
dx=pd.read_csv(U+"diagnoses.csv.gz",dtype={"icd_code":str}); alla=pd.read_csv(U+"admissions.csv.gz",parse_dates=["admittime"])[["hadm_id","admittime"]]
dx=dx.merge(alla,on="hadm_id"); idxt=adm.loc[H,["subject_id","admittime"]].reset_index().rename(columns={"admittime":"index_t","hadm_id":"index_hadm"})
dx=dx.merge(idxt,on="subject_id"); dx=dx[dx.admittime<dx.index_t]   # 只用既往住院诊断
code=dx.icd_code.str.replace(".","",regex=False)
pats={"prior_hf":r"^I50|^428","prior_ckd":r"^N18|^585","prior_dm":r"^E1[0-4]|^250","prior_mi":r"^I2[12]|^I252|^410|^412"}
for k,p in pats.items():
    s=set(dx[code.str.match(p)].index_hadm); FEAT[k]=(lm.hadm_id.isin(s).astype(float).values,"BASELINE")
FEAT["hours_since_admit"]=(lm.h.values.astype(float),"BASELINE")
names=list(FEAT); X=np.column_stack([FEAT[n][0] for n in names]).astype(np.float32); axis=[FEAT[n][1] for n in names]
pn=list(PROC); XP=np.column_stack([PROC[n] for n in pn]).astype(np.float32)
np.savez_compressed(f"full_features{SUF}.npz",X=X,XP=XP,y=lm.y.values,y_death=lm.y_death.values,y_circ=lm.y_circ.values,y_resp=lm.y_resp.values,
    patient=lm.subject_id.values,hadm=lm.hadm_id.values,h=lm.h.values,lead=((lm.event_time-lm.L).dt.total_seconds()/3600).values)
json.dump({"names":names,"axis":axis,"proc_names":pn},open(f"full_feature_meta{SUF}.json","w"),ensure_ascii=False)
cov=pd.Series(np.isfinite(X).mean(0),index=names)
print("X",X.shape,"XP",XP.shape,round(time.time()-t0))
print(cov[[n for n in names if n.endswith(".last") or "." not in n]].round(2).to_string())
