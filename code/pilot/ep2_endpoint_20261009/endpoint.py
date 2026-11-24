"""EP2：非计划恶化终点。输出每个预测时点的风险集资格与24h标签（主定义+两个敏感性定义）"""
import pandas as pd, numpy as np
U="/mnt/user-data/uploads/KG-v1/outputs/experiments/ep2_20261009/"
H=sorted(int(l.strip().strip(",")) for l in open(U+"ami_hadm_pat.txt"))
adm=pd.read_csv(U+"admissions.csv.gz",parse_dates=["admittime","dischtime","deathtime"]); adm=adm[adm.hadm_id.isin(H)].set_index("hadm_id")
P=lambda c:pd.to_datetime(c)
# 支持治疗区间
ie=pd.read_csv(U+"inputs.csv.gz"); ie["starttime"],ie["endtime"]=P(ie.starttime),P(ie.endtime)
ie=ie[(pd.to_numeric(ie.rate,errors="coerce")>0)&(ie.ordercategorydescription=="Continuous Med")&(ie.statusdescription!="Rewritten")&(ie.endtime>ie.starttime)]
VASO={221906,221289,229617,222315,221749,229630,229631,229632,221662,229709,229764}; INO={221653,221986}
ie["kind"]=np.where(ie.itemid.isin(INO),"inotrope","vasopressor")
pe=pd.read_csv(U+"procs.csv.gz"); pe["starttime"],pe["endtime"]=P(pe.starttime),P(pe.endtime)
pe=pe[(pe.statusdescription!="Rewritten")&(pe.endtime>pe.starttime)]
pe["kind"]=pe.itemid.map({225792:"imv",224272:"mcs",228169:"mcs",229529:"mcs",229530:"mcs"})
pe=pe[pe.kind.notna()&((pe.itemid!=225792)|(pd.to_numeric(pe.value,errors="coerce")>0))]
sup=pd.concat([ie[["hadm_id","starttime","endtime","kind"]],pe[["hadm_id","starttime","endtime","kind"]]],ignore_index=True)
sup=sup[sup.hadm_id.isin(H)]
# 计划内/操作相关判定
tr=pd.read_csv(U+"transfers.csv.gz",parse_dates=["intime","outtime"])
pacu=tr[tr.careunit.isin(["PACU"])][["hadm_id","intime","outtime"]]
px=pd.read_csv(U+"procedures_icd.csv.gz",dtype={"icd_code":str},parse_dates=["chartdate"])
c=px.icd_code.str.replace(".","",regex=False)
surg=((px.icd_version==10)&(c.str.match(r"^021|^02R|^02Q|^02U"))) | ((px.icd_version==9)&(c.str.match(r"^361|^35[0-2]")))
pci=((px.icd_version==10)&(c.str.match(r"^027|^4A023N|^B21"))) | ((px.icd_version==9)&(c.str.match(r"^0066|^3606|^3607|^372[2-3]|^8855|^8856|^8857")))
surgd=px[surg][["hadm_id","chartdate"]]; pcid=px[pci][["hadm_id","chartdate"]]
icu=pd.read_csv(U+"icustays.csv.gz",parse_dates=["intime","outtime"])
def classify(r):
    h,s=r.hadm_id,r.starttime
    p=pacu[pacu.hadm_id==h]
    if ((s>=p.intime-pd.Timedelta(hours=12))&(s<=p.outtime+pd.Timedelta(hours=12))).any(): return "periop_surgery"
    d=surgd[surgd.hadm_id==h].chartdate
    if ((s>=d)&(s<d+pd.Timedelta(hours=36))).any(): return "periop_surgery"
    d=pcid[pcid.hadm_id==h].chartdate
    onpci=((s>=d)&(s<d+pd.Timedelta(hours=30))).any()
    ic=icu[icu.hadm_id==h]; early=((s>=ic.intime)&(s<=ic.intime+pd.Timedelta(hours=6))).any()
    if onpci and early: return "peri_pci"
    if onpci: return "pci_day_other"
    return "unplanned"
sup["cls"]=sup.apply(classify,axis=1)
print("support records by class & kind:\n",pd.crosstab(sup.kind,sup.cls))
DEF={"primary":{"unplanned","pci_day_other"},"sens_all":{"unplanned","pci_day_other","peri_pci","periop_surgery"},"sens_strict":{"unplanned"}}
rows=[]
for h in H:
    a=adm.loc[h]; s=sup[sup.hadm_id==h]
    for dn,ok in DEF.items():
        ev=s[s.cls.isin(ok)]; first=ev.starttime.min() if len(ev) else pd.NaT
        firstk={k:(ev[ev.kind==k].starttime.min() if (ev.kind==k).any() else pd.NaT) for k in ["vasopressor","inotrope","mcs","imv"]}
        for hh in range(6,73,6):
            L=a.admittime+pd.Timedelta(hours=hh)
            if L>=a.dischtime or (pd.notna(a.deathtime) and a.deathtime<=L): continue
            if pd.notna(first) and first<=L: continue
            ongoing=((s.starttime<=L)&(s.endtime>L)).any()        # 正在接受任何支持治疗（含计划内）时不预测
            if ongoing: continue
            W=(L,L+pd.Timedelta(hours=24)); inw=lambda t:pd.notna(t) and W[0]<t<=W[1]
            death=inw(a.deathtime); circ=any(inw(firstk[k]) for k in ["vasopressor","inotrope","mcs"]); resp=inw(firstk["imv"])
            rows.append(dict(definition=dn,hadm_id=h,subject_id=int(a.subject_id),h=hh,L=L,y=int(death or circ or resp),y_death=int(death),y_circ=int(circ),y_resp=int(resp)))
lm=pd.DataFrame(rows); lm.to_csv("ep2_landmarks.csv.gz",index=False)
old=pd.read_csv("/mnt/user-data/uploads/KG-v1/outputs/experiments/ecg01_20261008/row_landmarks.csv")
print("OLD endpoint: landmarks",len(old),"pos",int(old.event.sum()),"event patients",old[old.event==1].patient.nunique())
for dn,g in lm.groupby("definition"):
    print(f"{dn:12s} landmarks {len(g)} pos {g.y.sum()} ({g.y.mean():.3%}) event patients {g[g.y==1].subject_id.nunique()} | death {g.y_death.sum()} circ {g.y_circ.sum()} resp {g.y_resp.sum()}")

# 诊断：整个住院期间（不限72h）首次非计划事件发生在入院后多少小时
ok=DEF["primary"]; ev=sup[sup.cls.isin(ok)].groupby("hadm_id").starttime.min()
a2=adm.loc[H]; t=(ev.reindex(H)-a2.admittime).dt.total_seconds()/3600
d=(a2.deathtime-a2.admittime).dt.total_seconds()/3600
first=pd.concat([t,d],axis=1).min(axis=1)
print("hadm with unplanned support or death during stay:",first.notna().sum())
print("timing of first unplanned event (h after admission):",pd.cut(first.dropna(),[0,6,30,54,78,96,168,336,1e5]).value_counts().sort_index().to_dict())
los=(a2.dischtime-a2.admittime).dt.total_seconds()/86400; print("LOS days median",los.median(),"p75",los.quantile(.75))
