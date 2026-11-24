"""EP2：非计划恶化终点。输出每个预测时点的风险集资格与24h标签（主定义+两个敏感性定义）"""
import pandas as pd, numpy as np
U="/mnt/user-data/uploads/KG-v1/outputs/experiments/full_v2_20261009/"
CO=pd.read_csv(U+"cohort.csv"); CO=CO[CO.period.isin(["development","later"])]; H=sorted(CO.hadm_id.astype(int))
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
EP={}
for h_,g in surgd.groupby("hadm_id"):
    st=pd.to_datetime(g.chartdate).min(); ic=icu[(icu.hadm_id==h_)&(icu.outtime>=st)&(icu.intime<=st+pd.Timedelta(hours=48))]
    en=(ic.outtime.max() if len(ic) else st+pd.Timedelta(hours=72))+pd.Timedelta(hours=12); EP[h_]=(st,en)
def classify2(r):
    h,s=r.hadm_id,r.starttime
    p=pacu[pacu.hadm_id==h]
    if ((s>=p.intime-pd.Timedelta(hours=12*SC_))&(s<=p.outtime+pd.Timedelta(hours=12*SC_))).any(): return "periop_surgery"
    if h in EP and EP[h][0]<=s<=EP[h][1]: return "periop_surgery"
    d=pcid[pcid.hadm_id==h].chartdate
    onpci=((s>=d)&(s<d+pd.Timedelta(hours=30*SC_))).any()
    ic=icu[icu.hadm_id==h]; early=((s>=ic.intime)&(s<=ic.intime+pd.Timedelta(hours=6*SC_))).any()
    if onpci and early: return "peri_pci"
    if onpci: return "pci_day_other"
    return "unplanned"
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
import sys
SC_=float(sys.argv[1]); sup["cls"]="x"; sup["cls2"]=sup.apply(classify2,axis=1)

DEF={"v2":("cls2",{"unplanned","pci_day_other"})}
rows=[]
for h in H:
    a=adm.loc[h]; s=sup[sup.hadm_id==h]
    for dn,(cc,ok) in DEF.items():
        ev=s[s[cc].isin(ok)]; first=ev.starttime.min() if len(ev) else pd.NaT
        firstk={k:(ev[ev.kind==k].starttime.min() if (ev.kind==k).any() else pd.NaT) for k in ["vasopressor","inotrope","mcs","imv"]}
        for hh in range(6,337,6):
            L=a.admittime+pd.Timedelta(hours=hh)
            if L>=a.dischtime or (pd.notna(a.deathtime) and a.deathtime<=L): continue
            if pd.notna(first) and first<=L: continue
            ongoing=((s.starttime<=L)&(s.endtime>L)).any()        # 正在接受任何支持治疗（含计划内）时不预测
            if ongoing: continue
            if dn=="v2" and h in EP and EP[h][0]<=L<=EP[h][1]: continue
            W=(L,L+pd.Timedelta(hours=24)); inw=lambda t:pd.notna(t) and W[0]<t<=W[1]
            death=inw(a.deathtime); circ=any(inw(firstk[k]) for k in ["vasopressor","inotrope","mcs"]); resp=inw(firstk["imv"])
            evt=min([t for t in [a.deathtime,first] if pd.notna(t)],default=pd.NaT)
            rows.append(dict(definition=dn,hadm_id=h,subject_id=int(a.subject_id),h=hh,L=L,event_time=evt,y=int(death or circ or resp),y_death=int(death),y_circ=int(circ),y_resp=int(resp)))
lm=pd.DataFrame(rows); lm[["hadm_id","h","y"]].to_csv(f"full_winsens_{SC_}.csv.gz",index=False)
dev=set(CO[CO.period=="development"].hadm_id)
for dn,g in lm[lm.hadm_id.isin(dev)].groupby("definition"):
    print("DEVELOPMENT ONLY",dn,len(g),g.y.sum(),g[g.y==1].hadm_id.nunique(),flush=True)
