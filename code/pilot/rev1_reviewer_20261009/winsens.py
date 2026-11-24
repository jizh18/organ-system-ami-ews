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

import sys
RES={}
for scale in [0.5,1.0,1.5]:
    WP,WS,WC,WI=12*scale,36*scale,30*scale,6*scale
    def classify(r):
        h,s=r.hadm_id,r.starttime
        p=pacu[pacu.hadm_id==h]
        if ((s>=p.intime-pd.Timedelta(hours=WP))&(s<=p.outtime+pd.Timedelta(hours=WP))).any(): return "periop_surgery"
        d=surgd[surgd.hadm_id==h].chartdate
        if ((s>=d)&(s<d+pd.Timedelta(hours=WS))).any(): return "periop_surgery"
        d=pcid[pcid.hadm_id==h].chartdate
        onpci=((s>=d)&(s<d+pd.Timedelta(hours=WC))).any()
        ic=icu[icu.hadm_id==h]; early=((s>=ic.intime)&(s<=ic.intime+pd.Timedelta(hours=WI))).any()
        if onpci and early: return "peri_pci"
        if onpci: return "pci_day_other"
        return "unplanned"
    
    sup["cls"]=sup.apply(classify,axis=1)
    ok={"unplanned","pci_day_other"}
    rows=[]
    for h in H:
        a=adm.loc[h]; s_=sup[sup.hadm_id==h]; ev=s_[s_.cls.isin(ok)]; first=ev.starttime.min() if len(ev) else pd.NaT
        for hh in range(6,337,6):
            L=a.admittime+pd.Timedelta(hours=hh)
            if L>=a.dischtime or (pd.notna(a.deathtime) and a.deathtime<=L): continue
            if pd.notna(first) and first<=L: continue
            if ((s_.starttime<=L)&(s_.endtime>L)).any(): continue
            W=(L,L+pd.Timedelta(hours=24)); inw=lambda t:pd.notna(t) and W[0]<t<=W[1]
            rows.append((h,hh,int(inw(a.deathtime) or inw(first))))
    d=pd.DataFrame(rows,columns=["hadm_id","h","y"]); d.to_csv(f"winsens_{scale}.csv.gz",index=False)
    print(scale,len(d),d.y.sum(),d[d.y==1].hadm_id.nunique(),flush=True)
