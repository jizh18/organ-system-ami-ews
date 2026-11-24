"""Amendment 3 analyses (post hoc). Writes SUPP_A3.json."""
import os
exec(open("full_common.py").read())
H=np.load("heldout_scores.npz"); m=np.zeros(len(y),bool); m[LATE]=True; md=np.zeros(len(y),bool); md[DEV]=True
P={a:sig(H[a]) for a in ["AXIS","WITHIN","FULL","ADD","LR"]}; P["HYBRID"]=mix(P["LR"],P["AXIS"]); P.update(SC)
dev=[np.load(f"cv_rep{r}.npz") for r in range(3)]
D={a:np.mean([sig(d[a]) for d in dev],0) for a in ["AXIS","WITHIN","FULL","ADD","LR"]}; D["HYBRID"]=mix(D["LR"],D["AXIS"]); D.update(SC)
lm=pd.read_csv("full_landmarks.csv.gz",parse_dates=["L","event_time"]); lm=lm[lm.definition=="v2"]
key=pd.DataFrame({"hadm_id":hadm,"h":hh}); lab=key.merge(lm,on=["hadm_id","h"],how="left")
assert (lab.y.values==y).all()
yd,yc,yr=lab.y_death.values.astype(bool),lab.y_circ.values.astype(bool),lab.y_resp.values.astype(bool)
px=pd.read_csv(U+"procedures_icd.csv.gz",dtype={"icd_code":str}); c=px.icd_code.str.replace(".","",regex=False)
surg_h=set(px[((px.icd_version==10)&(c.str.match(r"^021|^02R|^02Q|^02U")))|((px.icd_version==9)&(c.str.match(r"^361|^35[0-2]")))].hadm_id)
pci_h=set(px[((px.icd_version==10)&(c.str.match(r"^027|^4A023N|^B21")))|((px.icd_version==9)&(c.str.match(r"^0066|^3606|^3607|^372[2-3]|^8855|^8856|^8857")))].hadm_id)
surg=np.isin(hadm,list(surg_h)); pci=np.isin(hadm,list(pci_h))
R={}
r3=lambda v:round(float(v),3)
# ---------- S1/S2: structure refits
def ev(z):
    p=sig(z); full=np.full(len(y),np.nan); full[LATE]=p; return full
S={}
for f in sorted(os.listdir("supp")):
    if f.endswith(".npy"): S[f[:-4]]=ev(np.load("supp/"+f))
if "axis_P0" in S: R["repro_maxdiff_vs_primary"]=float(np.nanmax(np.abs(lgt(S["axis_P0"][m])-H["AXIS"][m])))
R["structure"]={k:{"AUROC":r3(AUC(y[m],v[m])),"AP":r3(AP(y[m],v[m]))} for k,v in S.items()}
R["structure_cmp"]={k:boot(P["AXIS"],v,m,B=500) for k,v in S.items() if not k.startswith("rand")}
rk=[k for k in S if k.startswith("rand")]
if rk:
    ra=np.array([AP(y[m],S[k][m]) for k in rk]); ru=np.array([AUC(y[m],S[k][m]) for k in rk])
    R["random"]={"n":len(rk),"AP_median":r3(np.median(ra)),"AP_5_95":[r3(np.percentile(ra,5)),r3(np.percentile(ra,95))],"AP_min_max":[r3(ra.min()),r3(ra.max())],
                 "AUROC_median":r3(np.median(ru)),"AUROC_5_95":[r3(np.percentile(ru,5)),r3(np.percentile(ru,95))],
                 "AXIS_AP_rank_pct":r3(np.mean(ra<AP(y[m],P["AXIS"][m]))*100),"AXIS_AUROC_rank_pct":r3(np.mean(ru<AUC(y[m],P["AXIS"][m]))*100)}
# ---------- S4: period comparison
def comp(mask):
    pos=mask&(y==1); return {"pos":int(pos.sum()),"death_pct":r3(yd[pos].mean()*100),"circ_pct":r3(yc[pos].mean()*100),"resp_pct":r3(yr[pos].mean()*100),
        "event_patients":int(len(np.unique(hadm[pos]))),"assess_ICU_pct":r3(ICU[mask].mean()*100),"assess_surgadm_pct":r3(surg[mask].mean()*100),"prev_pct":r3(y[mask].mean()*100)}
R["composition"]={"development":comp(md),"later":comp(m)}
day=np.select([hh<48,hh<96],[0,1],2); stratum=ICU.astype(int)*6+surg.astype(int)*3+day
ARMS=["AXIS","LR","HYBRID","FULL","GRACE","NEWS2"]
R["strata_auc"]={}
for nm,f in [("ICU",ICU),("ward",~ICU),("surgical_adm",surg),("nonsurgical_adm",~surg),("ICU_nonsurg",ICU&~surg),("ward_nonsurg",~ICU&~surg)]:
    R["strata_auc"][nm]={"dev_pos":int(y[md&f].sum()),"later_pos":int(y[m&f].sum()),**{a:[r3(AUC(y[md&f],D[a][md&f])),r3(AUC(y[m&f],P[a][m&f]))] for a in ARMS}}
pdv=np.bincount(stratum[md],minlength=12)/md.sum(); plt_=np.bincount(stratum[m],minlength=12)/m.sum()
w=np.where(plt_[stratum]>0,pdv[stratum]/np.maximum(plt_[stratum],1e-12),0)
R["reweighted_later_AUROC"]={a:{"dev":r3(AUC(y[md],D[a][md])),"later":r3(AUC(y[m],P[a][m])),"later_reweighted":r3(AUC(y[m],P[a][m],sample_weight=w[m]))} for a in ARMS}
# ---------- S5: alert burden
evh=np.unique(hadm[m&(y==1)]); order=np.lexsort((hh,hadm))
def burden(al):
    det=np.mean([(al&(hadm==q)&(y==1)).any() for q in evh])
    return {"alert_rate_pct":r3(al[m].mean()*100),"episodes_pct":r3(det*100),"PPV_pct":r3(y[m&al].mean()*100) if (m&al).any() else None,
            "patients_alerted_pct":r3(np.mean([al[hadm==q].any() for q in np.unique(hadm[m])])*100),"alerts_per_100pd":r3(al[m].sum()/(m.sum()/4)*100)}
devq=lambda a:float(np.quantile(D[a][md],0.9))
ax_al=m&(P["AXIS"]>=devq("AXIS")); target=np.mean([(ax_al&(hadm==q)&(y==1)).any() for q in evh])
R["burden"]={"AXIS_dev_threshold":burden(ax_al)}
for a in ["NEWS2","GRACE","LR"]:
    best=None
    for q in np.arange(0.995,0.40,-0.005):
        al=m&(P[a]>=np.quantile(P[a][m],q)); s=np.mean([(al&(hadm==h_)&(y==1)).any() for h_ in evh])
        if s>=target-1e-9: best=burden(al); best["quantile"]=r3(q); break
    R["burden"][f"{a}_matched_episode_sens"]=best
for t in [5,7]: R["burden"][f"NEWS2>={t}"]=burden(m&(SC["NEWS2"]>=t))
# near misses
evt=lab.event_time.values; Lt=lab.L.values
dt=(evt-Lt)/np.timedelta64(1,"h")
adm=pd.read_csv(U+"admissions.csv.gz",usecols=["hadm_id","deathtime"]).set_index("hadm_id"); died=np.isin(hadm,adm[adm.deathtime.notna()].index.values)
def nm_(al):
    fp=al&m&(y==0); return {"false_alerts":int(fp.sum()),"event_24_72h_pct":r3(np.mean((dt[fp]>24)&(dt[fp]<=72))*100),"died_in_hospital_pct":r3(died[fp].mean()*100),"later_event_any_pct":r3(np.mean(np.isfinite(dt[fp])&(dt[fp]>24))*100)}
nonal=m&~ax_al&(y==0)
R["near_miss"]={"AXIS_false_alerts":nm_(ax_al),"LR_false_alerts":nm_(m&(P["LR"]>=devq("LR"))),"non_alerted_reference":{"event_24_72h_pct":r3(np.mean((dt[nonal]>24)&(dt[nonal]<=72))*100),"died_in_hospital_pct":r3(died[nonal].mean()*100)}}
# ---------- S6: treatment-independent checks
R["death_only"]={a:{"AUROC":r3(AUC(yd[m],P[a][m])),"AP":r3(AP(yd[m],P[a][m])),"pos":int(yd[m].sum())} for a in ARMS}
R["death_only_dev"]={a:{"AUROC":r3(AUC(yd[md],D[a][md])),"AP":r3(AP(yd[md],D[a][md])),"pos":int(yd[md].sum())} for a in ARMS}
npc=m&~surg&~pci
R["no_procedure_adm"]={"pos":int(y[npc].sum()),"n":int(npc.sum()),**{a:{"AUROC":r3(AUC(y[npc],P[a][npc])),"AP":r3(AP(y[npc],P[a][npc]))} for a in ARMS}}
R["no_procedure_cmp"]={"AXIS-GRACE":boot(P["GRACE"],P["AXIS"],npc,B=500),"AXIS-LR":boot(P["LR"],P["AXIS"],npc,B=500)}
if "axis_imputed" in S: R["imputed_cmp"]=boot(P["AXIS"],S["axis_imputed"],m,B=500)
# ---------- S3: event-type specificity of contributions
G_=[k for k in H.files if k.startswith("C_") and k!="C_intercept"]; name=lambda k:k[2:].replace("_x_","×"); li=np.where(m)[0]
evset=set(evh); non=~np.isin(hadm[li],list(evset)); icu_l=ICU[li]; dd=np.minimum(hh[li]//24,7)
pos=y[li]==1; grp=np.where(yr[li],"resp",np.where(yc[li],"circ_only",np.where(yd[li],"death_only","none")))
S3={"n":{g:int((pos&(grp==g)).sum()) for g in ["resp","circ_only","death_only"]},"patients":{g:int(len(np.unique(hadm[li][pos&(grp==g)]))) for g in ["resp","circ_only","death_only"]},"terms":{}}
rng=np.random.default_rng(3)
for k in G_:
    cc=H[k]; ref=np.zeros(len(li))
    for a_ in [True,False]:
        for d_ in range(8):
            sel=non&(icu_l==a_)&(dd==d_); ref[(icu_l==a_)&(dd==d_)]=cc[sel].mean() if sel.sum()>0 else cc[non].mean()
    v=cc-ref; out={g:r3(np.exp(v[pos&(grp==g)].mean())) for g in ["resp","circ_only","death_only"]}
    # patient-cluster bootstrap of resp minus circ_only (log-OR)
    pr=np.unique(hadm[li][pos&(grp=="resp")]); pc=np.unique(hadm[li][pos&(grp=="circ_only")])
    ir={q:np.where(pos&(grp=="resp")&(hadm[li]==q))[0] for q in pr}; ic={q:np.where(pos&(grp=="circ_only")&(hadm[li]==q))[0] for q in pc}
    bs=[]
    for _ in range(500):
        a1=np.concatenate([ir[q] for q in rng.choice(pr,len(pr))]); a2=np.concatenate([ic[q] for q in rng.choice(pc,len(pc))]); bs.append(v[a1].mean()-v[a2].mean())
    out["resp_minus_circ_logOR"]=r3(v[pos&(grp=="resp")].mean()-v[pos&(grp=="circ_only")].mean()); out["ci"]=[r3(np.percentile(bs,2.5)),r3(np.percentile(bs,97.5))]
    S3["terms"][name(k)]=out
R["S3_event_type"]=S3
json.dump(R,open("SUPP_A3.json","w"),indent=1); print(json.dumps(R,indent=1)[:12000])
