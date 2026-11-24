"""Development (v2 endpoint) analysis of CV out-of-fold predictions."""
import sys, os
exec(open("full_common.py").read())
reps=[r for r in range(3) if os.path.exists(f"cv_rep{r}.npz")]
ARMS=["AXIS","WITHIN","FULL","ADD","LR"]
P={a:np.mean([sig(np.load(f"cv_rep{r}.npz")[a]) for r in reps],0) for a in ARMS}
for a in ["AXIS","WITHIN","FULL","ADD"]: P["LR+"+a]=mix(P["LR"],P[a])
for k,v in SC.items(): P[k]=v
m=np.zeros(len(y),bool); m[DEV]=True
R={"reps":reps,"n":int(m.sum()),"pos":int(y[m].sum()),"events":int(len(np.unique(hadm[m][y[m]==1])))}
R["metrics"]={a:metrics(P[a],m,prob=a not in SC) for a in P}
R["per_rep_AUROC"]={a:[float(AUC(y[m],np.load(f"cv_rep{r}.npz")[a][m])) for r in reps] for a in ARMS}
C={"H1 AXIS-FULL":("FULL","AXIS"),"H2 AXIS-LR":("LR","AXIS"),"H3 AXIS-WITHIN":("WITHIN","AXIS"),"WITHIN-ADD":("ADD","WITHIN"),"AXIS-ADD":("ADD","AXIS"),
   "H5 HYBRID(LR+AXIS)-LR":("LR","LR+AXIS"),"H6 LR+AXIS - LR+ADD":("LR+ADD","LR+AXIS"),"H7 LR+AXIS - LR+FULL":("LR+FULL","LR+AXIS"),"LR+WITHIN - LR+ADD":("LR+ADD","LR+WITHIN"),
   "HYBRID-AXIS":("AXIS","LR+AXIS"),"AXIS-GRACE":("GRACE","AXIS"),"AXIS-NEWS2":("NEWS2","AXIS"),"HYBRID-GRACE":("GRACE","LR+AXIS"),"FULL-LR":("LR","FULL")}
R["cmp"]={k:boot(P[a],P[b],m) for k,(a,b) in C.items()}
# subgroups
surg_h=set()
px=pd.read_csv(U+"procedures_icd.csv.gz",dtype={"icd_code":str}); c=px.icd_code.str.replace(".","",regex=False)
surg_h=set(px[((px.icd_version==10)&(c.str.match(r"^021|^02R|^02Q|^02U")))|((px.icd_version==9)&(c.str.match(r"^361|^35[0-2]")))].hadm_id)
surg=np.array([h_ in surg_h for h_ in hadm])
R["sub"]={}
for nm,mk in [("ICU",m&ICU),("ward",m&~ICU),("nonsurgical",m&~surg),("ecg6h",m&ecg6)]:
    R["sub"][nm]={"n":int(mk.sum()),"pos":int(y[mk].sum()),**{a:float(AUC(y[mk],P[a][mk])) for a in ["AXIS","LR","LR+AXIS","FULL","GRACE","NEWS2"]},
                  "AP":{a:float(AP(y[mk],P[a][mk])) for a in ["AXIS","LR","LR+AXIS","GRACE","NEWS2"]}}
R["sub_cmp"]={"ICU AXIS-GRACE":boot(P["GRACE"],P["AXIS"],m&ICU),"ICU AXIS-NEWS2":boot(P["NEWS2"],P["AXIS"],m&ICU),"nonsurg AXIS-GRACE":boot(P["GRACE"],P["AXIS"],m&~surg),"nonsurg HYBRID-LR":boot(P["LR"],P["LR+AXIS"],m&~surg)}
# window sensitivity (development only)
key=pd.DataFrame({"hadm_id":hadm,"h":hh,"i":np.arange(len(y))})
R["window"]={}
for s in ["0.5","1.5"]:
    fn=f"full_winsens_{s}.csv.gz"
    if not os.path.exists(fn): continue
    w=pd.read_csv(fn).merge(key,on=["hadm_id","h"]); w=w[np.isin(w.i,DEV)]; yy=w.y.values; ii=w.i.values
    R["window"][s]={"events":int(w[w.y==1].hadm_id.nunique()),"pos":int(yy.sum()),**{a:float(AUC(yy,P[a][ii])) for a in ["AXIS","LR","LR+AXIS"]}}
json.dump(R,open("FULL_DEV_V2.json","w"),indent=1)
print(R["n"],R["pos"],R["events"],reps)
for a,v in R["metrics"].items(): print(a,round(v["AUROC"],3),round(v["AP"],3),round(v["s10"],2))
for k,v in R["cmp"].items(): print(k,"dAUROC",round(v["dAUROC"],3),[round(x,3) for x in v["dAUROC_ci"]],"dAP",round(v["dAP"],3),[round(x,3) for x in v["dAP_ci"]])
print(json.dumps(R["sub"],indent=0)[:1500]); print({k:(round(v["dAUROC"],3),[round(x,3) for x in v["dAUROC_ci"]],round(v["dAP"],3),[round(x,3) for x in v["dAP_ci"]]) for k,v in R["sub_cmp"].items()}); print(R["window"])
