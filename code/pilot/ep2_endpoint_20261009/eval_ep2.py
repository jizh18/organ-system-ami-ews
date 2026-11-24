"""EP2 评价（事前锁定，见EP2_EVAL_PROTOCOL.md）"""
import numpy as np, json, time, warnings, sys
SUF=sys.argv[1] if len(sys.argv)>1 else ""
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, brier_score_loss
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore"); t0=time.time()
z=np.load(f"ep2_features{SUF}.npz"); meta=json.load(open(f"ep2_feature_meta{SUF}.json"))
X,XP,y,pat,hadm,lead=z["X"],z["XP"],z["y"],z["patient"],z["hadm"],z["lead"]; names,axis=meta["names"],np.array(meta["axis"])
keep=[j for j in range(X.shape[1]) if np.isfinite(X[:,j]).sum()>=200 and len(np.unique(X[np.isfinite(X[:,j]),j]))>=2]   # 技术修正：去掉有值时点<200或常数的列（均为罕见化验的变化量）
X=X[:,keep]; names=[names[j] for j in keep]; axis=axis[keep]
N=len(names); OLD59=set(range(N))
concept=lambda n:n.split(".")[0] if "." in n else n
ecg=[i for i in range(N) if axis[i]=="ELECTRICAL"]
newf={"shock_index","anion_gap","sf_ratio","gcs_total","urine_mlkgh_6h","urine_mlkgh_24h","map"}
old_idx=[i for i in range(N) if "." in names[i] and concept(names[i])!="map"]   # 原59概念（含新的5类数值特征）
def groups(idx,mode):
    idx=list(idx); ax=sorted(set(axis[idx]))
    if mode=="ADD":
        cs={}
        for i in idx: cs.setdefault(concept(names[i]),[]).append(i)
        return list(cs.values())
    G={a:[i for i in idx if axis[i]==a] for a in ax}
    out=[G[a] for a in ax]
    if mode=="AXIS":
        for a,b in [("INJURY","HEMO"),("HEMO","PERFUSION"),("PERFUSION","RESP"),("HEMO","RESP"),("HEMO","RENAL"),("PERFUSION","RENAL"),("ELECTRICAL","HEMO"),("NEURO","PERFUSION")]:
            if a in G and b in G: out.append(G[a]+G[b])
    return out
allp=list(range(N)); noecg=[i for i in allp if i not in ecg]
ARMS={"ADD":(allp,"ADD",False),"AXIS":(allp,"AXIS",False),"FULL":(allp,None,False),
      "AXIS_noECG":(noecg,"AXIS",False),"AXIS_old59":(old_idx,"AXIS",False),"AXIS+PROC":(allp,"AXIS",True)}
if SUF: ARMS={k:ARMS[k] for k in ["ADD","AXIS","FULL"]}
def design(idx,mode,proc):
    Xa=X[:,idx]; cst=None if mode is None else [[idx.index(i) for i in g] for g in groups(idx,mode)]
    if proc:
        Xa=np.column_stack([Xa,XP]); 
        if cst is not None: cst=cst+[list(range(len(idx),len(idx)+XP.shape[1]))]
    return Xa,cst
def make_model(cst,seed):
    return HGB(learning_rate=.05,max_iter=500,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0,early_stopping=True,
          n_iter_no_change=30,interaction_cst=cst,random_state=seed)
def grouped_split(tr,seed):
    # 修正：早停验证集按患者划分（15%患者），避免同一患者的时点同时出现在训练和验证中
    ti=np.where(tr)[0]; up_=np.unique(pat[ti]); r=np.random.default_rng(seed+77); v=set(r.choice(up_,max(1,int(.15*len(up_))),replace=False))
    isv=np.array([q in v for q in pat[ti]]); return ti[~isv],ti[isv]
def fit_model(Xa,cst,tr,seed):
    a,b=grouped_split(tr,seed); m=make_model(cst,seed).fit(Xa[a],y[a],X_val=Xa[b],y_val=y[b])
    zv=m.decision_function(Xa[b]); cal=LogisticRegression(C=1e6).fit(zv[:,None],y[b])   # Platt再校准，只用验证患者
    return m,cal
def fit_pred(Xa,cst,tr,te,seed):
    m,cal=fit_model(Xa,cst,tr,seed)
    return cal.predict_proba(m.decision_function(Xa[te])[:,None])[:,1]
up=np.unique(pat); OOF={a:np.zeros(len(y)) for a in ARMS}
for rep in range(3):
    rng=np.random.default_rng(1000+rep); s=up.copy(); rng.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}; f=np.array([fm[q] for q in pat])
    for a,(idx,mode,proc) in ARMS.items():
        Xa,cst=design(idx,mode,proc)
        for k in range(5): OOF[a][f==k]+=fit_pred(Xa,cst,f!=k,f==k,rep)/3
        print("rep",rep,a,round(time.time()-t0),flush=True)
np.savez_compressed(f"ep2_oof{SUF}.npz",**OOF)
# ---- 临床评分
def last(n): return X[:,names.index(n)]
def news2():
    rr,sp,t,sbp,hr,gcs=last("resp_rate.last"),last("spo2.last"),last("temperature.last"),last("sbp.last"),last("heart_rate.last"),last("gcs_total")
    fio=XP[:,meta["proc_names"].index("fio2_est")]
    sc=np.zeros(len(y))
    sc+=np.select([rr<=8,rr<=11,rr<=20,rr<=24,rr>24],[3,1,0,2,3],0)*np.isfinite(rr)
    sc+=np.select([sp<=91,sp<=93,sp<=95,sp>95],[3,2,1,0],0)*np.isfinite(sp)
    sc+=2*(fio>0.21)
    sc+=np.select([t<=35,t<=36,t<=38,t<=39,t>39],[3,1,0,1,2],0)*np.isfinite(t)
    sc+=np.select([sbp<=90,sbp<=100,sbp<=110,sbp<=219,sbp>219],[3,2,1,0,3],0)*np.isfinite(sbp)
    sc+=np.select([hr<=40,hr<=50,hr<=90,hr<=110,hr<=130,hr>130],[3,1,0,1,2,3],0)*np.isfinite(hr)
    sc+=3*(np.nan_to_num(gcs,nan=15)<15)
    return sc+np.random.default_rng(0).random(len(y))*1e-6
si=np.nan_to_num(last("shock_index"),nan=0.0)
qsofa=(np.nan_to_num(last("resp_rate.last"))>=22)+(np.nan_to_num(last("sbp.last"),nan=999)<=100)+(np.nan_to_num(last("gcs_total"),nan=15)<15)+np.random.default_rng(1).random(len(y))*1e-6
SCORES={"NEWS2":news2(),"shock_index":si,"qSOFA":qsofa.astype(float)}
# ---- 指标
ev_h=np.unique(hadm[y==1])
def metrics(p,prob=True):
    out=dict(prevalence=float(y.mean()),AP=float(AP(y,p)),AUROC=float(AUC(y,p)))
    out["AP_lift"]=out["AP"]/out["prevalence"]
    for q in (0.05,0.10,0.20):
        thr=np.quantile(p,1-q); al=p>=thr
        det=[];leads=[]
        for h in ev_h:
            m=(hadm==h)&(y==1)
            if (al&m).any(): det.append(1); leads.append(np.nanmax(lead[m&al]))
            else: det.append(0)
        ppv=float(y[al].mean())
        out[f"alert{int(q*100)}"]=dict(event_sensitivity=float(np.mean(det)),landmark_sensitivity=float(al[y==1].mean()),PPV=ppv,NNE=(1/ppv if ppv>0 else None),median_lead_h=float(np.median(leads)) if leads else None)
    if prob:
        pc=np.clip(p,1e-6,1-1e-6); lg=np.log(pc/(1-pc))
        out["Brier"]=float(brier_score_loss(y,pc))
        lr=LogisticRegression(C=1e6).fit(lg[:,None],y); out["cal_slope"]=float(lr.coef_[0,0])
        out["cal_in_the_large"]=float(y.mean()-pc.mean())
        nb={}
        for pt in (0.02,0.05,0.10):
            tp=((pc>=pt)&(y==1)).mean(); fp=((pc>=pt)&(y==0)).mean(); nb[str(pt)]=dict(model=float(tp-fp*pt/(1-pt)),treat_all=float(y.mean()-(1-y.mean())*pt/(1-pt)))
        out["net_benefit"]=nb
    return out
def boot_diff(pa,pb,B=1000,seed=7):
    up=np.unique(pat); idx={q:np.where(pat==q)[0] for q in up}; rng=np.random.default_rng(seed); dA=[];dU=[]
    for _ in range(B):
        r=np.concatenate([idx[q] for q in rng.choice(up,len(up))])
        if y[r].sum()==0: continue
        dA.append(AP(y[r],pb[r])-AP(y[r],pa[r])); dU.append(AUC(y[r],pb[r])-AUC(y[r],pa[r]))
    q=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    return dict(dAP=float(AP(y,pb)-AP(y,pa)),dAP_ci=q(dA),dAUROC=float(AUC(y,pb)-AUC(y,pa)),dAUROC_ci=q(dU))
RES={"n_landmarks":int(len(y)),"n_pos":int(y.sum()),"n_event_patients":int(len(np.unique(pat[y==1]))),"n_patients":int(len(up))}
RES["models"]={a:metrics(p) for a,p in OOF.items()}
RES["scores"]={a:metrics(p,prob=False) for a,p in SCORES.items()}
CMP=[("NEWS2","AXIS"),("shock_index","AXIS"),("FULL","AXIS"),("AXIS_noECG","AXIS"),("AXIS_old59","AXIS"),("ADD","AXIS"),("AXIS","AXIS+PROC")]
allp_={**OOF,**SCORES}
RES["comparisons"]={f"{b}_vs_{a}":boot_diff(allp_[a],allp_[b]) for a,b in CMP if a in allp_ and b in allp_}
RES["components_AXIS_AUROC"]={c:float(AUC(z[c][y==y] if False else z[c],OOF["AXIS"])) for c in ["y_death","y_circ","y_resp"]}
json.dump(RES,open(f"EP2_RESULTS{SUF}.json","w"),indent=1,ensure_ascii=False)
for a,m in {**RES["models"],**RES["scores"]}.items():
    print(f"{a:12s} AP {m['AP']:.3f} (x{m['AP_lift']:.1f}) AUROC {m['AUROC']:.3f} | alert10%: event-sens {m['alert10']['event_sensitivity']:.2f} PPV {m['alert10']['PPV']:.3f} lead {m['alert10']['median_lead_h']}")
for k,v in RES["comparisons"].items(): print(k,{kk:(round(vv,3) if isinstance(vv,float) else [round(x,3) for x in vv]) for kk,vv in v.items()})
print("components",RES["components_AXIS_AUROC"]); print("DONE",round(time.time()-t0))
