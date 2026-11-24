"""Amendment 5 analyses: score frozen MIMIC-IV models once on eICU and run the prespecified analyses."""
import numpy as np, pandas as pd, json, pickle, warnings, time, itertools
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from scipy.stats import spearmanr
warnings.filterwarnings("ignore"); t0=time.time(); log=lambda *a:print(*a,round(time.time()-t0),flush=True)
M=pickle.load(open("../full/final_models.pkl","rb")); meta=json.load(open("../full/full_feature_meta_v2.json"))
names0=meta["names"]; names=M["names"]; AXL=M["AXL"]; FA=np.array(M["FA"]); axis=np.array(M["axis"])
sig=lambda z:1/(1+np.exp(-z)); lgt=lambda p:np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))
def load(suf=""):
    z=np.load(f"eicu_features{suf}.npz",allow_pickle=True); D={k:z[k] for k in z.files}
    D["Xm"]=D["X"][:,[names0.index(n) for n in names]]; return D
D=load(); y=D["y"]; pat=D["hadm"]; hosp=D["hospital"]; lead=D["lead"]; X0=D["X"]; XP=D["XP"]; Xm=D["Xm"]; n=len(y)
def score_bag(models,X): return np.mean([A_*m.decision_function(X)+B_ for m,A_,B_ in models],0)
def score_lr(d,X):
    F=np.column_stack([np.where(np.isnan(X),d["med"],X),np.isnan(X[:,d["mc"]]).astype(np.float32)]); T=np.clip((F-d["mu"])/d["sd"],-8,8)
    return d["A"]*d["model"].decision_function(T)+d["B"]
def L_(n_,X=X0): return X[:,names0.index(n_)]
def news2(X,XP_):
    rr,sp,t,sbp,hr,gcs=[L_(c,X) for c in ["resp_rate.last","spo2.last","temperature.last","sbp.last","heart_rate.last","gcs_total"]]; fio=XP_[:,meta["proc_names"].index("fio2_est")]; sc=np.zeros(len(X))
    sc+=np.select([rr<=8,rr<=11,rr<=20,rr<=24,rr>24],[3,1,0,2,3],0)*np.isfinite(rr); sc+=np.select([sp<=91,sp<=93,sp<=95,sp>95],[3,2,1,0],0)*np.isfinite(sp)
    sc+=2*(fio>0.21); sc+=np.select([t<=35,t<=36,t<=38,t<=39,t>39],[3,1,0,1,2],0)*np.isfinite(t)
    sc+=np.select([sbp<=90,sbp<=100,sbp<=110,sbp<=219,sbp>219],[3,2,1,0,3],0)*np.isfinite(sbp)
    sc+=np.select([hr<=40,hr<=50,hr<=90,hr<=110,hr<=130,hr>130],[3,1,0,1,2,3],0)*np.isfinite(hr); sc+=3*(np.nan_to_num(gcs,nan=15)<15)
    return sc+np.random.default_rng(0).random(len(X))*1e-6
def grace(X,stemi,tni):
    age=L_("age",X); hr=L_("heart_rate.last",X); sbp=L_("sbp.last",X); cr=L_("creatinine.last",X); sf=L_("sf_ratio",X)
    sc=np.select([age<30,age<40,age<50,age<60,age<70,age<80,age<90,age>=90],[0,8,25,41,58,75,91,100],0)
    sc=sc+np.where(np.isfinite(hr),np.select([hr<50,hr<70,hr<90,hr<110,hr<150,hr<200,hr>=200],[0,3,9,15,24,38,46],9),9)
    sc=sc+np.where(np.isfinite(sbp),np.select([sbp<80,sbp<100,sbp<120,sbp<140,sbp<160,sbp<200,sbp>=200],[58,53,43,34,24,10,0],34),34)
    sc=sc+np.where(np.isfinite(cr),np.select([cr<.4,cr<.8,cr<1.2,cr<1.6,cr<2,cr<4,cr>=4],[1,4,7,10,13,21,28],7),7)
    sc=sc+np.where(np.nan_to_num(sf,nan=999)<315,39,0)
    enz=(np.nan_to_num(L_("troponin_t.last",X))>0.01)|(np.nan_to_num(tni)>0.04)
    sc=sc+28*stemi.astype(float)+14*enz
    return sc+np.random.default_rng(1).random(len(X))*1e-6
def all_scores(D):
    S={"AXIS":score_bag(M["final_AXIS"]["models"],D["Xm"]),"FULL":score_bag(M["final_FULL"]["models"],D["Xm"]),"LR":score_lr(M["final_LR"],D["Xm"]),
       "AXIS_dev":score_bag(M["dev_AXIS"]["models"],D["Xm"]),"GRACE":grace(D["X"],D["stemi"],D["tni"]),"NEWS2":news2(D["X"],D["XP"])}
    return S
S=all_scores(D); log("scored")
R={"n_assess":int(n),"n_pos":int(y.sum()),"n_patients":int(len(np.unique(pat))),"n_episodes":int(len(np.unique(pat[y==1]))),"n_hospitals":int(len(np.unique(hosp))),
   "prevalence":float(y.mean()),"y_death":int(D["y_death"].sum()),"y_circ":int(D["y_circ"].sum()),"y_resp":int(D["y_resp"].sum())}
def det(pp,m,al,yy=None):
    yy=y[m] if yy is None else yy[m]; hd=pat[m]; ld=lead[m]; ev=np.unique(hd[yy==1]); d=[];ls=[]
    for h_ in ev:
        mk=(hd==h_)&(yy==1)
        if (al&mk).any(): d.append(1); ls.append(np.nanmax(ld[mk&al]))
        else: d.append(0)
    return float(np.mean(d)),(float(np.median(ls)) if ls else None)
def metrics(p,m=None,yv=None):
    m=np.ones(n,bool) if m is None else m; yy=(y if yv is None else yv)[m]; pp=p[m]
    o=dict(n=int(m.sum()),pos=int(yy.sum()),AUROC=float(AUC(yy,pp)),AP=float(AP(yy,pp)))
    for q in (0.05,0.10,0.20):
        al=pp>=np.quantile(pp,1-q); s_,l_=det(pp,m,al,yv); o[f"s{int(q*100)}"]=s_; o[f"lead{int(q*100)}"]=l_; o[f"ppv{int(q*100)}"]=float(yy[al].mean())
    return o
def boot_ci(f,m=None,B=1000,seed=7,cluster=None):
    m=np.ones(n,bool) if m is None else m; cl=pat if cluster is None else cluster
    ii=np.where(m)[0]; cc=cl[ii]; uc=np.unique(cc); idx={q:ii[cc==q] for q in uc}; r=np.random.default_rng(seed); out=[]
    for _ in range(B):
        s=np.concatenate([idx[q] for q in r.choice(uc,len(uc))])
        if y[s].sum()==0 or y[s].sum()==len(s): continue
        out.append(f(s))
    out=np.array(out,dtype=float)
    if out.ndim==1: return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]
    return np.percentile(out,[2.5,97.5],axis=0).T.tolist()
def ci1(p,m=None,cluster=None,B=1000):
    v=boot_ci(lambda s:AUC(y[s],p[s]),m,B,cluster=cluster); return v
# ---- discrimination
R["models"]={}
for k,p in S.items():
    o=metrics(p); o["AUROC_ci"]=ci1(p); o["AP_ci"]=boot_ci(lambda s:AP(y[s],p[s])); R["models"][k]=o; log(k,round(o["AUROC"],3),round(o["AP"],3))
R["AXIS_AUROC_hospital_cluster_ci"]=ci1(S["AXIS"],cluster=hosp)
R["comparisons"]={}
for b in ["FULL","LR","GRACE","NEWS2"]:
    pa,pb=S[b],S["AXIS"]
    def f(s): return [AUC(y[s],pb[s])-AUC(y[s],pa[s]),AP(y[s],pb[s])-AP(y[s],pa[s])]
    ci=boot_ci(f)
    R["comparisons"][f"AXIS-{b}"]=dict(dAUROC=float(AUC(y,pb)-AUC(y,pa)),dAUROC_ci=ci[0],dAP=float(AP(y,pb)-AP(y,pa)),dAP_ci=ci[1]); log("cmp",b,R["comparisons"][f"AXIS-{b}"])
# ---- calibration
def calib(z,m=None):
    m=np.ones(n,bool) if m is None else m; zz=z[m]; yy=y[m]; p_=sig(zz)
    slope=float(LogisticRegression(C=1e6).fit(zz[:,None],yy).coef_[0,0])
    a=0.0
    for _ in range(50):
        pp_=sig(zz+a); g=np.sum(yy-pp_); hh=np.sum(pp_*(1-pp_)); a+=g/hh
    citl=float(a)
    return dict(mean_pred=float(p_.mean()),observed=float(yy.mean()),citl=citl,slope=slope,brier=float(brier_score_loss(yy,p_)))
R["calibration_as_is"]={k:calib(S[k]) for k in ["AXIS","FULL","LR","AXIS_dev"]}
uh=np.unique(hosp); r=np.random.default_rng(2026); half=set(r.choice(uh,len(uh)//2,replace=False)); h1=np.array([h in half for h in hosp])
zr=np.empty(n)
for tr_,te_ in [(h1,~h1),(~h1,h1)]:
    lr=LogisticRegression(C=1e6).fit(S["AXIS"][tr_][:,None],y[tr_]); zr[te_]=lr.decision_function(S["AXIS"][te_][:,None])
R["calibration_recalibrated_crossfit"]=calib(zr); R["recal_AUROC"]=float(AUC(y,zr))
# calibration curve (deciles) for figure
def curve(z):
    p_=sig(z); q=pd.qcut(p_,10,labels=False,duplicates="drop"); return [[float(p_[q==i].mean()),float(y[q==i].mean())] for i in np.unique(q)]
R["cal_curve_as_is"]=curve(S["AXIS"]); R["cal_curve_recal"]=curve(zr)
# ---- alerting at frozen thresholds
R["alerting"]={}
for k,key in [("AXIS","final_AXIS"),("FULL","final_FULL"),("LR","final_LR")]:
    al=S[k]>=M[key]["thr"]; s_,l_=det(S[k],np.ones(n,bool),al); R["alerting"][k]=dict(thr=M[key]["thr"],flagged=float(al.mean()),episodes_detected=s_,median_lead_h=l_,ppv=float(y[al].mean()),
        patients_ever_alerted=float(pd.Series(al).groupby(pat).any().mean()))
al=S["AXIS"]>=M["final_AXIS"]["thr"]; nz=~al
fp=al&(y==0)
for thr in (5,7):
    a2=S["NEWS2"]>=thr; s_,l_=det(S["NEWS2"],np.ones(n,bool),a2); R["alerting"][f"NEWS2>={thr}"]=dict(flagged=float(a2.mean()),episodes_detected=s_,median_lead_h=l_,ppv=float(y[a2].mean()),patients_ever_alerted=float(pd.Series(a2).groupby(pat).any().mean()))
target=R["alerting"]["AXIS"]["episodes_detected"]; qs=np.unique(S["NEWS2"].round(6)); best=None
for q in np.linspace(0.005,0.995,199):
    a2=S["NEWS2"]>=np.quantile(S["NEWS2"],1-q); s_,_=det(S["NEWS2"],np.ones(n,bool),a2)
    if s_>=target: best=dict(flagged=float(a2.mean()),episodes_detected=s_); break
R["alerting"]["NEWS2_matched"]=best; log("alerting",R["alerting"])
# ---- hospital heterogeneity
rows=[]
for h_ in uh:
    m=hosp==h_; ep=len(np.unique(pat[m&(y==1)]))
    if ep>=10 and 0<y[m].sum()<m.sum():
        a=AUC(y[m],S["AXIS"][m]); bs=[]; ii=np.where(m)[0]; pp=pat[ii]; up=np.unique(pp); idx={q:ii[pp==q] for q in up}; rg=np.random.default_rng(int(h_))
        for _ in range(300):
            s=np.concatenate([idx[q] for q in rg.choice(up,len(up))])
            if 0<y[s].sum()<len(s): bs.append(lgt(AUC(y[s],S["AXIS"][s])))
        rows.append(dict(hospital=int(h_),episodes=ep,n=int(m.sum()),AUROC=float(a),se_logit=float(np.std(bs))))
HT=pd.DataFrame(rows); th=lgt(HT.AUROC.values); v=HT.se_logit.values**2; w=1/v; mu_f=np.sum(w*th)/w.sum(); Q=np.sum(w*(th-mu_f)**2); k_=len(th)
tau2=max(0,(Q-(k_-1))/(w.sum()-np.sum(w**2)/w.sum())); ws=1/(v+tau2); mu=np.sum(ws*th)/ws.sum(); se=np.sqrt(1/ws.sum())
from scipy.stats import t as tdist
pi=tdist.ppf(.975,k_-2)*np.sqrt(tau2+se**2)
R["hospitals"]=dict(k=int(k_),pooled=float(sig(mu)),pooled_ci=[float(sig(mu-1.96*se)),float(sig(mu+1.96*se))],pred_int=[float(sig(mu-pi)),float(sig(mu+pi))],I2=float(max(0,(Q-(k_-1))/Q)) if Q>0 else 0.0,
    median=float(HT.AUROC.median()),range=[float(HT.AUROC.min()),float(HT.AUROC.max())],table=rows); log("hosp",R["hospitals"]["pooled"],R["hospitals"]["pred_int"])
# ---- subgroups / alternative outcomes
R["subgroups"]={}
for k in ["AXIS","FULL","LR","GRACE","NEWS2"]:
    R["subgroups"].setdefault("death_only",{})[k]=dict(AUROC=float(AUC(D["y_death"],S[k])),AP=float(AP(D["y_death"],S[k])),events=int(D["y_death"].sum()))
noproc=~D["pci"]&~D["surg"]; cic=D["cardiac_icu"]
for nm,m in [("no_PCI_no_surgery",noproc),("cardiac_ICU",cic),("other_ICU",~cic)]:
    R["subgroups"][nm]={k:dict(n=int(m.sum()),pos=int(y[m].sum()),AUROC=float(AUC(y[m],S[k][m])),AP=float(AP(y[m],S[k][m]))) for k in ["AXIS","FULL","LR","GRACE","NEWS2"]}
R["subgroups"]["no_PCI_no_surgery"]["AXIS"]["AUROC_ci"]=ci1(S["AXIS"],noproc)
# ---- sensitivity: without present-on-arrival rule
try:
    D0=load("_poa0"); S0=all_scores(D0); y0=D0["y"]
    R["sens_poa0"]=dict(n=int(len(y0)),pos=int(y0.sum()),episodes=int(len(np.unique(D0["hadm"][y0==1]))),**{k:dict(AUROC=float(AUC(y0,S0[k])),AP=float(AP(y0,S0[k]))) for k in S0})
except Exception as e: R["sens_poa0"]=str(e)
# ---- exact decomposition (dev model, compared with MIMIC held-out; and final model)
def decompose(m,Xt):
    nn=len(Xt); out={}
    for pr in m._predictors:
        nd=pr[0].nodes; node=np.zeros(nn,dtype=np.int64); mask=np.zeros(nn,dtype=np.int64)
        for _ in range(64):
            leaf=nd["is_leaf"][node].astype(bool)
            if leaf.all(): break
            f=nd["feature_idx"][node]; thr=nd["num_threshold"][node]; ml=nd["missing_go_to_left"][node].astype(bool)
            x=Xt[np.arange(nn),f]; left=np.where(np.isnan(x),ml,x<=thr)
            nxt=np.where(left,nd["left"][node],nd["right"][node]); mask=np.where(leaf,mask,mask|(1<<FA[f])); node=np.where(leaf,node,nxt)
        val=nd["value"][node]
        for mk in np.unique(mask):
            sel=mask==mk; lab="×".join(AXL[i] for i in range(len(AXL)) if mk>>i&1) or "intercept"
            out.setdefault(lab,np.zeros(nn))[sel]+=val[sel]
    return out
def terms(models,Xt):
    C={}
    for m,A_,B_ in models:
        o=decompose(m,Xt); base=float(m._baseline_prediction.ravel()[0])
        for g,v in o.items():
            if g=="intercept": continue
            C.setdefault(g,np.zeros(len(Xt))); C[g]+=A_*v/len(models)
        C.setdefault("intercept",np.zeros(len(Xt))); C["intercept"]+=(A_*base+B_)/len(models)
    return C
Cd=terms(M["dev_AXIS"]["models"],Xm); Cf=terms(M["final_AXIS"]["models"],Xm)
R["recon_err_final"]=float(np.max(np.abs(sum(Cf.values())-S["AXIS"])))
hz=np.load("../full/heldout_scores.npz"); mim={k[2:].replace("_x_","×"):np.nanstd(hz[k]) for k in hz.files if k.startswith("C_") and k!="C_intercept"}
ei={k:float(np.std(v)) for k,v in Cd.items() if k!="intercept"}; common=sorted(set(mim)&set(ei))
R["explain"]=dict(spearman_dev_model_eicu_vs_mimic=float(spearmanr([mim[c] for c in common],[ei[c] for c in common])[0]),n_terms=len(common),
    eicu_dev_top=sorted(ei.items(),key=lambda x:-x[1])[:10],mimic_top=sorted(mim.items(),key=lambda x:-x[1])[:10],
    eicu_final_top=sorted({k:float(np.std(v)) for k,v in Cf.items() if k!="intercept"}.items(),key=lambda x:-x[1])[:10])
log("explain",R["explain"]["spearman_dev_model_eicu_vs_mimic"],R["explain"]["eicu_final_top"][:3])
np.savez_compressed("eicu_scores.npz",**S,zr=zr,**{"Cf_"+k.replace("×","_x_"):v for k,v in Cf.items()})
json.dump(R,open("EICU_RESULTS.json","w"),indent=1,default=float); log("main done")
# ---- K2 replication: open graph within eICU, hospital halves
P0=dict(learning_rate=0.03,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0)
G={a:[i for i in range(len(names)) if axis[i]==a] for a in AXL}; pairs=list(itertools.combinations(AXL,2))
CST=[G[a] for a in AXL]+[G[a]+G[b] for a,b in pairs]
KG=set(tuple(sorted(p)) for p in M["PAIRS"])
def fit_bags(tr):
    out=[]
    for k in range(5):
        up=np.unique(pat[tr]); rg=np.random.default_rng(777*1000+10+k); v=set(rg.choice(up,max(1,int(.15*len(up))),replace=False)); isv=np.array([q in v for q in pat[tr]])
        a,b=tr[~isv],tr[isv]
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=CST,random_state=7770+k,**P0).fit(Xm[a],y[a],X_val=Xm[b],y_val=y[b])
        cal=LogisticRegression(C=1e6).fit(m.decision_function(Xm[b])[:,None],y[b]); out.append((m,float(cal.coef_[0,0]),float(cal.intercept_[0])))
    return out
RK=[]; BAGRANKS=[]
for tr_,te_ in [(np.where(h1)[0],np.where(~h1)[0]),(np.where(~h1)[0],np.where(h1)[0])]:
    bags=fit_bags(tr_); C=terms(bags,Xm[te_])
    st=np.array([np.std(C.get("×".join(sorted(p)),np.zeros(len(te_)))) for p in pairs]); RK.append((-st).argsort().argsort()+1)
    for m,A_,B_ in bags:
        Cb=terms([(m,A_,B_)],Xm[te_]); sb=np.array([np.std(Cb.get("×".join(sorted(p)),np.zeros(len(te_)))) for p in pairs]); BAGRANKS.append((-sb).argsort().argsort()+1)
    log("half done, open AUROC",float(AUC(y[te_],score_bag(bags,Xm[te_]))))
rk=np.mean(RK,0); order=rk.argsort().argsort()+1; isk=np.array([tuple(sorted(p)) in KG for p in pairs])
obs=order[isk].mean(); rg=np.random.default_rng(1); null=np.array([rg.choice(order,8,replace=False).mean() for _ in range(100000)])
top8_bags=(np.array(BAGRANKS)<=8).sum(0)
noel=np.array(["ELECTRICAL" not in p for p in pairs]); o2=rk[noel].argsort().argsort()+1; isk2=isk[noel]; null2=np.array([rg.choice(o2,isk2.sum(),replace=False).mean() for _ in range(100000)])
R["K2_eicu"]=dict(mean_rank_kg=float(obs),p_one_sided=float((null<=obs).mean()),
    pairs=sorted([dict(pair="×".join(sorted(p)),kg=bool(isk[i]),rank=int(order[i]),mean_rank_halves=float(rk[i]),top8_in_bags=int(top8_bags[i]),of_bags=len(BAGRANKS)) for i,p in enumerate(pairs)],key=lambda d:d["rank"]),
    posthoc_excluding_electrical=dict(n_pairs=int(noel.sum()),n_kg=int(isk2.sum()),mean_rank_kg=float(o2[isk2].mean()),p_one_sided=float((null2<=o2[isk2].mean()).mean())))
json.dump(R,open("EICU_RESULTS.json","w"),indent=1,default=float); log("ALL DONE")
