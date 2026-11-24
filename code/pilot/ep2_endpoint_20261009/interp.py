"""AXIS模型的可解释分析：折外的系统贡献分解、风险曲线、系统联动、出事前时间轨迹"""
import numpy as np, json, time, warnings
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.inspection import partial_dependence
warnings.filterwarnings("ignore"); t0=time.time()
exec(open("eval_ep2.py").read().split("up=np.unique(pat); OOF=")[0].replace('SUF=sys.argv[1] if len(sys.argv)>1 else ""','SUF=""'))
idx=allp; Xa,cst=design(idx,"AXIS",False); AXN=axis[idx]
def tree_groups(m):
    labs=[]
    for pr in m._predictors:
        n=pr[0].nodes; f=n["feature_idx"][~n["is_leaf"].astype(bool)]
        a=sorted(set(AXN[f])) if len(f) else []
        labs.append("intercept" if not a else ("×".join(a)))
    return labs
def decompose(m,Xt):
    st=[np.asarray(z).ravel() for z in m.staged_decision_function(Xt)]
    base=m._baseline_prediction.ravel()[0]; prev=np.full(len(Xt),base); out={}
    for lab,s in zip(tree_groups(m),st):
        out[lab]=out.get(lab,0)+(s-prev); prev=s
    return base,out
up=np.unique(pat); rng=np.random.default_rng(1000); s=up.copy(); rng.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}; f=np.array([fm[q] for q in pat])
CONTRIB={}; fold_imp=[]
for k in range(5):
    m=HGB(learning_rate=.05,max_iter=500,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0,early_stopping=True,validation_fraction=.15,n_iter_no_change=30,interaction_cst=cst,random_state=0).fit(Xa[f!=k],y[f!=k])
    base,out=decompose(m,Xa[f==k])
    for g,v in out.items():
        CONTRIB.setdefault(g,np.zeros(len(y)))[f==k]=v
    fold_imp.append({g:float(np.mean(np.abs(v))) for g,v in out.items()})
    print("fold",k,round(time.time()-t0),flush=True)
groups=sorted(CONTRIB,key=lambda g:-np.mean(np.abs(CONTRIB[g])))
imp={g:dict(mean_abs=float(np.mean(np.abs(CONTRIB[g]))),folds=[fi.get(g,0.0) for fi in fold_imp]) for g in groups}
# 时间轨迹：出事患者在事件前各时段的系统贡献（折外），非出事患者为参照
evp=np.isfinite(lead)&(lead>0)
bins=np.arange(0,73,6); traj={}
for g in groups:
    if g=="intercept": continue
    c=CONTRIB[g]; ref=float(np.mean(c[~evp])); rows=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        mk=evp&(lead>lo)&(lead<=hi)
        if mk.sum()>=10:
            v=c[mk]; se=v.std()/np.sqrt(len(v)); rows.append([float(lo),float(hi),float(v.mean()-ref),float(se),int(mk.sum())])
    traj[g]=rows
# 全数据拟合的模型：偏依赖曲线与二维联动
mfull=HGB(learning_rate=.05,max_iter=500,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0,early_stopping=True,validation_fraction=.15,n_iter_no_change=30,interaction_cst=cst,random_state=0).fit(Xa,y)
KEY=["lactate.last","map.last","sbp.last","heart_rate.last","shock_index","sf_ratio","resp_rate.last","creatinine.d_base","urea_nitrogen.last","bicarbonate.last","troponin_t.last","gcs_total","urine_mlkgh_24h","age","white_blood_cells.last","hemoglobin.last","potassium.last","anion_gap"]
nm=[names[i] for i in idx]; pdp={}
for kf in KEY:
    if kf not in nm: continue
    j=nm.index(kf); col=Xa[:,j]; ok=np.isfinite(col)
    grid=np.unique(np.nanpercentile(col,np.linspace(3,97,25)))
    r=partial_dependence(mfull,Xa,[j],custom_values={j:grid},kind="average",method="brute") if False else None
    # 手工PD（logit），只在该特征有值的时点上平均，避免缺失值混入
    sub=Xa[ok][np.random.default_rng(0).choice(ok.sum(),min(4000,ok.sum()),replace=False)].copy(); vals=[]
    for gv in grid:
        sub[:,j]=gv; vals.append(float(np.mean(mfull.decision_function(sub))))
    vals=np.array(vals); pdp[kf]=dict(grid=grid.tolist(),logit=vals.tolist(),coverage=float(ok.mean()),q=np.nanpercentile(col,[5,25,50,75,95]).tolist(),axis=str(AXN[j]))
    print("pd",kf,round(time.time()-t0),flush=True)
PAIRS=[("map.last","lactate.last"),("heart_rate.last","lactate.last"),("sf_ratio","resp_rate.last"),("troponin_t.last","heart_rate.last"),("map.last","creatinine.d_base")]
pd2={}
for a,b in PAIRS:
    ja,jb=nm.index(a),nm.index(b); ok=np.isfinite(Xa[:,ja])&np.isfinite(Xa[:,jb])
    if ok.sum()<300: continue
    ga=np.nanpercentile(Xa[ok,ja],np.linspace(5,95,12)); gb=np.nanpercentile(Xa[ok,jb],np.linspace(5,95,12))
    sub=Xa[ok][np.random.default_rng(1).choice(ok.sum(),min(2000,ok.sum()),replace=False)].copy()
    Z=np.zeros((12,12))
    for i,va in enumerate(ga):
        for jj,vb in enumerate(gb):
            sub[:,ja]=va; sub[:,jb]=vb; Z[i,jj]=np.mean(mfull.decision_function(sub))
    inter=Z-Z.mean(1,keepdims=True)-Z.mean(0,keepdims=True)+Z.mean()
    pd2[f"{a}|{b}"]=dict(ga=ga.tolist(),gb=gb.tolist(),Z=Z.tolist(),inter=inter.tolist(),n=int(ok.sum()))
    print("pd2",a,b,round(time.time()-t0),flush=True)
json.dump(dict(importance=imp,trajectory=traj,pdp=pdp,pd2=pd2,n_event_landmarks=int(evp.sum())),open("INTERP.json","w"),indent=1,ensure_ascii=False)
np.savez_compressed("interp_contrib.npz",**{g.replace("×","_x_"):v for g,v in CONTRIB.items()})
print("DONE",round(time.time()-t0)); print([(g,round(imp[g]["mean_abs"],3)) for g in groups])
