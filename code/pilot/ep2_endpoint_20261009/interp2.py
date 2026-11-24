"""按每个样本在每棵树中的实际路径，把预测logit精确分解到“单系统/系统对”"""
import numpy as np, json, time, warnings
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
warnings.filterwarnings("ignore"); t0=time.time()
exec(open("eval_ep2.py").read().split("up=np.unique(pat); OOF=")[0].replace('SUF=sys.argv[1] if len(sys.argv)>1 else ""','SUF=""'))
idx=allp; Xa,cst=design(idx,"AXIS",False); AXN=np.array(axis)[idx]
AXL=sorted(set(AXN)); aid={a:i for i,a in enumerate(AXL)}; FA=np.array([aid[a] for a in AXN])
def decompose(m,Xt):
    n=len(Xt); out={}
    for pr in m._predictors:
        nd=pr[0].nodes; node=np.zeros(n,dtype=np.int64); mask=np.zeros(n,dtype=np.int64)  # 路径上出现的系统(位掩码)
        for _ in range(64):
            leaf=nd["is_leaf"][node].astype(bool)
            if leaf.all(): break
            f=nd["feature_idx"][node]; thr=nd["num_threshold"][node]; ml=nd["missing_go_to_left"][node].astype(bool)
            x=Xt[np.arange(n),f]; left=np.where(np.isnan(x),ml,x<=thr)
            nxt=np.where(left,nd["left"][node],nd["right"][node])
            mask=np.where(leaf,mask,mask|(1<<FA[f])); node=np.where(leaf,node,nxt)
        val=nd["value"][node]
        for mk in np.unique(mask):
            sel=mask==mk; lab="×".join(AXL[i] for i in range(len(AXL)) if mk>>i&1) or "intercept"
            out.setdefault(lab,np.zeros(n))[sel]+=val[sel]
    return out
up=np.unique(pat); rng=np.random.default_rng(1000); s=up.copy(); rng.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}; f=np.array([fm[q] for q in pat])
C={}; check=[]
for k in range(5):
    m,_=fit_model(Xa,cst,f!=k,0)
    te=f==k; out=decompose(m,Xa[te]); base=float(m._baseline_prediction.ravel()[0])
    tot=base+sum(out.values()); check.append(float(np.max(np.abs(tot-m.decision_function(Xa[te])))))
    for g,v in out.items(): C.setdefault(g,np.zeros(len(y)))[te]=v
    print("fold",k,"max|recon err|",check[-1],round(time.time()-t0),flush=True)
groups=sorted(C,key=lambda g:-np.std(C[g]))
imp={g:dict(std=float(np.std(C[g])),mean=float(np.mean(C[g])),n_axes=len(g.split("×"))) for g in groups if g!="intercept"}
lead_=lead; evp=np.isfinite(lead_)&(lead_>0); bins=np.arange(0,73,6); traj={}
for g in groups:
    if g=="intercept": continue
    c=C[g]; ref=float(np.mean(c[~evp])); rows=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        mk=evp&(lead_>lo)&(lead_<=hi)
        if mk.sum()>=10: v=c[mk]; rows.append([float(lo),float(hi),float(v.mean()-ref),float(v.std()/np.sqrt(len(v))),int(mk.sum())])
    traj[g]=rows
J=json.load(open("INTERP.json")); J.update(importance=imp,trajectory=traj,recon_max_err=max(check),n_event_landmarks=int(evp.sum()),n_event_patients_with_lead=int(len(np.unique(pat[evp]))))
json.dump(J,open("INTERP.json","w"),indent=1,ensure_ascii=False)
np.savez_compressed("interp_contrib.npz",**{g.replace("×","_x_"):v for g,v in C.items()})
print("DONE",round(time.time()-t0)); print([(g,round(imp[g]["std"],3)) for g in groups if g!="intercept"])
