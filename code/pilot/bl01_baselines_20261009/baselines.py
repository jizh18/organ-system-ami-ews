"""BL01：对比基线（逻辑回归、MLP神经网络），与EP3同一患者划分、同一验证患者调参与Platt校准。"""
import numpy as np, json, time, itertools, warnings
src=open("ep3.py").read().split("RES={}")[0]; exec(src)
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")
Xall=X[:,allp]
def prep(tr_rows):
    med=np.nanmedian(Xall[tr_rows],0); med=np.where(np.isfinite(med),med,0)
    miss_cols=np.where(np.isnan(Xall[tr_rows]).any(0))[0]
    def f(rows):
        A=Xall[rows]; M=np.isnan(A[:,miss_cols]).astype(np.float32); A=np.where(np.isnan(A),med,A)
        return np.column_stack([A,M])
    Z=f(tr_rows); mu=Z.mean(0); sd=Z.std(0)+1e-6
    return lambda rows: np.clip((f(rows)-mu)/sd,-8,8)
def fit_lr(tr_idx,te_idx,seed):
    a,b=split_val(tr_idx,seed*1000+1); T=prep(a); best=None
    for C in [0.003,0.01,0.03,0.1,0.3]:
        m=LogisticRegression(C=C,max_iter=2000).fit(T(a),y[a]); s=ll(y[b],m.decision_function(T(b)))
        if best is None or s<best[0]: best=(s,C,m)
    m=best[2]; cal=LogisticRegression(C=1e6).fit(m.decision_function(T(b))[:,None],y[b])
    return cal.predict_proba(m.decision_function(T(te_idx))[:,None])[:,1],best[1]
GRID_MLP=[dict(hidden_layer_sizes=h,alpha=al,max_iter=it) for h,al,it in itertools.product([(64,),(128,64)],[1e-3,1e-1],[30,100])]
def fit_mlp(tr_idx,te_idx,seed):
    a,b=split_val(tr_idx,seed*1000+1); T=prep(a); Ta,Tb=T(a),T(b); best=None
    for g in GRID_MLP:
        m=MLPClassifier(random_state=seed,learning_rate_init=1e-3,batch_size=256,early_stopping=False,**g).fit(Ta,y[a])
        p=np.clip(m.predict_proba(Tb)[:,1],1e-6,1-1e-6); s=ll(y[b],np.log(p/(1-p)))
        if best is None or s<best[0]: best=(s,g)
    ps=[]
    for k in range(3):
        a,b=split_val(tr_idx,seed*1000+10+k); T=prep(a)
        m=MLPClassifier(random_state=seed*10+k,learning_rate_init=1e-3,batch_size=256,early_stopping=False,**best[1]).fit(T(a),y[a])
        zb=m.predict_proba(T(b))[:,1]; zb=np.log(np.clip(zb,1e-6,1-1e-6)/np.clip(1-zb,1e-6,1))
        cal=LogisticRegression(C=1e6).fit(zb[:,None],y[b]); zt=m.predict_proba(T(te_idx))[:,1]; zt=np.log(np.clip(zt,1e-6,1-1e-6)/np.clip(1-zt,1e-6,1))
        ps.append(cal.predict_proba(zt[:,None])[:,1])
    return np.mean(ps,0),best[1]
B={"LR":fit_lr,"MLP":fit_mlp}
up=np.unique(pat); OOF={a:np.zeros(len(y)) for a in B}; chosen={a:[] for a in B}
for rep in range(3):
    r=np.random.default_rng(1000+rep); s=up.copy(); r.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}; f=np.array([fm[q] for q in pat])
    for k in range(5):
        tr,te=np.where(f!=k)[0],np.where(f==k)[0]
        for a,fn in B.items():
            p,g=fn(tr,te,rep*10+k); OOF[a][te]+=p/3; chosen[a].append(str(g))
        print("rep",rep,"fold",k,round(time.time()-t0),flush=True)
O3=np.load("ep3_oof.npz"); RES={"cv":{a:metrics(p) for a,p in {**OOF,"AXIS":O3["AXIS"],"FULL":O3["FULL"]}.items()},"chosen":chosen}
RES["cv_cmp"]={f"AXIS_vs_{a}":boot(OOF[a],O3["AXIS"]) for a in B}
import pandas as pd
ptab=pd.read_csv("/mnt/user-data/uploads/KG-v1/outputs/experiments/ep2_20261009/patients.csv.gz").set_index("subject_id")
grp=pd.Series(pat).map(ptab.anchor_year_group).values; tr=np.where(grp!="2014 - 2016")[0]; te=np.where(grp=="2014 - 2016")[0]; mk=np.zeros(len(y),bool); mk[te]=True
TP={}
for a,fn in B.items():
    p=np.zeros(len(y)); p[te],_=fn(tr,te,777); TP[a]=p
E3=json.load(open("EP3_RESULTS.json"))
RES["temporal"]={a:metrics(p,mk) for a,p in TP.items()}
np.savez_compressed("bl_oof.npz",**OOF,**{f"temporal_{a}":p for a,p in TP.items()})
json.dump(RES,open("BL01_RESULTS.json","w"),indent=1)
for a,v in RES["cv"].items(): print("cv",a,round(v["AUROC"],3),round(v["AP"],3),round(v["alert10_event_sens"],2))
for a,v in RES["temporal"].items(): print("temporal",a,round(v["AUROC"],3),round(v["AP"],3))
print(RES["cv_cmp"]); print("DONE",round(time.time()-t0))
