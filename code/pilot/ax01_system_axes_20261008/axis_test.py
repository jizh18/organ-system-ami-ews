"""AX01：知识定义的系统轴能否把分散的跨指标组合信号收拢（事前锁定，见AX01_PROTOCOL.md）"""
import numpy as np, json, time, warnings, hashlib, itertools
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
warnings.filterwarnings("ignore")
D="/mnt/user-data/uploads/KG-v1/refine-logs/ceiling_test_20261008/extract/"
assert hashlib.sha256(open("axes.json","rb").read()).hexdigest()=="b6f82ab8daa9fd0de587bc0e15a0396435d849c230d60e1530f91f34edc9f972"
CFG=json.load(open("axes.json")); AX=CFG["axes"]; NAMES=list(AX)
ITERS=[25,50,100,200]; K5=5; NB=2000; NR=19
cols=lambda i:set(range(i,354,59)); ICU={354}
def axis_sets(assign):  # assign: dict axis->list of indicators
    U={a:set().union(*[cols(i) for i in ind]) for a,ind in assign.items()}
    within=[U[a] for a in NAMES]+[ICU]
    kg=within+[U[a]|U[b] for a,b,_ in CFG["knowledge_axis_pairs"]]
    allp=within+[U[a]|U[b] for a,b in itertools.combinations(NAMES,2)]
    return within,kg,allp
rate=np.load("present_rate.npy"); order=np.argsort(rate); strata=np.array_split(order,6)
def rand_assign(rng):
    m=np.arange(59)
    for s in strata: m[s]=rng.permutation(s)
    return {a:[int(m[i]) for i in ind] for a,ind in AX.items()}
W,KG,ALLP=axis_sets(AX)
rng=np.random.default_rng(20261008); RAND=[axis_sets(rand_assign(rng))[1] for _ in range(NR)]
arms={"ADD":[cols(i) for i in range(59)]+[ICU],"FULL":None,"WITHIN":W,"AXIS_KG":KG,"AXIS_ALLPAIRS":ALLP}
arms.update({f"RAND_{r}":c for r,c in enumerate(RAND)})
def w_pat(p):
    u,inv,c=np.unique(p,return_inverse=True,return_counts=True); return 1.0/c[inv]
nll_rows=lambda y,z:np.logaddexp(0,z)-y*z
def summ(y,p,z):
    w=w_pat(p); return dict(NLL=float((nll_rows(y,z)*w).sum()/w.sum()),AP=float(AP(y,z)),AUROC=float(AUC(y,z)))
def fit_staged(Xtr,ytr,wtr,Xte,cst,iters=ITERS):
    m=HGB(learning_rate=.1,max_iter=max(iters),max_leaf_nodes=8,min_samples_leaf=30,l2_regularization=1.0,
          early_stopping=False,interaction_cst=[sorted(s) for s in cst] if cst else None,random_state=0).fit(Xtr,ytr,sample_weight=wtr)
    return {i:np.asarray(z).ravel().copy() for i,z in enumerate(m.staged_decision_function(Xte),1) if i in iters}
def boot(y,p,za,zb,seed=1):
    up,inv=np.unique(p,return_inverse=True); idx=[np.where(inv==i)[0] for i in range(len(up))]
    w=w_pat(p); pa=np.bincount(inv,nll_rows(y,za)*w); pb=np.bincount(inv,nll_rows(y,zb)*w)
    rng=np.random.default_rng(seed); dn=[];da=[]
    for _ in range(NB):
        s=rng.integers(0,len(up),len(up)); dn.append((pb[s]-pa[s]).mean())
        r=np.concatenate([idx[j] for j in s]); da.append(AP(y[r],zb[r])-AP(y[r],za[r]))
    q=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    return dict(dAP=float(AP(y,zb)-AP(y,za)),dAP_ci=q(da),dNLL=float((pb-pa).mean()),dNLL_ci=q(dn))
R={}
for fo in range(3):
    t0=time.time(); d=np.load(D+f"fold_{fo}.npz")
    X=np.column_stack([d["shared"],d["icu"]]).astype(np.float32); tr=d["role"]==1
    Xt,yt,pt=X[tr],d["label"][tr].astype(int),d["patient"][tr]
    up=np.unique(pt); rr=np.random.default_rng(0); rr.shuffle(up)
    fmap={q:i%K5 for i,q in enumerate(up)}; f=np.array([fmap[q] for q in pt])
    Z={a:{i:np.zeros(len(yt)) for i in ITERS} for a in arms}
    for k in range(K5):
        a_,b_=f!=k,f==k; wa=w_pat(pt[a_])
        for a,c in arms.items():
            o=fit_staged(Xt[a_],yt[a_],wa,Xt[b_],c)
            for i in ITERS: Z[a][i][b_]=o[i]
        print(fo,"inner",k,round(time.time()-t0),flush=True)
    best={a:min(ITERS,key=lambda i:summ(yt,pt,Z[a][i])["NLL"]) for a in arms}
    Zs={a:Z[a][best[a]] for a in arms}
    res={"best_iter":best,"oof":{a:summ(yt,pt,z) for a,z in Zs.items()},
         "cmp":{f"{b}-ADD":boot(yt,pt,Zs["ADD"],Zs[b]) for b in ["WITHIN","AXIS_KG","AXIS_ALLPAIRS","FULL"]}}
    res["cmp"]["FULL-AXIS_KG"]=boot(yt,pt,Zs["AXIS_KG"],Zs["FULL"])
    R[f"fold_{fo}"]=res; json.dump(R,open("AX01_RESULTS.json","w"),indent=1)
    o=res["oof"]; print(fo,{a:(round(o[a]["AP"],4),round(o[a]["NLL"],5)) for a in ["ADD","WITHIN","AXIS_KG","AXIS_ALLPAIRS","FULL"]},
          "RAND AP med",round(float(np.median([o[f"RAND_{r}"]["AP"] for r in range(NR)])),4),
          "AXIS_KG NLL rank",1+sum(o[f"RAND_{r}"]["NLL"]<=o["AXIS_KG"]["NLL"] for r in range(NR)),round(time.time()-t0),flush=True)
F=[R[f"fold_{fo}"] for fo in range(3)]
pool=lambda a,m:float(np.mean([r["oof"][a][m] for r in F]))
S={a:{m:pool(a,m) for m in ["NLL","AP","AUROC"]} for a in arms}
rn=[S[f"RAND_{r}"]["NLL"] for r in range(NR)]; ra=[S[f"RAND_{r}"]["AP"] for r in range(NR)]
rec=lambda a,m:(S[a][m]-S["ADD"][m])/(S["FULL"][m]-S["ADD"][m])
summary=dict(pooled=S,
 A1_recovery_NLL=rec("AXIS_KG","NLL"),A1_recovery_AP=rec("AXIS_KG","AP"),
 A2_rand_p_NLL=(1+sum(x<=S["AXIS_KG"]["NLL"] for x in rn))/(NR+1),A2_rand_p_AP=(1+sum(x>=S["AXIS_KG"]["AP"] for x in ra))/(NR+1),
 A3_beats_ADD_all_folds=all(r["oof"]["AXIS_KG"]["NLL"]<r["oof"]["ADD"]["NLL"] and r["oof"]["AXIS_KG"]["AP"]>r["oof"]["ADD"]["AP"] for r in F),
 recovery_WITHIN_AP=rec("WITHIN","AP"),recovery_ALLPAIRS_AP=rec("AXIS_ALLPAIRS","AP"))
summary["PASS"]=bool(summary["A1_recovery_NLL"]>=.6 and summary["A1_recovery_AP"]>=.6 and summary["A2_rand_p_NLL"]<=.05 and summary["A3_beats_ADD_all_folds"])
R["summary"]=summary; json.dump(R,open("AX01_RESULTS.json","w"),indent=1); print(json.dumps(summary,indent=1,ensure_ascii=False)); print("DONE")
