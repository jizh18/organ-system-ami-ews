"""Amendment 3 refits (post hoc). Same hyperparameters/seed as the primary organ-system model; only structure or input differs.
Each variant saves held-out logits to supp/<name>.npy (resume-safe)."""
import os, sys
exec(open("full_common.py").read())
os.makedirs("supp", exist_ok=True)
P0=dict(learning_rate=0.03,max_leaf_nodes=8,min_samples_leaf=40,l2_regularization=1.0)
G={a:[i for i in range(N) if axis[i]==a] for a in AXL}
def cst_from(groups,pairs): return [groups[a] for a in groups]+[groups[a]+groups[b] for a,b in pairs if groups[a] and groups[b]]
V={}
V["axis_P0"]=(CST["AXIS"],P0)
V["within_P0"]=(cst_from(G,[]),P0)
V["full_P0"]=(None,P0)
V["full_depth2"]=(None,dict(P0,max_depth=2))
for k,pr in enumerate(PAIRS): V[f"drop_{pr[0]}_{pr[1]}"]=(cst_from(G,[p for p in PAIRS if p!=pr]),P0)
V["all55"]=(cst_from(G,list(itertools.combinations(AXL,2))),P0)
# random concept groupings, sizes matched to real systems
cons=sorted(set(concept(n) for n in names)); cidx={c:[i for i in range(N) if concept(names[i])==c] for c in cons}
sizes=[len(G[a]) for a in AXL]
for r in range(20):
    rng=np.random.default_rng(1000+r); order=rng.permutation(len(cons)); groups={a:[] for a in AXL}; gi=0
    for ci in order:
        while gi<len(AXL)-1 and len(groups[AXL[gi]])>=sizes[gi]: gi+=1
        groups[AXL[gi]]+=cidx[cons[ci]]
    allp=list(itertools.combinations(AXL,2)); pick=[allp[j] for j in rng.choice(len(allp),8,replace=False)]
    V[f"rand{r:02d}"]=(cst_from(groups,pick),P0)
only=sys.argv[1:]
for name,(cst,par) in V.items():
    if only and not any(name.startswith(o) for o in only): continue
    fn=f"supp/{name}.npy"
    if os.path.exists(fn): continue
    z_,th,_=bag(cst,DEV,LATE,777,par); np.save(fn,z_); log(name)
# S6b: missingness masked (median-imputed, no indicators)
fn="supp/axis_imputed.npy"
if (not only or "axis_imputed" in only) and not os.path.exists(fn):
    med=np.nanmedian(X[DEV],0); med=np.where(np.isfinite(med),med,0); X[:]=np.where(np.isnan(X),med,X)
    z_,th,_=bag(CST["AXIS"],DEV,LATE,777,P0); np.save(fn,z_); log("axis_imputed")
log("SUPP DONE")
