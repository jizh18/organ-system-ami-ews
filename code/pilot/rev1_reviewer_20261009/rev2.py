"""REV2 (prespecified before running, 9 Oct 2026 17:05):
Question: at equal accuracy, are organ-system explanations from constrained models (AXIS, WITHIN) more concentrated
and more reproducible than system-level explanations of the unconstrained black box (FULL)?
Attribution (identical for all models): each leaf value is credited to the set S of systems split on along the path the
assessment takes; for system-level vectors, a set term is split equally among its |S| systems (Shapley value of a pure |S|-way term).
Final models: EP3/REV1 chosen hyperparameters, 5 bags, Platt slope applied, logit average (as REV1).
Metrics:
 M1 concentration: share of between-assessment variance of the logit carried by terms with >=3 systems; number of distinct system sets.
 M2 reproducibility across the 3 CV repeats (different training sets): per-assessment Pearson r of the 11-system attribution
    vector between repeat pairs (median over assessments; mean over 3 pairs); top-system agreement among the 10% highest-risk
    assessments; Spearman of global importance; reference: Pearson r of the logits themselves.
 Pre-declared criterion for an "explanation-stability advantage": AXIS minus FULL difference in M2 per-assessment r and in
 top-system agreement both > 0 with patient-bootstrap 95% CI excluding 0.
"""
import numpy as np, json, time, os
src=open("rev1.py").read().split("OUT={}")[0]; exec(src)
from scipy.stats import spearmanr
CHW=json.load(open("rev1_B_chosen.json"))
NS=len(AXL)
def decompose_sets(m,Xt):
    n=len(Xt); out={}
    for pr in m._predictors:
        nd=pr[0].nodes; node=np.zeros(n,dtype=np.int64); mask=np.zeros(n,dtype=np.int64)
        for _ in range(64):
            leaf=nd["is_leaf"][node].astype(bool)
            if leaf.all(): break
            f=nd["feature_idx"][node]; thr=nd["num_threshold"][node]; ml=nd["missing_go_to_left"][node].astype(bool)
            x=Xt[np.arange(n),f]; left=np.where(np.isnan(x),ml,x<=thr)
            nxt=np.where(left,nd["left"][node],nd["right"][node])
            mask=np.where(leaf,mask,mask|(1<<FA[f])); node=np.where(leaf,node,nxt)
        val=nd["value"][node]
        for mk in np.unique(mask):
            sel=mask==mk; out.setdefault(int(mk),np.zeros(n))[sel]+=val[sel]
    return out
def fit_sets(Xa,cst,tr,te,seed,params):
    S={}
    for k in range(NBAG):
        a,b=split_val(tr,seed*1000+10+k)
        m=HGB(max_iter=800,early_stopping=True,n_iter_no_change=40,interaction_cst=cst,random_state=seed*10+k,**params).fit(Xa[a],y[a],X_val=Xa[b],y_val=y[b])
        cal=LogisticRegression(C=1e6).fit(m.decision_function(Xa[b])[:,None],y[b]); A=float(cal.coef_[0,0])
        for mk,v in decompose_sets(m,Xa[te]).items(): S.setdefault(mk,np.zeros(len(te))); S[mk]+=A*v/NBAG
    return S
ARMS={"AXIS":(design(allp,"AXIS",False),lambda i:CH["AXIS"][i]),"FULL":(design(allp,None,False),lambda i:CH["FULL"][i]),"WITHIN":(design(allp,"WITHIN",False),lambda i:CHW[i])}
CK="rev2_ckpt.npz"
if os.path.exists(CK):
    c=np.load(CK,allow_pickle=True); SYS=c["SYS"].item(); SETVAR=c["SETVAR"].item(); done=set(c["done"].tolist())
else: SYS={a:np.zeros((3,len(y),NS)) for a in ARMS}; SETVAR={a:{} for a in ARMS}; done=set()
for rep in range(3):
    f=FOLDS[rep]
    for k in range(5):
        if rep*5+k in done: continue
        tr,te=np.where(f!=k)[0],np.where(f==k)[0]
        for a,((Xa,cst),par) in ARMS.items():
            S=fit_sets(Xa,cst,tr,te,rep*10+k,par(rep*5+k))
            for mk,v in S.items():
                if mk==0: continue
                bits=[i for i in range(NS) if mk>>i&1]
                for i in bits: SYS[a][rep,te,i]+=v/len(bits)
                SETVAR[a].setdefault(mk,np.zeros((3,len(y))))[rep,te]+=v
        done.add(rep*5+k); np.savez(CK,SYS=np.array(SYS,dtype=object),SETVAR=np.array(SETVAR,dtype=object),done=np.array(sorted(done)))
        log("rep",rep,"fold",k)
# ---------------- metrics
R={}
for a in ARMS:
    terms=SETVAR[a]; tot=sum(terms.values())  # (3,n) logit minus intercepts
    var_tot=np.mean([np.var(tot[r]) for r in range(3)])
    big=[mk for mk in terms if bin(mk).count("1")>=3]
    v3=np.mean([np.var(sum(terms[mk][r] for mk in big)) for r in range(3)]) if big else 0.0
    cov3=np.mean([np.cov(sum(terms[mk][r] for mk in big),tot[r])[0,1] for r in range(3)]) if big else 0.0
    nsets=sum(1 for mk in terms if np.mean([np.std(terms[mk][r]) for r in range(3)])>0.01)
    R[a]={"n_system_sets":len(terms),"n_sets_sd_gt_0.01":nsets,"share_var_ge3_systems":float(cov3/var_tot),"max_systems_on_path":int(max(bin(mk).count("1") for mk in terms))}
pairs=[(0,1),(0,2),(1,2)]
def per_assess_r(Z,i,j):
    A_=Z[i]-Z[i].mean(1,keepdims=True); B_=Z[j]-Z[j].mean(1,keepdims=True)
    return (A_*B_).sum(1)/np.sqrt((A_**2).sum(1)*(B_**2).sum(1)+1e-12)
logit=lambda a,r:SYS[a][r].sum(1)
def stab(a,idx=None):
    idx=np.arange(len(y)) if idx is None else idx; Z=SYS[a][:,idx,:]
    pr=np.mean([np.median(per_assess_r(Z,i,j)) for i,j in pairs])
    lg=np.mean([np.corrcoef(Z[i].sum(1),Z[j].sum(1))[0,1] for i,j in pairs])
    hi=np.where(Z.sum(2).mean(0)>=np.quantile(Z.sum(2).mean(0),0.9))[0]
    top=np.mean([np.mean(np.argmax(Z[i][hi],1)==np.argmax(Z[j][hi],1)) for i,j in pairs])
    gi=np.mean([spearmanr(Z[i].std(0),Z[j].std(0)).correlation for i,j in pairs])
    return dict(per_assessment_r=float(pr),top_system_agreement_top10pct=float(top),global_rank_spearman=float(gi),logit_r=float(lg))
for a in ARMS: R[a].update(stab(a))
up_=np.unique(pat); idxp={q:np.where(pat==q)[0] for q in up_}; rng=np.random.default_rng(5); B=[]
for _ in range(300):
    s=np.concatenate([idxp[q] for q in rng.choice(up_,len(up_))]); sa=stab("AXIS",s); sf=stab("FULL",s); sw=stab("WITHIN",s)
    B.append([sa["per_assessment_r"]-sf["per_assessment_r"],sa["top_system_agreement_top10pct"]-sf["top_system_agreement_top10pct"],sw["per_assessment_r"]-sf["per_assessment_r"],sw["top_system_agreement_top10pct"]-sf["top_system_agreement_top10pct"],sa["per_assessment_r"]-sw["per_assessment_r"]])
B=np.array(B); q=lambda c:[float(np.percentile(B[:,c],2.5)),float(np.percentile(B[:,c],97.5))]
R["diff"]={"AXIS-FULL_per_assessment_r":q(0),"AXIS-FULL_top_agreement":q(1),"WITHIN-FULL_per_assessment_r":q(2),"WITHIN-FULL_top_agreement":q(3),"AXIS-WITHIN_per_assessment_r":q(4)}
R["criterion_met_AXIS"]=bool(q(0)[0]>0 and q(1)[0]>0)
json.dump(R,open("REV2_RESULTS.json","w"),indent=1); print(json.dumps(R,indent=1)); log("DONE")
