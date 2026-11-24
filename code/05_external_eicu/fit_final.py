"""Amendment 5: final models refitted on all MIMIC-IV (frozen hyperparameters), plus development-only models, saved for eICU scoring."""
import pickle, ast
exec(open("full_common.py").read())
P=json.load(open("heldout_params.json")); ALL=np.concatenate([DEV,LATE])
OUT={"names":names,"axis":list(axis),"AXL":AXL,"FA":FA,"PAIRS":PAIRS}
for tag,tr in [("final",ALL),("dev",DEV)]:
    for a in ["AXIS","FULL"]:
        g=ast.literal_eval(P[a]); z_,th,models=bag(CST[a],tr,tr[:10],777,g,keep_models=True)
        OUT[f"{tag}_{a}"]=dict(models=models,thr=th); log(tag,a,"thr",th)
    # LR with stored transform
    a_,b_=split_val(tr,777*1000+1); med=np.nanmedian(X[a_],0); med=np.where(np.isfinite(med),med,0); mc=np.where(np.isnan(X[a_]).any(0))[0]
    F=lambda M:np.column_stack([np.where(np.isnan(M),med,M),np.isnan(M[:,mc]).astype(np.float32)]); Za=F(X[a_]); mu=Za.mean(0); sd=Za.std(0)+1e-6
    T=lambda M:np.clip((F(M)-mu)/sd,-8,8); C=P["LR"]
    m=LogisticRegression(C=C,max_iter=3000).fit(T(X[a_]),y[a_]); cal=LogisticRegression(C=1e6).fit(m.decision_function(T(X[b_]))[:,None],y[b_])
    A_,B_=float(cal.coef_[0,0]),float(cal.intercept_[0]); th=float(np.quantile(A_*m.decision_function(T(X[b_]))+B_,0.9))
    OUT[f"{tag}_LR"]=dict(model=m,med=med,mc=mc,mu=mu,sd=sd,A=A_,B=B_,thr=th); log(tag,"LR thr",th)
    pickle.dump(OUT,open("final_models.pkl","wb"))
# sanity: dev AXIS reproduces frozen held-out scores
z=np.load("heldout_scores.npz"); d=OUT["dev_AXIS"]["models"]
zz=np.mean([A_*m.decision_function(X[LATE])+B_ for m,A_,B_ in d],0); log("max diff vs frozen heldout AXIS",float(np.max(np.abs(zz-z["AXIS"][LATE]))))
log("DONE")
