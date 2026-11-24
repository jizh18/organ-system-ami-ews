"""FULL01 development cross-validation: 5-fold patient-grouped CV x 3 repeats within the development period only."""
import os, sys
exec(open("full_common.py").read())
ARMS=["AXIS","WITHIN","FULL","ADD"]
reps=[int(a) for a in sys.argv[1:]] or [0,1,2]
upd=np.unique(pat[DEV])
for rep in reps:
    r=np.random.default_rng(1000+rep); s=upd.copy(); r.shuffle(s); fm={q:i%5 for i,q in enumerate(s)}
    f=np.full(len(y),-1); f[DEV]=[fm[q] for q in pat[DEV]]
    CK=f"cv_rep{rep}_ckpt.npz"
    if os.path.exists(CK):
        c=np.load(CK,allow_pickle=True); Z=c["Z"].item(); TH=c["TH"].item(); CH=c["CH"].item(); done=set(c["done"].tolist())
    else:
        Z={a:np.full(len(y),np.nan) for a in ARMS+["LR"]}; TH={a:np.full(len(y),np.nan) for a in ARMS+["LR"]}; CH={a:[] for a in ARMS+["LR"]}; done=set()
    for k in range(5):
        if k in done: continue
        tr=np.where((f>=0)&(f!=k))[0]; te=np.where(f==k)[0]; seed=rep*10+k
        for a in ARMS:
            g=tune(CST[a],tr,seed); z_,th,_=bag(CST[a],tr,te,seed,g); Z[a][te]=z_; TH[a][te]=th; CH[a].append(g)
        z_,th,C,_=lr_fit(tr,te,seed); Z["LR"][te]=z_; TH["LR"][te]=th; CH["LR"].append(C)
        done.add(k); np.savez(CK,Z=np.array(Z,dtype=object),TH=np.array(TH,dtype=object),CH=np.array(CH,dtype=object),done=np.array(sorted(done)))
        log("rep",rep,"fold",k,{a:round(AUC(y[te],Z[a][te]),3) for a in ARMS+["LR"]})
    np.savez_compressed(f"cv_rep{rep}.npz",**{a:v for a,v in Z.items()},**{"thr_"+a:v for a,v in TH.items()})
    json.dump({a:[str(x) for x in v] for a,v in CH.items()},open(f"cv_rep{rep}_params.json","w"))
log("CV DONE",reps)
