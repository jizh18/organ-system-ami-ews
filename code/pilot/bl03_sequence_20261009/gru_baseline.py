"""BL03: GRU sequence baseline. Same patient folds (3 reps x 5), same inner 15% validation patients for
early stopping / model selection, Platt recalibration, 3-seed bagging; plus temporal split (train 2008-2013, test 2014-2016).
Input: bl03_input.npz (exported from the cloud analysis). Output: BL03_RESULTS.json, bl03_oof.npz, bl03_log.txt.
Run:  python gru_baseline.py
"""
import numpy as np, json, time, itertools, sys, os
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP, brier_score_loss
from sklearn.linear_model import LogisticRegression
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
LOG=open("bl03_log.txt","a")
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu"); log("device",DEV,torch.__version__)
D=np.load("bl03_input.npz"); X=D["X"]; y=D["y"].astype(np.float32); pat=D["pat"]; hadm=D["hadm"]; hh=D["h"]; lead=D["lead"]
folds=D["folds"]; tt=D["temporal_test"]; t0=time.time()
# ---- sequences: one per admission, landmarks sorted by hour
order=np.lexsort((hh,hadm)); adm_ids,starts=np.unique(hadm[order],return_index=True); ends=np.r_[starts[1:],len(order)]
SEQ=[order[s:e] for s,e in zip(starts,ends)]; adm_pat=pat[order][starts]; T=max(len(s) for s in SEQ)
def split_val(idx_pat,seed):   # idx_pat: array of patient ids in training part
    up_=np.unique(idx_pat); r=np.random.default_rng(seed); v=set(r.choice(up_,max(1,int(.15*len(up_))),replace=False)); return v
def make_prep(rows):
    med=np.nanmedian(X[rows],0); med=np.where(np.isfinite(med),med,0); A=np.where(np.isnan(X[rows]),med,X[rows]); mu=A.mean(0); sd=A.std(0)+1e-6
    def f(r):
        M=np.isnan(X[r]).astype(np.float32); A=np.where(np.isnan(X[r]),med,X[r]); A=np.clip((A-mu)/sd,-8,8)
        return np.concatenate([A,M,(hh[r][:,None]/336.0).astype(np.float32)],1).astype(np.float32)
    return f
def tensors(seq_idx,f):
    n=len(seq_idx); F=f(np.arange(len(y))) ; d=F.shape[1]
    Xs=np.zeros((n,T,d),np.float32); Ys=np.zeros((n,T),np.float32); Ms=np.zeros((n,T),np.float32); R=-np.ones((n,T),np.int64)
    for i,k in enumerate(seq_idx):
        r=SEQ[k]; L=len(r); Xs[i,:L]=F[r]; Ys[i,:L]=y[r]; Ms[i,:L]=1; R[i,:L]=r
    return torch.tensor(Xs),torch.tensor(Ys),torch.tensor(Ms),R
class Net(nn.Module):
    def __init__(s,d,hdim,drop):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(d,hdim),nn.ReLU(),nn.Dropout(drop)); s.gru=nn.GRU(hdim,hdim,batch_first=True); s.out=nn.Sequential(nn.Dropout(drop),nn.Linear(hdim,1))
    def forward(s,x): z,_=s.gru(s.inp(x)); return s.out(z).squeeze(-1)
def train(tr,va,hdim,drop,wd,seed,maxep=80,pat_=8):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt,Yt,Mt,_=tr; Xv,Yv,Mv,_=va; net=Net(Xt.shape[2],hdim,drop).to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=wd)
    bce=nn.BCEWithLogitsLoss(reduction="none"); best=(1e9,None,0); n=len(Xt)
    Xv_,Yv_,Mv_=Xv.to(DEV),Yv.to(DEV),Mv.to(DEV)
    for ep in range(maxep):
        net.train(); perm=torch.randperm(n)
        for b in range(0,n,64):
            ii=perm[b:b+64]; xb,yb,mb=Xt[ii].to(DEV),Yt[ii].to(DEV),Mt[ii].to(DEV)
            loss=(bce(net(xb),yb)*mb).sum()/mb.sum(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        net.eval()
        with torch.no_grad(): vl=float((bce(net(Xv_),Yv_)*Mv_).sum()/Mv_.sum())
        if vl<best[0]-1e-5: best=(vl,{k:v.detach().clone() for k,v in net.state_dict().items()},ep)
        elif ep-best[2]>=pat_: break
    net.load_state_dict(best[1]); return net,best[0]
def logits(net,te):
    Xe,_,Me,R=te; net.eval(); out=[]
    with torch.no_grad():
        for b in range(0,len(Xe),256): out.append(net(Xe[b:b+256].to(DEV)).cpu().numpy())
    Z=np.concatenate(out); m=R>=0; return R[m],Z[m]
GRID=[dict(hdim=h,drop=dr,wd=wd) for h,dr,wd in itertools.product([64,128],[0.3],[1e-4,1e-2])]
def fit_predict(train_seq,test_seq,seed):
    tp=adm_pat[train_seq]; v=split_val(tp,seed*1000+1); a=train_seq[~np.isin(tp,list(v))]; b=train_seq[np.isin(tp,list(v))]
    f=make_prep(np.concatenate([SEQ[k] for k in a])); A,B,E=tensors(a,f),tensors(b,f),tensors(test_seq,f)
    best=None
    for g in GRID:
        _,vl=train(A,B,seed=seed,**g)
        if best is None or vl<best[0]: best=(vl,g)
    ps=[]
    for k in range(3):
        net,_=train(A,B,seed=seed*10+k,**best[1]); rb,zb=logits(net,B); re,ze=logits(net,E)
        cal=LogisticRegression(C=1e6).fit(zb[:,None],y[rb]); p=np.zeros(len(y)); p[re]=cal.predict_proba(ze[:,None])[:,1]; ps.append(p)
    return np.mean(ps,0),best[1]
ev_h=np.unique(hadm[y==1])
def metrics(p,m):
    yy,pp,hd=y[m],p[m],hadm[m]; out=dict(n=int(m.sum()),pos=int(yy.sum()),AUROC=float(AUC(yy,pp)),AP=float(AP(yy,pp)))
    al=pp>=np.quantile(pp,0.9); ev=np.unique(hd[yy==1]); out["alert10_event_sens"]=float(np.mean([(al&(hd==q)&(yy==1)).any() for q in ev])); out["alert10_PPV"]=float(yy[al].mean())
    pc=np.clip(pp,1e-6,1-1e-6); lg=np.log(pc/(1-pc)); out["Brier"]=float(brier_score_loss(yy,pc)); out["cal_slope"]=float(LogisticRegression(C=1e6).fit(lg[:,None],yy).coef_[0,0]); return out
def boot(pa,pb,m,B=1000,seed=7):   # pb minus pa
    ii=np.where(m)[0]; pp=pat[ii]; up_=np.unique(pp); idx={q:ii[pp==q] for q in up_}; r=np.random.default_rng(seed); dA=[];dP=[]
    for _ in range(B):
        s=np.concatenate([idx[q] for q in r.choice(up_,len(up_))])
        if y[s].sum()==0: continue
        dA.append(AUC(y[s],pb[s])-AUC(y[s],pa[s])); dP.append(AP(y[s],pb[s])-AP(y[s],pa[s]))
    return dict(dAUROC=float(AUC(y[m],pb[m])-AUC(y[m],pa[m])),dAUROC_ci=[float(np.percentile(dA,2.5)),float(np.percentile(dA,97.5))],dAP=float(AP(y[m],pb[m])-AP(y[m],pa[m])),dAP_ci=[float(np.percentile(dP,2.5)),float(np.percentile(dP,97.5))])
CK="bl03_ckpt.npz"
OOF=np.zeros(len(y)); chosen=[]; done=set()
if os.path.exists(CK): c=np.load(CK,allow_pickle=True); OOF=c["OOF"]; done=set(c["done"].tolist()); chosen=c["chosen"].tolist()
adm_fold=[folds[:,SEQ[k][0]] for k in range(len(SEQ))]; adm_fold=np.array(adm_fold)
for rep in range(3):
    for k in range(5):
        key=rep*5+k
        if key in done: continue
        tr=np.where(adm_fold[:,rep]!=k)[0]; te=np.where(adm_fold[:,rep]==k)[0]
        p,g=fit_predict(tr,te,rep*10+k); rows=np.concatenate([SEQ[q] for q in te]); OOF[rows]+=p[rows]/3; chosen.append(str(g)); done.add(key)
        np.savez(CK,OOF=OOF,done=np.array(sorted(done)),chosen=np.array(chosen,dtype=object))
        log("rep",rep,"fold",k,g,round(time.time()-t0))
allm=np.ones(len(y),bool)
RES={"cv":{"GRU":metrics(OOF,allm),"AXIS":metrics(D["axis_oof"],allm)},"chosen":chosen}
RES["cv_cmp"]={"AXIS_vs_GRU":boot(OOF,D["axis_oof"],allm),"FULL_vs_GRU":boot(OOF,D["full_oof"],allm)}
adm_tt=np.array([tt[SEQ[k][0]] for k in range(len(SEQ))]); tr=np.where(~adm_tt)[0]; te=np.where(adm_tt)[0]
pt,_=fit_predict(tr,te,777); RES["temporal"]={"GRU":metrics(pt,tt)}
np.savez_compressed("bl03_oof.npz",GRU=OOF,temporal_GRU=pt)
json.dump(RES,open("BL03_RESULTS.json","w"),indent=1)
log("cv GRU",round(RES["cv"]["GRU"]["AUROC"],3),round(RES["cv"]["GRU"]["AP"],3),"| temporal GRU",round(RES["temporal"]["GRU"]["AUROC"],3),round(RES["temporal"]["GRU"]["AP"],3))
log(RES["cv_cmp"]); log("DONE",round(time.time()-t0))
