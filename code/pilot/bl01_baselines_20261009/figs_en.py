"""English manuscript figures + alerting table (EP3 tuned OOF + BL01 baselines)."""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score as AUC, average_precision_score as AP
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#8a8984","axes.labelcolor":"#33332f","xtick.color":"#33332f","ytick.color":"#33332f","axes.grid":True,
  "grid.color":"#ecebe7","grid.linewidth":0.8,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
BLUE,ORANGE,AQUA,YELLOW,VIOLET,GREY="#2a78d6","#eb6834","#1baf7a","#eda100","#4a3aa7","#8a8984"; INK,INK2="#0b0b0b","#52514e"
import os; os.makedirs("fig_en",exist_ok=True)
def save(fig,n): fig.savefig(f"fig_en/{n}.png",dpi=300,bbox_inches="tight"); plt.close(fig)
src=open("ep3.py").read().split("RES={}")[0]; exec(src)   # loads y, hadm, lead, pat, SC (NEWS2, GRACE)
O3=np.load("ep3_oof.npz"); BL=np.load("bl_oof.npz"); R3=json.load(open("EP3_RESULTS.json")); RB=json.load(open("BL01_RESULTS.json")); RL=json.load(open("BL02_LR_RESULTS.json")); BL2=np.load("bl2_oof.npz")
P={"AXIS (knowledge-constrained)":O3["AXIS"],"Unconstrained GBM":O3["FULL"],"Logistic regression":BL2["LR"],"MLP":BL["MLP"],"Dynamic GRACE":SC["GRACE"],"NEWS2":SC["NEWS2"]}
COL={"AXIS (knowledge-constrained)":BLUE,"Unconstrained GBM":GREY,"Logistic regression":AQUA,"MLP":VIOLET,"Dynamic GRACE":YELLOW,"NEWS2":ORANGE}
ev_h=np.unique(hadm[y==1])
# ---------- alerting table
rows=[]
for nmd,p in P.items():
    r=dict(model=nmd,AUROC=AUC(y,p),AP=AP(y,p))
    for q in (0.05,0.10,0.20):
        al=p>=np.quantile(p,1-q); det=[];leads=[]
        for h in ev_h:
            m=(hadm==h)&(y==1)
            if (al&m).any(): det.append(1); leads.append(np.nanmax(lead[m&al]))
            else: det.append(0)
        ppv=y[al].mean(); r[f"s{int(q*100)}"]=np.mean(det); r[f"ppv{int(q*100)}"]=ppv; r[f"nne{int(q*100)}"]=1/ppv; r[f"lead{int(q*100)}"]=np.median(leads)
    rows.append(r)
json.dump(rows,open("ALERT_TABLE_EP3.json","w"),indent=1,default=float)
for r in rows: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()})
# ---------- Fig 1 endpoint
fig,ax=plt.subplots(figsize=(8.6,4.2))
defs=["All support starts","Primary: unplanned","Strict"]; comp=["Death","Circulatory support","Invasive ventilation"]
vals={"All support starts":[27,250,216],"Primary: unplanned":[28,88,21],"Strict":[28,79,16]}; tot={"All support starts":286,"Primary: unplanned":113,"Strict":104}
w=0.26; cols=[INK2,ORANGE,BLUE]
for j,c in enumerate(comp):
    xs=np.arange(3)+(j-1)*w; v=[vals[d][j] for d in defs]
    ax.bar(xs,v,w*0.92,color=cols[j],label=c)
    for x_,v_ in zip(xs,v): ax.text(x_,v_+4,str(v_),ha="center",fontsize=9,color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{d}\n(n = {tot[d]} patients)" for d in defs]); ax.grid(axis="x",visible=False)
ax.set_ylabel("Patients with event component"); ax.legend(frameon=False,loc="upper right")
ax.set_title("Patients with each event component under three endpoint definitions",loc="left",fontsize=11.5,fontweight="bold",color=INK)
save(fig,"Fig1_endpoint")
# ---------- Fig 2 ROC + forest
fig,axs=plt.subplots(1,2,figsize=(12,5.2),gridspec_kw=dict(width_ratios=[1,1.15]))
ax=axs[0]
for nmd,p in P.items():
    f,t,_=roc_curve(y,p); ax.plot(f,t,color=COL[nmd],lw=2.6 if nmd.startswith("AXIS") else 1.6,label=f"{nmd} ({AUC(y,p):.3f})")
ax.plot([0,1],[0,1],ls="--",color=GREY,lw=1); ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.legend(frameon=False,fontsize=8.8,loc="lower right",title="AUROC, cross-validation",title_fontsize=9)
ax.set_title("A  ROC curves (patient-grouped 5-fold CV × 3)",loc="left",fontsize=11,fontweight="bold")
ax=axs[1]; items=[]
cvc={"Unconstrained GBM":R3["cv_cmp"]["AXIS_vs_FULL"],"Logistic regression":RL["cv_cmp"]["AXIS_vs_LR"],"MLP":RB["cv_cmp"]["AXIS_vs_MLP"],"Dynamic GRACE":R3["cv_cmp"]["AXIS_vs_GRACE"],"NEWS2":R3["cv_cmp"]["AXIS_vs_NEWS2"]}
tc={"Unconstrained GBM":R3["temporal_cmp"]["AXIS_vs_FULL"],"Dynamic GRACE":R3["temporal_cmp"]["AXIS_vs_GRACE"],"NEWS2":R3["temporal_cmp"]["AXIS_vs_NEWS2"]}
tc={("Logistic regression" if k=="AXIS_vs_LR" else "MLP" if k=="AXIS_vs_MLP" else k):v for k,v in tc.items()}
names=list(cvc); yy=np.arange(len(names))[::-1]
for i,n in enumerate(names):
    c=cvc[n]; ax.errorbar(c["dAUROC"],yy[i]+0.13,xerr=[[c["dAUROC"]-c["dAUROC_ci"][0]],[c["dAUROC_ci"][1]-c["dAUROC"]]],fmt="o",color=BLUE,capsize=3,ms=6,label="Cross-validation" if i==0 else None)
    if n in tc:
        c=tc[n]; ax.errorbar(c["dAUROC"],yy[i]-0.13,xerr=[[c["dAUROC"]-c["dAUROC_ci"][0]],[c["dAUROC_ci"][1]-c["dAUROC"]]],fmt="s",color=ORANGE,capsize=3,ms=6,label="Temporal validation (2014–2016)" if i==0 else None)
ax.axvline(0,color=INK2,lw=1); ax.axvspan(-0.02,0.02,color="#f0efec",zorder=0)
ax.set_yticks(yy); ax.set_yticklabels([f"vs {n}" for n in names]); ax.grid(axis="y",visible=False)
ax.set_xlabel("Δ AUROC, AXIS minus comparator (95% patient-bootstrap CI)"); ax.set_ylim(-0.6,len(names)-0.2); ax.legend(frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=2)
ax.set_title("B  AXIS versus each comparator (shaded: ±0.02 margin)",loc="left",fontsize=11,fontweight="bold")
fig.tight_layout(); save(fig,"Fig2_discrimination")
# ---------- Fig 3 alert-rate curves
rates=np.linspace(0.01,0.30,30)
def ev_sens(p):
    return np.array([np.mean([((p>=np.quantile(p,1-q))&(hadm==h)&(y==1)).any() for h in ev_h]) for q in rates])
fig,ax=plt.subplots(figsize=(7.4,4.8))
for nmd,p in P.items():
    s=ev_sens(p); ax.plot(rates*100,s*100,color=COL[nmd],lw=2.6 if nmd.startswith("AXIS") else 1.6,label=nmd)
ax.axvline(10,color=GREY,lw=1,ls=":"); ax.set_xlim(0,30); ax.set_ylim(0,100)
ax.set_xlabel("Alert rate (% of 6-hourly assessments flagged)"); ax.set_ylabel("Deterioration episodes alerted within prior 24 h (%)")
ax.legend(frameon=False,fontsize=9,loc="lower right"); ax.set_title("Event-level sensitivity at a fixed alert burden",loc="left",fontsize=11.5,fontweight="bold")
save(fig,"Fig3_alerting")
# ---------- Fig 4 calibration + DCA (AXIS)
p=O3["AXIS"]; fig,axs=plt.subplots(1,2,figsize=(11,4.4))
q=np.quantile(p,np.linspace(0,1,11)); bi=np.clip(np.searchsorted(q,p,side="right")-1,0,9)
mp=np.array([p[bi==i].mean()*100 for i in range(10)]); mo=np.array([y[bi==i].mean()*100 for i in range(10)]); n_=np.array([(bi==i).sum() for i in range(10)])
se=np.sqrt(mo/100*(1-mo/100)/n_)*100
ax=axs[0]; mx=max(mp.max(),mo.max())*1.15; ax.plot([0,mx],[0,mx],ls="--",color=GREY,lw=1); ax.errorbar(mp,mo,yerr=1.96*se,fmt="o-",color=BLUE,lw=2,ms=6,capsize=2)
ax.set_xlabel("Predicted risk, % (deciles)"); ax.set_ylabel("Observed event rate, %")
ax.set_title(f"A  Calibration (slope {R3['cv']['AXIS']['cal_slope']:.2f}, Brier {R3['cv']['AXIS']['Brier']:.4f})",loc="left",fontsize=11,fontweight="bold")
ax=axs[1]; th=np.linspace(0.005,0.12,60); prev=y.mean()
nb=lambda pp:np.array([((pp>=t)&(y==1)).mean()-((pp>=t)&(y==0)).mean()*t/(1-t) for t in th])*1000
ax.plot(th*100,nb(p),color=BLUE,lw=2.6,label="AXIS"); ax.plot(th*100,nb(O3["FULL"]),color=GREY,lw=1.4,label="Unconstrained GBM")
ax.plot(th*100,np.array([prev-(1-prev)*t/(1-t) for t in th])*1000,color=INK2,lw=1.2,ls="--",label="Alert all"); ax.axhline(0,color=INK,lw=1,label="Alert none")
ax.set_ylim(-2,nb(p).max()*1.3); ax.set_xlabel("Threshold probability, %"); ax.set_ylabel("Net benefit per 1,000 assessments"); ax.legend(frameon=False,fontsize=9)
ax.set_title("B  Decision curve",loc="left",fontsize=11,fontweight="bold")
fig.tight_layout(); save(fig,"Fig4_calibration_dca")
# ---------- Fig 5 importance, Fig 6 trajectory (decomposition of AXIS)
J=json.load(open("INTERP.json"))
EN={"INJURY":"Injury","HEMO":"Haemodynamic","PERFUSION":"Perfusion","RESP":"Respiratory","RENAL":"Renal","HEME":"Haematology","INFLAM":"Inflammation","META":"Metabolic/hepatic","NEURO":"Neurological","ELECTRICAL":"Electrical (ECG)","BASELINE":"Baseline"}
cn=lambda g:" × ".join(EN[a] for a in g.split("×"))
imp=sorted([(g,v["std"],v["n_axes"]) for g,v in J["importance"].items() if v["std"]>=0.03],key=lambda t:t[1])
fig,ax=plt.subplots(figsize=(7.6,5.6))
ax.barh(range(len(imp)),[t[1] for t in imp],color=[ORANGE if t[2]==2 else BLUE for t in imp],height=0.62)
for i,t in enumerate(imp): ax.text(t[1]+0.004,i,f"{t[1]:.2f}",va="center",fontsize=9,color=INK2)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([cn(t[0]) for t in imp]); ax.grid(axis="y",visible=False)
ax.set_xlabel("SD of log-odds contribution across assessments")
ax.legend(handles=[Patch(color=BLUE,label="Single system"),Patch(color=ORANGE,label="System pair")],frameon=False,loc="lower right")
ax.set_title("Contribution of organ systems and system pairs to predicted risk",loc="left",fontsize=11.5,fontweight="bold")
save(fig,"Fig5_importance")
sc=[]
for g,rr in J["trajectory"].items():
    r=np.array(rr); sc.append((r[r[:,1]<=24][:,2].mean()-r[r[:,0]>=36][:,2].mean(),g))
show=[g for _,g in sorted(sc,reverse=True)[:6]]
fig,axs=plt.subplots(2,3,figsize=(12,6),sharey=True)
for ax,g in zip(axs.ravel(),show):
    r=np.array(J["trajectory"][g]); xm=-(r[:,0]+r[:,1])/2; m=r[:,2]; se=r[:,3]
    ax.fill_between(xm,np.exp(m-1.96*se),np.exp(m+1.96*se),color=BLUE,alpha=.15,lw=0); ax.plot(xm,np.exp(m),"o-",color=BLUE,lw=2,ms=4)
    ax.axhline(1,color=GREY,lw=1); ax.axvspan(-24,0,color="#f0efec",zorder=0); ax.set_xlim(-72,0); ax.set_xticks([-72,-48,-24,0]); ax.set_title(cn(g),loc="left",fontsize=10.5)
for ax in axs[1]: ax.set_xlabel("Hours before event")
for ax in axs[:,0]: ax.set_ylabel("Odds ratio vs non-event mean")
fig.suptitle(f"Contribution trajectories before deterioration ({J['n_event_patients_with_lead']} patients, {J['n_event_landmarks']} assessments)",x=0.01,ha="left",fontsize=11.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); save(fig,"Fig6_trajectory")
print("ok",show)
