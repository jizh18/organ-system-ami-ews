import os, matplotlib
matplotlib.use("Agg")
exec(open("full_common.py").read())
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve
plt.rcParams.update({"font.family":["Liberation Sans","Arial","DejaVu Sans"],"font.size":7,"axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":6,"axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"lines.linewidth":1.2,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#8a8984","axes.grid":True,"grid.color":"#ecebe7","figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})
BLUE,ORANGE,AQUA,YELLOW,VIOLET,GREY,LIGHT="#2a78d6","#eb6834","#1baf7a","#eda100","#4a3aa7","#8a8984","#7fb0e6"; INK,INK2="#0b0b0b","#52514e"
os.makedirs("fig_nc",exist_ok=True)
def save(fig,n):
    for ext,kw in [("pdf",{}),("tif",{"dpi":600,"pil_kwargs":{"compression":"tiff_lzw"}}),("png",{"dpi":300})]: fig.savefig(f"fig_nc/{n}.{ext}",bbox_inches="tight",**kw)
    plt.close(fig)
D=json.load(open("FULL_DEV_V2.json")); Hj=json.load(open("FULL_HELDOUT.json")); H=np.load("heldout_scores.npz")
m=np.zeros(len(y),bool); m[LATE]=True
P={a:sig(H[a]) for a in ["AXIS","WITHIN","FULL","ADD","LR"]}; P["LR+AXIS"]=mix(P["LR"],P["AXIS"]); P.update(SC)
NM={"AXIS":"Organ-system model","FULL":"Unconstrained model","WITHIN":"Within-system model","ADD":"Additive model","LR":"Logistic regression","LR+AXIS":"Hybrid","GRACE":"GRACE (approximate)","NEWS2":"NEWS2"}
COL={"AXIS":BLUE,"FULL":GREY,"WITHIN":LIGHT,"ADD":VIOLET,"LR":AQUA,"LR+AXIS":"#c2185b","GRACE":YELLOW,"NEWS2":ORANGE}
# Fig 1 endpoint
fig,axs=plt.subplots(1,2,figsize=(7.09,2.6),gridspec_kw=dict(width_ratios=[1.2,1]))
ax=axs[0]; comp=["Death","Circulatory support","Invasive ventilation"]; v1=[70,218,50]; v2=[67,137,45]; w=0.36
for j,(lab,vals,c) in enumerate([("Fixed-hour windows (v1; 289 patients)",v1,GREY),("Episode-based rule (v2, primary; 204 patients)",vals2:=v2,BLUE)]):
    xs=np.arange(3)+(j-.5)*w; ax.bar(xs,vals,w*.92,color=c,label=lab)
    for x_,v_ in zip(xs,vals): ax.text(x_,v_+3,str(v_),ha="center",fontsize=6,color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels(comp); ax.grid(axis="x",visible=False); ax.set_ylabel("Development patients with component"); ax.set_ylim(0,290); ax.legend(frameon=False,fontsize=6,loc="upper right")
ax.set_title("a",loc="left",fontsize=8,fontweight="bold")
ax=axs[1]; sc=["×0.5","×1.0","×1.5"]; ev1=[None,None,None]
evv2=[D["window"]["0.5"]["events"],D["events"],D["window"]["1.5"]["events"]]; au=[D["window"]["0.5"]["AXIS"],D["metrics"]["AXIS"]["AUROC"],D["window"]["1.5"]["AXIS"]]
ax.bar(range(3),evv2,0.55,color=BLUE)
for i,(e,a) in enumerate(zip(evv2,au)): ax.text(i,e+3,f"{e} patients\nAUROC {a:.3f}",ha="center",fontsize=6,color=INK2)
ax.set_xticks(range(3)); ax.set_xticklabels(["×0.5","×1.0 (primary)","×1.5"],fontsize=6); ax.set_xlabel("Post-anaesthesia and coronary-intervention window length"); ax.set_ylim(0,max(evv2)*1.3); ax.grid(axis="x",visible=False); ax.set_ylabel("Development patients with event")
ax.set_title("b",loc="left",fontsize=8,fontweight="bold")
fig.tight_layout(); save(fig,"Fig2_endpoint")
# Fig 2 held-out ROC + forests
fig=plt.figure(figsize=(7.09,4.6)); gs=fig.add_gridspec(2,2,width_ratios=[1,1.25]); axs=[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
ax=axs[0]
for a in ["AXIS","FULL","LR","LR+AXIS","GRACE","NEWS2"]:
    f,t,_=roc_curve(y[m],P[a][m]); ax.plot(f,t,color=COL[a],lw=1.6 if a=="AXIS" else 1.5,label=f"{NM[a]} ({Hj['metrics'][a]['AUROC']:.3f})")
ax.plot([0,1],[0,1],ls="--",color=GREY,lw=1); ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.legend(frameon=False,fontsize=5.5,loc="lower right",title="AUROC, held-out 2017–2022",title_fontsize=6)
ax.set_title("a",loc="left",fontsize=8,fontweight="bold")
pairs=[("FULL","H1 AXIS-FULL"),("WITHIN","H3 AXIS-WITHIN"),("ADD","AXIS-ADD"),("LR","H2 AXIS-LR"),("GRACE",("AXIS-GRACE","H4 AXIS-GRACE")),("NEWS2",("AXIS-NEWS2","H4 AXIS-NEWS2"))]
for j,(metric,ttl) in enumerate([("AUROC","b"),("AP","c")]):
    ax=axs[j+1]; yy=np.arange(len(pairs))[::-1]
    for i,(nm,key) in enumerate(pairs):
        kd,kh=(key if isinstance(key,tuple) else (key,key))
        for off,src,k,c,mk,lab in [(0.14,D["cmp"],kd,BLUE,"o","Development CV (3,924 patients)"),(-0.14,Hj["cmp"],kh,ORANGE,"s","Held-out 2017–2022 (2,470 patients)")]:
            v=src[k]; e=v["d"+metric]; lo,hi=v["d"+metric+"_ci"]
            ax.errorbar(e,yy[i]+off,xerr=[[e-lo],[hi-e]],fmt=mk,color=c,capsize=1.5,ms=3.5,label=lab if i==0 else None)
    ax.axvline(0,color=INK2,lw=1); ax.set_yticks(yy); ax.set_yticklabels([f"vs {NM[n]}" for n,_ in pairs],fontsize=6); ax.grid(axis="y",visible=False)
    ax.set_xlabel(f"Δ {metric}, organ-system model minus comparator (95% CI)"); ax.set_title(ttl,loc="left",fontsize=8,fontweight="bold")
    if j==1: ax.legend(frameon=False,fontsize=6,loc="upper center",bbox_to_anchor=(0.45,-0.22),ncol=2)
fig.tight_layout(); save(fig,"Fig3_discrimination")
# Fig 3 alert curves held-out
rates=np.linspace(0.01,0.30,30); ev_h=np.unique(hadm[m][y[m]==1])
def ev_sens(p):
    pm=np.where(m,p,-np.inf); out=[]
    for q in rates:
        thr=np.quantile(p[m],1-q); al=(pm>=thr); out.append(np.mean([(al&(hadm==h_)&(y==1)).any() for h_ in ev_h]))
    return np.array(out)
fig,ax=plt.subplots(figsize=(3.46,2.8))
for a in ["AXIS","FULL","LR","LR+AXIS","GRACE","NEWS2"]:
    ax.plot(rates*100,ev_sens(P[a])*100,color=COL[a],lw=1.6 if a=="AXIS" else 1.5,label=NM[a])
ax.axvline(10,color=GREY,ls=":",lw=1); ax.set_xlim(0,30); ax.set_ylim(0,100); ax.set_xlabel("Alert rate (% of 6-hourly assessments flagged)"); ax.set_ylabel("Episodes alerted within prior 24 h (%)")
ax.legend(frameon=False,fontsize=6,loc="lower right"); None; save(fig,"Fig4_alerting")
# Fig 4 calibration + DCA held-out
p=P["AXIS"][m]; yy=y[m]; fig,axs=plt.subplots(1,2,figsize=(7.09,2.8))
q=np.quantile(p,np.linspace(0,1,11)); bi=np.clip(np.searchsorted(q,p,side="right")-1,0,9)
mp=np.array([p[bi==i].mean()*100 for i in range(10)]); mo=np.array([yy[bi==i].mean()*100 for i in range(10)]); n_=np.array([(bi==i).sum() for i in range(10)]); se=np.sqrt(mo/100*(1-mo/100)/n_)*100
ax=axs[0]; mx=max(mp.max(),mo.max())*1.15; ax.plot([0,mx],[0,mx],ls="--",color=GREY,lw=1); ax.errorbar(mp,mo,yerr=1.96*se,fmt="o-",color=BLUE,lw=1.2,ms=3.5,capsize=2)
c=Hj["cal"]["AXIS"]; ax.set_xlabel("Predicted risk, % (deciles)"); ax.set_ylabel("Observed event rate, %")
ax.set_title("a",loc="left",fontsize=8,fontweight="bold")
ax=axs[1]; th=np.linspace(0.003,0.08,60); prev=yy.mean()
nb=lambda pp:np.array([((pp>=t)&(yy==1)).mean()-((pp>=t)&(yy==0)).mean()*t/(1-t) for t in th])*1000
for a in ["AXIS","LR","FULL"]: ax.plot(th*100,nb(P[a][m]),color=COL[a],lw=1.6 if a=="AXIS" else 1.4,label=NM[a])
ax.plot(th*100,np.array([prev-(1-prev)*t/(1-t) for t in th])*1000,color=INK2,ls="--",lw=1.2,label="Alert all"); ax.axhline(0,color=INK,lw=1,label="Alert none")
ax.set_ylim(-2,nb(P["AXIS"][m]).max()*1.3); ax.set_xlabel("Threshold probability, %"); ax.set_ylabel("Net benefit per 1,000 assessments"); ax.legend(frameon=False,fontsize=6)
ax.set_title("b",loc="left",fontsize=8,fontweight="bold"); fig.tight_layout(); save(fig,"Fig5_calibration_dca")
# Fig 5 importance held-out
EN={"INJURY":"Injury","HEMO":"Haemodynamic","PERFUSION":"Perfusion","RESP":"Respiratory","RENAL":"Renal","HEME":"Haematology","INFLAM":"Inflammation","META":"Metabolic/hepatic","NEURO":"Neurological","ELECTRICAL":"Electrical (ECG)","BASELINE":"Baseline"}
cn=lambda g:" × ".join(EN[a] for a in g.split("×"))
imp=sorted([(g,v["std"],v["n_axes"]) for g,v in Hj["importance"].items() if v["std"]>=0.02],key=lambda t:t[1])
fig,ax=plt.subplots(figsize=(4.72,3.4)); ax.barh(range(len(imp)),[t[1] for t in imp],color=[ORANGE if t[2]==2 else BLUE for t in imp],height=0.62)
for i,t in enumerate(imp): ax.text(t[1]+0.004,i,f"{t[1]:.2f}",va="center",fontsize=6,color=INK2)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([cn(t[0]) for t in imp]); ax.grid(axis="y",visible=False); ax.set_xlabel("SD of log-odds contribution across held-out assessments")
ax.legend(handles=[Patch(color=BLUE,label="Single system"),Patch(color=ORANGE,label="System pair")],frameon=False,loc="lower right")
None; save(fig,"Fig6_importance")
# Fig 6 trajectories held-out
show=["PERFUSION×RENAL","PERFUSION×RESP","HEMO×PERFUSION","PERFUSION","ELECTRICAL×HEMO","INFLAM"]
fig,axs=plt.subplots(2,3,figsize=(7.09,4.0),sharey=True)
for k,(ax,g) in enumerate(zip(axs.ravel(),show)):
    r=np.array(Hj["trajectory"][g]); xm=-(r[:,0]+r[:,1])/2; mm_=r[:,2]; se=r[:,3]
    ax.fill_between(xm,np.exp(mm_-1.96*se),np.exp(mm_+1.96*se),color=BLUE,alpha=.15,lw=0); ax.plot(xm,np.exp(mm_),"o-",color=BLUE,lw=1.2,ms=2.5)
    ax.axhline(1,color=GREY,lw=1); ax.axvspan(-24,0,color="#f0efec",zorder=0); ax.set_xlim(-72,0); ax.set_xticks([-72,-48,-24,0]); ax.set_title(cn(g),loc="left",fontsize=7); ax.text(-0.18,1.08,"abcdef"[k],transform=ax.transAxes,fontsize=8,fontweight="bold",va="bottom")
    for x_,n_ in zip(xm,r[:,5]): ax.text(x_,0.02,str(int(n_)),transform=ax.get_xaxis_transform(),ha="center",fontsize=5,color=INK2)
for ax in axs[1]: ax.set_xlabel("Hours before event")
for ax in axs[:,0]: ax.set_ylabel("OR vs matched non-events")
None
fig.tight_layout(rect=[0,0,1,0.95]); save(fig,"Fig7_trajectory"); print("ok")
