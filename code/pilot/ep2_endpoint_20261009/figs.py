import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import roc_curve
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family":"Noto Sans CJK SC","font.size":11,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#a3a29c","axes.labelcolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","axes.grid":True,
  "grid.color":"#ecebe7","grid.linewidth":0.8,"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","savefig.facecolor":"#fcfcfb","figure.dpi":110})
BLUE,ORANGE,AQUA,YELLOW,VIOLET="#2a78d6","#eb6834","#1baf7a","#eda100","#4a3aa7"; INK,INK2,MUTED,GRID="#0b0b0b","#52514e","#a3a29c","#ecebe7"
z=np.load("ep2_features.npz"); y=z["y"]; hadm=z["hadm"]; lead=z["lead"]; pat=z["patient"]
meta=json.load(open("ep2_feature_meta.json")); nm=meta["names"]; X=z["X"]
O=np.load("ep2_oof.npz"); R=json.load(open("EP2_RESULTS.json")); J=json.load(open("INTERP.json"))
def title(ax,t,s=None):
    ax.set_title(t,loc="left",fontsize=13,color=INK,pad=22 if s else 10,fontweight="bold")
    if s: ax.text(0,1.02,s,transform=ax.transAxes,fontsize=10,color=INK2)
def save(fig,n): fig.savefig(f"fig/{n}.png",dpi=200,bbox_inches="tight"); plt.close(fig)
def news2():
    def L(n): return X[:,nm.index(n)]
    rr,sp,t,sbp,hr,gcs=L("resp_rate.last"),L("spo2.last"),L("temperature.last"),L("sbp.last"),L("heart_rate.last"),L("gcs_total")
    fio=z["XP"][:,meta["proc_names"].index("fio2_est")]; sc=np.zeros(len(y))
    sc+=np.select([rr<=8,rr<=11,rr<=20,rr<=24,rr>24],[3,1,0,2,3],0)*np.isfinite(rr); sc+=np.select([sp<=91,sp<=93,sp<=95,sp>95],[3,2,1,0],0)*np.isfinite(sp)
    sc+=2*(fio>0.21); sc+=np.select([t<=35,t<=36,t<=38,t<=39,t>39],[3,1,0,1,2],0)*np.isfinite(t)
    sc+=np.select([sbp<=90,sbp<=100,sbp<=110,sbp<=219,sbp>219],[3,2,1,0,3],0)*np.isfinite(sbp)
    sc+=np.select([hr<=40,hr<=50,hr<=90,hr<=110,hr<=130,hr>130],[3,1,0,1,2,3],0)*np.isfinite(hr); sc+=3*(np.nan_to_num(gcs,nan=15)<15)
    return sc+np.random.default_rng(0).random(len(y))*1e-6
NEWS=news2(); SI=np.nan_to_num(X[:,nm.index("shock_index")],nan=0.0)
# ---------- F1 终点清理
fig,ax=plt.subplots(figsize=(8,3.6))
labs=["旧终点\n（前72小时，含计划内）","剔除计划内后\n（前72小时）","新终点\n（住院全程，至14天）"]; vals=[178,73,113]; cols=[MUTED,MUTED,BLUE]
b=ax.barh(range(3)[::-1],vals,color=cols,height=0.55)
for i,v in enumerate(vals): ax.text(v+3,2-i,f"{v} 名出事患者",va="center",color=INK,fontsize=11)
ax.set_yticks(range(3)[::-1]); ax.set_yticklabels(labs); ax.set_xlim(0,230); ax.grid(axis="y",visible=False); ax.set_xlabel("出事患者数")
title(ax,"旧终点里约六成“事件”是心外科术后的计划内支持治疗","剔除后把预测窗口延长到住院全程，事件数回升到113例")
save(fig,"f1_endpoint")
# ---------- F2 ROC
fig,ax=plt.subplots(figsize=(6.4,5.6))
for p,lab,c,lw in [(O["FULL"],"黑箱模型",MUTED,1.6),(NEWS,"NEWS2",ORANGE,2),(SI,"休克指数",YELLOW,2),(O["AXIS"],"系统轴模型",BLUE,2.6)]:
    fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,color=c,lw=lw,label=lab)
au={"系统轴模型":R["models"]["AXIS"]["AUROC"],"黑箱模型":R["models"]["FULL"]["AUROC"],"NEWS2":R["scores"]["NEWS2"]["AUROC"],"休克指数":R["scores"]["shock_index"]["AUROC"]}
ax.plot([0,1],[0,1],ls="--",color=MUTED,lw=1)
h,l=ax.get_legend_handles_labels(); ax.legend(h,[f"{x}（AUROC {au[x]:.3f}）" for x in l],frameon=False,loc="lower right",fontsize=10)
ax.set_xlabel("假阳性率（1 − 特异度）"); ax.set_ylabel("灵敏度"); ax.set_xlim(0,1); ax.set_ylim(0,1)
title(ax,f"系统轴模型 AUROC {au['系统轴模型']:.3f}，显著高于 NEWS2（{au['NEWS2']:.3f}）",f"与黑箱模型（{au['黑箱模型']:.3f}）无显著差异；按患者分组5折交叉验证×3次")
save(fig,"f2_roc")
# ---------- F3 报警率-事件灵敏度
ev=np.unique(hadm[y==1]); rates=np.linspace(0.01,0.30,30)
def ev_sens(p):
    out=[]
    for q in rates:
        al=p>=np.quantile(p,1-q); out.append(np.mean([(al&(hadm==h)&(y==1)).any() for h in ev]))
    return np.array(out)
fig,ax=plt.subplots(figsize=(7.6,4.8))
curves=[(NEWS,"NEWS2",ORANGE),(SI,"休克指数",YELLOW),(O["AXIS"],"系统轴模型",BLUE)]
for p,lab,c in curves:
    s=ev_sens(p); ax.plot(rates*100,s*100,color=c,lw=2.6 if lab=="系统轴模型" else 2,label=lab); ax.text(30.4,s[-1]*100,lab,color=INK2,va="center",fontsize=10)
ax.axvline(10,color=MUTED,lw=1,ls=":"); 
s_ax=ev_sens(O["AXIS"]); s_nw=ev_sens(NEWS); i10=np.argmin(np.abs(rates-0.10))
ax.text(10.6,6,f"报警10%时：系统轴 {s_ax[i10]*100:.0f}%，NEWS2 {s_nw[i10]*100:.0f}%",color=INK,fontsize=10)
ax.set_xlabel("报警率（被报警的预测时点占比，%）"); ax.set_ylabel("出事前24小时内被报警的出事患者（%）"); ax.set_xlim(0,34); ax.set_ylim(0,100)
title(ax,"同样的报警负担下，系统轴模型多抓到约一半的出事病人","事件级灵敏度：出事前24小时内至少报警一次")
save(fig,"f3_alert")
# ---------- F4 校准 + 决策曲线
fig,axs=plt.subplots(1,2,figsize=(11,4.6))
p=O["AXIS"]; q=np.quantile(p,np.linspace(0,1,11)); bi=np.clip(np.searchsorted(q,p,side="right")-1,0,9)
mp=[p[bi==i].mean()*100 for i in range(10)]; mo=[y[bi==i].mean()*100 for i in range(10)]
ax=axs[0]; mx=max(mp+mo)*1.1; ax.plot([0,mx],[0,mx],ls="--",color=MUTED,lw=1); ax.plot(mp,mo,"o-",color=BLUE,lw=2,ms=7)
ax.set_xlabel("预测风险（%，按十分位分组）"); ax.set_ylabel("实际发生率（%）"); title(ax,f"校准：预测风险与实际发生率基本一致",f"校准斜率 {R['models']['AXIS']['cal_slope']:.2f}（理想为1），Brier {R['models']['AXIS']['Brier']:.4f}")
ax=axs[1]; th=np.linspace(0.005,0.15,60); prev=y.mean()
nb=lambda pp:[((pp>=t)&(y==1)).mean()-((pp>=t)&(y==0)).mean()*t/(1-t) for t in th]
ax.plot(th*100,np.array(nb(p))*1000,color=BLUE,lw=2.6,label="系统轴模型")
ax.plot(th*100,np.array([prev-(1-prev)*t/(1-t) for t in th])*1000,color=MUTED,lw=1.6,ls="--",label="全部报警")
ax.axhline(0,color=INK2,lw=1,label="都不报警")
ax.set_ylim(-3,max(nb(p))*1000*1.3); ax.set_xlabel("决策阈值（预测风险，%）"); ax.set_ylabel("净获益（每千个时点）"); ax.legend(frameon=False)
title(ax,"决策曲线：常用阈值范围内净获益均为正","高于“全部报警”和“都不报警”两种策略")
fig.tight_layout(); save(fig,"f4_calib_dca")
# ---------- F5 系统贡献（标准差）
CN={"INJURY":"心肌损伤","HEMO":"循环","PERFUSION":"灌注/酸碱","RESP":"呼吸","RENAL":"肾/电解质","HEME":"血液/凝血","INFLAM":"炎症","META":"代谢/其他","NEURO":"神经","ELECTRICAL":"心电","BASELINE":"基线"}
cn=lambda g:" × ".join(CN[a] for a in g.split("×"))
imp=[(g,v["std"],v["n_axes"]) for g,v in J["importance"].items() if v["std"]>=0.05]
imp.sort(key=lambda t:t[1])
fig,ax=plt.subplots(figsize=(8.4,6.2))
ax.barh(range(len(imp)),[t[1] for t in imp],color=[ORANGE if t[2]==2 else BLUE for t in imp],height=0.62)
for i,t in enumerate(imp): ax.text(t[1]+0.006,i,f"{t[1]:.2f}",va="center",fontsize=9.5,color=INK2)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([cn(t[0]) for t in imp]); ax.grid(axis="y",visible=False)
ax.set_xlabel("对风险的贡献在患者间的波动（logit 标准差）")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="单个系统"),Patch(color=ORANGE,label="系统间联动")],frameon=False,loc="lower right")
npair=sum(1 for t in imp[-10:] if t[2]==2)
title(ax,f"贡献最大的10项中有{npair}项是系统间联动，循环系统几乎只通过联动起作用","折外样本上把每次预测精确分解到各系统与系统对（分解误差<1e-13）")
save(fig,"f5_importance")
# ---------- F6 系统贡献随关键指标变化
C=np.load("interp_contrib.npz"); keys=list(C.keys())
def axis_total(a): return sum(C[k] for k in keys if a in k.split("_x_"))
panels=[("lactate.last","PERFUSION","乳酸（mmol/L）"),("bicarbonate.last","PERFUSION","碳酸氢根（mmol/L）"),("shock_index","HEMO","休克指数（心率/收缩压）"),("sf_ratio","RESP","SpO2/FiO2"),
        ("urea_nitrogen.last","RENAL","尿素氮（mg/dL）"),("troponin_t.last","INJURY","肌钙蛋白T（ng/mL）"),("hemoglobin.last","HEME","血红蛋白（g/dL）"),("white_blood_cells.last","INFLAM","白细胞（K/µL）")]
fig,axs=plt.subplots(2,4,figsize=(14,6.4),sharey=True)
for ax,(f,a,lab) in zip(axs.ravel(),panels):
    x=X[:,nm.index(f)]; c=axis_total(a); ok=np.isfinite(x); x,c=x[ok],c[ok]; base=c.mean()
    qs=np.unique(np.quantile(x,np.linspace(0,1,11))); b=np.clip(np.searchsorted(qs,x,side="right")-1,0,len(qs)-2)
    xm=np.array([np.median(x[b==i]) for i in range(len(qs)-1)]); cm=np.array([c[b==i].mean()-base for i in range(len(qs)-1)]); se=np.array([c[b==i].std()/np.sqrt((b==i).sum()) for i in range(len(qs)-1)])
    ax.fill_between(xm,np.exp(cm-1.96*se),np.exp(cm+1.96*se),color=BLUE,alpha=.15,lw=0); ax.plot(xm,np.exp(cm),"o-",color=BLUE,lw=2,ms=4)
    ax.axhline(1,color=MUTED,lw=1); ax.set_yscale("log"); ax.set_ylim(0.6,2.2); ax.set_yticks([0.67,1,1.5,2]); ax.set_yticklabels(["0.67","1","1.5","2"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if f=="troponin_t.last": ax.set_xscale("log")
    ax.set_title(f"{lab}\n{CN[a]}系统（有值时点 {ok.mean()*100:.0f}%）",fontsize=10.5,color=INK,loc="left")
for ax in axs[:,0]: ax.set_ylabel("风险相对倍数（odds，对该系统平均）")
fig.suptitle("关键指标与系统风险贡献：碳酸氢根降低、SpO2/FiO2下降、休克指数和白细胞升高时风险上升",x=0.01,ha="left",fontsize=13,fontweight="bold",color=INK)
fig.text(0.01,0.925,"按十分位分组，折外贡献的平均值与95%区间；系统贡献包含它参与的联动项",fontsize=10,color=INK2)
fig.tight_layout(rect=[0,0,1,0.91]); save(fig,"f6_curves")
# ---------- F7 联动热图（数据驱动选特征）
pairs=[("PERFUSION_x_RESP","bicarbonate.last","resp_rate.last","碳酸氢根（mmol/L）","呼吸频率（次/分）"),
       ("PERFUSION_x_RENAL","bicarbonate.last","urea_nitrogen.last","碳酸氢根（mmol/L）","尿素氮（mg/dL）")]
fig,axs=plt.subplots(1,2,figsize=(11.5,5))
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list("div",["#1c5cab","#86b6ef","#f0efec","#f2a07f","#c4402a"])
for ax,(g,fa,fb,la,lb) in zip(axs,pairs):
    c=C[g]; xa=X[:,nm.index(fa)]; xb=X[:,nm.index(fb)]; ok=np.isfinite(xa)&np.isfinite(xb)
    qa=np.quantile(xa[ok],np.linspace(0,1,6)); qb=np.quantile(xb[ok],np.linspace(0,1,6))
    ia=np.clip(np.searchsorted(qa,xa[ok],side="right")-1,0,4); ib=np.clip(np.searchsorted(qb,xb[ok],side="right")-1,0,4)
    M=np.full((5,5),np.nan); cc=c[ok]-c[ok].mean()
    for i in range(5):
        for j in range(5):
            s=(ia==i)&(ib==j)
            if s.sum()>=30: M[j,i]=np.exp(cc[s].mean())
    lim=np.nanmax(np.abs(np.log(M)))
    im=ax.imshow(np.log(M),origin="lower",cmap=cmap,norm=TwoSlopeNorm(0,-lim,lim),aspect="auto")
    for i in range(5):
        for j in range(5):
            if np.isfinite(M[j,i]): ax.text(i,j,f"{M[j,i]:.2f}",ha="center",va="center",fontsize=9,color=INK)
    fmt=lambda q:[f"{q[k]:.3g}–{q[k+1]:.3g}" for k in range(5)]
    ax.set_xticks(range(5)); ax.set_xticklabels(fmt(qa),rotation=30,fontsize=8.5); ax.set_yticks(range(5)); ax.set_yticklabels(fmt(qb),fontsize=8.5)
    ax.set_xlabel(la); ax.set_ylabel(lb); ax.grid(False)
    ax.set_title(f"{cn(g.replace('_x_','×'))}\n联动项的风险倍数（n={ok.sum()}）",loc="left",fontsize=11,color=INK)
fig.suptitle("系统间联动：低碳酸氢根合并呼吸增快或尿素氮升高时风险最高",x=0.01,ha="left",fontsize=13,fontweight="bold",color=INK)
fig.text(0.01,0.895,"格内数字为该联动项相对平均的风险倍数（只含联动项本身，效应幅度约±20%）；按五分位分组，少于30个时点的格子留空",fontsize=10,color=INK2)
fig.tight_layout(rect=[0,0,1,0.88]); save(fig,"f7_interactions")
# ---------- F8 出事前的时间轨迹
sc=[]
for g,rows in J["trajectory"].items():
    r=np.array(rows); sc.append((r[r[:,1]<=24][:,2].mean()-r[r[:,0]>=36][:,2].mean(),g))
show=[g for _,g in sorted(sc,reverse=True)[:6]]   # 按“事件前24小时相对36–72小时的抬升”排序，数据决定展示哪6项
fig,axs=plt.subplots(2,3,figsize=(13,6.4),sharey=True)
for ax,g in zip(axs.ravel(),show):
    rows=np.array(J["trajectory"][g]); xm=-(rows[:,0]+rows[:,1])/2; m=rows[:,2]; se=rows[:,3]
    ax.fill_between(xm,np.exp(m-1.96*se),np.exp(m+1.96*se),color=BLUE,alpha=.15,lw=0); ax.plot(xm,np.exp(m),"o-",color=BLUE,lw=2,ms=4)
    ax.axhline(1,color=MUTED,lw=1); ax.axvspan(-24,0,color="#f0efec",zorder=0); ax.set_xlim(-72,0); ax.set_xticks([-72,-48,-24,0])
    ax.set_title(cn(g),loc="left",fontsize=11,color=INK)
for ax in axs[1]: ax.set_xlabel("距事件的小时数")
for ax in axs[:,0]: ax.set_ylabel("相对未出事患者的风险倍数")
fig.suptitle("出事前约24–30小时，以灌注为中心的系统联动信号率先抬升",x=0.01,ha="left",fontsize=13,fontweight="bold",color=INK)
fig.text(0.01,0.925,f"出事患者各时段的折外系统贡献（{J['n_event_patients_with_lead']}名出事患者，{J['n_event_landmarks']}个时点），以未出事患者的平均为1；阴影为事件前24小时；展示抬升最明显的6项",fontsize=10,color=INK2)
fig.tight_layout(rect=[0,0,1,0.91]); save(fig,"f8_trajectory")
print("ok")
