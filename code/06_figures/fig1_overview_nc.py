import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"Liberation Sans","font.size":7,"pdf.fonttype":42})
W,H=180/25.4,100/25.4
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,180); ax.set_ylim(0,100); ax.axis("off")
BLUE,ORANGE,GREY,INK,LIGHT,PALE="#2a78d6","#eb6834","#8a8984","#1d1d1b","#eaf2fb","#fbeee8"
def box(x,y,w,h,fc,ec="none",r=2): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=0.6))
def lab(x,y,s): ax.text(x,y,s,fontsize=8,fontweight="bold",color=INK,va="top")
def arrow(x0,x1,y): ax.annotate("",xy=(x1,y),xytext=(x0,y),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=1))
# a data and design
lab(2,99,"a"); box(4,4,38,90,LIGHT); ax.text(23,90,"Data and design",ha="center",fontsize=7.5,fontweight="bold")
rows=[("MIMIC-IV, single centre",""),("Development","3,924 patients, 2008–2016"),("Held-out later period","2,470 patients, 2017–2022\nopened once, after freezing"),("eICU-CRD, 108 hospitals",""),("External validation","4,977 ICU patients, 2014–2015\nprotocol amended before use")]
y=83
for a,b in rows:
    if b=="": ax.text(7,y,a,fontsize=7,fontweight="bold",color=BLUE); y-=6
    else:
        ax.text(9,y,a,fontsize=6.5,fontweight="bold",color=INK); ax.text(9,y-3.6,b,fontsize=6,color=INK,va="top"); y-=13 if "\n" in b else 10
ax.text(23,13,"Every 6 h: predict death or unplanned\nvasopressor, inotrope, MCS or\ninvasive ventilation within 24 h",ha="center",va="center",fontsize=6,color=INK)
arrow(42.5,46.5,49)
# b model
lab(46,99,"b"); box(47,4,44,90,"white",ec=GREY); ax.text(69,90,"Organ-system model",ha="center",fontsize=7.5,fontweight="bold")
systems=["Haemodynamic","Perfusion","Respiratory","Renal","Myocardial injury","Electrical","Haematological","Inflammatory","Metabolic","Neurological","Baseline"]
for i,s in enumerate(systems):
    c,r=i%2,i//2; x=50+c*19.5; yy=83-r*6.6
    box(x,yy-2.4,18,5,LIGHT,r=1); ax.text(x+9,yy+0.1,s,ha="center",va="center",fontsize=5.8)
ax.text(69,38,"Boosted trees may combine predictors\nonly within 11 systems and\n8 prespecified system pairs",ha="center",va="center",fontsize=6)
box(51,10,36,18,ORANGE,r=1.5)
ax.text(69,22.5,"log-odds = sum of named",ha="center",va="center",color="white",fontsize=6.5,fontweight="bold")
ax.text(69,17.5,"system and pair terms",ha="center",va="center",color="white",fontsize=6.5,fontweight="bold")
ax.text(69,12.8,"(exact decomposition)",ha="center",va="center",color="white",fontsize=5.8)
arrow(91.5,95.5,49)
# c temporal
lab(95,99,"c"); box(96,4,40,90,LIGHT); ax.text(116,90,"Held-out period",ha="center",fontsize=7.5,fontweight="bold")
ax.text(99,84,"AUROC",fontsize=6.5,fontweight="bold"); ax.text(133,84,"bars from 0.5",fontsize=5,color=GREY,ha="right")
vals=[("Organ-system",0.873,BLUE),("Unconstrained",0.870,GREY),("Logistic reg.",0.842,GREY),("GRACE",0.763,GREY),("NEWS2",0.715,GREY)]
for i,(n,v,c) in enumerate(vals):
    yy=79-i*5.2; ax.text(99,yy,n,va="center",fontsize=6); L=(v-0.5)*40
    ax.add_patch(plt.Rectangle((116,yy-1.6),L,3.2,color=c)); ax.text(116.8+L,yy,f"{v:.3f}",va="center",fontsize=6)
ax.text(99,50,"Assessments flagged to detect\n84% of episodes",va="top",fontsize=6.5,fontweight="bold")
for i,(n,v,c) in enumerate([("Organ-system",13.5,BLUE),("NEWS2",48.5,ORANGE)]):
    yy=38-i*5.5; ax.text(99,yy,n,va="center",fontsize=6); L=v/50*12
    ax.add_patch(plt.Rectangle((116,yy-1.7),L,3.4,color=c)); ax.text(116.8+L,yy,f"{v}%",va="center",fontsize=6)
ax.text(116,20,"Median lead time 18 h",ha="center",fontsize=6.5,fontweight="bold",color=BLUE)
ax.text(116,12,"Largest term:\nperfusion–renal",ha="center",va="center",fontsize=6)
arrow(136.5,140.5,49)
# d external
lab(140,99,"d"); box(141,4,37,90,PALE); ax.text(159.5,90,"External: eICU-CRD",ha="center",fontsize=7.5,fontweight="bold")
ax.text(144,84,"AUROC, without retraining",fontsize=6.5,fontweight="bold")
ext=[("Organ-system",0.615,BLUE),("Unconstrained",0.613,GREY),("GRACE",0.587,GREY),("NEWS2",0.599,GREY)]
for i,(n,v,c) in enumerate(ext):
    yy=79-i*5.2; ax.text(144,yy,n,va="center",fontsize=6); L=(v-0.5)*40
    ax.add_patch(plt.Rectangle((160,yy-1.6),L,3.2,color=c)); ax.text(160.8+L,yy,f"{v:.3f}",va="center",fontsize=6)
ax.text(144,54,"AUROC, refitted on half of\nhospitals (post hoc)",va="top",fontsize=6.5,fontweight="bold")
for i,(n,v,c) in enumerate([("Organ-system",0.752,BLUE),("Unconstrained",0.756,GREY)]):
    yy=42-i*5.2; ax.text(144,yy,n,va="center",fontsize=6); L=(v-0.5)*40
    ax.add_patch(plt.Rectangle((160,yy-1.6),L,3.2,color=c)); ax.text(160.8+L,yy,f"{v:.3f}",va="center",fontsize=6)
ax.text(159.5,22,"Term ranking preserved\n(Spearman ρ = 0.93)",ha="center",va="center",fontsize=6,fontweight="bold",color=BLUE)
ax.text(159.5,11,"Structure transfers;\nparameters need local updating",ha="center",va="center",fontsize=6)
fig.savefig("fig_nc/Fig1_overview.pdf"); fig.savefig("fig_nc/Fig1_overview.png",dpi=300); fig.savefig("fig_nc/Fig1_overview.tif",dpi=600,pil_kwargs={"compression":"tiff_lzw"})
print("ok")
