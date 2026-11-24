import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"Liberation Sans","font.size":10})
W,H=18/2.54,12.5/2.54
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,180); ax.set_ylim(0,125); ax.axis("off")
BLUE,ORANGE,GREY,INK,LIGHT="#2a78d6","#eb6834","#8a8984","#1d1d1b","#eaf2fb"
def box(x,y,w,h,fc,ec="none",r=2.5): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=1))
# title
box(0,111,180,14,BLUE,r=0); ax.text(90,118,"Interpretable organ-system early warning after acute myocardial infarction",ha="center",va="center",color="white",fontsize=12,fontweight="bold")
# panel 1 data
box(4,8,48,98,LIGHT); ax.text(28,100,"Data",ha="center",fontsize=11,fontweight="bold",color=INK)
for i,(a,b) in enumerate([("MIMIC-IV","first AMI admissions"),("3,924 patients","development 2008–2016"),("2,470 patients","held-out 2017–2022"),("Every 6 h","labs, vital signs, urine,\nECG, baseline")]):
    yy=90-i*17; ax.text(28,yy,a,ha="center",fontsize=11,fontweight="bold",color=BLUE); ax.text(28,yy-5,b,ha="center",va="top",fontsize=9,color=INK)
ax.text(28,16,"Outcome: death or unplanned\nvasopressor, inotrope, MCS or\nventilation within 24 h",ha="center",va="center",fontsize=8.5,color=INK)
ax.annotate("",xy=(60,57),xytext=(53,57),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=2))
# panel 2 model
box(61,8,54,98,"white",ec=GREY); ax.text(88,100,"Organ-system model",ha="center",fontsize=11,fontweight="bold",color=INK)
systems=["Haemodynamic","Perfusion","Respiratory","Renal","Injury","Electrical","Haematology","Inflammation","Metabolic","Neurological","Baseline"]
for i,s in enumerate(systems):
    c,r=i%2,i//2; x=65+c*24.5; y=90-r*8.2
    box(x,y-3,22,6.2,LIGHT,r=1.5); ax.text(x+11,y+0.1,s,ha="center",va="center",fontsize=8,color=INK)
ax.text(88,38,"Trees combine variables only within\n11 systems + 8 prespecified pairs",ha="center",va="center",fontsize=8.8,color=INK)
box(65,13,46,17,ORANGE,r=2); ax.text(88,24.5,"Risk = exact sum of",ha="center",va="center",color="white",fontsize=9.5,fontweight="bold")
ax.text(88,18,"named organ-system terms",ha="center",va="center",color="white",fontsize=9.5,fontweight="bold")
ax.annotate("",xy=(123,57),xytext=(116,57),arrowprops=dict(arrowstyle="-|>",color=GREY,lw=2))
# panel 3 results
box(124,8,52,98,LIGHT); ax.text(150,100,"Held-out results",ha="center",fontsize=11,fontweight="bold",color=INK)
ax.text(128,92,"AUROC",fontsize=9,fontweight="bold",color=INK)
vals=[("Organ-system",0.873,BLUE),("Unconstrained",0.870,GREY),("Logistic reg.",0.842,GREY),("GRACE",0.763,GREY),("NEWS2",0.715,GREY)]
for i,(n,v,c) in enumerate(vals):
    y=86-i*6.3; ax.text(128,y,n,va="center",fontsize=8.5,color=INK); L=(v-0.6)/0.3*20
    ax.add_patch(plt.Rectangle((150,y-2),L,4,color=c)); ax.text(151+L,y,f"{v:.3f}",va="center",fontsize=8.5,color=INK)
ax.text(128,55,"Assessments flagged to\ndetect 84% of episodes",va="top",fontsize=9,fontweight="bold",color=INK)
for i,(n,v,c) in enumerate([("Organ-system",13.5,BLUE),("NEWS2",48.5,ORANGE)]):
    y=40-i*7; ax.text(128,y,n,va="center",fontsize=8.5,color=INK); L=v/50*14
    ax.add_patch(plt.Rectangle((150,y-2.2),L,4.4,color=c)); ax.text(151+L,y,f"{v}%",va="center",fontsize=8.5,color=INK)
ax.text(150,24,"Median lead time 18 h",ha="center",fontsize=9.5,fontweight="bold",color=BLUE)
ax.text(150,15,"Leading signal:\nperfusion–renal compromise",ha="center",va="center",fontsize=9,color=INK)
fig.savefig("fig_acvc/Graphical_abstract.tif",dpi=600,pil_kwargs={"compression":"tiff_lzw"}); fig.savefig("fig_acvc/Graphical_abstract.png",dpi=300); fig.savefig("fig_acvc/Graphical_abstract.pdf")
print("ok")
