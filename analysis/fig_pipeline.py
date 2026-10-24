import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","font.size":8.5})
fig,ax=plt.subplots(figsize=(10,4.3)); ax.set_xlim(0,100); ax.set_ylim(-5,47); ax.axis("off")
def box(x,y,w,h,txt,fc="#f4f4f4",ec="#555",fs=8.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,weight="bold" if bold else "normal",wrap=True)
def arr(x1,y1,x2,y2,txt=None,c="#333",dy=1.2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=10,color=c,lw=1))
    if txt: ax.text((x1+x2)/2,(y1+y2)/2+dy,txt,ha="center",va="bottom",fontsize=7.2,color=c)
# clients
for i,(y,lab) in enumerate([(33,"Client 1 (silo)"),(19,"Client 2 (silo)"),(3,"Client n (silo)")]):
    box(1,y,17,9,f"{lab}\nlocal flow records $D_i$\n(never leave the silo)",fc="#e8eef7")
ax.text(9.5,14.5,"...",ha="center",fontsize=14)
# stages
box(27,34,26,9.5,"Step 0 — DP bin sketch\nnoisy 1024-cell histograms\nper feature (1 release)",fc="#fbeee0")
box(27,19,26,11,"Stage 1 — DP-FedSGD warm start\n(optional)\nclipped, noised logistic\ngradients ($T_1$ releases)",fc="#fbeee0")
box(27,2,26,12.5,"Stage 2 — federated histogram\nGBDT, level-wise\nnoised $G$/$H$ histograms\nper node, feature, bin\n($T\\cdot D$ releases)",fc="#f6dcdc")
for y in [38.5,24.5,8]: arr(18.8,y,26.3,y)
ax.text(22.5,44.5,"each client adds Gaussian noise to every upload",ha="center",fontsize=7.2,color="#a33")
# server
box(62,14,17,20,"Server\n\nrobust aggregation\n(centred clipping; or\nMulti-Krum / trimmed\nmean / median)\n\nsplit selection,\nleaf values",fc="#eaeaea")
for y in [38.5,24.5,8]: arr(53.8,y,61.4,min(max(y,16),32))
ax.plot([70.5,70.5,9.5],[13.4,-3,-3],ls="--",color="#2b5c9e",lw=1)
arr(9.5,-3,9.5,2.4,c="#2b5c9e")
ax.text(40,-3,"server broadcasts bin edges, warm-start weights and split rules; clients route their own records",ha="center",fontsize=7.4,color="#2b5c9e",va="center",bbox=dict(fc="white",ec="none",pad=1))
# outputs
box(84,26,15,10,"Global model\n$F_0$ + 50 trees\n(release = post-\nprocessing)",fc="#e3f0e3")
box(84,8,15,12,"TreeSHAP\nglobal / local\nattributions\n(no extra\nprivacy cost)",fc="#e3f0e3")
arr(79.6,28,83.4,30); arr(91.5,25.4,91.5,20.6)
ax.text(70.5,40.5,"accountant: R = 1 + $T_1$ + $T\\cdot D$ Gaussian releases\n→ record-level ($\\varepsilon,\\delta$)-DP (RDP)",ha="center",fontsize=7.6,
        bbox=dict(fc="white",ec="#a33",boxstyle="round,pad=0.3"))
plt.savefig("../figures/fig_pipeline.png",dpi=300,bbox_inches="tight"); print("ok")
