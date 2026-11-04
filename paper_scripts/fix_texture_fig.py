"""Clean the Kaggle texture figure (overlapping auto-titles): re-letter panels and re-draw y labels."""
from PIL import Image, ImageDraw, ImageFont
import matplotlib, os, io, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt
src, dst = sys.argv[1], sys.argv[2]
im=Image.open(src).convert("RGB"); W,H=im.size; d=ImageDraw.Draw(im)
d.rectangle([0,0,W,123],fill="white"); d.rectangle([0,1160,W,1289],fill="white")
d.rectangle([126,124,202,1150],fill="white"); d.rectangle([1740,124,1816,1150],fill="white")
fig=plt.figure(figsize=(1,8),dpi=300); fig.text(0.5,0.5,r"log$_{10}$ normalised power",rotation=90,ha="center",va="center",fontsize=16.5,family="DejaVu Sans")
buf=io.BytesIO(); fig.savefig(buf,format="png",facecolor="white",bbox_inches="tight",pad_inches=0.02); plt.close(fig)
lab=Image.open(buf).convert("RGB")
for xc in (166,1780): im.paste(lab,(xc-lab.width//2, 484-lab.height//2))
fp=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/DejaVuSans.ttf")
fb=os.path.join(os.path.dirname(matplotlib.__file__),"mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
f=ImageFont.truetype(fp,58); fB=ImageFont.truetype(fb,62)
for (x,y,tag,txt) in [(330,38,"A","Normal: radially averaged power spectrum"),(2000,38,"B","Cancer: radially averaged power spectrum"),
                      (330,1205,"C","Normal: uniform LBP (P = 8, R = 1)"),(2000,1205,"D","Cancer: uniform LBP (P = 8, R = 1)")]:
    d.text((x,y),tag,font=fB,fill="black"); d.text((x+80,y+3),txt,font=f,fill="black")
im.save(dst)
