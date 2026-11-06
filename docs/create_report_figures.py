from pathlib import Path
import csv, math
from PIL import Image, ImageDraw, ImageFont
script=Path(__file__).resolve(); root=next((p for p in script.parents if (p/'results').is_dir() and (p/'docs').is_dir() and (p/'ml').is_dir()), script.parents[1]/'outputs'/'ObjectLab')
res=root/'results'
series=[]
for name,label,color in [('baseline_aug_1','Scratch CNN','#277DA1'),('mobilenetv4_aug_0','MobileNetV4 • aug off','#43AA8B'),('mobilenetv4_aug_1','MobileNetV4 • aug on','#F9844A')]:
    with (res/f'{name}_history.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    series.append((label,color,[{k:float(r[k]) for k in ('epoch','train_loss','val_loss','train_accuracy','val_accuracy')} for r in rows]))
W,H=1500,850
im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
fontpath=Path('C:/Windows/Fonts/arial.ttf'); boldpath=Path('C:/Windows/Fonts/arialbd.ttf')
def font(sz,b=False): return ImageFont.truetype(str(boldpath if b else fontpath),sz)
d.text((W//2,20),'Training histories (seed 2)',fill='#16283F',font=font(30,True),anchor='mt')
plots=[('Loss', 'train_loss','val_loss',0,1.5),('Accuracy','train_accuracy','val_accuracy',0,1.0)]
for pi,(title,trainkey,valkey,ymin,ymax) in enumerate(plots):
    x0,y0=100,95+pi*365; pw,ph=1280,270
    d.text((x0,y0),title,fill='#16283F',font=font(24,True))
    top=y0+40; left=x0+55; right=x0+pw; bottom=top+ph
    d.line((left,top,left,bottom,right,bottom),fill='#596A7C',width=2)
    for j in range(6):
       v=ymin+(ymax-ymin)*j/5; yy=bottom-(v-ymin)/(ymax-ymin)*ph
       d.line((left,yy,right,yy),fill='#E2E7EB',width=1)
       d.text((left-12,yy),f'{v:.1f}',fill='#596A7C',font=font(16),anchor='rm')
    maxepoch=max(len(s[2]) for s in series)
    for label,color,rows in series:
      n=len(rows)
      for key,dash in ((trainkey,False),(valkey,True)):
        pts=[]
        for r in rows:
          x=left+(r['epoch']-1)/max(1,maxepoch-1)*(right-left)
          y=bottom-(r[key]-ymin)/(ymax-ymin)*ph
          pts.append((int(x),int(max(top,min(bottom,y)))) )
        if not dash:
          d.line(pts,fill=color,width=4)
        else:
          for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
             length=math.hypot(x2-x1,y2-y1); steps=max(1,int(length/10))
             for k in range(0,steps,2):
                a=k/steps;b=min(1,(k+1)/steps)
                d.line((x1+(x2-x1)*a,y1+(y2-y1)*a,x1+(x2-x1)*b,y1+(y2-y1)*b),fill=color,width=3)
    # x ticks
    for e in (1,10,20,30):
      if e<=maxepoch:
       xx=left+(e-1)/max(1,maxepoch-1)*(right-left)
       d.text((xx,bottom+8),str(e),fill='#596A7C',font=font(15),anchor='mt')
    d.text(((left+right)//2,bottom+28),'Epoch',fill='#596A7C',font=font(16),anchor='mt')
# legend
y=825; x=115
for label,color,_ in series:
    d.line((x,y,x+42,y),fill=color,width=4); d.text((x+50,y),label,fill='#273444',font=font(16),anchor='lm'); x+=370
im.save(res/'training_curves_comparison.png',quality=95)

# A compact qualitative panel: one correct bottle and one scissors-to-pens
# error from the four-case held-out Grad-CAM figure prepared by the ML pipeline.
cam_source=res/'qualitative_cam_examples.png'
if cam_source.exists():
    source=Image.open(cam_source).convert('RGB')
    selected=[source.crop((0,80,source.width,690)),source.crop((0,1280,source.width,1885))]
    panel=Image.new('RGB',(source.width,1295),'white')
    draw=ImageDraw.Draw(panel)
    draw.text((source.width//2,18),'MobileNetV4 · held-out examples · Grad-CAM overlays',font=ImageFont.truetype(str(boldpath),34),fill='#16283F',anchor='mt')
    panel.paste(selected[0],(0,60)); panel.paste(selected[1],(0,690))
    panel.save(res/'qualitative_cam_selected.png')


