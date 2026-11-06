"""Create post-hoc held-out examples with the exported SOTA ONNX heatmaps."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import onnxruntime as ort
from train import CLASSES,image_tensor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',default='outputs/ObjectLab/results/qualitative_cam_examples.png');ap.add_argument('--model',default='outputs/ObjectLab/models/sota.onnx')
    a=ap.parse_args();data=Path(a.data);rows=list(csv.DictReader(open('outputs/ObjectLab/results/mobilenetv4_aug_1_predictions.csv',encoding='utf-8-sig')))
    correct=[r for r in rows if r['true_label']==r['predicted_label']]
    errors=[r for r in rows if r['true_label']!=r['predicted_label']]
    chosen=[correct[0],correct[1],*errors[:2]]
    sess=ort.InferenceSession(a.model,providers=['CPUExecutionProvider']);fig,axes=plt.subplots(len(chosen),4,figsize=(14,3.6*len(chosen)))
    for i,r in enumerate(chosen):
        path=data/Path(r['path']); x=image_tensor(path).unsqueeze(0).numpy(); probs,maps=sess.run(None,{'images':x})
        with Image.open(path) as src: canvas=ImageOps.pad(ImageOps.exif_transpose(src).convert('RGB'),(224,224),method=Image.Resampling.LANCZOS,color=(127,127,127))
        img=np.asarray(canvas); true_i=CLASSES.index(r['true_label']);pred_i=CLASSES.index(r['predicted_label'])
        axes[i,0].imshow(img);axes[i,0].set_title(f"{r['true_label']} · {Path(r['path']).name}\n{'correct' if true_i==pred_i else 'incorrect'}")
        for j,ci,title in [(1,true_i,'True-class'),(2,pred_i,'Predicted-class')]:
            heat=Image.fromarray(np.uint8(np.clip(maps[0,ci],0,1)*255)).resize((224,224),Image.Resampling.BILINEAR)
            axes[i,j].imshow(img);axes[i,j].imshow(np.asarray(heat),cmap='inferno',alpha=.46,vmin=0,vmax=255)
            axes[i,j].set_title(f'{title} CAM · {CLASSES[ci]}')
        p=np.asarray(probs[0]);axes[i,3].barh(CLASSES,p,color=['#8eb9d4' if c!=r['predicted_label'] else '#d47a58' for c in CLASSES]);axes[i,3].invert_yaxis();axes[i,3].set_xlim(0,1);axes[i,3].set_xlabel('Probability');axes[i,3].set_title('ONNX prediction')
        for ax in axes[i]:ax.axis('off') if ax is not axes[i,3] else None
        axes[i,3].axis('on')
    fig.suptitle('MobileNetV4 held-out test examples · test set n=81 · Grad-CAM overlays',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180,bbox_inches='tight');plt.close(fig)
    print(out)
if __name__=='__main__':main()
