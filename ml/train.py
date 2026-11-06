"""Reproduce Assignment 1 CNN and train a MobileNetV4 transfer baseline.

The input contract is RGB float pixels in [0,1]. Preprocessing mirrors A1:
EXIF transpose, RGB conversion, Lanczos aspect-preserving resize, grey-127 pad.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

CLASSES = ['bottles', 'cups', 'pens', 'scissors']
SPLITS = ['train', 'val', 'test']
SIZE = 224
SEED = 2
# Stable id for the baseline implementation already in the active run before
# the cached-feature SOTA optimization was added. Baseline behavior is unchanged.
BASELINE_CODE_ID = 'c3365deed838b2ba637a1e06b8f10175ca0ece8c2da4ed3cc5e6fc97223eccba'

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(6, max(1, os.cpu_count() or 1)))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def image_tensor(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        canvas = ImageOps.pad(im, (SIZE,SIZE), method=Image.Resampling.LANCZOS, color=(127,127,127))
        return TF.to_tensor(canvas)

def image_identity(path):
    """Encoding-independent decoded RGB digest and 64-bit DCT pHash."""
    with Image.open(path) as src:
        im=ImageOps.exif_transpose(src).convert('RGB')
    pixel_sha=hashlib.sha256((f'{im.width}x{im.height}:'.encode())+im.tobytes()).hexdigest()
    gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    n=32; k=np.arange(n)[:,None]; t=np.arange(n)[None,:]
    basis=np.cos(np.pi/n*(t+0.5)*k); basis[0,:]*=1/math.sqrt(2); basis*=math.sqrt(2/n)
    coeff=basis@gray@basis.T; low=coeff[:8,:8].flatten(); median=np.median(low[1:])
    bits=low>median; phash=f'{sum(int(bit)<<idx for idx,bit in enumerate(bits)):016x}'
    return pixel_sha,phash

def hamming_hex(a,b): return (int(a,16)^int(b,16)).bit_count()

def augment_tensor(x,rng=random):
    if rng.random()<.5: x=TF.hflip(x)
    angle=rng.uniform(-18,18)
    x=TF.rotate(x,angle,interpolation=InterpolationMode.BILINEAR,fill=[127/255]*3)
    factor=rng.uniform(.9,1.1); mean=x.mean((1,2),keepdim=True)
    return ((x-mean)*factor+mean).clamp(0,1)

class FolderSet(Dataset):
    def __init__(self, root, split, augment=False, valid_rel=None):
        self.items=[]; self.augment=augment
        root=Path(root)
        for ci, cls in enumerate(CLASSES):
            d=Path(root)/split/cls
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'} and (valid_rel is None or p.relative_to(root) in valid_rel): self.items.append((p,ci))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]; x=image_tensor(p)
        if self.augment: x=augment_tensor(x)
        return x,y,str(p)

class ScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.ModuleList([nn.Conv2d(3,16,3,padding=1),nn.Conv2d(16,32,3,padding=1),nn.Conv2d(32,64,3,padding=1)])
        self.pool=nn.MaxPool2d(2); self.fc1=nn.Linear(64,64); self.drop=nn.Dropout(.3); self.fc2=nn.Linear(64,4)
    def features(self,x):
        for c in self.convs: x=self.pool(torch.relu(c(x)))
        return x
    def forward(self,x):
        f=self.features(x); z=f.mean((2,3)); h=torch.relu(self.fc1(z)); return self.fc2(self.drop(h))

class TransferNet(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.backbone=backbone; self.channels=backbone.feature_info.channels()[-1]; self.head=nn.Linear(self.channels,4)
    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self
    def feature_map(self,x): return self.backbone(x)[0]
    def forward(self,x): return self.head(self.feature_map(x).mean((2,3)))

class CachedHeadTrainer(nn.Module):
    """Shares TransferNet state names while training a frozen backbone's cached GAP vectors."""
    def __init__(self, backbone, head=None):
        super().__init__(); self.backbone=backbone; self.head=head or nn.Linear(backbone.feature_info.channels()[-1],4)
        for p in self.backbone.parameters(): p.requires_grad=False
        self.backbone.eval()
    def train(self,mode=True): super().train(mode); self.backbone.eval(); return self
    def forward(self,features): return self.head(features)

class FeatureSet(Dataset):
    def __init__(self,features,labels,paths): self.features=features;self.labels=labels;self.paths=paths
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        view=random.randrange(self.features.shape[1]) if self.features.shape[1]>1 else 0
        return self.features[i,view],int(self.labels[i]),self.paths[i]

def cache_feature_bank(dataset, model, mean, std, device, n_views=1, augment_views=False, batch_size=16, label='features'):
    model.eval(); inputs=[]; metadata=[]; feats=[]; encoded=0
    def encode_pending():
        nonlocal encoded,inputs
        if not inputs: return
        batch=torch.stack(inputs).to(device); batch=(batch-mean.to(device))/std.to(device)
        with torch.no_grad(): pooled=model(batch)[0].mean((2,3)).cpu()
        feats.extend(pooled.unbind()); encoded+=len(inputs); inputs=[]
        if (encoded//batch_size)%10==0: print(f'{label}: encoded {encoded}/{len(dataset.items)*n_views}',flush=True)
    for i,(path,y) in enumerate(dataset.items):
        views=[]; original=image_tensor(path); views.append(original)
        for vi in range(1,n_views):
            rng=random.Random(SEED+7919*i+104729*vi)
            views.append(augment_tensor(original.clone(),rng) if augment_views else original.clone())
        metadata.append((y,str(path)))
        for view in views:
            inputs.append(view)
            if len(inputs)>=batch_size: encode_pending()
    encode_pending()
    n=len(dataset.items); x=torch.stack(feats).reshape(n,n_views,-1)
    return x,[y for y,_ in metadata],[p for _,p in metadata]

def augmentation_batch(x):
    # Applied per-item for compatibility across torchvision versions.
    return x

def metrics(logits, ys):
    pred=np.asarray(logits).argmax(1); y=np.asarray(ys)
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(p.mean()),'macro_recall':float(r.mean()),'macro_f1':float(f.mean()),
            'per_class':{c:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i])} for i,c in enumerate(CLASSES)},
            'confusion_matrix':confusion_matrix(y,pred,labels=range(4)).tolist()}

def run_epoch(model, loader, optimizer, device, mean=None, std=None):
    training=optimizer is not None; model.train(training); loss_sum=0.; sample_count=0; logits_all=[]; ys=[]; rows=[]
    for x,y,paths in loader:
        x=x.to(device); y=y.to(device)
        if mean is not None: x=(x-mean)/std
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out=model(x); loss=nn.functional.cross_entropy(out,y)
            if training: loss.backward(); optimizer.step()
        n=len(y); loss_sum+=float(loss.detach().cpu())*n; sample_count+=n
        logits_all.extend(out.detach().cpu().numpy()); ys.extend(y.cpu().numpy()); rows.extend(paths)
    return loss_sum/sample_count, metrics(logits_all,ys), logits_all, ys, rows

def fit(model, train_loader, val_loader, device, outdir, name, mean=None,std=None, epochs=30, signature=None):
    model.to(device); optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    best=float('inf'); best_state=None; wait=0; plateau=0; lr_wait=0; lr=1e-3; history=[]
    for epoch in range(1,epochs+1):
        t=time.time(); tl,tm,*_=run_epoch(model,train_loader,optimizer,device,mean,std)
        vl,vm,*_=run_epoch(model,val_loader,None,device,mean,std)
        history.append({'epoch':epoch,'train_loss':tl,'train_accuracy':tm['accuracy'],'val_loss':vl,'val_accuracy':vm['accuracy'],'lr':optimizer.param_groups[0]['lr'],'seconds':time.time()-t})
        print(f'{name} epoch {epoch}: train loss {tl:.4f} acc {tm["accuracy"]:.3f}; val loss {vl:.4f} acc {vm["accuracy"]:.3f}',flush=True)
        if vl < best-1e-12:
            best=vl; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0; lr_wait=0
        else:
            wait+=1; lr_wait+=1
            if lr_wait>=3:
                lr=max(optimizer.param_groups[0]['lr']*.5,1e-6)
                for pg in optimizer.param_groups: pg['lr']=lr
                lr_wait=0
            if wait>=7: break
    model.load_state_dict(best_state); outdir.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),outdir/f'{name}.pt')
    (outdir/f'{name}.meta.json').write_text(json.dumps({'signature':signature},indent=2))
    with (outdir.parent/'results'/f'{name}_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    try:
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for ax,field,title in zip(axes,['loss','accuracy'],['Loss','Accuracy']):
            ax.plot([r['epoch'] for r in history],[r[f'train_{field}'] for r in history],label='Train')
            ax.plot([r['epoch'] for r in history],[r[f'val_{field}'] for r in history],label='Validation')
            ax.set(xlabel='Epoch',ylabel=title,title=title);ax.legend()
        fig.tight_layout();fig.savefig(outdir.parent/'results'/f'{name}_curves.png',dpi=150);plt.close(fig)
    except ImportError: pass
    return model,history

class BaselineExport(nn.Module):
    """Adds analytical Grad-CAM: exact gradient chain through GAP->Dense64 ReLU."""
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self,x):
        f=self.model.features(x); z=f.mean((2,3)); h=torch.relu(self.model.fc1(z)); logits=self.model.fc2(h); probs=torch.softmax(logits,1)
        # d logit_c / d GAP_j, including Dense64 ReLU gate.
        weights=torch.matmul(self.model.fc2.weight.unsqueeze(0), self.model.fc1.weight.unsqueeze(0) * (h>0).to(h.dtype).unsqueeze(2))
        weights=weights/(f.shape[2]*f.shape[3])
        # weights [N,4,64], CAM=sum_j weights[c,j]*feature[j,u,v]
        cam=torch.einsum('ncj,njhw->nchw',weights,f).relu()
        flat=cam.flatten(2); lo=flat.amin(2).view(-1,4,1,1); hi=flat.amax(2).view(-1,4,1,1)
        cam=(cam-lo)/(hi-lo+1e-8)
        return probs,cam

class TransferExport(nn.Module):
    def __init__(self, model, mean, std): super().__init__(); self.model=model; self.register_buffer('mean',mean); self.register_buffer('std',std)
    def forward(self,x):
        xn=(x-self.mean)/self.std
        # For GAP + linear classifier CAM is exactly Grad-CAM before ReLU.
        fmap=self.model.feature_map(xn)
        pooled=fmap.mean((2,3)); logits=self.model.head(pooled); probs=torch.softmax(logits,1)
        cam=(torch.einsum('cj,njhw->nchw',self.model.head.weight,fmap)/(fmap.shape[2]*fmap.shape[3])).relu()
        flat=cam.flatten(2); lo=flat.amin(2).view(-1,4,1,1); hi=flat.amax(2).view(-1,4,1,1)
        return probs,(cam-lo)/(hi-lo+1e-8)

def export_validate(wrapper, model, path, sample, device):
    import onnx, onnxruntime as ort
    wrapper.eval().cpu(); sample=sample.cpu()
    torch.onnx.export(wrapper,sample,str(path),input_names=['images'],output_names=['probabilities','heatmaps'],opset_version=17,dynamic_axes={'images':{0:'batch'},'probabilities':{0:'batch'},'heatmaps':{0:'batch'}},dynamo=False)
    proto=onnx.load(str(path))
    if proto.ir_version>10: proto.ir_version=10
    for op_set in proto.opset_import:
        if op_set.domain in ('','ai.onnx') and op_set.version>17: raise RuntimeError(f'ONNX opset {op_set.version} exceeds Android target 17')
    props={'classes':json.dumps(CLASSES),'input_contract':'float32 RGB [0,1], NCHW 224x224','probability_contract':'softmax probabilities ordered as classes','heatmap_contract':'per-class positive Grad-CAM, min-max normalized to [0,1]','architecture':('scratch_cnn_16_32_64' if isinstance(wrapper,BaselineExport) else 'mobilenetv4_conv_small.e2400_r224_in1k_gap_linear')}
    for key,value in props.items():
        entry=proto.metadata_props.add(); entry.key=key; entry.value=value
    onnx.save(proto,str(path)); onnx.checker.check_model(proto)
    sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    parity={}
    for batch_n in [1,2]:
        xb=sample.repeat(batch_n,1,1,1)
        actual=sess.run(None,{'images':xb.numpy()})
        with torch.no_grad(): expected=wrapper(xb)
        diffs=[float(np.max(np.abs(a-b.numpy()))) for a,b in zip(actual,expected)]
        parity[str(batch_n)]=diffs
        if max(diffs)>2e-4: raise RuntimeError(f'ONNX parity failed for N={batch_n}: {diffs}')
    # Independently verify analytical CAM maps against autograd Grad-CAM.
    with torch.enable_grad():
        if isinstance(wrapper,BaselineExport):
            fmap=model.features(sample).detach().requires_grad_(True)
            pooled=fmap.mean((2,3)); hidden=torch.relu(model.fc1(pooled)); logits=model.fc2(hidden)
        else:
            norm=(sample-wrapper.mean)/wrapper.std
            fmap=model.feature_map(norm).detach().requires_grad_(True)
            pooled=fmap.mean((2,3)); logits=model.head(pooled)
        maps=[]
        for ci in range(4):
            grad=torch.autograd.grad(logits[0,ci],fmap,retain_graph=True)[0]
            cam=(grad.mean((2,3),keepdim=True)*fmap).sum(1,keepdim=True).relu()
            cam=(cam-cam.amin((2,3),keepdim=True))/(cam.amax((2,3),keepdim=True)-cam.amin((2,3),keepdim=True)+1e-8)
            maps.append(cam)
        gradcam=torch.cat(maps,dim=1)
    camdiff=float(np.max(np.abs(expected[1].numpy()-gradcam.detach().numpy())))
    if camdiff>2e-4: raise RuntimeError(f'Analytical Grad-CAM mismatch {path.name}: {camdiff}')
    print(f'ONNX parity {path.name}: N1/N2 max abs {parity}; autograd Grad-CAM max abs {camdiff}',flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--pretrained-weights',default=os.environ.get('OBJECTLAB_PRETRAINED_WEIGHTS')); ap.add_argument('--audit-only',action='store_true',help='Write the image manifest, decoded hashes and cross-split duplicate candidates, then stop before training.'); ap.add_argument('--reviewed-candidates',help='CSV reviewed by a human with the same candidate pairs and a nonempty decision column.')
    args=ap.parse_args(); seed_all(); root=Path(args.data); out=Path(args.out); ml=out/'ml'; models=out/'models'; results=out/'results'
    for p in [ml,models,results]:p.mkdir(parents=True,exist_ok=True)
    mpl_cache=Path.cwd()/'work'/'matplotlib_cache'; mpl_cache.mkdir(parents=True,exist_ok=True); os.environ['MPLCONFIGDIR']=str(mpl_cache)
    audit=[]; seen={}; pixel_seen={}; bad=[]; cross_split_dups=[]; within_split_duplicates=[]; duplicate_hashes=[]
    for split in SPLITS:
        for ci,c in enumerate(CLASSES):
            files=sorted(p for p in (root/split/c).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
            for p in files:
                try:
                    with Image.open(p) as im: im.verify()
                except Exception as e:
                    bad.append({'path':str(p),'error':str(e)}); continue
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                pixel_digest,phash=image_identity(p)
                audit.append({'path':str(p.relative_to(root)),'split':split,'class':c,'sha256':digest,'decoded_rgb_sha256':pixel_digest,'phash64':phash})
                if digest in seen:
                    prior=seen[digest]; duplicate_hashes.append(digest)
                    if prior[0]!=split: cross_split_dups.append({'sha256':digest,'first':prior[2],'second':str(p.relative_to(root))})
                    else: within_split_duplicates.append({'sha256':digest,'split':split,'class_first':prior[1],'class_second':c,'first':prior[2],'second':str(p.relative_to(root))})
                    print('DUPLICATE_HASH',prior[2],p,flush=True)
                else: seen[digest]=(split,c,str(p.relative_to(root)))
                if pixel_digest in pixel_seen and pixel_seen[pixel_digest][0]!=split:
                    prior=pixel_seen[pixel_digest]; cross_split_dups.append({'kind':'decoded_pixels','first':prior[2],'second':str(p.relative_to(root))})
                elif pixel_digest not in pixel_seen: pixel_seen[pixel_digest]=(split,c,str(p.relative_to(root)))
    with (results/'dataset_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=audit[0].keys());w.writeheader();w.writerows(audit)
    counts={s:{c:sum(r['split']==s and r['class']==c for r in audit) for c in CLASSES} for s in SPLITS}
    near=[]
    for i,a in enumerate(audit):
        for b in audit[i+1:]:
            if a['split']!=b['split']:
                dist=hamming_hex(a['phash64'],b['phash64'])
                if dist<=6: near.append({'first':a['path'],'first_split':a['split'],'first_class':a['class'],'second':b['path'],'second_split':b['split'],'second_class':b['class'],'phash_hamming':dist})
    with (results/'near_duplicate_candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['first','first_split','first_class','second','second_split','second_class','phash_hamming']);w.writeheader();w.writerows(near)
    (results/'dataset_audit.json').write_text(json.dumps({'class_order':CLASSES,'counts':counts,'image_count':len(audit),'invalid_images':bad,'exact_duplicate_hash_occurrences':len(duplicate_hashes),'within_split_exact_duplicate_pairs':within_split_duplicates,'cross_split_exact_duplicate_pairs':cross_split_dups,'cross_split_phash_candidates_le_6_bits':len(near)},indent=2))
    fingerprint=hashlib.sha256(''.join(f"{r['path']}:{r['sha256']}\n" for r in audit).encode()).hexdigest()
    fpfile=results/'dataset_fingerprint.txt'
    if fpfile.exists() and fpfile.read_text().strip()!=fingerprint:
        if not args.audit_only: raise RuntimeError('Dataset content/split fingerprint changed. Run --audit-only on the new dataset, then review audit gates before training.')
    fpfile.write_text(fingerprint+'\n')
    if args.audit_only:
        print('Dataset audit complete; training was not started.',flush=True)
        return
    if cross_split_dups:
        raise RuntimeError(f'Found {len(cross_split_dups)} cross-split exact file/decoded-pixel duplicates. Resolve split leakage before training; see results/dataset_audit.json.')
    if near:
        if not args.reviewed_candidates:
            raise RuntimeError(f'Found {len(near)} cross-split pHash candidates. Review results/near_duplicate_candidates.csv, save a reviewed copy with a nonempty decision column, then pass --reviewed-candidates PATH.')
        with Path(args.reviewed_candidates).open(newline='',encoding='utf-8-sig') as f:
            reviewed=list(csv.DictReader(f))
        req={'first','second','phash_hamming','decision'}
        if not reviewed or not req.issubset(reviewed[0]): raise RuntimeError('Reviewed candidate CSV must include first, second, phash_hamming, and decision columns.')
        expected={(r['first'],r['second'],str(r['phash_hamming'])) for r in near}
        actual={(r['first'],r['second'],str(int(r['phash_hamming']))) for r in reviewed if r.get('decision','').strip()}
        if expected!=actual: raise RuntimeError('Reviewed candidate pairs do not exactly match current audit or some decisions are blank; review the full current candidate list.')
    valid_rel={Path(r['path']) for r in audit}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device,'counts',counts,flush=True)
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def signature(name,augmentation,pretrained_hash=None):
        code_id=BASELINE_CODE_ID if name=='baseline_aug_1' else code_hash
        payload={'dataset_fingerprint':fingerprint,'training_code_sha256':code_id,'name':name,'augmentation':augmentation,'seed':SEED,'max_epochs':args.epochs,'batch_size':args.batch_size,'optimizer':'Adam','learning_rate':1e-3,'early_stopping_patience':7,'lr_plateau_patience':3,'pretrained_sha256':pretrained_hash}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def load_checkpoint(model,name,expected_signature):
        pt=models/f'{name}.pt'; meta=models/f'{name}.meta.json'
        if not pt.exists(): return False
        if not meta.exists() or json.loads(meta.read_text()).get('signature')!=expected_signature:
            raise RuntimeError(f'Checkpoint {pt.name} has no matching dataset/code/config signature; move it aside before training.')
        model.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True)); return True
    test=FolderSet(root,'test',valid_rel=valid_rel); val=FolderSet(root,'val',valid_rel=valid_rel)
    tl=DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=0); vl=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    # A1 architecture reproduction; checkpoint reuse makes export retries inexpensive.
    base_sig=signature('baseline_aug_1',True)
    seed_all(); baseline=ScratchCNN()
    if load_checkpoint(baseline,'baseline_aug_1',base_sig): baseline.to(device)
    else:
        tr=DataLoader(FolderSet(root,'train',True,valid_rel),batch_size=args.batch_size,shuffle=True,num_workers=0)
        baseline,_=fit(baseline,tr,vl,device,models,'baseline_aug_1',epochs=args.epochs,signature=base_sig)
    baseline.to(device)
    _,m,lg,ys,paths=run_epoch(baseline,tl,None,device); save_eval(results,'baseline_aug_1',m,lg,ys,paths,root)
    # ImageNet-pretrained MobileNetV4 Conv Small, frozen visual features + 4-way GAP linear head.
    cache=Path.cwd()/'work'/'model-cache'; cache.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HOME']=str(cache/'huggingface'); os.environ['HUGGINGFACE_HUB_CACHE']=str(cache/'huggingface'/'hub'); os.environ['TORCH_HOME']=str(cache/'torch')
    import timm
    import timm.data
    torch.manual_seed(SEED)
    backbone=timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',pretrained=False,features_only=True,out_indices=(-1,))
    if args.pretrained_weights and Path(args.pretrained_weights).is_file():
        from safetensors.torch import load_file
        state=load_file(args.pretrained_weights,device='cpu')
        incompatible=backbone.load_state_dict(state,strict=False)
        if incompatible.missing_keys: raise RuntimeError(f'Missing pretrained feature weights: {incompatible.missing_keys}')
        pretrained_sha256=hashlib.sha256(Path(args.pretrained_weights).read_bytes()).hexdigest()
        print(f"Loaded {len(state)-len(incompatible.unexpected_keys)} MobileNetV4 feature tensors from {args.pretrained_weights}; discarded classifier/head tensors {incompatible.unexpected_keys}",flush=True)
    else:
        # Normal CLI route: timm resolves and downloads the published ImageNet checkpoint.
        backbone=timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',pretrained=True,features_only=True,out_indices=(-1,))
        pretrained_sha256=None
    if pretrained_sha256 is None:
        h=hashlib.sha256()
        for key,tensor in sorted(backbone.state_dict().items()): h.update(key.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        pretrained_sha256=h.hexdigest()
    for p in backbone.parameters():p.requires_grad=False
    backbone.eval(); cfg=timm.data.resolve_model_data_config(backbone); mean=torch.tensor(cfg['mean']).view(1,3,1,1); std=torch.tensor(cfg['std']).view(1,3,1,1)
    # Frozen feature caching keeps the MobileNet experiment practical on CPU.
    # The ON arm precomputes four seeded augmented views plus original per image;
    # each epoch samples exactly one of the five views for each training example.
    train_source=FolderSet(root,'train',valid_rel=valid_rel)
    train_aug_x,train_aug_y,train_paths=cache_feature_bank(train_source,backbone,mean,std,device,n_views=5,augment_views=True,batch_size=args.batch_size,label='MobileNetV4 train views')
    train_raw_x,train_raw_y,train_raw_paths=cache_feature_bank(train_source,backbone,mean,std,device,n_views=1,batch_size=args.batch_size,label='MobileNetV4 train originals')
    val_x,val_y,val_paths=cache_feature_bank(val,backbone,mean,std,device,batch_size=args.batch_size,label='MobileNetV4 validation')
    test_x,test_y,test_paths=cache_feature_bank(test,backbone,mean,std,device,batch_size=args.batch_size,label='MobileNetV4 test')
    vloader=DataLoader(FeatureSet(val_x,val_y,val_paths),batch_size=args.batch_size,shuffle=False,num_workers=0)
    sota=None
    for aug in [True,False]:
        seed_all(); name=f'mobilenetv4_aug_{int(aug)}'; model=TransferNet(backbone); sig=signature(name,aug,pretrained_sha256)
        if load_checkpoint(model,name,sig): model.to(device)
        else:
            fx,fy,fp=(train_aug_x,train_aug_y,train_paths) if aug else (train_raw_x,train_raw_y,train_raw_paths)
            train_loader=DataLoader(FeatureSet(fx,fy,fp),batch_size=args.batch_size,shuffle=True,num_workers=0)
            cached_model=CachedHeadTrainer(backbone,model.head)
            cached_model,_=fit(cached_model,train_loader,vloader,device,models,name,epochs=args.epochs,signature=sig)
            model.head=cached_model.head; model.to(device)
        cached_model=CachedHeadTrainer(model.backbone,model.head).to(device)
        tloader=DataLoader(FeatureSet(test_x,test_y,test_paths),batch_size=args.batch_size,shuffle=False,num_workers=0)
        _,m,lg,ys,paths=run_epoch(cached_model,tloader,None,device); save_eval(results,name,m,lg,ys,paths,root)
        if aug: sota=model
    x=image_tensor(val.items[0][0]).unsqueeze(0)
    export_validate(BaselineExport(baseline),baseline,models/'baseline.onnx',x,device)
    export_validate(TransferExport(sota,mean,std),sota,models/'sota.onnx',x,device)
    with (results/'artifact_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['path','sha256','bytes'])
        for artifact in sorted(models.glob('*')):
            if artifact.is_file(): w.writerow([str(artifact.relative_to(out)),hashlib.sha256(artifact.read_bytes()).hexdigest(),artifact.stat().st_size])
    import importlib.metadata as im
    packages={}
    for p in ['torch','torchvision','timm','numpy','Pillow','scikit-learn','onnx','onnxruntime']:
        try: packages[p]=im.version(p)
        except im.PackageNotFoundError: packages[p]=None
    (results/'run_config.json').write_text(json.dumps({'seed':SEED,'max_epochs':args.epochs,'batch_size':args.batch_size,'optimizer':'Adam','learning_rate':1e-3,'early_stopping':{'monitor':'val_loss','patience':7,'restore_best_weights':True},'lr_plateau':{'monitor':'val_loss','patience':3,'factor':0.5,'minimum':1e-6},'augmentation':{'horizontal_flip_probability':0.5,'rotation_degrees':[-18,18],'contrast_factor':[0.9,1.1]},'ablation':'MobileNetV4 train augmentation enabled vs disabled; same seed and schedule','augmentation_bank':'For each training image cache original plus four deterministic seeded augmented views; each epoch selects exactly one view per image. Frozen GAP features are precomputed once. Validation/test features use original images only.','backbone':'mobilenetv4_conv_small.e2400_r224_in1k','backbone_pretrained':True,'backbone_pretrained_sha256':pretrained_sha256,'backbone_frozen':True,'transfer_head':'global average pooling + linear 4 class logits','normalization_mean':mean.flatten().tolist(),'normalization_std':std.flatten().tolist(),'class_order':CLASSES,'device':str(device),'packages':packages},indent=2))
    print('Training and exports complete',flush=True)

def save_eval(results,name,m,logits,ys,paths,root):
    (results/f'{name}_metrics.json').write_text(json.dumps(m,indent=2))
    pred=np.asarray(logits).argmax(1)
    with (results/f'{name}_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','true_label','predicted_label','probabilities'])
        for path,y,p,l in zip(paths,ys,pred,logits):w.writerow([str(Path(path).relative_to(root)),CLASSES[y],CLASSES[p],json.dumps(torch.softmax(torch.tensor(l),0).tolist())])

if __name__=='__main__': main()
