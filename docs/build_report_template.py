from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path
import argparse

SCRIPT=Path(__file__).resolve()
REPO=None
for candidate in SCRIPT.parents:
 if (candidate/'docs').is_dir() and (candidate/'ml').is_dir() and (candidate/'models').is_dir():
  REPO=candidate; break
 if (candidate/'outputs'/'ObjectLab').is_dir():
  REPO=candidate/'outputs'/'ObjectLab'; break
if REPO is None: raise RuntimeError('Could not locate ObjectLab repository relative to builder script')
OUT=REPO.parent if REPO.parent.name=='outputs' else REPO
DOCS=REPO/'docs'; DOCS.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser(description='Build the ObjectLab report template DOCX')
parser.add_argument('--output',type=Path,default=OUT/'Report_Template.docx')
DOC=parser.parse_args().output.resolve(); DOC.parent.mkdir(parents=True,exist_ok=True)
NAVY='16283F'; TEAL='237C83'; PALE='EAF2F3'; LIGHT='F2F5F8'; MUTED='596A7C'; GOLD='C18B3A'

doc=Document(); sec=doc.sections[0]; sec.page_width=Inches(8.5); sec.page_height=Inches(11); sec.top_margin=Inches(.62); sec.bottom_margin=Inches(.62); sec.left_margin=Inches(.72); sec.right_margin=Inches(.72)
styles=doc.styles
normal=styles['Normal']; normal.font.name='Aptos'; normal.font.size=Pt(10.5); normal.font.color.rgb=RGBColor.from_string(NAVY); normal.paragraph_format.space_after=Pt(4); normal.paragraph_format.line_spacing=1.04
for s,size,color in [('Title',29,NAVY),('Heading 1',19,NAVY),('Heading 2',12,TEAL),('Heading 3',10,NAVY)]:
 st=styles[s]; st.font.name='Aptos Display' if s!='Normal' else 'Aptos'; st.font.size=Pt(size); st.font.bold=True; st.font.color.rgb=RGBColor.from_string(color); st.paragraph_format.space_before=Pt(5); st.paragraph_format.space_after=Pt(4)
styles['Title'].font.color.rgb=RGBColor.from_string('000000')
# header/footer
h=sec.header.paragraphs[0]; h.text='OBJECTLAB  /  COMPUTER VISION'; h.style='Caption'; h.runs[0].font.color.rgb=RGBColor.from_string(TEAL); h.runs[0].font.bold=True
f=sec.footer.paragraphs[0]; f.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=f.add_run('ASSIGNMENT 2     •     '); run.font.size=Pt(8); run.font.color.rgb=RGBColor.from_string(MUTED)
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); f._p.append(fld)

def shade(cell,color):
 tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),color); tcPr.append(shd)
def set_cell(cell,text,bold=False,color=NAVY,size=8):
 cell.text=''; p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(2); r=p.add_run(text); r.bold=bold; r.font.name='Aptos'; r.font.size=Pt(size); r.font.color.rgb=RGBColor.from_string(color); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(headers, rows, widths=None):
 t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
 if widths:
  for c,w in zip(t.columns,widths): c.width=Inches(w)
 for i,x in enumerate(headers): shade(t.rows[0].cells[i],NAVY); set_cell(t.rows[0].cells[i],x,True,'FFFFFF',9)
 trPr=t.rows[0]._tr.get_or_add_trPr(); repeat=OxmlElement('w:tblHeader'); repeat.set(qn('w:val'),'true'); trPr.append(repeat)
 for ri,row in enumerate(rows):
  cells=t.add_row().cells
  if widths:
   for c,w in zip(cells,widths): c.width=Inches(w)
  for i,x in enumerate(row):
   shade(cells[i], 'F2F5F8' if ri%2==0 else 'FFFFFF'); set_cell(cells[i],str(x),False,NAVY,9)
 t.style='Table Grid'
 tblPr=t._tbl.tblPr
 borders=OxmlElement('w:tblBorders')
 for edge in ('top','left','bottom','right','insideH','insideV'):
  el=OxmlElement(f'w:{edge}'); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:space'),'0'); el.set(qn('w:color'),'D6DCE3'); borders.append(el)
 tblPr.append(borders)
 margins=OxmlElement('w:tblCellMar')
 for edge,value in [('top','75'),('start','90'),('bottom','75'),('end','90')]:
  el=OxmlElement(f'w:{edge}'); el.set(qn('w:w'),value); el.set(qn('w:type'),'dxa'); margins.append(el)
 tblPr.append(margins)
 doc.add_paragraph().paragraph_format.space_after=Pt(1)
 return t

def p(text='',bold_lead=None,style=None):
 x=doc.add_paragraph(style=style)
 if bold_lead and text.startswith(bold_lead):
  x.add_run(bold_lead).bold=True; x.add_run(text[len(bold_lead):])
 else:x.add_run(text)
 return x

def placeholder(text):
 x=doc.add_paragraph(); r=x.add_run('['+text+']'); r.italic=True; r.font.color.rgb=RGBColor.from_string('737D88'); return x

def heading(k,title,sub):
 x=doc.add_paragraph(); x.paragraph_format.space_after=Pt(1); r=x.add_run(k.upper()); r.bold=True; r.font.size=Pt(8); r.font.color.rgb=RGBColor.from_string(TEAL)
 doc.add_heading(title,1); p(sub)
def page(): doc.add_page_break()

# 1
x=doc.add_paragraph(); x.paragraph_format.space_before=Pt(12); r=x.add_run('ASTANA IT UNIVERSITY   /   COMPUTER VISION'); r.bold=True; r.font.size=Pt(9); r.font.color.rgb=RGBColor.from_string(TEAL)
doc.add_paragraph('Household Object\nRecognition',style='Title')
p('Assignment 2  |  Transfer learning and Android demonstration')
table(['AUTHOR','COURSE','INSTRUCTOR'],[['Alina Kovalenko','ST-2402','Nurym Kaiyrkhan']], [2.2,2.0,2.7])
p('Submission date: 3 November 2026  |  Source repository: https://github.com/alina-kovalenko/cv_assignment2_app (confirm this remote contains the final submitted source)  |  Demo: [FILL URL]')
doc.add_heading('Project summary',2)
p('This project extends the Assignment 1 household-object classifier for bottles, cups, pens, and scissors. I compare the historical A1 results and a fresh PyTorch reproduction of its scratch CNN with an ImageNet-pretrained MobileNetV4 Conv Small classifier. The Android app presents a prediction and a class activation heatmap. The held-out test set is used only for final evaluation.')
placeholder('IN 2–3 SENTENCES: summarize your measured result after training. State the metric that supports the conclusion, and mention the most important limitation.')
doc.add_heading('Dataset at a glance',2)
table(['ITEM','A1 REPORTED','UPDATED DATASET'],[['Images','441 total','645 raw → 623 clean'],['Split','320 / 61 / 60','481 / 83 / 81 raw → 461 / 81 / 81 clean'],['Classes','Bottles, cups, pens, scissors','Same four labels'],['Sources','Manual + internet','Drive snapshot; [FILL provenance]']], [1.25,2.15,3.5])
p('A1 reported test performance: Logistic Regression accuracy 33.33%, macro-F1 0.33; scratch CNN accuracy 36.67%, macro-F1 0.32. The refreshed Drive snapshot contained 645 images. Before training, 21 visually matched copies across splits were removed (keeping test, then validation, then train), plus one exact duplicate within train. The cleaned set has 623 images. Report historical A1 results separately from current-run scores.')
placeholder('Confirm source provenance and cite the exclusion log in ObjectLab/results. All 623 cleaned images decode successfully; the clean split has no remaining exact or cross-split perceptual-hash candidates.')

# 2
page(); heading('01  /  Method','Approach and motivation','The central question is whether pretrained visual features improve recognition when the labeled dataset is small.')
p('The A1 topic and four labels are retained. The reference baseline is the A1 scratch CNN architecture retrained in PyTorch because the original TensorFlow checkpoint was not supplied. Historical A1 scores are shown as a separate reference; they are not measurements from the new checkpoint.')
p('The completed transfer arm uses timm MobileNetV4 Conv Small with ImageNet-1k weights, a frozen backbone, and a new global-average-pooling linear head for four classes. A third ArtAug tab is implemented with an explicit empty state until approved edits are imported. After import, it previews source/edit pairs and can run the existing MobileNetV4 classifier on both images; this works before any ArtAug retraining.')
p('Transfer learning reuses ImageNet-pretrained CNN features with a replaced classifier head [4]; the selected MobileNetV4 implementation and weights are documented in [1] and [2].')
doc.add_heading('Pipeline diagram',2)
t=doc.add_table(rows=1,cols=5); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
labs=['Image\nEXIF → RGB','Resize + pad\n224 × 224','Model\nCNN / V4','Output\nclass + score','Android tabs\nBaseline / SOTA / ArtAug*']
for i,l in enumerate(labs):
 c=t.rows[0].cells[i]; shade(c,TEAL if i in (0,2,4) else PALE); set_cell(c,l,True,'FFFFFF' if i in (0,2,4) else NAVY,8)
 if i<4:
  pass

doc.add_heading('Why this model',2)
p('“SOTA” names the selected contemporary classifier architecture; the project does not claim state-of-the-art performance on this task. ArtAug flow (planned data, implemented app tab): original train photo + approved prompt → FLUX.1-Kontext-dev-lora-ArtAug edit in Colab → human QC → saved edit + provenance → import into the Android ArtAug tab. The tab is implemented with an empty state until import; after import it can classify source and edit with the existing MobileNetV4 before any retraining. No generation or ArtAug classifier-training results are available yet.')
p('The heatmap is Grad-CAM for the scratch CNN and class activation mapping for MobileNetV4, whose global-average-pooling linear head supports a class activation map.')


# 3
page(); heading('02  /  Experimental setup','Data and training protocol','Record the exact dataset snapshot and software environment used for the reported run.')
doc.add_heading('Dataset handling',2)
p('Images are read from fixed train, validation, and test folders. Preprocessing applies EXIF orientation correction, RGB conversion, Lanczos resizing with aspect ratio preserved, and centered grey padding (RGB 127) to 224 × 224. Training-only augmentation consists of horizontal flips, rotations up to 18 degrees, and contrast changes within ±10%. The no-augmentation ablation disables all three transformations.')
table(['CLEAN SPLIT','BOTTLES','CUPS','PENS','SCISSORS','TOTAL'],[['Train','113','113','120','115','461'],['Validation','21','20','21','19','81'],['Test','20','20','21','20','81']], [.95,.95,.85,.85,1.0,.8])
p('The raw Drive snapshot had 645 images (train 481, validation 83, test 81). The clean set has 623 after removing 21 visually confirmed cross-split copies and one exact within-train duplicate. Lower-priority copies were removed in test > validation > train order, preserving the test occurrence where available.')
placeholder('Cite ObjectLab/results/leakage_removals.csv and record any remaining grouping limitations or source-provenance uncertainty.')
doc.add_heading('Training protocol',2)
table(['SETTING','SCRATCH CNN REPRODUCTION','MOBILENETV4 TRANSFER'],[['Initialization','Random','ImageNet-1k pretrained'],['Trainable layers','All','Four-class linear head only'],['Optimizer / LR','Adam / 1e-3','Adam / 1e-3'],['Batch / max epochs','16 / 30','16 / 30'],['Selection','Best val loss; patience 7','Best val loss; patience 7']], [1.35,2.6,2.9])
p('Both arms use seed 2 and the same split. The scratch CNN applies its augmentation online. For MobileNetV4, a cached feature bank stores each training image’s original view and four seeded augmented views; one view is sampled per image per epoch. The augmentation-off ablation uses the original view only. Both transfer arms use the same head initialization and optimization, run for at most 30 epochs, and select by validation loss with early stopping. The test split is not used for model selection. The scratch CNN stopped after epoch 16 (best validation-loss epoch 9); both transfer runs reached the 30-epoch cap.')
doc.add_heading('Environment and reproducibility',2)
p('Run environment: Python 3.12.7, PyTorch 2.15.0, timm 1.0.30, CPU. Weight identifier and checksum are in results/model_card.json; run settings, commands, and artifact hashes are in results/run_config.json and results/artifact_manifest.csv.')

# 4
page(); heading('03  /  Results','Held-out test comparison','Results on the fixed clean test split (n = 81); metrics are computed from saved predictions.')
p('A1 scores below are historical values from the Assignment 1 report. “A1 CNN reproduction” is a new PyTorch training run and must be labeled with its current metrics. Report accuracy and macro precision, recall, and F1 to show both overall and class-balanced performance.')
table(['MODEL / SOURCE','ACCURACY','MACRO P','MACRO R','MACRO F1','N TEST'],[['A1 Logistic Regression, historical','.3333','.34','.33','.33','60'],['A1 scratch CNN, historical','.3667','.32','.37','.32','60'],['Scratch CNN, PyTorch run','.3333','.3095','.3304','.3074','81'],['MobileNetV4, aug OFF','.9753','.9783','.9750','.9755','81'],['MobileNetV4, aug ON','.9753','.9783','.9750','.9755','81']], [2.25,.85,.85,.85,.85,1.05])
p('On this split, each MobileNetV4 run classified 79/81 images correctly versus 27/81 for the scratch CNN reproduction. The two transfer variants produced identical test predictions. The A1 report’s CNN result (36.67%, macro-F1 0.3169, n=60) is historical and comes from a different dataset snapshot and framework; it is not a directly paired run.')
doc.add_heading('Per-class results',2)
table(['TRANSFER MODEL†','CLASS','PRECISION','RECALL','F1','N'],[['MobileNetV4','bottles','1.0000','1.0000','1.0000','20'],['MobileNetV4','cups','1.0000','1.0000','1.0000','20'],['MobileNetV4','pens','.9130','1.0000','.9545','21'],['MobileNetV4','scissors','1.0000','.9000','.9474','20']], [1.6,1.2,1.05,1.05,.95,.9])
p('† Augmentation ON and OFF have the same per-class scores and confusion matrix. In both, the two errors are scissors predicted as pens; no other class is misclassified. The scratch baseline spreads errors across all four labels. The small test support (20–21 images per class) limits how broadly to generalize.')
table(['ACTUAL → PREDICTED','BOTTLES','CUPS','PENS','SCISSORS'],[['Scratch · bottles','2','5','13','0'],['Scratch · cups','3','9','2','6'],['Scratch · pens','3','1','12','5'],['Scratch · scissors','2','6','8','4'],['MobileNetV4 · bottles','20','0','0','0'],['MobileNetV4 · cups','0','20','0','0'],['MobileNetV4 · pens','0','0','21','0'],['MobileNetV4 · scissors','0','0','2','18']], [2.3,1.05,1.05,1.05,1.05])

# 5
page(); heading('04  /  Analysis','Learning curves and ablation','Measured seed-2 histories and a paired augmentation comparison on the same split.')
doc.add_heading('Training curves',2)
p('Training and validation curves (solid = training, dashed = validation). MobileNetV4 validation accuracy reached 98.8%; both transfer runs reached the 30-epoch cap.')
doc.add_picture(str(REPO/'results'/'training_curves_comparison.png'),width=Inches(6.55))
doc.add_heading('Ablation on training augmentation',2)
p('Only training augmentation changed; split, seed, head initialization, optimizer, and selection were fixed. Interpret this single-seed comparison cautiously.')
table(['TRANSFER MODEL','TEST ACCURACY','TEST MACRO-F1','BEST EPOCH'],[['Augmentation OFF','.9753','.9755','30'],['Augmentation ON','.9753','.9755','30'],['Difference (ON − OFF)','0.0 pp','0.0000','—']], [2.25,1.45,1.55,1.3])
doc.add_heading('Interpretation',2)
p('Both variants scored identically on this split: two scissors images were predicted as pens. With n = 81 and one seed, this tie does not show that augmentation has no general benefit.')
doc.add_heading('ArtAug pilot status',2)
p('The four prompts are approved. A Colab T4 is allocated and small BF16/FP16/FP8 checks pass; generation still awaits gated weight access and full-pipeline validation. DiffSynth-Studio FLUX.1-Kontext-dev-lora-ArtAug is an image-editing LoRA, distinct from the 2024 ArtAug text-to-image paper [7, 8]. The pilot has five training photos per class (20 total). Source QC excludes train/pens/77.jpg; a replacement, train/pens/78.jpg, preserves the count. No edits or classifier retraining have been completed.')

# 6
page(); heading('05  /  Android demo','Prediction and visual explanation','The Android app runs the exported ONNX model locally and displays the selected model’s predicted class and heatmap.')
doc.add_heading('User flow',2)
p('For the completed classifier, choose a photo → select Baseline or SOTA → view label, confidence, inference time, and heatmap. The implemented ArtAug tab has an explicit empty state until saved Colab edits are imported. After import, it previews each source/edit pair and can run both existing MobileNetV4 predictions; this does not require retraining. The tab does not generate on the phone. No synthetic-data classifier retraining has been completed.')
table(['DESKTOP ONNX GRAPH BENCHMARK','BASELINE','SOTA'],[['Device / runtime','i5-12450H · ORT 1.31.1','i5-12450H · ORT 1.31.1'],['Mean / median (ms)','2.64 / 2.42','3.42 / 3.31'],['p95 (ms)','3.63','4.25'],['Protocol','5 warm-ups; 50 runs; N=1','5 warm-ups; 50 runs; N=1']], [2.4,2.15,2.15])
p('Scope: desktop ONNX graph only, including class scores and heatmaps; excludes image decode, resize/pad, app overhead, and ArtAug editing. This is not phone latency or end-to-end FPS. Android device/version, installed APK, phone latency and throughput: [FILL after on-device testing].')
doc.add_heading('Explainability example',2)
doc.add_picture(str(REPO/'results'/'qualitative_cam_selected.png'),width=Inches(3.0))
placeholder('Add a real phone screenshot showing a completed Baseline or SOTA prediction. If ArtAug outputs are generated and imported, add a separate source/edit screenshot and label it synthetic; do not use a mock output.')
doc.add_heading('Export validation',2)
p('Both exported ONNX graphs were compared with recorded PyTorch predictions under ONNX Runtime 1.20.1 on CPU: all 81 test images were checked per model, probabilities were finite and normalized, and heatmaps stayed within [0,1]. Maximum probability differences: 1.2e-7 (baseline), 1.5e-6 (SOTA). This is desktop compatibility validation, not Android device testing.')
placeholder('Add two or three sentences describing the in-app interaction and any tested failure case.')

# 7
page(); heading('06  /  Responsible use','Ethics and limitations','Object classification may mislead when the photo contains unfamiliar objects, clutter, or ambiguous labels.')
doc.add_heading('Data, privacy, and consent',2)
p('The dataset combines manually collected photos and internet images as described in the A1 report. Document source, license or permission, and whether identifiable people appear in any images. The demo processes user-selected photos locally; it does not upload or retain them [confirm this matches the actual implementation]. Obtain consent before using another person’s image and avoid including private or sensitive content in the demo video.')
doc.add_heading('Bias and misuse',2)
p('The model represents only four household classes and the collected examples. It may learn background, lighting, camera, or source-specific patterns instead of object shape. A prediction is a classroom demonstration, not a safety-critical decision. Confidence is not a guarantee of correctness. Heatmaps show model sensitivity at a coarse spatial scale and can be misleading.')
doc.add_heading('Limitations',2)
table(['LIMITATION','EFFECT ON INTERPRETATION'],[['Small held-out test set (n = 81)','A few images can shift accuracy and macro-F1.'],['One fixed split and seed','Results do not establish a stable ranking.'],['Dataset version differs from A1','Historical and current scores are not directly paired.'],['Frozen ImageNet backbone','No evidence that end-to-end fine-tuning would perform similarly.'],['Class labels only; no boxes','mAP@0.5 is not measured for this classifier.'],['ArtAug generation pending','No claim about synthetic-image benefit yet. Generation and controlled retraining remain pending.']], [2.35,4.4])
placeholder('State the largest limitation actually observed in the app or dataset. Distinguish measurement from hypothesis.')
doc.add_heading('ArtAug provenance, QC, and evaluation',2)
table(['STEP','RECORD / RULE'],[['Generation manifest','Save each source/edit pair, exact approved prompt, deterministic seed, adapter and base-model revision/hash, runtime/GPU, duration, and hashes. Record the 20 selected sources and the one QC-excluded source.'],['Human review','Approve only edits that preserve object identity, count, and class label; record rejected files and reasons. Prefer no people; reject any with people unless consent is documented and label permitted synthetic-person images clearly. Check photo rights and model license before sharing.'],['Controlled classifier study','Compare MobileNetV4 trained on original train images vs original + QC-approved edits. Keep split, seed, initialization, and schedule fixed; use untouched real validation/test images only. Record accuracy, macro-F1, per-class metrics, and confusion matrix after retraining.']], [1.65,5.05])
p('Current labels have no bounding boxes, so mAP@0.5 is unavailable. The latency table is desktop ONNX graph timing, not end-to-end FPS or phone speed. The Colab notebook and hash-checked importer document how saved edits move into the empty ArtAug tab.')

# 8
page(); heading('07  /  Conclusion','Findings and reproducibility','Summarize what the evidence supports, then provide enough detail for another student to rerun the experiment.')
doc.add_heading('Conclusion',2)
placeholder('In 4–6 sentences: answer the research question using completed test metrics; explain class-level behavior and the conventional augmentation tie; note the 81-image test set and dataset-version difference. If the ArtAug pilot is later completed, describe its measured result separately and state the QC/sample limits.')
doc.add_heading('Reproduce the project',2)
p('1. Install dependencies in ObjectLab/ml/requirements.txt.  2. Prepare a clean copy: python ml/prepare_dataset.py --source <raw_dataset> --output <clean_dataset>. The script verifies source hashes against results/leakage_removals.csv and leaves the raw data unchanged.  3. Train: python ml/train.py --data <clean_dataset> --out . --epochs 30.  4. Export the ONNX models and build the Android app as described in the repository README. Record exact versions, commands, commit, and APK hash.')
p('Repository: https://github.com/alina-kovalenko/cv_assignment2_app (verify final source is pushed)   Commit / tag: [FILL]   APK SHA-256: [FILL]   Demo video: [FILL]')
doc.add_heading('References',2)
p('[1] A. Howard et al., “MobileNetV4: Universal Models for the Mobile Ecosystem,” 2024. https://arxiv.org/abs/2404.10518')
p('[2] timm model card, mobilenetv4_conv_small.e2400_r224_in1k. https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k')
p('[3] R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” ICCV 2017. https://arxiv.org/abs/1610.02391')
p('[4] Neurohive, “Convolutional Neural Network (CNN): How It Works, With PyTorch,” https://neurohive.io/en/guides/convolutional-neural-network/. General background on ImageNet transfer learning and mobile/edge CNN use; the specific model and weights used here are documented in [1] and [2].')
p('[5] Computer Vision Dataset, author-provided Google Drive folder, accessed 3 November 2026. https://drive.google.com/drive/folders/1t0GP7m-EcSfGtFkbcQQrgaKOHIeLcDr2. Original photo provenance and licenses: [document available records].')
p('[6] ObjectLab dataset audit and exclusion records: results/dataset_manifest.csv, results/dataset_fingerprint.txt, results/leakage_removals.csv; clean copy preparation: ml/prepare_dataset.py.')
p('[7] DiffSynth-Studio, “FLUX.1-Kontext-dev-lora-ArtAug: Aesthetic Enhancement - Kontext Image Editing LoRA,” model card, https://huggingface.co/DiffSynth-Studio/FLUX.1-Kontext-dev-lora-ArtAug. Selected adapter and usage example; generation remains pending.')
p('[8] Z. Duan et al., “ArtAug: Enhancing Text-to-Image Generation through Synthesis-Understanding Interaction,” arXiv:2412.12888, 2024, https://arxiv.org/abs/2412.12888. A distinct text-to-image model-enhancement paper, not the selected Kontext editing adapter.')

# metadata
doc.core_properties.title='Household Object Recognition Assignment 2 Report Template'
doc.core_properties.subject='Eight-page fill-in report template for ObjectLab transfer-learning project'
doc.core_properties.author='Alina Kovalenko'
doc.save(DOC)
print(DOC)












