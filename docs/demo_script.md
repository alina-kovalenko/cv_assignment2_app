# ObjectLab demo video script (3–5 minutes)

Target length: about 4 minutes. Record the real Android app on the device used for the latency measurements. Replace bracketed cues with observed values. Do not stage a correct prediction if the model produces a different result; show the actual output and explain it.

The app tabs are labeled **Baseline**, **Enchanced Model**, and **Generated Photos** (the second label is spelled that way in the app). Generated Photos contains four reviewed, saved offline pairs. Its **Generate** button advances through the saved pairs in this order: bottles, cups, pens, scissors, then wraps. It does not generate images on the phone. The released adapter generated these four examples, but the separate classifier retraining experiment has not been run; do not claim an ArtAug accuracy or augmentation benefit.

## 0:00–0:25 — Introduction

**On screen:** App home page, Baseline tab selected.

**Say:** “This is ObjectLab, an Android demo for my Computer Vision Assignment 2. It classifies four household-object classes: bottles, cups, pens, and scissors. The project continues my Assignment 1 dataset and compares a scratch CNN reproduction with a pretrained MobileNetV4 model.”

## 0:25–0:55 — Dataset and comparison

**On screen:** Briefly show the project’s dataset summary or the report table.

**Say:** “The refreshed Drive snapshot contained 645 images. After removing 21 visually confirmed cross-split copies and one same-split exact duplicate, the clean experiment set has 623 images: 461 training, 81 validation, and 81 test images. Assignment 1 reported 441 images and a 320/61/60 split, so I report its historical numbers separately. The transfer model uses ImageNet pretrained features with a frozen backbone and a new four-class head.”

## 0:55–1:45 — Baseline prediction

**On screen:** In Baseline tab, select a test or demonstration photo. Keep the input visible. Wait for classification and heatmap.

**Say:** “The baseline predicts [actual class] with [actual confidence] confidence in [actual latency] milliseconds on this [device / Android version]. The heatmap highlights regions associated with the selected class. It is a coarse model visualization, not proof of the model’s reasoning.”

**If the prediction is wrong:** “This example is misclassified as [predicted class] instead of [true class]. The highlighted area and surrounding context may help explain why it is difficult, but one example does not identify the cause.”

## 1:45–2:35 — SOTA transfer prediction

**On screen:** Select the same image in **Enchanced Model**.

**Say:** “Using the same photo and preprocessing, MobileNetV4 predicts [actual class] with [actual confidence] confidence in [actual latency] milliseconds. This side-by-side view makes the comparison concrete. The selected model is a contemporary pretrained architecture; this project does not claim state-of-the-art results for this task.”

## 2:35–3:15 — Saved edits, explainability, and ablation

**On screen:** Select **Generated Photos** and tap **Generate** to show the saved pairs in order: bottles, cups, pens, then scissors. Point out the original and AI-edited views and the manifest prompt/seed if available. The button navigates saved examples; it does not run the generator. Also show a heatmap and the report's conventional-augmentation ablation table.

**Say:** “The transfer model’s class activation map shows coarse spatial score support; it does not prove the model’s reasoning. I compared conventional flip, rotation, and contrast augmentation with no such training transforms. Both runs had test macro-F1 0.9755 and made the same two scissors-to-pens errors. This is a single split and seed, not a universal conclusion. Generated Photos is an offline gallery of four reviewed edits made with the released ArtAug adapter. These examples have not been used to retrain the classifier, so I make no claim that ArtAug improves classification.”

## 3:15–3:45 — Results and limitations

**On screen:** Show baseline/SOTA metrics and confusion matrices from the report.

**Say:** “On the held-out test set, the reproduced scratch CNN reached 33.33% accuracy and 0.3074 macro-F1, while MobileNetV4 reached 97.53% accuracy and 0.9755 macro-F1. MobileNetV4 correctly classified all bottles, cups, and pens; it misclassified two of the 20 scissors images as pens. The main limitations are the small test set, a single fixed split and seed, and the difference between the current folder and the A1-reported dataset.”

## 3:45–4:00 — Close

**On screen:** Return to app home and briefly show repository / run instructions.

**Say:** “The repository includes the training and export code, model artifacts, Android project, and reproducibility instructions. This demo is an educational classifier; predictions can be wrong, and photos are processed [locally / describe actual behavior].”

## Recording checks

- Show the app name and the exact tabs **Baseline**, **Enchanced Model**, and **Generated Photos**; include a real prediction, a heatmap, and measured latency.
- Capture one correct and one incorrect or uncertain example if available; report the true label honestly.
- Keep personal notifications, private photos, and account details off-screen.
- Show clear text at full-screen resolution and record the device / Android version in the report.
- Make sure the final recording is 3–5 minutes and the shared video link is accessible to the instructor.


