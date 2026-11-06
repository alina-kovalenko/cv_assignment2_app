# Assignment 2 submission checklist

## Report (6–8 pages)

- [ ] English report is 6–8 pages total, including title page and references.
- [ ] Project title, author, course, instructor, submission date, repository link, and demo link are filled in.
- [ ] Current dataset size, class counts, split counts, source mix, augmentation, and duplicate/repeated-object check are verified.
- [ ] A1's reported 441 images and 320/61/60 split are distinguished from the refreshed Drive snapshot: 645 raw images, split 481/83/81, and the cleaned training set: 623 images, split 461/81/81.
- [ ] Clean class counts are recorded: train 113/113/120/115; validation 21/20/21/19; test 20/20/21/20 (bottles/cups/pens/scissors).
- [ ] The 645-image Drive snapshot was verified as valid JPEGs; 21 visually matched cross-split copies and one same-split exact duplicate were removed, with test > validation > train priority. Cite the exclusion log and confirm no unresolved duplicate/group leakage.
- [ ] Historical A1 scores are identified as historical; the PyTorch scratch CNN is identified as a fresh reproduction because the original checkpoint was unavailable.
- [ ] MobileNetV4 pretrained model identifier, weight source, frozen-backbone/head setup, software versions, seed, and training schedule are recorded.
- [x] Test accuracy and macro precision/recall/F1 use actual current run values; per-class scores and confusion matrices are included.
- [x] Current clean-test metrics are populated: scratch CNN 27/81 (accuracy .3333, macro-F1 .3074); MobileNetV4 with and without augmentation 79/81 (accuracy .9753, macro-F1 .9755); both transfer runs made the same two scissors-to-pens errors.
- [x] Training and validation curves are inserted and legible.
- [x] Measured training/validation histories are combined in `results/training_curves_comparison.png`; the selected correct and incorrect Grad-CAM examples are in `results/qualitative_cam_selected.png`.
- [x] Augmentation ablation changes only the stated augmentation setting; test scores are not used for model selection.
- [x] Existing ablation is clearly labeled as the conventional flip/rotation/contrast transforms and is kept separate from the planned ArtAug editing study.
- [x] The four class prompts are approved. The executed Kaggle notebook generated four of its ten selected train photos on a Tesla T4; the four reviewed pairs and their source/prompt/seed/model provenance are recorded in the manifest.
- [ ] The planned 20-image pilot is incomplete: four reviewed pairs are saved (one per class), and the other selected photos have not been generated. Preserve originals and record prompt, seed, revisions/hashes, runtime/GPU, duration, outputs, and QC decisions in the manifest.
- [x] QC of the four imported pairs confirms that each original is byte-identical to its frozen training source and the edit preserves the object identity and scene. Images show hands but no identifiable face. The remaining pilot photos still need review if generated.
- [ ] Retrain MobileNetV4 on original-only versus original-plus-QC-edits training, with the same split/seed/schedule and untouched real validation/test photos. Do not fill pending ArtAug metrics until measured.
- [x] The third tab, labeled **Generated Photos**, contains four reviewed offline original/edit pairs. Tapping **Generate** advances in this order: bottles, cups, pens, scissors, then wraps. It is a saved-image browser; it does not create images or classify the pair.
- [x] Built and verified APK 1.2 (code 3), including the four reviewed originals and edits. Build, signature, asset checks and device status are in `results/apk_verification.json`.
- [ ] Verify the rebuilt app shows the four pairs in **Generated Photos**. Classifier retraining with edited data is a separate, unperformed experiment; do not report augmentation-benefit metrics until it is run.
- [x] mAP@0.5 is not claimed because the current dataset has class labels but no bounding boxes.
- [ ] Android inference time includes device/model identity, sample count, warmup, and measurement scope.
- [x] Desktop ONNX graph latency is recorded separately (5 warm-ups, 50 runs, N=1); this is clearly labeled as non-Android and must not substitute for an on-device phone measurement.
- [ ] At least one app example shows input, prediction, confidence, and heatmap, with correct/incorrect status stated.
- [ ] Ethics, privacy, consent, limitations, bias risks, and intended educational use are discussed.
- [x] References include the primary MobileNetV4 paper, model card, Grad-CAM paper, and verified Neurohive page required by the assignment.
- [ ] Replace remaining student fields (phone screenshots/latency, device, links, reflection, photo provenance) before final submission; do not substitute guessed values. Check all URLs and licensing evidence.

## Android app and repository

- [ ] The displayed tabs are identified by their exact labels and order: **Baseline**, **Enchanced Model**, **Generated Photos**.
- [ ] Generated Photos shows the four saved source/edit pairs and their AI-edited labeling. Do not imply generation or pair classification occurs on-device.
- [ ] Photo selection works and both ONNX models load successfully.
- [ ] Input preprocessing matches training: EXIF correction, RGB, aspect-preserving resize, grey padding, and correct normalization.
- [ ] Predicted label, confidence, inference latency, and visual explanation appear for both models.
- [ ] ONNX outputs match the source model within the documented tolerance.
- [ ] App tested on the named Android device; app behavior for invalid or oversized images is documented.
- [ ] Repository contains source, model artifacts or download instructions, training script, dependency list, and reproducibility command.
- [ ] APK installs on a clean device; package/version and SHA-256 are recorded.
- [ ] No dataset image is redistributed without permission; no private data or credentials are committed.

## Demo video

- [ ] Video is 3–5 minutes and shows the actual installed app.
- [ ] Shows baseline and SOTA prediction using the same input.
- [ ] Shows heatmap, latency, held-out metrics, and one limitation.
- [ ] All spoken metrics match the final report and results files.
- [ ] Video link opens for the instructor.

## Moodle submission

- [ ] Upload final report PDF.
- [ ] Provide APK or local run guide.
- [ ] Provide repository link.
- [ ] Provide demo video link.
- [ ] Submit before 7 November 2026, 23:59 Asia/Almaty, and prepare to defend the work in practice lessons.


