# ObjectLab

An offline Android app for classifying bottles, cups, pens and scissors. **Baseline** and **Enchanced Model** accept gallery and camera input and show four class probabilities, a heatmap and inference time. In **Generated Photos**, tap **Generate** to compare a training photo with its saved ArtAug edit. The original appears above the image marked **Enhanced by AI**. Successive taps show the bottle, cup, pens and scissors, then return to the bottle.

The Baseline tab runs a freshly trained reproduction of the Assignment 1 CNN architecture. Enchanced Model runs an ImageNet-pretrained MobileNetV4 feature backbone with a trained four-class head. Existing experiment files use the key `sota` for this transfer model.

The app has no Internet permission. User photos are processed locally. It always chooses among four known classes and has no independently validated unknown-object detector. A high softmax score is not a guarantee of a correct prediction.

## ArtAug pilot status

Four photographs from the frozen training split were edited on a Kaggle Tesla T4 with **FLUX.1-Kontext-dev + ArtAug LoRA**, using 30 inference steps each. The app includes these four reviewed pairs, one per class. The classifier scores reported below come from the baseline and MobileNetV4 experiments on the real image split; the generated edits are not part of those scores.

`ml/ObjectLab_ArtAug_Kaggle_Executed.ipynb` preserves the executed Kaggle run, including the four generated photo outputs and logs; Kaggle widget payloads were removed for static portability. `ml/ObjectLab_ArtAug_Kaggle_10.ipynb` is the clean, output-free rerun copy with the same four generation sections. Both notebooks contain an embedded input bundle with ten candidate training photos and four prompts, but only photos 1–4 have generation sections in the submitted notebooks; photos 5–10 are not included as output examples. The embedded input bundle is distinct from `ArtAug_Pilot_Output.zip`, which stores generated pairs. A new run requires a Kaggle GPU, access to the gated Hugging Face model weights, and the recorded model/runtime dependencies. The approved export manifest in `android/app/src/main/assets/artaug/manifest.json` records each prompt, seed, model revision, generation duration and image hash. `results/artaug_status.json` records the executed run and exported assets.

Generate displays a packaged pair on the phone. The gallery shows saved synthetic edits; it does not run the generator on-device.

## Android build and install

Use a full JDK 17 or newer, Android SDK platform 35 and build-tools 35.0.1. Both trained ONNX files must be present in `models/`.

```bash
cd android
./gradlew assembleDebug assembleDebugAndroidTest lintDebug
# Windows: gradlew.bat assembleDebug assembleDebugAndroidTest lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The installable APK supports Android 8.0 (API 26) and newer. It is signed with a development certificate for coursework distribution. Keep the same signing key if you want future builds to install as updates. Private signing keys and machine-specific SDK paths are excluded from Git.

To run the included device tests on a connected Android device or emulator:

```bash
./gradlew connectedDebugAndroidTest
```

The instrumentation suite checks class ordering, image padding, both model output contracts, probability/heatmap validity, heatmap toggling and clearing results when switching tabs. A successful build alone does not prove device tests passed; consult `docs/validation.md` for the checks actually completed.

## Dataset and experiment

The updated shared Drive snapshot initially contained **645 readable JPEG images**: 481 training, 83 validation and 81 test. A visual review confirmed 21 same-photograph pairs across splits; one additional byte-identical copy occurred within training. The experiment uses **623 images: 461 training, 81 validation and 81 test**, preserving test copies and excluding the redundant training/validation copies. `results/dataset_audit.json` and `results/leakage_removals.csv` give the final counts and exclusions. Any requested cleanup of the original Drive files is tracked separately in `results/drive_cleanup_plan.json`; the experiment is bound to its frozen input manifest.

Historical Assignment 1 used 441 reported images and a 60-image test split. This project uses the updated test split. Compare the retrained Baseline and SOTA results within this experiment; the historical A1 results are context, not a controlled estimate of improvement due only to the architecture.

Dataset images are not included in Git. `data/download_manifest.json` freezes file IDs, relative paths and SHA-256 hashes for the prepared experiment inputs. Download them with:

```bash
python tools/download_data.py --output data/prepared
```

The download fails if a file changes or becomes inaccessible, so a silently changed dataset cannot masquerade as this experiment.

## Reproduce

Create a Python environment and install `ml/requirements-lock.txt` for the recorded package versions (or `ml/requirements.txt` for the broader supported dependencies), then run:

```powershell
python ml/train.py --data data/prepared --out runs/reproduction --epochs 30
```

To inspect split counts, decoded-pixel hashes, and cross-split perceptual-hash candidates before fitting, add `--audit-only`. Training hard-stops on cross-split exact file or decoded-pixel duplicates. If perceptual-hash candidates exist, review `results/near_duplicate_candidates.csv`, save a reviewed copy with a nonempty `decision` column, then provide it with `--reviewed-candidates PATH`.

The trainer sets seed 2, uses at most six CPU threads, keeps test data out of model selection, applies validation-loss early stopping (patience 7) and halves the learning rate after three non-improving epochs. `baseline_aug_1` is the A1 scratch CNN architecture reproduction (three 3x3 16/32/64 convolution + max-pool blocks, GAP, Dense 64 ReLU, dropout .3, four logits, Adam 1e-3). The transfer experiments use `mobilenetv4_conv_small.e2400_r224_in1k` ImageNet weights from timm with a frozen feature backbone and trainable GAP linear four-class head; `mobilenetv4_aug_0` and `_aug_1` differ only in train-time augmentation and use the same schedule. To keep frozen-feature training practical on CPU, the enabled arm caches original plus four seeded augmented feature views per training image and samples one view per image each epoch; validation/test features are cached from originals once.

The provided A1 TensorFlow checkpoint is not part of the source materials. These weights are a **PyTorch architecture-level retraining**, not the original A1 checkpoint or a bit-exact TensorFlow rerun. Framework initialization and augmentation sampling differ. Historical scores remain those stated in the assignment (A1 CNN test accuracy 36.67%, macro-F1 0.32); they are not represented as results of these trained weights. The transfer backbone uses the [timm MobileNetV4 Conv Small model card](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k), released under Apache-2.0, and the [MobileNetV4 paper](https://arxiv.org/abs/2404.10518).

## Input and outputs

Both ONNX files accept `images`: float32 `[N,3,224,224]`, RGB, pixel range `[0,1]`. Each graph performs model-specific ImageNet normalization internally where needed. Outputs are `probabilities` `[N,4]` in class order `[bottles,cups,pens,scissors]`, and normalized nonnegative `heatmaps` `[N,4,H,W]` in `[0,1]`. Baseline maps are analytical Grad-CAM at the final pooled convolutional feature map, with gradients through GAP, Dense64 ReLU and the class head. MobileNetV4's GAP-linear classifier has CAM equal to Grad-CAM. Export checks probability and heatmap parity against PyTorch and saves ONNX Runtime CPU-compatible models.

Python image preparation uses EXIF transpose, RGB conversion, Lanczos resizing preserving aspect ratio, and centered grey (127) padding to 224 square. Training augmentation follows A1's horizontal flip, up to 18 degree rotation, and ±10% contrast; validation and test are unaugmented. Android uses integer-centered padding with Canvas bilinear resizing and bounded image decoding. The interpolation difference means Python test-set scores are not a measured APK accuracy score; see `android/README.md`.

## Artifacts

- `models/*.pt` and matching `*.meta.json`: trained state dictionaries and signatures bound to dataset fingerprint, training code hash, pretrained weights and schedule to prevent accidental stale checkpoint reuse.
- `models/*.onnx`: deployable classifier and class heatmaps.
- `results/*_history.csv`: observed epoch losses, accuracies, learning rate, and durations.
- `results/*_metrics.json`, `*_predictions.csv`: observed held-out test metrics and per-image probabilities.
- `results/dataset_manifest.csv`, `dataset_audit.json`, `dataset_fingerprint.txt`: relative input paths, SHA-256 hashes, split/class counts, cross-split exact-duplicate audit, and a fingerprint that prevents accidentally reusing weights with another split.
- `results/artifact_manifest.csv`, `run_config.json`: output hashes and the actual runtime/training configuration.
- `results/leakage_removals.csv`, `.json`: reviewed exclusions and data-cleaning provenance. Rebuild the clean split copy from the supplied source using `python ml/prepare_dataset.py --source /path/to/original/dataset --output /path/to/new/clean-copy`.
- `MODEL_CARD.md` and `results/model_card.json`: best epochs, data fingerprint, test metrics, ONNX contract, compatibility checks, and desktop graph latency.
- `results/qualitative_cam_examples.png`: true and predicted class Grad-CAM overlays for held-out correct and incorrect cases.

## Report and submission

The report template predates the Kaggle run. When filling its ArtAug section, use the four reviewed gallery pairs and the generation record in `results/artaug_status.json`. Keep the effect on classifier accuracy marked as unmeasured.

The report template covers the required content: dataset update, method and diagram, comparison, curves and ablation, explainability, ethics and reproducibility. The PDF companion has been typeset and verified at 8 pages. The DOCX is editable and has eight intended sections, but Word pagination was not rendered here and may reflow with different Word/font versions. Complete the remaining bracketed fields with your interpretation, device evidence and submission links; the template is not a finished report while those fields remain.

- `docs/demo_script.md`: a 3–5 minute demonstration plan.
- `docs/submission_checklist.md`: requirements mapped to submission evidence.
- `docs/validation.md`: actual build/model/device verification status.
- `docs/build_report_template.py`: portable report template generator.
- `docs/create_report_figures.py`: rebuilds the report’s combined training-curves and selected Grad-CAM panels from saved results.
- `docs/build_report_pdf.py`: independently typesets the editable DOCX as an 8-page PDF companion; it is not a Word-render proof.
- `ml/ObjectLab_Training.ipynb`: Colab training entry point.

To rebuild the report assets from this folder, install `docs/report-requirements.txt`, then run `python docs/create_report_figures.py`, `python docs/build_report_template.py`, and `python docs/build_report_pdf.py` in order. The report template already contains the verified experiment metrics; complete the remaining student reflection, phone test, screenshots, repository and demo links before submission.

## Repository

The selected repository is [alina-kovalenko/cv_assignment2_app](https://github.com/alina-kovalenko/cv_assignment2_app). The source ZIP includes the Android app, trained ONNX models, training code, notebooks and experiment records. Dataset photos, build output and signing keys are excluded. Use the `Reproduce` section above to prepare a local data copy and run the classifier experiments. The workflow in `.github/workflows/android.yml` builds the APK and checks Android lint on push.

## References

- [MobileNetV4 paper, ECCV 2024](https://arxiv.org/abs/2404.10518)
- [Exact pretrained timm model card](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391)
- [ONNX Runtime Java documentation](https://onnxruntime.ai/docs/get-started/with-java.html)

Model and dependency notices are listed in `THIRD_PARTY_NOTICES.md`.
