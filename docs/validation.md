# Verification record

The classifier experiments were checked on 3 November 2026. APK 1.2 (version code 3), built on 6 November, includes four reviewed ArtAug pairs.

## Completed

| Check | Evidence and outcome |
|---|---|
| Dataset integrity | All 645 downloaded source images decoded. Twenty-one visually confirmed cross-split copies and one same-split exact copy were excluded. The frozen experiment contains 623 images (461/81/81). No remaining cross-split exact/decoded duplicates or pHash candidates at Hamming distance ≤6 were found. See `results/dataset_audit.json` and `results/leakage_removals.csv`. |
| Real training | Baseline best validation-loss epoch 9, stopped at epoch 16. Both MobileNetV4 arms reached the predeclared 30-epoch limit, with their best validation loss at epoch 30. Test data did not choose epochs. Histories and predictions are saved. |
| Held-out test | Baseline 27/81 correct, accuracy 33.33%, macro-F1 0.3074. MobileNetV4 augmentation ON and OFF both 79/81, accuracy 97.53%, macro-F1 0.9755. No test accuracy/F1 improvement was observed from augmentation in this single-seed comparison. |
| Export parity | PyTorch/ONNX probability and heatmap comparisons and analytical/autograd Grad-CAM comparisons passed. Exact tolerances and measured errors are recorded with the ML artifacts. |
| Older-runtime compatibility | ONNX Runtime **1.20.1 for Windows CPU** reproduced all 81 recorded test predictions with batch sizes 1 and 2. Maximum probability differences were 1.20e-7 (baseline) and 1.50e-6 (SOTA). Probability sums and finite heatmap ranges passed. See `results/runtime_compatibility.json` and `tools/verify_onnx.py`. |
| Android build | The application APK and instrumentation APK for version 1.2 (code 3) compiled with Android SDK 35. Android lint completed. See `results/apk_verification.json` for counts. |
| Packaged gallery | Four approved training originals and four corresponding ArtAug edits were checked against their SHA-256 hashes. The packaged manifest and all eight JPEGs match the reviewed assets. |
| APK contents | Both ONNX classifier files match the model exports byte-for-byte. Class order is bottles, cups, pens and scissors. |
| APK signature and permissions | Package `com.objectlab.app` has one v2 signer matching the previous APK. The minimum API is 26, target API is 35, and the manifest requests no Internet access. |

The universal APK includes arm64-v8a, armeabi-v7a, x86 and x86_64 native runtimes. Its size and SHA-256 are in `results/apk_verification.json`.

## ArtAug outputs and pending classifier experiment

The app tab labels, in order, are **Baseline**, **Enchanced Model**, and **Generated Photos**. The Generated Photos tab reads approved offline original/edit pairs and labels the edited view. Tap **Generate** to advance through available saved examples by class order: bottles, cups, pens, scissors, then wrap to bottles. The current four pairs are bottles `83265545fffeaf0d`, cups `d7b947bddc0ae98d`, pens `749d97e16adbec01`, and scissors `cae86b8c27dded0a`. The button navigates saved files; it does not generate on-device. This gallery does not run a classifier on the pair images. The separate classifier retraining comparison using edited images has not been performed, and no ArtAug accuracy/F1 result exists.

The supplied executed Kaggle notebook records four successful generations from its ten selected frozen training photos. Each generated result completed 30/30 steps on a Tesla T4. The four output originals were byte-identical to their source training images, source and generated hashes matched the ZIP manifest, and the reviewed pairs were imported as approved offline assets. The other selected photos were not generated in this run. These edits demonstrate the released adapter workflow; they are not a retrained classifier result.

## Device verification still required

See `results/apk_verification.json` for the packaged build checks. The instrumentation APK compiled. There was no connected Android device, and the emulator hypervisor was unavailable. Device installation, visual layout, camera/gallery input and phone inference latency still need a phone or functioning emulator.

Install the APK on Android 8.0 or newer, then check:

1. Both models load; select the same object photo in each tab and run inference.
2. Camera captures a full-resolution image; canceling capture returns safely to the app.
3. The heatmap toggle changes the displayed image; switching tabs clears the old prediction.
4. A rotated portrait photo displays upright; landscape orientation and large text remain usable.
5. Record the device model, Android version, model, number of measured runs, and the shown average latency. The app times model execution including heatmap output, excluding image decode/resize and rendering. Its displayed running average includes the first inference.
6. For automated checks, connect the device and run `android/gradlew.bat -p android connectedDebugAndroidTest` (or the equivalent shell command on macOS/Linux).
7. Install the rebuilt version 1.2 (version code 3). Open **Generated Photos** and tap **Generate** four times to confirm the bottle, cup, pen, and scissors pairs appear in that order, with the AI-edited label and offline source/edit views. The gallery has no classifier action. Confirm the three displayed tab labels and capture the gallery screenshot; verify classifier retraining separately before reporting any augmentation metrics.

Python evaluation uses Lanczos interpolation; Android uses bilinear bitmap resizing and bounded decode subsampling. Their geometric padding and normalization agree, but pixel-level inputs and confidence values can differ. Report the 81-image results as Python experiment scores, not measured APK accuracy.

## Limits of the evidence

Perceptual-hash screening cannot guarantee separation of every physical object, background or shooting session. Similar objects photographed separately can remain in different splits. The 81-image test is small and includes only four known classes; 97.53% is not a guarantee for arbitrary phone photos. There is no validated unknown-object detector, calibration study, multi-seed significance test, or segmentation ground truth.

GitHub publication and Drive trash operations are external actions. Their status must be reported only after verification in the respective service; local packaging is not evidence of publication or deletion.
