# ObjectLab Android

Native Java app for four-class household image classification. Inference runs locally with ONNX Runtime Android 1.20.0; the manifest requests no Internet permission. The exact tab labels and order are **Baseline**, **Enchanced Model**, and **Generated Photos**. The last tab browses four approved offline original/edit pairs; it does not classify those pairs or generate images on-device. Tapping **Generate** advances through saved examples in this order: bottles, cups, pens, scissors, then wraps. No classifier retraining with edited images or enhancement-benefit result is claimed. APK version 1.2 (version code 3) includes all four pairs.

## Build

From this directory:

```powershell
.\gradlew.bat :app:assembleDebug
.\gradlew.bat :app:assembleDebugAndroidTest
```

The app build stages only `*.onnx` files from `../models` into its generated model assets. It expects `baseline.onnx` and `sota.onnx`, with input `images` and outputs `probabilities` and `heatmaps`. Class order is listed in `app/src/main/assets/labels.txt`. Optional sample images can be placed in `app/src/main/assets/samples/`. The current `app/src/main/assets/artaug/manifest.json` and `images/` contain four reviewed pairs with `approved: true`: one each for bottles, cups, pens, and scissors. The manifest stores source hashes, output hashes, prompt, seed, model revisions, runtime, and generation metadata. To import another reviewed bundle, use `../tools/import_artaug.py`; it refuses to overwrite an existing `assets/artaug` directory. The app only displays saved files and does not fabricate examples.

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`. Instrumentation tests check label order, letterboxing, both ONNX model output contracts, and heatmap/model-tab state transitions. Model inference tests need the ONNX files and run on an Android device or emulator. The Generated Photos UI test covers empty and populated gallery states; rebuild version 1.2 and run it with the current approved image assets before release.

## Image preprocessing note

Gallery and camera images are EXIF-oriented, converted to RGB, scaled to fit 224×224 while preserving aspect ratio, and centered on RGB 127 padding. The app uses Android bitmap bilinear filtering and power-of-two decode subsampling for large images. The training/evaluation path uses Pillow Lanczos resizing. The image geometry and padding color match, but interpolation and large-image resampling can differ slightly, so Android confidence scores may not exactly match offline Python inference. No Android-device measurement is claimed until run on a device or emulator.
