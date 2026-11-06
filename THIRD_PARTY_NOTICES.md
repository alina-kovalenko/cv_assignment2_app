# Third-party notices

## Offline ArtAug gallery

The Kaggle and Colab notebooks reference `DiffSynth-Studio/FLUX.1-Kontext-dev-lora-ArtAug` (adapter card: Apache-2.0) and Black Forest Labs FLUX.1-Kontext-dev / FLUX.1-dev auxiliary weights, which have their own gated access and FLUX.1-dev non-commercial terms. Those generator weights are not bundled in the APK or source ZIP. Consult the linked model cards and terms before downloading. The adapter license does not replace the base-model terms. Only manually reviewed, clearly labelled generated images are eligible for import. This is the released Kontext photo-editing adapter, not a reproduction of the original ArtAug training method.

- https://huggingface.co/DiffSynth-Studio/FLUX.1-Kontext-dev-lora-ArtAug
- https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://github.com/modelscope/DiffSynth-Studio

ObjectLab is a coursework project by Alina Kovalenko. No new license for the student's original source or dataset is implied by these dependency notices.

| Component | Use | License / source |
|---|---|---|
| timm MobileNetV4 Conv Small ImageNet checkpoint | Frozen visual feature backbone; its original classification head is replaced | Apache-2.0; https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k |
| PyTorch Image Models (timm) | Model definition and pretrained feature extraction | Apache-2.0; `licenses/timm-Apache-2.0.txt` |
| ONNX Runtime Android 1.20.0 | Local inference in the Android app | MIT; `licenses/onnxruntime-MIT.txt`; https://github.com/microsoft/onnxruntime |
| AndroidX Core 1.13.1 and ExifInterface 1.3.7 | Insets, FileProvider and photo orientation | Apache-2.0; https://android.googlesource.com/platform/frameworks/support/ |
| Gradle wrapper | Reproducible Android builds | Apache-2.0; https://github.com/gradle/gradle |

Python training packages remain subject to their respective upstream licenses; see the exact versions in `ml/requirements-lock.txt`. They are installed as dependencies rather than vendored in this repository.

The dataset was supplied by the student through a shared Drive folder. The source package contains a download manifest, hashes and analysis outputs, not a grant of rights to redistribute the photographs. Confirm ownership, collection permission and any third-party image licenses in the report before public redistribution. Cropped/annotated example figures retain the rights of their underlying photographs.

No generative image or video model is used in this experiment. Flips, rotations and contrast changes are conventional transformations of training photos. Grad-CAM overlays visualize model responses; they are not segmentation ground truth.
