# ObjectLab model card

Dataset fingerprint: `d407cc89aa2b3f16a87e3a40b694272a47625626416adfd79b726d3a8b2de07d`
Final training source SHA-256: `de366fc53e4d85056917fe467051a34a0256a7a74ddf7a7d16973f7964a24c92`
Baseline checkpoint code ID: `c3365deed838b2ba637a1e06b8f10175ca0ece8c2da4ed3cc5e6fc97223eccba` (the baseline run completed before the SOTA feature-cache refactor). Checkpoint signatures and provenance are in `results/code_provenance.json`.

Class order: bottles, cups, pens, scissors.

Split counts: train 461, validation 81, test 81; 623 valid images.

Data review removed 21 visually confirmed cross-split duplicate copies and 1 byte-identical same-split duplicate. Test images were preserved.

## Results

| Model | Epochs / best epoch | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Scratch CNN (A1 architecture retraining) | 16 / 9 | 0.3333 | 0.3095 | 0.3304 | 0.3074 |
| MobileNetV4 (augmentation bank) | 30 / 30 | 0.9753 | 0.9783 | 0.9750 | 0.9755 |
| MobileNetV4 (no train augmentation) | 30 / 30 | 0.9753 | 0.9783 | 0.9750 | 0.9755 |

The MobileNetV4 arms share the same ImageNet-frozen feature extractor and schedule. Augmentation ON and OFF achieved the same results on this split. The two mistakes are scissors classified as pens. Per-class metrics and confusion matrices are in `results/*_metrics.json`.

## Inference contract

Input `images`: [N,3,224,224] RGB pixels in [0,1]. Outputs are `probabilities` [N,4] and `heatmaps` [N,4,H,W] with H×W 28×28 for the scratch model and 7×7 for MobileNetV4. Both models include heatmap computation in their ONNX graph.

Desktop CPU graph-only median latency (50 runs, five warmups, two intra-op threads): baseline 2.42 ms; MobileNetV4 3.31 ms. See `results/latency_benchmark.json` for environment and percentiles.

ONNX Runtime 1.20.1 validated all 81 test predictions and N=1/N=2 inputs; max probability error was 1.2e-07 for baseline and 1.5e-06 for MobileNetV4. This is desktop Python verification, not device-specific timing.

## Reference and limitations

The historical A1 CNN score (36.67% test accuracy, macro-F1 about 0.32 on its original 60-image test set) is provided for context only; the assignment dataset was later expanded. These models were trained on 461 training images, selected using 81 validation images, and evaluated on 81 held-out test images.

The MobileNetV4 checkpoint uses the [timm model card](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k) (Apache-2.0) and [paper](https://arxiv.org/abs/2404.10518). The original A1 TensorFlow weights were unavailable, so the baseline is a PyTorch architecture-level retraining.

These results use one seed and one manually curated split. They describe the supplied data and do not estimate performance across new capture sources.
