# ArtAug photo pilot

This workflow prepares a small, auditable photo-enhancement pilot. It uses the released **FLUX.1-Kontext ArtAug LoRA** for image-to-image aesthetic editing. It does not reproduce the 2024 ArtAug research training procedure and it does not train or fine-tune the generator.

## Current state

The executed Kaggle run is recorded in `ml/ObjectLab_ArtAug_Kaggle_Executed.ipynb`; its four generated image outputs and runtime logs are preserved. `ml/ObjectLab_ArtAug_Kaggle_10.ipynb` is the output-free rerun copy with the same four generation sections. Both notebooks contain an embedded input bundle with ten candidate training photos; only photos 1–4 have generation sections, and photos 5–10 were not generated. The 6 November 2026 run loaded the pinned 14-file model bundle and generated four training photos on a Tesla T4. Each generation completed 30 steps. The outputs and run metadata are in the reviewed `artaug_pilot` bundle imported into the Android app assets. This validates the recorded run, not compatibility with every GPU/runtime.

The notebook contains the selected photo inputs and pinned manifest, so a separate pilot-input ZIP or `ObjectLab_Source.zip` upload is not needed for the four-photo run. Attach a prior `ArtAug_Pilot_Output.zip` only if you want to restore existing generated pairs in a new session. The approved asset manifest and `results/artaug_status.json` record the provenance of the four generated pairs.

The four class-specific prompts are recorded with each generated item. In the executed run, the manifest records the pinned FLUX.1-Kontext-dev, FLUX.1-dev, and ArtAug adapter revisions; the adapter digest; local manifest hash; 14 verified model files; DiffSynth 2.1.12; PyTorch 2.11.0+cu130; BF16 compute; FP8 offload; T4 runtime; actual dimensions; 30 steps; guidance; and per-image duration. Four reviewed examples are approved in the imported bundle; the other six selected photos were not generated in that run. The separate classifier retraining experiment and its accuracy/F1 comparison remain pending.

The project artifact stores dataset manifests, not the image files. Upload the current `ObjectLab_Source.zip` project archive at notebook startup; extraction rejects paths that escape the destination. The executed notebook selected ten deterministic examples from the frozen **training** split, verified their source files against recorded SHA-256 values, and previewed them before loading model weights. Validation and test rows are never eligible for the pilot. Drive IDs come from `data/download_manifest.json`; the frozen split and hashes come from `results/dataset_manifest.csv`.

## Runtime access and limitations

The adapter's official card documents `FluxImagePipeline`, BF16 CUDA inference, the `kontext_images` argument, and the instruction `Enhance the aesthetic quality of this image.` The current notebook follows the official DiffSynth FLUX.1-Kontext pipeline API and applies the adapter at alpha 1. Runtime loading requires a GPU runtime with CUDA and BF16 support; this is a preflight check, not a minimum-VRAM guarantee. The recorded Kaggle T4 run completed four generations; one successful environment does not guarantee another runtime will work.

The base model and auxiliary FLUX.1-dev text encoder/VAE repositories require gated Hugging Face access. The notebook checks authenticated access to their actual weight files without downloading them. During Run All, the authentication cell stops with setup instructions if the `HF_TOKEN` secret is unavailable or lacks the required read access. Before running model setup, accept the Black Forest Labs terms for both gated repositories: [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). Create a Hugging Face user access token with the `read` role, or a fine-grained token with read-only access to the required model repositories. Add it in Kaggle Secrets for the ten-photo notebook, or Colab Secrets for the older workflow, under the exact name `HF_TOKEN` and enable notebook access. The notebook reads that secret without printing it; it does not use Hugging Face browser OAuth, `login()`, or Git credentials. Never paste the token into a cell or save it in Drive. The base model is under the FLUX.1-dev non-commercial license; the adapter's Apache-2.0 label does not replace the base-model terms. The official adapter card currently has no Hugging Face hosted inference provider. See the [adapter card](https://huggingface.co/DiffSynth-Studio/FLUX.1-Kontext-dev-lora-ArtAug), the [base model card/license](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), and the [DiffSynth low-VRAM Kontext example](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux/model_inference_low_vram/FLUX.1-Kontext-dev.py).

The adapter model card's import path (`flux_image_new`) differs from the current official DiffSynth Kontext example (`flux_image`). The notebook pins the DiffSynth Python package version and uses the current documented `flux_image` module; it records the installed version and model revisions into its output metadata. Loading the base model also requires HF access to the FLUX.1-dev text encoders and VAE referenced by the official adapter recipe. Treat a failed preflight or API compatibility check as a stop condition; do not silently substitute another model or generate untracked outputs.

## Selection and review protocol

The executed notebook selected ten training photos from the frozen manifest with a fixed seed. A manual pilot-only QC exclusion removes `train/pens/77.jpg` because the dominant scissors alongside the pen makes its class visually ambiguous; this does not change the frozen dataset, splits, or classifier evaluation. Each edit gets a deterministic seed derived from the pilot seed and its stable ID. Four of the ten selected photos have saved, reviewed outputs: bottles `83265545fffeaf0d`, cups `d7b947bddc0ae98d`, pens `749d97e16adbec01`, and scissors `cae86b8c27dded0a`. Prompt text and seed are stored with every generated image.

The four saved original/edit pairs were reviewed and marked approved. Their originals match the frozen training files byte-for-byte; the edited JPEG hashes match the output manifest. The visible objects and scene composition are retained. Leave `approved` false for any future artifact that changes object class, count, shape, labels, or scene in a way that makes the photo unsuitable. Approved outputs remain marked synthetic and retain their original-source path and hash. The importer rejects validation/test sources, changed source hashes, unsafe paths, missing files, and unapproved-only bundles. It copies only manually approved image pairs into `android/app/src/main/assets/artaug/manifest.json` plus `images/`.

The app's **Generated Photos** tab is a saved-pair gallery. Its **Generate** button advances through installed examples in class order—bottles, cups, pens, scissors—then wraps; it does not run image generation or classification on-device. If the four reviewed edits are later used for a classifier experiment, add approved edits to the training split only, keep validation and test unchanged, and compare against the original-only baseline under the same controlled schedule. That retraining has not been run. Do not claim better metrics until the classifier has actually been retrained and evaluated on the frozen test split. With four generated examples, treat the outputs as a workflow demonstration and QC sample, not as broad augmentation coverage.

## Files

- `ml/ObjectLab_ArtAug_Kaggle_Executed.ipynb` — the saved Kaggle run with four generated image outputs and logs.
- `ml/ObjectLab_ArtAug_Kaggle_10.ipynb` — the output-free, four-photo rerun notebook; its embedded input bundle contains ten candidate source photos.
- `ml/ObjectLab_ArtAug_Colab.ipynb` — staged Colab workflow; final prompts are in place and `RUN_GENERATION` starts false.
- `ml/artaug_prepare.py` — deterministic training-only source selection and bundle validation.
- `tools/import_artaug.py` — hash-checked importer for manually approved examples.

Run the importer only after exporting the Colab output folder:

```powershell
python tools/import_artaug.py --bundle path/to/artaug_output
```

It will refuse to overwrite an existing app `assets/artaug` directory. Review the copied manifest before building the APK.


