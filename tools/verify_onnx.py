"""Check shipped ONNX files against recorded predictions on the frozen test split.

Run with the runtime version being validated. Does not train or select models.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def prepare(path):
    with Image.open(path) as raw:
        rgb = ImageOps.exif_transpose(raw).convert("RGB")
        square = ImageOps.pad(rgb, (224, 224), method=Image.Resampling.LANCZOS, color=(127, 127, 127))
        return np.asarray(square, dtype=np.float32).transpose(2, 0, 1) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    classes = ["bottles", "cups", "pens", "scissors"]
    summary = {"runtime_version": ort.__version__, "provider": "CPUExecutionProvider", "scope": "Desktop Python ONNX Runtime; not Android device verification", "models": {}}
    for deployed, experiment in [("baseline", "baseline_aug_1"), ("sota", "mobilenetv4_aug_1")]:
        model_path = args.project / "models" / (deployed + ".onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        assert session.get_inputs()[0].name == "images"
        assert json.loads(session.get_modelmeta().custom_metadata_map["classes"]) == classes
        with (args.project / "results" / (experiment + "_predictions.csv")).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        samples = [prepare(args.data / Path(r["path"].replace("\\", "/"))) for r in rows]
        max_probability_error = 0.0
        all_predictions = []
        for begin in range(0, len(samples), 2):
            batch = np.stack(samples[begin:begin + 2])
            probabilities, heatmaps = session.run(["probabilities", "heatmaps"], {"images": batch})
            assert probabilities.shape == (len(batch), 4)
            assert heatmaps.shape[:2] == (len(batch), 4) and heatmaps.ndim == 4
            assert np.isfinite(probabilities).all() and np.isfinite(heatmaps).all()
            assert probabilities.min() >= 0 and probabilities.max() <= 1
            assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            assert heatmaps.min() >= -1e-6 and heatmaps.max() <= 1 + 1e-6
            expected = np.array([json.loads(r["probabilities"]) for r in rows[begin:begin + len(batch)]])
            max_probability_error = max(max_probability_error, float(np.abs(probabilities - expected).max()))
            all_predictions.extend(probabilities.argmax(axis=1).tolist())
        assert max_probability_error < 1e-4, (deployed, max_probability_error)
        predicted = [classes[i] for i in all_predictions]
        assert predicted == [r["predicted_label"] for r in rows]
        correct = sum(p == r["true_label"] for p, r in zip(predicted, rows))
        summary["models"][deployed] = {"test_images": len(rows), "correct": correct, "accuracy": correct / len(rows), "max_probability_error_vs_recorded_pytorch": max_probability_error, "batch_sizes_checked": [1, 2], "probabilities_finite_normalized": True, "heatmaps_finite_in_0_1": True, "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    output = args.output or args.project / "results" / "runtime_compatibility.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
