"""PyTorch vs TensorFlow comparison (evaluation/pytorch_vs_tensorflow.py).

Scores both backends on the SAME test split with the SAME shared metrics
(evaluation/metrics.py), and adds inference time + model size so the
frameworks can be compared fairly (project spec section 5.2):

  training/validation accuracy -> training history CSVs + metrics JSONs
  precision/recall/F1          -> per-attribute metrics in both JSONs
  inference time               -> measured here on the test split
  model size                   -> checkpoint file size
  memory usage                 -> reported from the metrics JSONs if present

Missing attribute labels are ignored per attribute, never treated as errors.
The TensorFlow side is skipped (with a clear note) if TF is not installed.

Usage:  python evaluation/pytorch_vs_tensorflow.py --garment top
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import attribute_metrics, format_report, mean_accuracy  # noqa: E402
from shared.data import load_split  # noqa: E402
from shared.schema import load_spec  # noqa: E402


def model_size_mb(path):
    path = Path(path)
    return round(path.stat().st_size / 1e6, 1) if path.exists() else None


def evaluate_pytorch(test, spec, garment):
    """Returns (preds, seconds_per_image, versions)."""
    sys.path.insert(0, str(ROOT / "2_train" / "pytorch"))
    import torch  # noqa: E402
    from inference import load_model, predict_pil  # noqa: E402
    import torchvision  # noqa: E402

    ck_path = ROOT / "weights" / f"{garment}_pytorch.pt"
    bundle, _ = load_model(ck_path)
    preds, t0 = [], time.perf_counter()
    for s in test:
        from PIL import Image
        _, labels = predict_pil(bundle, spec, Image.open(s["path"]).convert("RGB"),
                                s.get("box"))
        preds.append(labels)
    elapsed = (time.perf_counter() - t0) / max(len(test), 1)
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}
    return preds, elapsed, versions, ck_path


def evaluate_tensorflow(test, spec, garment):
    """Returns (preds, seconds_per_image, versions). Skips if TF is missing."""
    try:
        import tensorflow as tf  # noqa: E402
    except ImportError:
        print("TensorFlow is not installed in this environment - TF side skipped.")
        print("(on Windows-native Python install tensorflow==2.10, or use WSL2)")
        return None, None, None, None
    sys.path.insert(0, str(ROOT / "2_train" / "tensorflow"))
    from inference import load_model, predict_pil  # noqa: E402

    ck_path = ROOT / "weights" / f"{garment}_tensorflow.keras"
    if not ck_path.exists():
        print(f"No TF checkpoint at {ck_path} - TF side skipped.")
        return None, None, None, None
    bundle, _ = load_model(ck_path)
    preds, t0 = [], time.perf_counter()
    for s in test:
        from PIL import Image
        _, labels = predict_pil(bundle, spec, Image.open(s["path"]).convert("RGB"),
                                s.get("box"))
        preds.append(labels)
    elapsed = (time.perf_counter() - t0) / max(len(test), 1)
    return preds, elapsed, {"tensorflow": tf.__version__}, ck_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment", choices=("top", "bottom"), default="top")
    ap.add_argument("--limit", type=int, default=0,
                    help="only evaluate the first N test samples (0 = all)")
    ap.add_argument("--out", default=None, help="output JSON (default: evaluation/results_<garment>.json)")
    a = ap.parse_args()

    spec = load_spec(a.garment)
    test = load_split(ROOT / "dataset" / a.garment / "splits" / "test.json")
    if not test:
        raise SystemExit(f"No test split for {a.garment}. Run shared/split.py first.")
    if a.limit:
        test = test[:a.limit]
    truth = [s["attributes"] for s in test]
    print(f"Test samples: {len(test)}\n")

    results = {"garment": a.garment, "test_samples": len(test), "backends": {}}

    for name, fn in (("pytorch", evaluate_pytorch), ("tensorflow", evaluate_tensorflow)):
        print(f"=== {name} ===")
        preds, sec_per_img, versions, ck_path = fn(test, spec, a.garment)
        if preds is None:
            results["backends"][name] = {"status": "skipped"}
            continue
        metrics = attribute_metrics(truth, preds, spec)
        acc = mean_accuracy(metrics)
        print(f"mean accuracy: {'N/A' if acc is None else f'{acc:.3f}'} | "
              f"inference {sec_per_img * 1000:.0f} ms/image | "
              f"model size {model_size_mb(ck_path)} MB")
        print(format_report(metrics))
        results["backends"][name] = {
            "status": "ok",
            "mean_accuracy": acc,
            "metrics": metrics,
            "seconds_per_image": round(sec_per_img, 4),
            "model_size_mb": model_size_mb(ck_path),
            "versions": versions,
        }
        print()

    out = Path(a.out) if a.out else ROOT / "evaluation" / f"results_{a.garment}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Comparison -> {out}")


if __name__ == "__main__":
    main()
