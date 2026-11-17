"""Training - TensorFlow (2_train/tensorflow/train.py).

Two-phase fine-tuning (common transfer-learning recipe):
  1. frozen backbone, train the new heads with the full LR
  2. unfreeze the backbone, continue with a 10x smaller LR
EarlyStopping keeps the best validation loss; the best model is saved in
Keras format plus a sidecar JSON (spec + config). The final test split is
scored with the SAME shared metrics as the PyTorch backend
(evaluation/metrics.py) and written to <garment>_tensorflow_metrics.json.

Usage:  python 2_train/tensorflow/train.py --garment top
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
TF = Path(__file__).resolve().parent
for p in (ROOT, TF):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import make_dataset  # noqa: E402
from model import build_model, compile_model, predict_dict, unfreeze_backbone  # noqa: E402
from evaluation.metrics import attribute_metrics, format_report, mean_accuracy  # noqa: E402
from shared.data import load_split  # noqa: E402
from shared.preprocessing import config  # noqa: E402
from shared.schema import load_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment", choices=("top", "bottom"), default="top")
    ap.add_argument("--test-only", action="store_true", help="score the test split, no training")
    ap.add_argument("--epochs", type=int, default=10, help="epochs per phase")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--show", type=int, default=0,
                    help="popup with Grad-CAM boxes on N random images when done")
    ap.add_argument("--save-visual", default=None,
                    help="with --show: save PNGs to this folder instead of popup")
    a = ap.parse_args()
    test_only(a) if a.test_only else _train(a)


def _train(a):
    # show where training runs: TF >= 2.11 has no CUDA on native Windows,
    # so this prints GPU inside WSL2 but CPU on normal Windows
    gpus = tf.config.list_physical_devices("GPU")
    device = f"GPU x{len(gpus)}" if gpus else \
        "CPU (native-Windows TF has no CUDA; use WSL2 for GPU - see README)"
    print(f"Device: {device}")
    spec = load_spec(a.garment)
    splits = ROOT / "dataset" / a.garment / "splits"
    data = {n: load_split(splits / f"{n}.json") for n in ("train", "valid")}
    if not data["train"]:
        raise SystemExit(f"No data. Run: python shared/split.py --garment {a.garment}")

    tr, n_tr = make_dataset(data["train"], a.garment, a.batch, train=True)
    va, n_va = make_dataset(data["valid"], a.garment, a.batch, train=False)

    model = build_model(spec, input_size=(288, 384))
    compile_model(model, spec, lr=a.lr)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                            restore_best_weights=True)

    print("Phase 1: frozen backbone")
    model.fit(tr, validation_data=va, epochs=a.epochs, callbacks=[stop])

    print("Phase 2: unfrozen backbone, 10x smaller LR")
    unfreeze_backbone(model)
    compile_model(model, spec, lr=a.lr * 0.1)
    model.fit(tr, validation_data=va, epochs=a.epochs, callbacks=[stop])

    out_dir = ROOT / "weights"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{a.garment}_tensorflow.keras"
    model.save(out)
    side = out.with_suffix(".json")
    side.write_text(json.dumps({
        "spec": spec,
        "config": config(a.garment, "mobilenetv2", extra={
            "args": {k: v for k, v in vars(a).items()},
            "num_samples": {n: len(v) for n, v in data.items()},
            "versions": {"tensorflow": tf.__version__,
                         "keras": tf.keras.__version__,
                         "python": sys.version.split()[0]},
        }),
    }, indent=2), encoding="utf-8")
    print(f"Saved {out} (+ spec/config json)")

    test_only(a)
    if a.show:
        sys.path.insert(0, str(TF))  # local visualize.py (shared/visualize.py also exists)
        from visualize import popup_from_samples
        shown = random.sample(data["valid"], min(a.show, len(data["valid"])))
        popup_from_samples(shown, a.garment, a.save_visual)


def predictions_for_split(test, garment, batch):
    """Test samples -> (truth, preds) label dicts scored by shared metrics."""
    spec = load_spec(garment)
    model = tf.keras.models.load_model(
        ROOT / "weights" / f"{garment}_tensorflow.keras")
    ds, _ = make_dataset(test, garment, batch, train=False)
    preds = []
    for x, _, _ in ds:  # (image, labels, weights)
        outs = predict_dict(model, x)
        for i in range(len(x)):
            p_row = {}
            for k, cat in spec.items():
                if cat["mode"] == "multi":
                    picked = [cat["options"][j] for j in
                              np.where(outs[k][i] > 0.5)[0]]
                    p_row[k] = picked or [cat["options"][int(np.argmax(outs[k][i]))]]
                else:
                    p_row[k] = cat["options"][int(np.argmax(outs[k][i]))]
            preds.append(p_row)
    truth = [s["attributes"] for s in test]
    return truth, preds


def test_only(args):
    """Final unbiased check on the never-touched test split."""
    spec = load_spec(args.garment)
    test = load_split(ROOT / "dataset" / args.garment / "splits" / "test.json")
    if not test:
        raise SystemExit("No test split. Run: python shared/split.py --garment <g>")

    t0 = time.perf_counter()
    truth, preds = predictions_for_split(test, args.garment, args.batch)
    elapsed = (time.perf_counter() - t0) / max(len(test), 1)

    metrics = attribute_metrics(truth, preds, spec)
    acc = mean_accuracy(metrics)
    print(f"TEST mean acc {'N/A' if acc is None else f'{acc:.3f}'} | "
          f"{elapsed * 1000:.0f} ms/image")
    print(format_report(metrics))
    out = ROOT / "weights" / f"{args.garment}_tensorflow_metrics.json"
    out.write_text(json.dumps({
        "garment": args.garment, "backend": "tensorflow",
        "metrics": metrics, "mean_accuracy": acc,
        "seconds_per_image": round(elapsed, 4),
        "versions": {"tensorflow": tf.__version__,
                     "python": sys.version.split()[0]},
        "args": {k: v for k, v in vars(args).items()},
    }, indent=2), encoding="utf-8")
    print(f"Metrics -> {out}")
    if args.show:
        sys.path.insert(0, str(TF))  # local visualize.py (shared/visualize.py also exists)
        from visualize import popup_from_samples
        shown = random.sample(test, min(args.show, len(test)))
        popup_from_samples(shown, args.garment, args.save_visual)


if __name__ == "__main__":
    main()
