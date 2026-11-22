#!/usr/bin/env python3
"""
Is the model overfitting, underfitting, or about right? And how accurate is it?

    python evaluate.py neck_crew
    python evaluate.py neck_crew --weights runs/neck_crew/weights/best.pt --split test

Reads  runs/<task>/results.csv  (training curves) and re-scores the model on the
train / val / test splits built by build_dataset.py.
Writes runs/<task>/evaluation/report.json and curves.png.

The verdict uses simple rules of thumb (thresholds at the top of the file), not a
statistical proof. With few images the numbers are noisy: look at the sample sizes.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# ---- rule-of-thumb thresholds -------------------------------------------------
GAP_DETECT, GAP_CLS = 0.15, 0.08        # train score - val score above this => overfitting
LOW_DETECT, LOW_CLS = 0.40, 0.70        # val score below this (and train low too) => underfitting
VAL_LOSS_RISE = 1.10                    # last val loss > 110% of its minimum => diverging


def wilson(k, n, z=1.96):
    """95% confidence interval for an accuracy k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_curves(run: Path):
    df = pd.read_csv(run / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    tr = [c for c in df.columns if c.startswith("train/") and "loss" in c]
    va = [c for c in df.columns if c.startswith("val/") and "loss" in c]
    metric = next((c for c in df.columns if c.startswith("metrics/mAP50-95")), None) or \
        next((c for c in df.columns if "accuracy_top1" in c), None)
    train_loss = df[tr].sum(axis=1) if tr else None
    val_loss = df[va].sum(axis=1) if va else None
    return df, train_loss, val_loss, (df[metric] if metric else None), metric


# ---- scoring ------------------------------------------------------------------
def kwargs_dev(device):
    return {"device": device} if device else {}


def score_detect(model, data_yaml, split, imgsz, device, save_dir):
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, plots=(split != "train"), verbose=False,
                  project=str(save_dir), name=f"val_{split}", exist_ok=True, **kwargs_dev(device))
    summary = {"map50_95": float(m.box.map), "map50": float(m.box.map50),
               "precision": float(m.box.mp), "recall": float(m.box.mr)}
    return summary, m


def count_labels(ds: Path, split, names):
    counts = {n: 0 for n in names}
    for f in (ds / "labels" / split).glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                counts[names[int(line.split()[0])]] += 1
    return counts


def score_cls(model, ds: Path, split, imgsz, device):
    inv = {v: k for k, v in model.names.items()}
    files, y_true = [], []
    for cname in inv:
        for f in sorted((ds / split / cname).glob("*")):
            if f.suffix.lower() in IMG_EXT:
                files.append(str(f))
                y_true.append(inv[cname])
    y_pred = []
    for i in range(0, len(files), 64):
        for r in model.predict(files[i:i + 64], imgsz=imgsz, verbose=False, **kwargs_dev(device)):
            y_pred.append(int(r.probs.top1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    k = int((y_true == y_pred).sum()) if n else 0
    cm = np.zeros((len(inv), len(inv)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    rows = []
    for i in range(len(inv)):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        rows.append({"class": model.names[i], "n": int(cm[i, :].sum()), "precision": float(prec),
                     "recall": float(rec), "f1": float(2 * prec * rec / (prec + rec + 1e-9))})
    lo, hi = wilson(k, n)
    return {"accuracy": k / n if n else 0.0, "n": n, "ci95": [lo, hi]}, rows, cm, [model.names[i] for i in range(len(inv))]


# ---- diagnosis ----------------------------------------------------------------
def diagnose(is_det, train_score, val_score, train_loss, val_loss, val_curve):
    gap_thr, low_thr = (GAP_DETECT, LOW_DETECT) if is_det else (GAP_CLS, LOW_CLS)
    gap = train_score - val_score
    reasons = []
    overfit = gap > gap_thr
    if overfit:
        reasons.append(f"train score {train_score:.3f} is {gap:.3f} above val score {val_score:.3f} (limit {gap_thr})")

    n = len(val_loss) if val_loss is not None else 0
    if n >= 10:
        best = int(val_loss.values.argmin())
        rising = val_loss.iloc[-1] > val_loss.min() * VAL_LOSS_RISE and best < n * 0.8
        train_down = train_loss is not None and train_loss.iloc[-1] < train_loss.iloc[best]
        if rising and train_down:
            overfit = True
            reasons.append(f"val loss bottomed at epoch {best + 1} then rose while train loss kept falling")

    underfit = val_score < low_thr and train_score < low_thr + 0.10
    if underfit:
        reasons.append(f"both train ({train_score:.3f}) and val ({val_score:.3f}) are low (val < {low_thr})")

    improving = False
    if val_curve is not None and len(val_curve) >= 10:
        k = max(5, len(val_curve) // 5)
        slope = np.polyfit(range(k), val_curve.values[-k:], 1)[0]
        improving = slope > 0.002 and val_curve.iloc[-1] >= val_curve.max() * 0.99
        if improving:
            reasons.append("val score was still rising at the last epochs")

    if overfit and not underfit:
        verdict = "OVERFITTING"
        advice = ["Add more labeled images (most effective), especially varied backgrounds, lighting and poses.",
                  "Use a smaller model (yolo26n) or fewer epochs; keep patience for early stopping.",
                  "Check for near-duplicate photos split across train and val."]
    elif underfit:
        verdict = "UNDERFITTING"
        advice = ["Train longer" + (" (curves were still improving)." if improving else "."),
                  "Try a larger model (yolo26s / yolo26m) or a bigger --imgsz.",
                  "Check label quality: inconsistent or ambiguous labels look like underfitting. Re-read the labeling rules.",
                  "If labels are clean and training is long, you probably need more data."]
    else:
        verdict = "GOOD FIT" + (" (could still improve with more epochs)" if improving else "")
        advice = ["Train and val agree. To improve further: more data, then test on photos unlike your training set."]
    return verdict, reasons, advice, gap


def plot_curves(df, train_loss, val_loss, val_curve, metric, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = df["epoch"] if "epoch" in df else range(len(df))
    if train_loss is not None:
        ax[0].plot(x, train_loss, label="train loss")
    if val_loss is not None:
        ax[0].plot(x, val_loss, label="val loss")
    ax[0].set_title("Loss (summed)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    if val_curve is not None:
        ax[1].plot(x, val_curve, label=f"val {metric}")
        ax[1].set_title(metric); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--weights")
    ap.add_argument("--split", default="test", choices=["val", "test"], help="held-out split for final statistics")
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "tasks.yaml").read_text("utf-8"))[args.task]
    is_det = cfg["mode"] == "detect"
    run = ROOT / "runs" / args.task
    ds = ROOT / "datasets" / args.task
    weights = Path(args.weights) if args.weights else run / "weights" / "best.pt"
    if not weights.exists() or not ds.exists():
        sys.exit("Need a trained model and a built dataset. Run build_dataset.py then train.py first.")
    rep = json.loads((ds / "dataset_report.json").read_text())
    names = rep["names"]
    imgsz = cfg.get("imgsz", 640)
    out = run / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    split = args.split
    if rep["images"].get(split, 0) == 0:
        print(f"(no {split} images, falling back to val)")
        split = "val"

    model = YOLO(str(weights))
    report = {"task": args.task, "weights": str(weights), "split_used": split, "images": rep["images"]}

    # 1) training curves
    df, train_loss, val_loss, val_curve, metric = load_curves(run)
    plot_curves(df, train_loss, val_loss, val_curve, metric, out / "curves.png")

    # 2) scores on train / val / held-out
    per_class, cm, cm_names = None, None, None
    if is_det:
        s_train, _ = score_detect(model, ds / "data.yaml", "train", imgsz, args.device, out)
        s_val, _ = score_detect(model, ds / "data.yaml", "val", imgsz, args.device, out)
        s_hold, m = score_detect(model, ds / "data.yaml", split, imgsz, args.device, out)
        counts = count_labels(ds, split, names)
        per_class = []
        for i, ci in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class.append({"class": names[int(ci)], "n": counts[names[int(ci)]], "precision": float(p),
                              "recall": float(r), "f1": float(2 * p * r / (p + r + 1e-9)),
                              "map50": float(ap50), "map50_95": float(ap)})
        key = "map50_95"
        t_score, v_score = s_train[key], s_val[key]
        report["confusion_matrix_png"] = str(out / f"val_{split}" / "confusion_matrix_normalized.png")
    else:
        s_train, _, _, _ = score_cls(model, ds, "train", imgsz, args.device)
        s_val, _, _, _ = score_cls(model, ds, "val", imgsz, args.device)
        s_hold, per_class, cm, cm_names = score_cls(model, ds, split, imgsz, args.device)
        key = "accuracy"
        t_score, v_score = s_train[key], s_val[key]

    verdict, reasons, advice, gap = diagnose(is_det, t_score, v_score, train_loss, val_loss, val_curve)
    report.update({"train": s_train, "val": s_val, "heldout": s_hold, "per_class": per_class,
                   "gap_train_minus_val": gap, "verdict": verdict, "reasons": reasons, "advice": advice})

    # 3) data warnings
    warn = []
    if rep["images"].get(split, 0) < 30:
        warn.append(f"only {rep['images'].get(split, 0)} {split} images: treat the numbers below as rough")
    tr_counts = rep["instances"]["train"]
    for n_, c in tr_counts.items():
        if c < 50:
            warn.append(f"class '{n_}' has {c} training instances (<50)")
    report["data_warnings"] = warn
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- print ----
    line = "=" * 60
    print(f"\n{line}\n  {args.task}  |  {'detection (mAP50-95)' if is_det else 'classification (accuracy)'}\n{line}")
    print(f"  Train   : {t_score:.3f}\n  Val     : {v_score:.3f}\n  Gap     : {gap:+.3f}")
    print(f"\n  VERDICT : {verdict}")
    for r in reasons:
        print(f"   - {r}")
    print("  Suggestions:")
    for a in advice:
        print(f"   * {a}")
    print(f"\n  Held-out ({split}) statistics")
    if is_det:
        print(f"   mAP50-95 {s_hold['map50_95']:.3f} | mAP50 {s_hold['map50']:.3f} | "
              f"precision {s_hold['precision']:.3f} | recall {s_hold['recall']:.3f}")
    else:
        lo, hi = s_hold["ci95"]
        print(f"   accuracy {s_hold['accuracy']:.3f} on {s_hold['n']} crops  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"   {'class':<16}{'n':>6}{'prec':>8}{'recall':>8}{'F1':>8}")
    for r in per_class:
        print(f"   {r['class']:<16}{r['n']:>6}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}")
    if cm is not None:
        print("\n   Confusion matrix (rows = true, cols = predicted)")
        print("   " + " " * 14 + "".join(f"{n[:8]:>9}" for n in cm_names))
        for n, row in zip(cm_names, cm):
            print(f"   {n[:12]:<14}" + "".join(f"{v:>9}" for v in row))
    for w in warn:
        print(f"\n  ! {w}")
    print(f"\n  Saved: {out / 'report.json'}  and  {out / 'curves.png'}\n")


if __name__ == "__main__":
    main()
