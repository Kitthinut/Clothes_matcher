#!/usr/bin/env python3
"""
Is the model overfitting, underfitting, or about right? And how accurate is it?

    python evaluate.py                             # popup picker (default: TEST split)
    python evaluate.py neck_crew
    python evaluate.py neck_crew --weights runs/neck_crew/weights/best.pt --split test
    python evaluate.py multi                       # legacy multi.pt via config.json
    python evaluate.py model_v1                    # versioned multi checkpoint
    python evaluate.py --models model_v1,model_v0  # compare versions
    python evaluate.py --all                       # every trained model
    python evaluate.py --cli                       # list available models, no GUI

Reads  runs/<task>/results.csv  (training curves) and re-scores the model on the
train / val / test splits built by build_dataset.py. For a versioned model the
class list, dataset path and weights come from
train/multi-models/model_vN.meta.json (no tasks.yaml entry required).
Writes runs/<task>/evaluation/report.json and curves.png.

The verdict uses simple rules of thumb (thresholds at the top of the file), not a
statistical proof. With few images the numbers are noisy: look at the sample sizes.
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from ultralytics import YOLO

from select_models import (
    find_all_weights,
    is_version_name,
    model_label,
    pick_models,
    task_name,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
MULTI = ROOT / "multi-models"
MULTI_DATASET = MULTI / "dataset"
MULTI_CONFIG = MULTI / "config.json"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---- rule-of-thumb thresholds -------------------------------------------------
GAP_DETECT, GAP_CLS = 0.15, 0.08        # train score - val score above this => overfitting
LOW_DETECT, LOW_CLS = 0.40, 0.70        # val score below this (and train low too) => underfitting
VAL_LOSS_RISE = 1.10                    # last val loss > 110% of its minimum => diverging

# Printed at the TOP of every report so the numbers are self-explanatory.
FORMULAS = {
    "IoU": "intersection(box_pred, box_true) / union(box_pred, box_true)",
    "TP/FP/FN": "a detection is TP when IoU >= 0.5 and the class matches; otherwise FP; "
                "an unmatched ground-truth box is FN",
    "Precision": "TP / (TP + FP)",
    "Recall": "TP / (TP + FN)",
    "F1": "2 * Precision * Recall / (Precision + Recall)",
    "AP50": "101-point interpolated area under the precision-recall curve at IoU=0.5",
    "mAP50 / mAP50-95": "mean AP50 over classes / mean AP over IoU 0.50,0.55,...,0.95",
    "Accuracy": "correct_predictions / total_predictions (held-out split)",
    "Wilson 95% CI": "p_adj = (p + z^2/2n) / (1 + z^2/n); "
                     "half = z*sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n); z = 1.96",
    "Consistency": "matched_detections_with_predicted_class_equal_to_sidecar_attribute "
                   "/ matched_detections_checked (IoU >= 0.5; tasks.yaml aliases applied)",
    "Train-val gap": "train_score - val_score; overfit if > 0.15 (detect) or > 0.08 (cls)",
}

FORMULA_MD = """## Formulas used for every number in this report

```
IoU          = intersection(pred, true) / union(pred, true)
TP / FP / FN = IoU >= 0.5 and class matches -> TP; extra detection -> FP;
               unmatched ground-truth box -> FN
Precision    = TP / (TP + FP)
Recall       = TP / (TP + FN)
F1           = 2 * Precision * Recall / (Precision + Recall)
AP50         = 101-point interpolated area under the precision-recall curve (IoU=0.5)
mAP50-95     = mean AP over IoU 0.50, 0.55, ..., 0.95 and over classes
Accuracy     = correct / total on the held-out split
Wilson CI    = adjusted p +/- z*sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n), z=1.96
Consistency  = matched detections whose predicted class equals the original JSON
               sidecar attribute / matched detections checked (aliases from tasks.yaml)
Train-val gap= train_score - val_score  (overfit if > 0.15 detect / > 0.08 cls)
```

"""


def format_score(score) -> str:
    """Readable one-line score (all metrics stay in the JSON report)."""
    if not isinstance(score, dict):
        return "n/a"
    parts = []
    for key, label in (("accuracy", "accuracy"), ("map50", "mAP50"),
                       ("map50_95", "mAP50-95"), ("precision", "P"),
                       ("recall", "R")):
        if score.get(key) is not None:
            parts.append(f"{label} {float(score[key]):.3f}")
    if score.get("n") is not None:
        parts.append(f"n={score['n']}")
    if score.get("ci95"):
        lo, hi = score["ci95"]
        parts.append(f"95% CI {lo:.3f}-{hi:.3f}")
    return " | ".join(parts) if parts else "n/a"


def headline_score(result) -> tuple[str, float]:
    """The one headline metric: accuracy for classifiers, mAP50 for detectors."""
    held = result.get("heldout") or {}
    if "accuracy" in held and held["accuracy"] is not None:
        return "accuracy", float(held["accuracy"])
    return "mAP50", float(held.get("map50", 0.0) or 0.0)


def mean_f1(result) -> float:
    values = [float(row.get("f1", 0.0) or 0.0) for row in (result.get("per_class") or [])]
    return float(np.mean(values)) if values else 0.0


def summary_rows(results) -> list[tuple[str, ...]]:
    """Rows shared by the console summary and the Markdown summary table."""
    rows = []
    for result in results:
        metric, value = headline_score(result)
        agreement = float((result.get("consistency") or {}).get("agreement", 0.0) or 0.0)
        rows.append((str(result.get("task", "?")), str(result.get("split_used", "?")),
                     f"{metric} {value:.3f}", f"{mean_f1(result):.3f}",
                     f"{agreement:.3f}", str(result.get("verdict", "n/a"))))
    return rows


SUMMARY_HEADER = ("Model", "Split", "mAP50 / accuracy", "F1", "Consistency", "Verdict")


def print_selected_summary(results):
    """Compact console table: one line per model, then any data warnings."""
    rows = summary_rows(results)
    widths = [max([len(header)] + [len(row[i]) for row in rows])
              for i, header in enumerate(SUMMARY_HEADER)]
    line = "-" * (sum(widths) + 5 * 3)

    def render(cells):
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    print(f"\n{line}\n{render(SUMMARY_HEADER)}\n{line}")
    for row in rows:
        print(render(row))
    print(line)
    for result in results:
        for warning in result.get("data_warnings") or []:
            print(f"  ! {result.get('task', '?')}: {warning}")


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


# ---- versioned multi model metadata -------------------------------------------
def version_meta_path(version: str) -> Path:
    """train/multi-models/model_vN.meta.json for a version like 'model_v1'."""
    return MULTI / f"{version}.meta.json"


def load_version_meta(version: str) -> dict | None:
    """Read a versioned checkpoint's metadata; None if absent or malformed."""
    path = version_meta_path(version)
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _resolve_path(value, *bases: Path) -> Path:
    """First existing candidate for a path stored in metadata (relative or absolute)."""
    path = Path(str(value))
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return (bases[0] / path) if bases else path


def resolve_version_weights(version: str, meta: dict) -> Path:
    """Weights named by meta['weights'], with model_vN.pt as the safe fallback."""
    raw = meta.get("weights") or (MULTI / f"{version}.pt")
    return _resolve_path(raw, MULTI, ROOT, ROOT.parent, Path.cwd())


def resolve_version_dataset(meta: dict) -> Path:
    """Dataset named by the metadata, defaulting to the multi-models dataset."""
    raw = (meta.get("dataset") or meta.get("dataset_path")
           or meta.get("dataset_dir") or (MULTI / "dataset"))
    return _resolve_path(raw, MULTI, ROOT, ROOT.parent)


def version_cfg(meta: dict, version: str) -> dict:
    """Build a config from a version's meta.json, using config.json as fallback."""
    cfg = dict(load_multi_config())
    cfg["mode"] = str(meta.get("mode") or cfg.get("mode") or "detect")
    cfg["version"] = version
    classes, regions = meta.get("classes"), meta.get("regions")
    if isinstance(classes, list):
        cfg["model_classes"] = list(classes)      # keep 'classes' mapping-shaped
    elif isinstance(classes, dict):
        cfg["classes"] = dict(classes)
    if isinstance(regions, list):
        cfg["model_regions"] = list(regions)
    params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
    imgsz = (meta.get("imgsz") or meta.get("image_size")
             or params.get("imgsz") or params.get("image_size"))
    if imgsz:
        cfg["imgsz"] = imgsz                 # metadata wins over config's default
    model = meta.get("model") or params.get("model")
    if model and not cfg.get("model"):
        cfg["model"] = model
    return cfg


def version_report_names(meta: dict, fallback) -> list[str]:
    """Class order from the version metadata when it is available."""
    classes = meta.get("classes") if isinstance(meta, dict) else None
    if isinstance(classes, list) and classes:
        return [str(c) for c in classes]
    return list(fallback)


# ---- multi-attribute model paths / config -------------------------------------
def load_multi_config() -> dict:
    """Read train/multi-models/config.json (no tasks.yaml entry required).

    config.json may now be a compatibility pointer to the latest version's
    metadata, so merge that metadata's useful fields when such a pointer exists.
    """
    fallback = {"mode": "detect", "imgsz": 640}
    if not MULTI_CONFIG.exists():
        return fallback
    try:
        cfg = json.loads(MULTI_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    if not isinstance(cfg, dict):
        return fallback
    inner = cfg.get("config")
    if isinstance(inner, dict):
        cfg = {**cfg, **inner}
    pointer = cfg.get("latest") or cfg.get("version")
    if isinstance(pointer, dict):
        pointer = pointer.get("latest") or pointer.get("version")
    if isinstance(pointer, str) and is_version_name(pointer):
        meta = load_version_meta(pointer)
        if meta:
            params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
            if isinstance(meta.get("classes"), list):
                cfg.setdefault("classes", meta["classes"])
            if isinstance(meta.get("regions"), list):
                cfg.setdefault("regions", meta["regions"])
            for key in ("imgsz", "image_size", "model"):
                if not cfg.get(key):
                    value = meta.get(key) or params.get(key)
                    if value:
                        cfg[key] = value
            cfg.setdefault("version", pointer)
    cfg.setdefault("mode", "detect")
    if not cfg.get("imgsz"):
        cfg["imgsz"] = cfg.get("image_size", 640)
    return cfg


def find_results_csv(run: Path, task: str) -> Path | None:
    """Locate results.csv for a task (the multi trainer may nest it)."""
    candidates = [run / "results.csv", run / "train" / "results.csv",
                  run / task / "results.csv", ROOT / "runs" / task / "results.csv"]
    if task == "multi" or is_version_name(task):
        # Versioned trainers write train/multi-models/<version>_run/results.csv;
        # older multi checkpoints kept curves directly under multi-models/.
        candidates += [MULTI / f"{task}_run" / "results.csv",
                       MULTI / "results.csv", MULTI / task / "results.csv"]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_curves_safe(run: Path, task: str):
    """load_curves() that tolerates a missing results.csv (multi model)."""
    csv = find_results_csv(run, task)
    if csv is None:
        return None, None, None, None, None
    try:
        return load_curves(csv.parent)
    except Exception:
        return None, None, None, None, None


def training_instance_counts(rep) -> dict:
    """Individual datasets nest counts under 'train'; the multi one is flat."""
    instances = rep.get("instances") or {}
    if isinstance(instances.get("train"), dict):
        return instances["train"]
    if any(k in instances for k in ("train", "val", "test")):
        return {}
    return {k: v for k, v in instances.items() if isinstance(v, (int, float))}


def diagnose_safe(is_det, train_score, val_score, train_loss, val_loss, val_curve, has_curves):
    """Same diagnosis when curves exist; explicit note when they don't."""
    if not has_curves:
        return ("N/A - no training curves",
                ["results.csv not found: train-val gap cannot be judged"],
                ["Re-train with curve logging, or judge the held-out metrics below directly."],
                None)
    return diagnose(is_det, train_score, val_score, train_loss, val_loss, val_curve)


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


def plot_curves(df, train_loss, val_loss, val_curve, metric, path, results_jpg=None):
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
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if results_jpg is not None:
        fig.savefig(results_jpg, dpi=130, format="jpg")
    plt.close(fig)


# ---- selected-model evaluation ------------------------------------------------
def iou_boxes(a, b):
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def sidecar_index():
    """Map annotation sidecar stem -> JSON, across data/**/*.json."""
    index = {}
    if not DATA.exists():
        return index
    for path in DATA.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "objects" in data:
            index[path.stem] = data
    return index


def class_aliases(cfg):
    """Attribute value -> task class name, using tasks.yaml groups."""
    aliases = {}
    for label, values in (cfg.get("classes") or {}).items():
        for value in values:
            aliases[str(value)] = str(label)
    return aliases


def sidecar_boxes(data, attribute):
    """Return regions in a sidecar that carry *attribute*."""
    out = []
    for obj in data.get("objects", []):
        region = str(obj.get("region", ""))
        attrs = obj.get("attrs") or {}
        if region == attribute or region.startswith(f"{attribute}_") or attribute in attrs:
            out.append(obj)
    return out


def consistency_detect(model, dataset, split, cfg, index, imgsz, device, names):
    """Matched-detection agreement with the original sidecar attribute value."""
    attribute = str(cfg.get("attribute", ""))
    aliases = class_aliases(cfg)
    stats = {"compared": 0, "agreed": 0, "matched": 0, "unmatched_gt": 0,
             "wrong_class": 0, "missing_sidecar": 0}
    per_class = {name: [0, 0] for name in names}
    image_dir = dataset / "images" / split
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in IMG_EXT:
            continue
        sidecar = index.get(image_path.stem)
        if sidecar is None:
            stats["missing_sidecar"] += 1
            continue
        with Image.open(image_path) as src:
            width, height = src.size
        targets = []
        for obj in sidecar_boxes(sidecar, attribute):
            box = obj.get("box") or []
            if len(box) != 4:
                continue
            cx, cy, w, h = (float(v) for v in box)
            targets.append({
                "box": ((cx - w / 2) * width, (cy - h / 2) * height,
                        (cx + w / 2) * width, (cy + h / 2) * height),
                "true": aliases.get(str((obj.get("attrs") or {}).get(attribute, "")), None),
            })
        targets = [target for target in targets if target["true"] is not None]
        if not targets:
            continue

        dev = kwargs_dev(device)
        results = model(str(image_path), imgsz=imgsz, verbose=False, **dev)
        detections = []
        if results:
            model_names = results[0].names or {}
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                detections.append({
                    "label": str(model_names.get(cls_id, cls_id)),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                    "conf": float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 1.0,
                })
        used = [False] * len(targets)
        for detection in sorted(detections, key=lambda item: -item["conf"]):
            best_iou, best_idx = 0.0, -1
            for idx, target in enumerate(targets):
                if used[idx]:
                    continue
                score = iou_boxes(detection["xyxy"], target["box"])
                if score > best_iou:
                    best_iou, best_idx = score, idx
            if best_idx < 0 or best_iou < 0.5:
                continue
            used[best_idx] = True
            stats["matched"] += 1
            expected = targets[best_idx]["true"]
            per_class[expected][1] += 1
            if detection["label"] == expected:
                stats["compared"] += 1
                stats["agreed"] += 1
                per_class[expected][0] += 1
            else:
                stats["wrong_class"] += 1
        stats["unmatched_gt"] += sum(1 for was_used in used if not was_used)
    return {
        "agreement": round(stats["agreed"] / stats["compared"], 4) if stats["compared"] else 0.0,
        "matched_boxes": stats["matched"], "label_checked": stats["compared"],
        "wrong_class": stats["wrong_class"],
        "unmatched_ground_truth": stats["unmatched_gt"],
        "missing_sidecar": stats["missing_sidecar"],
        "by_class": {
            name: {"checked": per_class[name][1],
                   "agreement": round(per_class[name][0] / per_class[name][1], 4)
                   if per_class[name][1] else 0.0}
            for name in names if per_class[name][1]
        },
    }


# ---- consistency for the multi-attribute model --------------------------------
SCHEMA = ROOT.parent / "annotate" / "labeling_config" / "schema.yaml"


def region_prefixes() -> list[str]:
    """Bare and object-qualified region prefixes from schema.yaml, longest first."""
    try:
        schema = yaml.safe_load(SCHEMA.read_text(encoding="utf-8")) or {}
    except Exception:
        schema = {}
    prefixes = set()
    for obj, regions in (schema.get("regions") or {}).items():
        for region in regions or []:
            region = str(region)
            prefixes.add(region)
            prefixes.add(f"{obj}_{region}")
    return sorted(prefixes, key=len, reverse=True)


def expected_multi_labels(obj, value, name_set, prefixes) -> set:
    """Candidate '<region>_<value>' classes for one sidecar region/attribute."""
    region = str(obj.get("region") or "").strip()
    cls = str(obj.get("cls") or "").strip()
    bases = []
    if region:
        bases += [region, f"{cls}_{region}" if cls else ""]
    for attr in (obj.get("attrs") or {}).keys():
        attr = str(attr).strip()
        if attr:
            bases += [attr, f"{cls}_{attr}" if cls else ""]
    candidates = {f"{base}_{value}".strip("_") for base in bases if base}
    return {c for c in candidates
            if c in name_set and any(c.startswith(p + "_") for p in prefixes)}


def consistency_multi(model, dataset, split, index, imgsz, device, names):
    """Match detections to sidecar boxes and compare region-prefixed classes."""
    prefixes = region_prefixes()
    name_set = {str(n) for n in names}
    stats = {"compared": 0, "agreed": 0, "matched": 0, "unmatched_gt": 0,
             "missing_sidecar": 0, "no_class_match": 0}
    per_class = {name: [0, 0] for name in names}
    image_dir = dataset / "images" / split
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in IMG_EXT:
            continue
        sidecar = index.get(image_path.stem)
        if sidecar is None:
            stats["missing_sidecar"] += 1
            continue
        try:
            with Image.open(image_path) as src:
                width, height = src.size
        except OSError:
            continue
        targets = []
        for obj in sidecar.get("objects", []):
            box = obj.get("box") or []
            if len(box) != 4:
                continue
            expected = set()
            for attr_value in (obj.get("attrs") or {}).values():
                if attr_value:
                    expected.update(expected_multi_labels(obj, str(attr_value), name_set, prefixes))
            if not expected:
                stats["no_class_match"] += 1
                continue
            try:
                cx, cy, w, h = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            targets.append({
                "box": ((cx - w / 2) * width, (cy - h / 2) * height,
                        (cx + w / 2) * width, (cy + h / 2) * height),
                "true": expected,
            })
        if not targets:
            continue

        results = model(str(image_path), imgsz=imgsz, verbose=False, **kwargs_dev(device))
        detections = []
        if results:
            model_names = results[0].names or {}
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                detections.append({
                    "label": str(model_names.get(cls_id, cls_id)),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                    "conf": float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 1.0,
                })
        used = [False] * len(targets)
        for detection in sorted(detections, key=lambda item: -item["conf"]):
            best_iou, best_idx = 0.0, -1
            for idx, target in enumerate(targets):
                if used[idx]:
                    continue
                score = iou_boxes(detection["xyxy"], target["box"])
                if score > best_iou:
                    best_iou, best_idx = score, idx
            if best_idx < 0 or best_iou < 0.5:
                continue
            used[best_idx] = True
            stats["matched"] += 1
            stats["compared"] += 1
            expected = targets[best_idx]["true"]
            canonical = sorted(expected)[0]
            per_class[canonical][1] += 1
            if detection["label"] in expected:
                stats["agreed"] += 1
                per_class[canonical][0] += 1
        stats["unmatched_gt"] += sum(1 for was_used in used if not was_used)
    return {
        "note": "multi model: predicted '<region>_<value>' class compared with the "
                "value of the matching sidecar region/attribute box",
        "agreement": round(stats["agreed"] / stats["compared"], 4) if stats["compared"] else 0.0,
        "matched_boxes": stats["matched"], "label_checked": stats["compared"],
        "wrong_class": stats["compared"] - stats["agreed"],
        "unmatched_ground_truth": stats["unmatched_gt"],
        "missing_sidecar": stats["missing_sidecar"],
        "regions_without_matching_class": stats["no_class_match"],
        "by_class": {
            name: {"checked": per_class[name][1],
                   "agreement": round(per_class[name][0] / per_class[name][1], 4)
                   if per_class[name][1] else 0.0}
            for name in names if per_class[name][1]
        },
    }


def evaluate_selected_one(weights, cfg, *, split="test", device=None, index, task_cfg,
                          version_meta=None):
    """Evaluate one selected weight (individual task, 'multi' or model_vN) and report."""
    task = task_name(weights)
    is_version = is_version_name(task)
    is_multi = task == "multi" or is_version
    meta = {}
    if is_version:
        meta = version_meta if isinstance(version_meta, dict) else (load_version_meta(task) or {})
        cfg = version_cfg(meta, task)
        dataset = resolve_version_dataset(meta)
        run = ROOT / "runs" / task
        out = run / "evaluation"
    elif is_multi:
        cfg = load_multi_config()
        dataset = MULTI_DATASET
        run = MULTI
        out = ROOT / "runs" / "multi" / "evaluation"
    else:
        dataset = ROOT / "datasets" / task
        run = ROOT / "runs" / task
        out = run / "evaluation"
    is_det = str(cfg.get("mode", "detect")) == "detect"
    if not dataset.exists() or not (dataset / "dataset_report.json").exists():
        raise FileNotFoundError(
            f"No built dataset for {task}: "
            f"{'run train.py first' if is_multi else 'run build_dataset.py first'}"
        )
    rep = json.loads((dataset / "dataset_report.json").read_text(encoding="utf-8"))
    names = version_report_names(meta, rep["names"]) if is_version else rep["names"]
    imgsz = int(cfg.get("imgsz", 640))
    out.mkdir(parents=True, exist_ok=True)
    rep_images = rep.get("images") or {}
    used_split = split if rep_images.get(split, 0) else "val"

    weights = Path(weights)
    if is_version and not weights.is_file():
        weights = resolve_version_weights(task, meta)
    model = YOLO(str(weights))
    results_csv = find_results_csv(run, task)
    has_curves = results_csv is not None
    df, train_loss, val_loss, val_curve, metric = load_curves_safe(run, task)
    if results_csv is not None:
        plot_curves(df, train_loss, val_loss, val_curve, metric, out / "curves_selected.png",
                    results_jpg=results_csv.with_name("results.jpg"))
    result = {
        "task": task, "mode": cfg.get("mode", "detect"), "weights": str(weights),
        "split_used": used_split, "split_requested": split,
        "images": rep_images, "classes": names,
        "generated_at": datetime.now(timezone.utc).isoformat(), "formulas": FORMULAS,
    }
    if is_version:
        result.update({
            "version": task, "dataset": str(dataset),
            "model_classes": list(meta.get("classes") or []),
            "model_regions": list(meta.get("regions") or []),
        })
        if isinstance(meta.get("evaluation"), dict):
            result["prior_evaluation"] = meta["evaluation"]
    if not has_curves:
        result["curves_note"] = "results.csv not found: train-val gap/verdict unavailable"

    if is_det:
        s_train, _ = score_detect(model, dataset / "data.yaml", "train", imgsz, device, out)
        s_val, _ = score_detect(model, dataset / "data.yaml", "val", imgsz, device, out)
        s_hold, val_result = score_detect(model, dataset / "data.yaml", used_split, imgsz, device, out)
        counts = count_labels(dataset, used_split, names)
        per_class = []
        for i, cls_index in enumerate(val_result.box.ap_class_index):
            p, r, ap50, ap = val_result.box.class_result(i)
            class_name = names[int(cls_index)]
            f1 = float(2 * p * r / (p + r + 1e-9))
            per_class.append({"class": class_name, "n": counts.get(class_name, 0),
                              "precision": float(p), "recall": float(r), "f1": f1,
                              "map50": float(ap50), "map50_95": float(ap)})
        key = "map50_95"
        result["confusion_matrix_png"] = str(out / f"val_{used_split}" / "confusion_matrix_normalized.png")
        if is_multi:
            try:
                result["consistency"] = consistency_multi(
                    model, dataset, used_split, index, imgsz, device, names
                )
            except Exception as exc:
                result["consistency"] = {
                    "note": f"multi consistency skipped: {exc}",
                    "agreement": 0.0, "label_checked": 0,
                }
        else:
            result["consistency"] = consistency_detect(model, dataset, used_split, cfg, index, imgsz, device, names)
    else:
        s_train, _, _, _ = score_cls(model, dataset, "train", imgsz, device)
        s_val, _, _, _ = score_cls(model, dataset, "val", imgsz, device)
        s_hold, per_class, cm, cm_names = score_cls(model, dataset, used_split, imgsz, device)
        key = "accuracy"
        lo, hi = wilson(int(round(s_hold["accuracy"] * s_hold["n"])), s_hold["n"])
        s_hold["ci95"] = [lo, hi]
        result["confusion_matrix"] = cm.tolist() if cm is not None else None
        result["confusion_labels"] = cm_names
        result["consistency"] = {
            "note": "classification mode: held-out accuracy is the label-consistency measure",
            "agreement": s_hold["accuracy"], "label_checked": s_hold["n"],
        }

    verdict, reasons, advice, gap = diagnose_safe(
        is_det, s_train[key], s_val[key], train_loss, val_loss, val_curve, has_curves
    )
    result.update({"train": s_train, "val": s_val, "heldout": s_hold,
                   "per_class": per_class, "gap_train_minus_val": gap,
                   "verdict": verdict, "reasons": reasons, "advice": advice})
    warnings = []
    if rep_images.get(used_split, 0) < 30:
        warnings.append(f"only {rep_images.get(used_split, 0)} {used_split} images — treat numbers as rough")
    for class_name, count in training_instance_counts(rep).items():
        if count < 50:
            warnings.append(f"class '{class_name}' has {count} training instances (<50)")
    result["data_warnings"] = warnings
    (out / "report_selected.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def format_single_markdown(report):
    """Markdown version of the original single-task report (with formulas)."""
    gap = report.get("gap_train_minus_val")
    gap_text = f"{gap:.4f}" if gap is not None else "n/a (no results.csv)"
    metric, value = headline_score(report)
    lines = ["# Evaluation report", "", f"Model: **{report['task']}**",
             f"Generated: {report.get('generated_at', '')}", "",
             FORMULA_MD,
             f"**{metric} {value:.4f} (held-out {report['split_used']}) · "
             f"{report['verdict']}** · gap {gap_text}", "",
             "| Split | Score |",
             "|---|---|",
             f"| train | {format_score(report['train'])} |",
             f"| val | {format_score(report['val'])} |",
             f"| held-out ({report['split_used']}) | {format_score(report['heldout'])} |", ""]
    if "consistency" in report:
        consistency = report["consistency"]
        lines += [f"Consistency: {consistency.get('agreement', 0.0):.4f} "
                  f"on {consistency.get('label_checked', 0)} matched boxes", ""]
        if consistency.get("note"):
            lines.append(f"({consistency['note']})")
    if report.get("reasons"):
        lines += ["Reasons:", ""]
        lines += [f"- {reason}" for reason in report["reasons"]]
    if report.get("advice"):
        lines += ["", "Suggestions:", ""]
        lines += [f"- {advice}" for advice in report["advice"]]
    lines += ["", "## Per-class results", "",
              "| Class | n | Precision | Recall | F1 | mAP50 | mAP50-95 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["per_class"]:
        lines.append(
            f"| {row['class']} | {row['n']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row.get('map50', 0):.3f} | {row.get('map50_95', 0):.3f} |"
        )
    for warning in report["data_warnings"]:
        lines.append(f"- Warning: {warning}")
    return "\n".join(lines) + "\n"


def format_selected_markdown(results):
    """Clean combined report: summary table first, then one section per model."""
    lines = ["# Selected-model evaluation report", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             FORMULA_MD,
             "## Summary (held-out split)", "",
             "| Model | Split | mAP50 / accuracy | F1 | Consistency | Verdict |",
             "|---|---|---|---:|---:|---|"]
    for row in summary_rows(results):
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "> F1 and consistency are means over matched classes / boxes; "
              "see each model's section for the detail.", ""]

    for r in results:
        gap = r.get("gap_train_minus_val")
        gap_text = f"{gap:.4f}" if gap is not None else "n/a (no results.csv)"
        consistency = r.get("consistency") or {}
        lines += [f"## {r['task']}", "",
                  f"- Weights: `{r['weights']}` · mode: {r.get('mode', 'detect')} · "
                  f"split: {r['split_used']}",
                  f"- Train: {format_score(r['train'])}",
                  f"- Val: {format_score(r['val'])}",
                  f"- Held-out ({r['split_used']}): {format_score(r['heldout'])}",
                  f"- Consistency: {consistency.get('agreement', 0.0):.4f} on "
                  f"{consistency.get('label_checked', 0)} matched boxes",
                  f"- Gap train−val: {gap_text} · **{r['verdict']}**", ""]
        if consistency.get("note"):
            lines += [f"_{consistency['note']}_", ""]
        lines += ["### Per-class results", ""]
        detect = str(r.get("mode", "detect")) == "detect"
        if detect:
            lines += ["| Class | n | Precision | Recall | F1 | mAP50 | mAP50-95 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
        else:
            lines += ["| Class | n | Precision | Recall | F1 |",
                      "|---|---:|---:|---:|---:|"]
        for row in r["per_class"]:
            cells = [str(row["class"]), str(row["n"]), f"{row['precision']:.3f}",
                     f"{row['recall']:.3f}", f"{row['f1']:.3f}"]
            if detect:
                cells += [f"{row.get('map50', 0):.3f}", f"{row.get('map50_95', 0):.3f}"]
            lines.append("| " + " | ".join(cells) + " |")
        if r.get("reasons"):
            lines += ["", "Reasons:", ""] + [f"- {reason}" for reason in r["reasons"]]
        if r.get("advice"):
            lines += ["", "Suggestions:", ""] + [f"- {advice}" for advice in r["advice"]]
        if r.get("data_warnings"):
            lines += ["", "Warnings:", ""] + [f"- {warning}" for warning in r["data_warnings"]]
        lines.append("")
    return "\n".join(lines) + "\n"


def run_selected(argv):
    """CLI for multi-model evaluation (selection popup / --all / --models)."""
    ap = argparse.ArgumentParser(description="Evaluate selected Raja-Fashion models")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated task/version names (e.g. model_v1,model_v0)")
    ap.add_argument("--all", action="store_true", help="Evaluate every trained model")
    ap.add_argument("--select", action="store_true", help="Use the popup picker (default when no flags)")
    ap.add_argument("--cli", action="store_true", help="List available models and exit")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="held-out split: test by default (falls back to val if empty)")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args(argv)

    found = find_all_weights()
    if args.cli:
        for weight in found:
            print(f"{model_label(weight):<28s} {weight}")
        return
    if not found:
        sys.exit("No weights found under train/runs/*/weights/best.pt, "
                 "train/multi-models/multi.pt or train/multi-models/model_v*.pt")

    if args.models:
        requested = [part.strip() for part in args.models.split(",") if part.strip()]
        selected = [w for w in found if task_name(w) in requested]
        for name in requested:
            if not any(task_name(w) == name for w in found):
                print(f"Skip {name}: no weights found")
    elif args.all:
        selected = found
    else:
        try:
            selected = pick_models(found, title="Select models to evaluate")
        except Exception as exc:
            print(f"Popup unavailable ({exc}); evaluating all models")
            selected = found
    if not selected:
        sys.exit("No models selected")

    # Versioned models use their meta.json, so a missing tasks.yaml is not fatal.
    try:
        task_cfg = yaml.safe_load((ROOT / "tasks.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        task_cfg = {}
    index = sidecar_index()
    results = []
    print(f"Held-out split: {args.split} · {len(selected)} model(s) selected")
    for weights in selected:
        task = task_name(weights)
        version_meta = None
        if is_version_name(task):
            version_meta = load_version_meta(task)
            if version_meta is None:
                print(f"Skip {task}: missing {version_meta_path(task).name}")
                continue
            cfg = version_cfg(version_meta, task)
        else:
            cfg = task_cfg.get(task)
            if cfg is None and task != "multi":
                print(f"Skip {task}: no entry in tasks.yaml")
                continue
        print(f"Evaluating {task} …")
        try:
            results.append(evaluate_selected_one(weights, cfg, split=args.split,
                                                 device=args.device, index=index,
                                                 task_cfg=task_cfg,
                                                 version_meta=version_meta))
        except Exception as exc:
            print(f"FAILED {task}: {exc}")
    if not results:
        sys.exit("No models could be evaluated")

    out = ROOT / "runs" / "evaluation_selected"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(format_selected_markdown(results), encoding="utf-8")
    print_selected_summary(results)
    print(f"\nWrote {out / 'report.md'} and {out / 'report.json'}\n")


# ---- single task (original CLI) ----------------------------------------------
def run_single_task(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--weights")
    ap.add_argument("--split", default="test", choices=["val", "test"], help="held-out split for final statistics")
    ap.add_argument("--device")
    args = ap.parse_args(argv)

    is_version = is_version_name(args.task)
    is_multi = args.task == "multi" or is_version
    version_meta = None
    if is_version:
        # Versioned multi checkpoints carry their own metadata: no tasks.yaml entry.
        version_meta = load_version_meta(args.task)
        if version_meta is None:
            sys.exit(f"Missing {version_meta_path(args.task).name}; "
                     "train the version or pass --weights explicitly.")
        cfg = version_cfg(version_meta, args.task)
        run = ROOT / "runs" / args.task
        ds = resolve_version_dataset(version_meta)
        out = run / "evaluation"
    elif is_multi:
        # The multi model has no tasks.yaml entry: config + dataset live together.
        cfg = load_multi_config()
        run = MULTI
        ds = MULTI_DATASET
        out = ROOT / "runs" / "multi" / "evaluation"
    else:
        cfg = yaml.safe_load((ROOT / "tasks.yaml").read_text("utf-8"))[args.task]
        run = ROOT / "runs" / args.task
        ds = ROOT / "datasets" / args.task
        out = run / "evaluation"
    is_det = str(cfg.get("mode", "detect")) == "detect"
    if is_version:
        default_weights = resolve_version_weights(args.task, version_meta)
    else:
        default_weights = MULTI / "multi.pt" if is_multi else run / "weights" / "best.pt"
    weights = Path(args.weights) if args.weights else default_weights
    if not weights.exists() or not (ds / "dataset_report.json").exists():
        sys.exit("Need a trained model and a built dataset. "
                 + ("Run train.py first." if is_multi
                    else "Run build_dataset.py then train.py first."))
    rep = json.loads((ds / "dataset_report.json").read_text(encoding="utf-8"))
    names = version_report_names(version_meta, rep["names"]) if is_version else rep["names"]
    imgsz = int(cfg.get("imgsz", 640))
    out.mkdir(parents=True, exist_ok=True)
    rep_images = rep.get("images") or {}

    split = args.split
    if rep_images.get(split, 0) == 0:
        print(f"(no {split} images, falling back to val)")
        split = "val"

    model = YOLO(str(weights))
    report = {"task": args.task, "weights": str(weights), "split_used": split,
              "images": rep_images, "classes": names, "formulas": FORMULAS,
              "generated_at": datetime.now(timezone.utc).isoformat()}
    if is_version:
        report.update({
            "version": args.task, "dataset": str(ds),
            "model_classes": list(version_meta.get("classes") or []),
            "model_regions": list(version_meta.get("regions") or []),
        })
        if isinstance(version_meta.get("evaluation"), dict):
            report["prior_evaluation"] = version_meta["evaluation"]

    # 1) training curves (optional for the multi model)
    results_csv = find_results_csv(run, args.task)
    has_curves = results_csv is not None
    df, train_loss, val_loss, val_curve, metric = load_curves_safe(run, args.task)
    if results_csv is not None:
        plot_curves(df, train_loss, val_loss, val_curve, metric, out / "curves.png",
                    results_jpg=results_csv.with_name("results.jpg"))
    else:
        report["curves_note"] = "results.csv not found: train-val gap/verdict unavailable"

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

    verdict, reasons, advice, gap = diagnose_safe(
        is_det, t_score, v_score, train_loss, val_loss, val_curve, has_curves
    )
    report.update({"train": s_train, "val": s_val, "heldout": s_hold, "per_class": per_class,
                   "gap_train_minus_val": gap, "verdict": verdict, "reasons": reasons, "advice": advice})
    if is_multi and is_det:
        try:
            report["consistency"] = consistency_multi(
                model, ds, split, sidecar_index(), imgsz, args.device, names
            )
        except Exception as exc:
            report["consistency"] = {"note": f"multi consistency skipped: {exc}",
                                     "agreement": 0.0, "label_checked": 0}

    # 3) data warnings
    warn = []
    if rep_images.get(split, 0) < 30:
        warn.append(f"only {rep_images.get(split, 0)} {split} images: treat the numbers below as rough")
    tr_counts = training_instance_counts(rep)
    for n_, c in tr_counts.items():
        if c < 50:
            warn.append(f"class '{n_}' has {c} training instances (<50)")
    report["data_warnings"] = warn
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(format_single_markdown(report), encoding="utf-8")

    # ---- print (formulas first) ----
    print(FORMULA_MD)
    line = "=" * 60
    print(f"\n{line}\n  {args.task}  |  {'detection (mAP50-95)' if is_det else 'classification (accuracy)'}\n{line}")
    gap_text = f"{gap:+.3f}" if gap is not None else "n/a (no results.csv)"
    print(f"  Train   : {t_score:.3f}\n  Val     : {v_score:.3f}\n  Gap     : {gap_text}")
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
    if "consistency" in report:
        c = report["consistency"]
        print(f"\n   Consistency: {c.get('agreement', 0.0):.4f} "
              f"on {c.get('label_checked', 0)} matched boxes")
        if c.get("note"):
            print(f"   ({c['note']})")
    for w in warn:
        print(f"\n  ! {w}")
    saved = f"{out / 'report.json'}, {out / 'report.md'}"
    if has_curves:
        saved += f" and {out / 'curves.png'}"
    print(f"\n  Saved: {saved}\n")


def main(argv=None):
    """Dispatch: a positional task (individual, multi or model_vN) runs the
    single-task report; otherwise use the selector (--all / --models / popup / --cli)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and not argv[0].startswith("-"):
        run_single_task(argv)
    else:
        run_selected(argv)


if __name__ == "__main__":
    main()
