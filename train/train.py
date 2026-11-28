#!/usr/bin/env python3
"""Train ONE versioned multi-attribute YOLO detector.

    python train.py                       # review popup, then train
    python train.py --cli                 # headless: no popups
    python train.py --cli --epochs 50 --batch 8
    python train.py --fresh               # ignore old versions, start from yolo11n.pt
    python train.py --version model_v0    # continue from a specific version

Every successful run writes a new version under ``train/multi-models/``:

    model_vN.pt                versioned weights (N starts at 0)
    model_vN.meta.json         version + weights + classes + regions + fingerprint
                               + base_model + params + evaluation
    model_vN_run/              Ultralytics run directory (results.csv, best.pt, ...)
    latest.json                {"latest": "model_vN"}
    config.json                compatibility pointer to the latest version's metadata
                               (read by evaluate.py)
    multi.pt, best.pt          aliases of the latest version's weights

The dataset is always rebuilt from ``data/**/*.json`` sidecars with the shared
70/15/15 hash split and lives at ``train/multi-models/dataset/``. Training always
uses the current FULL dataset; there is no separate "individual" training path.
The only real training entry point is this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from build_dataset import (
    SCHEMA_PATH, TEST_PCT, TRAIN_PCT, VAL_PCT,
    build_multi_dataset, validate_split,
)

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
BASE_MODEL = "yolo11n.pt"
FRESH = "fresh base"
VERSION_RE = re.compile(r"^model_v(\d+)\.pt$")


# Dataset building lives exclusively in build_dataset.py (imported above).

def load_schema() -> dict:
    try:
        import yaml
    except ImportError as exc:
        sys.exit(f"PyYAML is required to read schema.yaml: {exc}\n"
                 "Install dependencies: pip install -r requirements.txt")
    if not SCHEMA_PATH.is_file():
        sys.exit(f"Schema not found: {SCHEMA_PATH}")
    return yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fingerprints / versions / metadata
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def dataset_entries(ds: Path) -> list[list[str]]:
    """Sorted [[relative image path, sha256 of YOLO .txt label], ...]."""
    entries: list[list[str]] = []
    for split in SPLITS:
        img_dir = ds / "images" / split
        if not img_dir.is_dir():
            continue
        label_dir = ds / "labels" / split
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            label = label_dir / (img.stem + ".txt")
            digest = (hashlib.sha256(label.read_bytes()).hexdigest()
                      if label.is_file() else "")
            entries.append([f"images/{split}/{img.name}", digest])
    return sorted(entries)


def fingerprint_of(entries) -> dict:
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
    return {
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "files": len(entries),
    }


def fingerprint_text(fp) -> str:
    if not isinstance(fp, dict) or not fp.get("digest"):
        return "none"
    return f"{fp['digest'][:12]} ({fp.get('files', '?')} files)"


def changed_images(current, previous) -> tuple[list[str], list[str]]:
    """Return (new image paths, relabeled image paths) relative to the old set."""
    old = ({str(pair[0]): str(pair[1]) for pair in previous}
           if isinstance(previous, list) else {})
    new, relabeled = [], []
    for pair in current:
        path, digest = str(pair[0]), str(pair[1])
        if path not in old:
            new.append(path)
        elif old[path] != digest:
            relabeled.append(path)
    return new, relabeled


def normalize_entries(entries) -> list[list[str]]:
    if not isinstance(entries, list):
        return []
    return [[str(pair[0]), str(pair[1])]
            for pair in entries
            if isinstance(pair, (list, tuple)) and len(pair) >= 2]


def discover_versions(out: Path) -> list[dict]:
    """All model_vN.pt versions, oldest first, with their metadata."""
    found = []
    for path in out.glob("model_v*.pt"):
        m = VERSION_RE.match(path.name)
        if not m:
            continue
        version = int(m.group(1))
        meta_path = out / f"model_v{version}.meta.json"
        found.append({
            "version": version,
            "name": f"model_v{version}",
            "weights": path,
            "meta_path": meta_path,
            "meta": read_json(meta_path),
        })
    return sorted(found, key=lambda item: item["version"])


def latest_version(versions: list[dict]) -> dict | None:
    return versions[-1] if versions else None


def old_state(latest: dict | None, config: dict) -> tuple[list[list[str]], dict | None]:
    """Fingerprint entries from the latest version's meta (legacy config fallback)."""
    if latest:
        entries = normalize_entries(latest["meta"].get("fingerprint_entries"))
        if entries:
            return entries, latest["meta"].get("fingerprint")
    return normalize_entries(config.get("fingerprint_entries")), config.get("fingerprint")


def base_weights_value(model_arg: str):
    """Fresh base: local path when available, else the raw name for Ultralytics."""
    candidates = [Path(model_arg).expanduser()]
    if not candidates[0].is_absolute():
        candidates += [Path.cwd() / model_arg, ROOT.parent / model_arg, ROOT / model_arg]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return model_arg


def resolve_weights(value: str) -> Path | None:
    p = Path(value).expanduser()
    candidates = [p] if p.is_absolute() else [
        Path.cwd() / p, ROOT.parent / p, ROOT / p,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def compat_config(meta: dict) -> dict:
    """config.json = latest metadata + the flat keys evaluate.py looks for."""
    cfg = dict(meta)
    params = meta.get("params") or {}
    cfg.update({
        "mode": "detect",
        "model": meta.get("base_model", BASE_MODEL),
        "imgsz": params.get("imgsz", 640),
        "weights": meta.get("weights"),
    })
    return cfg


def sync_aliases(out: Path, weights: Path) -> None:
    """Keep select_models.py / evaluate.py standalone flows working."""
    for alias in (out / "multi.pt", out / "best.pt"):
        try:
            shutil.copy2(weights, alias)
        except OSError as exc:
            print(f"Warning: could not update alias {alias}: {exc}")


# ---------------------------------------------------------------------------
# Evaluation: reuse evaluate.py's functions without editing them
# ---------------------------------------------------------------------------

def _number(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compact_evaluation(full: dict) -> dict:
    """Small, readable slice of evaluate.py's report for the popups and meta."""
    held = full.get("heldout") or {}
    consistency = full.get("consistency") or {}
    p = _number(held.get("precision"))
    r = _number(held.get("recall"))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    verdict = str(full.get("verdict") or "n/a")
    simple = ("OVERFITTING" if "OVERFIT" in verdict.upper() else
              "UNDERFITTING" if "UNDERFIT" in verdict.upper() else
              "NO OVERFITTING" if verdict else "n/a")
    return {
        "map50": round(_number(held.get("map50")), 4),
        "map50_95": round(_number(held.get("map50_95")), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "consistency": round(_number(consistency.get("agreement")), 4),
        "verdict": verdict,
        "overfit_verdict": simple,
        "gap_train_minus_val": full.get("gap_train_minus_val"),
        "split": full.get("split_used"),
        "train": full.get("train"),
        "val": full.get("val"),
        "reasons": list(full.get("reasons") or [])[:5],
        "source": "evaluate.evaluate_selected_one",
    }


def evaluate_weights(weights: Path, out: Path, *, tag: str | None = None,
                     data_dir: Path | None = None, meta: dict | None = None,
                     device: str | None = None, split: str = "test") -> dict:
    """Call evaluate.py on one of our versioned weights; return a compact report.

    evaluate.py hard-codes its multi paths and recognises ``multi.pt`` by name.
    We redirect those module constants in-process only (no file edits) so it
    evaluates ``model_vN.pt`` against the dataset we just built.
    """
    try:
        import evaluate
    except Exception as exc:
        raise RuntimeError(f"evaluate.py could not be imported: {exc}") from exc

    patched = ("MULTI", "MULTI_DATASET", "MULTI_CONFIG", "DATA",
               "task_name", "find_results_csv", "load_multi_config")
    saved = {name: getattr(evaluate, name) for name in patched if hasattr(evaluate, name)}
    orig_find = getattr(evaluate, "find_results_csv", None)
    orig_task_name = getattr(evaluate, "task_name", None)

    params = (meta or {}).get("params") or {}
    cfg = {
        "mode": "detect",
        "imgsz": int(params.get("imgsz", 640)),
        "classes": (meta or {}).get("classes") or [],
        "regions": (meta or {}).get("regions") or [],
    }

    try:
        evaluate.MULTI = out
        evaluate.MULTI_DATASET = out / "dataset"
        evaluate.MULTI_CONFIG = out / "config.json"
        if data_dir is not None:
            evaluate.DATA = Path(data_dir)
        # Task name decides which dataset evaluate.py uses: force the multi path.
        evaluate.task_name = lambda _weights: "multi"
        evaluate.load_multi_config = lambda: cfg

        if orig_find is not None:
            def find_results(run, task, _orig=orig_find):
                if tag:
                    csv = out / f"{tag}_run" / "results.csv"
                    if csv.is_file():
                        return csv
                return _orig(run, task)
            evaluate.find_results_csv = find_results

        index = evaluate.sidecar_index()
        full = evaluate.evaluate_selected_one(
            Path(weights), cfg, split=split, device=device,
            index=index, task_cfg={},
        )
        return compact_evaluation(full)
    finally:
        for name, value in saved.items():
            setattr(evaluate, name, value)
        # Restore the two helper names even if they were absent before.
        if orig_find is not None:
            evaluate.find_results_csv = orig_find
        if orig_task_name is not None:
            evaluate.task_name = orig_task_name


def evaluation_note(exc: Exception) -> dict:
    return {
        "verdict": "n/a", "overfit_verdict": "n/a",
        "note": f"Evaluation unavailable: {exc}"[:300],
        "source": "evaluate.py",
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_yolo_class():
    """Import Ultralytics lazily so --help and dataset-only runs stay light."""
    from ultralytics import YOLO
    return YOLO


def train_model(start_weights, data_yaml: Path, out: Path, tag: str,
                names: list[str], params: dict) -> Path:
    """Train from the selected weights and return the Ultralytics run directory."""
    YOLO = get_yolo_class()
    model = YOLO(str(start_weights))

    kw = dict(
        data=str(data_yaml),
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        batch=params["batch"],
        patience=params["patience"],
        fliplr=params["fliplr"],
        rect=params["rect"],
        project=str(out),
        name=f"{tag}_run",
        exist_ok=True,
        seed=params["seed"],
    )
    if params.get("device"):
        kw["device"] = params["device"]

    print(f"\n{'=' * 56}")
    print("Training ONE versioned multi-attribute detector")
    print(f"  new version : {tag}")
    print(f"  start from  : {start_weights}")
    print(f"  data        : {data_yaml}")
    print(f"  classes     : {len(names)}")
    print(f"  imgsz       : {params['imgsz']}   rect = {params['rect']}"
          + ("   (4:3 letterbox ~%d px tall)" % int(params["imgsz"] * 4 / 3)
             if params["rect"] else ""))
    print(f"  fliplr      : {params['fliplr']}")
    print(f"  epochs      : {params['epochs']}  batch = {params['batch']}  "
          f"patience = {params['patience']}  seed = {params['seed']}")
    print("=" * 56)
    model.train(**kw)

    run_dir = out / f"{tag}_run"
    best = run_dir / "weights" / "best.pt"
    dest = out / f"{tag}.pt"
    if best.is_file():
        shutil.copy2(best, dest)
    else:
        model.save(dest)  # never fall back to a fresh pretrained model
        print(f"Warning: no best.pt in {run_dir}; saved in-memory model instead.")
    return run_dir


# ---------------------------------------------------------------------------
# Reports / popups
# ---------------------------------------------------------------------------

def build_plan_summary(names, regions, report, cur_fp, old_fp, new_paths,
                       relabeled_paths, old_name, old_eval, start_choice) -> str:
    images = report.get("images") or {}
    verdict = (old_eval or {}).get("overfit_verdict", "n/a")
    raw = (old_eval or {}).get("verdict", "")
    lines = [
        "Current dataset (rebuilt from data/**/*.json)",
        f"  classes : {len(names)}   regions: {len(regions)}",
        f"  images  : train={images.get('train', 0)}  "
        f"val={images.get('val', 0)}  test={images.get('test', 0)}",
        f"  split   : {report.get('split_percent', {})}",
        f"  fingerprint : {fingerprint_text(cur_fp)}",
        f"  previous    : {fingerprint_text(old_fp)}",
        f"  new images since previous : {len(new_paths)}",
        f"  relabeled images          : {len(relabeled_paths)}",
        "",
        f"Old model selected : {old_name}",
        f"Old model on current dataset : {verdict}"
        + (f"  [{raw}]" if raw and raw != verdict else ""),
    ]
    if old_eval and old_eval.get("map50_95") is not None:
        lines.append(
            f"  mAP50-95={old_eval.get('map50_95')}  "
            f"F1={old_eval.get('f1')}  consistency={old_eval.get('consistency')}"
        )
    if old_eval and old_eval.get("note"):
        lines.append(f"  note: {old_eval['note']}")
    lines += [
        "",
        f"Starting weights : {start_choice}",
        "Training always uses the full current dataset (no incremental-only path).",
    ]
    return "\n".join(lines)


def build_result_summary(tag: str, evaluation: dict | None, fp: dict,
                         weights: Path, meta_path: Path) -> str:
    ev = evaluation or {}
    lines = [
        f"Training finished: {tag}",
        f"  weights  : {weights}",
        f"  metadata : {meta_path}",
        f"  fingerprint : {fingerprint_text(fp)}",
        "",
        "Evaluation (evaluate.py, held-out split "
        f"{ev.get('split', 'test')})",
        f"  mAP50       : {ev.get('map50', 'n/a')}",
        f"  mAP50-95    : {ev.get('map50_95', 'n/a')}",
        f"  precision   : {ev.get('precision', 'n/a')}",
        f"  recall      : {ev.get('recall', 'n/a')}",
        f"  F1          : {ev.get('f1', 'n/a')}",
        f"  consistency : {ev.get('consistency', 'n/a')}",
        f"  verdict     : {ev.get('verdict', 'n/a')}"
        + (f"  ({ev.get('overfit_verdict')})" if ev.get("overfit_verdict") else ""),
    ]
    if ev.get("note"):
        lines.append(f"  note: {ev['note']}")
    for reason in ev.get("reasons") or []:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def plan_popup(summary: str, versions: list[str], default_version: str,
               params: dict):
    """Tk popup shown BEFORE training. Returns settings dict, None or 'cancelled'."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        root = tk.Tk()
    except Exception as exc:
        print(f"Popup unavailable ({exc}); continuing headlessly.")
        return None

    result: dict = {"status": None, "values": None}

    def cancel():
        result["status"] = "cancelled"
        root.destroy()

    def confirm():
        try:
            values = {
                "epochs": int(epochs_var.get()),
                "batch": int(batch_var.get()),
                "imgsz": int(imgsz_var.get()),
                "patience": int(patience_var.get()),
                "seed": int(seed_var.get()),
                "device": device_var.get().strip(),
                "version": version_var.get(),
            }
            if min(values["epochs"], values["batch"], values["imgsz"]) <= 0:
                raise ValueError("epochs, batch and imgsz must be positive")
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid setting", str(exc), parent=root)
            return
        result["status"] = "ok"
        result["values"] = values
        root.destroy()

    root.title("raja_fashion - Review dataset and train new version")
    root.geometry("720x650")
    root.minsize(600, 520)
    root.configure(bg="#eef1f5")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    head = ttk.Frame(root, padding=(16, 12, 16, 4))
    head.pack(fill="x")
    ttk.Label(head, text="Before training - dataset & old model report",
              font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(head, text="Review, adjust the settings below, then press Train.",
              foreground="#667085").pack(anchor="w", pady=(2, 0))

    text_frame = ttk.Frame(root, padding=(16, 4, 16, 8))
    text_frame.pack(fill="both", expand=True)
    text = tk.Text(text_frame, wrap="word", height=17, font=("Consolas", 10),
                   bg="white", relief="flat", highlightthickness=1,
                   highlightbackground="#d9dfe8")
    scroll = ttk.Scrollbar(text_frame, command=text.yview)
    text.config(yscrollcommand=scroll.set)
    text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="left", fill="y")
    text.insert("1.0", summary)
    text.config(state="disabled")

    fields = ttk.LabelFrame(root, text="Training settings", padding=(16, 8))
    fields.pack(fill="x", padx=16, pady=(0, 8))
    epochs_var = tk.StringVar(value=str(params["epochs"]))
    batch_var = tk.StringVar(value=str(params["batch"]))
    imgsz_var = tk.StringVar(value=str(params["imgsz"]))
    patience_var = tk.StringVar(value=str(params["patience"]))
    seed_var = tk.StringVar(value=str(params["seed"]))
    device_var = tk.StringVar(value=params.get("device") or "")
    version_var = tk.StringVar(value=default_version)

    row = 0
    def add(label, var, width=10, column=1):
        nonlocal row
        ttk.Label(fields, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(fields, textvariable=var, width=width).grid(row=row, column=column, sticky="w")
        row += 1

    add("epochs", epochs_var)
    add("batch", batch_var)
    add("imgsz", imgsz_var)
    add("patience", patience_var)
    add("seed", seed_var)
    add("device (blank = auto)", device_var)
    ttk.Label(fields, text="start from").grid(row=row, column=0, sticky="w", padx=(0, 6))
    ttk.Combobox(fields, textvariable=version_var, values=versions,
                 state="readonly", width=24).grid(row=row, column=1, sticky="w")

    footer = ttk.Frame(root, padding=(16, 4, 16, 14))
    footer.pack(fill="x")
    ttk.Button(footer, text="Cancel", command=cancel).pack(side="right")
    ttk.Button(footer, text="Train", command=confirm).pack(side="right", padx=(0, 8))

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.bind("<Escape>", lambda _e: cancel())
    root.bind("<Return>", lambda _e: confirm())
    try:
        root.mainloop()
    except tk.TclError:
        return None
    if result["status"] == "cancelled":
        return "cancelled"
    return result["values"]


def result_popup(text: str) -> None:
    """Result popup after training; OK closes it."""
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
    except Exception as exc:
        print(f"Popup unavailable ({exc}).")
        return
    root.title("raja_fashion - New model evaluation")
    root.geometry("660x460")
    root.minsize(520, 320)
    root.configure(bg="#eef1f5")
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="New version - evaluation result",
              font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
    box = tk.Text(frame, wrap="word", font=("Consolas", 10), bg="white",
                  relief="flat", highlightthickness=1, highlightbackground="#d9dfe8")
    scroll = ttk.Scrollbar(frame, command=box.yview)
    box.config(yscrollcommand=scroll.set)
    box.pack(side="left", fill="both", expand=True)
    scroll.pack(side="left", fill="y")
    box.insert("1.0", text)
    box.config(state="disabled")
    ttk.Button(frame, text="OK", command=root.destroy).pack(pady=(8, 0))
    root.bind("<Return>", lambda _e: root.destroy())
    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train one versioned multi-attribute YOLO detector "
                    "(single training entry point).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", default=str(ROOT.parent / "data"),
                    help="folder with images + JSON sidecars (searched recursively)")
    ap.add_argument("--out", default=str(ROOT / "multi-models"),
                    help="output folder for dataset, versions and metadata")
    ap.add_argument("--val", type=int, default=VAL_PCT,
                    help=f"percent for validation (default split "
                         f"{TRAIN_PCT}/{VAL_PCT}/{TEST_PCT})")
    ap.add_argument("--test", type=int, default=TEST_PCT,
                    help="percent for test")
    ap.add_argument("--model", default=BASE_MODEL,
                    help="fresh base weights when no old version is selected")
    ap.add_argument("--version", help="old version to continue from, e.g. model_v0")
    ap.add_argument("--fresh", "--scratch", action="store_true",
                    help="ignore old versions and start from the fresh base")
    ap.add_argument("--init", metavar="PATH", help="start from this specific .pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", help="0 for first GPU, cpu, mps ... (blank = auto)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--square", action="store_true",
                    help="force square training (default rect=True, 4:3 letterbox)")
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="held-out split used by evaluate.py")
    ap.add_argument("--cli", "--yes", action="store_true", dest="cli",
                    help="skip both popups (headless)")
    ap.add_argument("--skip-old-eval", action="store_true",
                    help="skip the old-model evaluation in the pre-training report")
    ap.add_argument("--skip-eval", action="store_true",
                    help="skip the post-training evaluation")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_split(args.val, args.test)
    except ValueError as exc:
        sys.exit(f"Invalid split: {exc}")

    data_dir = Path(args.data).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        sys.exit(f"Data folder not found: {data_dir} (override with --data).")

    base_value = base_weights_value(args.model)
    versions = discover_versions(out)
    latest = latest_version(versions)
    config_before = read_json(out / "config.json")

    # Capture the old fingerprint BEFORE build_multi_dataset() wipes dataset/.
    previous_entries, previous_fp = old_state(latest, config_before)

    names, regions, report = build_multi_dataset(
        data_dir, out, load_schema(), args.val, args.test
    )
    if not names:
        sys.exit("No usable labeled region boxes found. Check --data and schema.yaml.")

    ds = out / "dataset"
    entries = dataset_entries(ds)
    fp = fingerprint_of(entries)
    new_paths, relabeled_paths = changed_images(entries, previous_entries)

    params = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device or "",
        "patience": args.patience,
        "seed": args.seed,
        "rect": not args.square,
        "fliplr": 0.0 if any("left" in n.lower() or "right" in n.lower()
                              for n in names) else 0.5,
        "val_pct": args.val,
        "test_pct": args.test,
    }

    start_choice = args.version or (latest["name"] if latest else FRESH)
    if args.fresh:
        start_choice = FRESH

    old_eval = None
    old_meta = latest["meta"] if latest else {}
    if not args.cli and not args.skip_old_eval and start_choice != FRESH:
        old_weights = out / f"{start_choice}.pt"
        if old_weights.is_file():
            try:
                old_eval = evaluate_weights(
                    old_weights, out, tag=start_choice, data_dir=data_dir,
                    meta=old_meta, device=args.device, split=args.split,
                )
            except Exception as exc:
                old_eval = evaluation_note(exc)

    old_label = start_choice if start_choice != FRESH else (
        f"none (fresh base: {args.model})")
    summary = build_plan_summary(
        names, regions, report, fp, previous_fp, new_paths, relabeled_paths,
        old_label, old_eval, start_choice,
    )

    if not args.cli:
        choices = [FRESH] + [v["name"] for v in versions]
        answer = plan_popup(summary, choices, start_choice, params)
        if answer == "cancelled":
            sys.exit("Cancelled: no training performed.")
        if isinstance(answer, dict):
            for key in ("epochs", "batch", "imgsz", "patience", "seed", "device"):
                params[key] = answer[key]
            start_choice = answer["version"]
        else:
            print("No interactive display; using the settings below.")
    print("\n" + summary)

    if args.init:
        start_weights = resolve_weights(args.init)
        if start_weights is None:
            sys.exit(f"--init checkpoint not found: {args.init}")
        source = "init"
    elif start_choice == FRESH:
        start_weights = base_value
        source = FRESH
    else:
        start_weights = out / f"{start_choice}.pt"
        if not start_weights.is_file():
            sys.exit(f"Old version not found: {start_weights}")
        source = start_choice

    version_no = latest["version"] + 1 if latest else 0
    tag = f"model_v{version_no}"
    meta_path = out / f"{tag}.meta.json"
    weights_path = out / f"{tag}.pt"

    params["start_weights"] = str(start_weights)
    params["start_source"] = source
    params["base_model"] = args.model

    run_dir = train_model(start_weights, ds / "data.yaml", out, tag, names, params)

    meta = {
        "version": version_no,
        "weights": weights_path.name,
        "classes": names,
        "regions": regions,
        "fingerprint": fp,
        "fingerprint_entries": entries,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.model,
        "params": params,
        "evaluation": None,
    }
    write_json(meta_path, meta)
    write_json(out / "latest.json", {"latest": tag})
    write_json(out / "config.json", compat_config(meta))
    sync_aliases(out, weights_path)
    print(f"\nSaved {weights_path}  (Ultralytics run: {run_dir})")
    print(f"Metadata: {meta_path}")
    print(f"Compatibility pointers: latest.json, config.json, multi.pt")

    evaluation = None
    if not args.skip_eval:
        try:
            evaluation = evaluate_weights(
                weights_path, out, tag=tag, data_dir=data_dir, meta=meta,
                device=params.get("device") or None, split=args.split,
            )
            meta["evaluation"] = evaluation
            write_json(meta_path, meta)
            write_json(out / "config.json", compat_config(meta))
        except Exception as exc:
            evaluation = evaluation_note(exc)
            print(f"Evaluation failed ({exc}); continuing with n/a.")

    result_text = build_result_summary(tag, evaluation, fp, weights_path, meta_path)
    print("\n" + result_text)
    if not args.cli:
        result_popup(result_text)


if __name__ == "__main__":
    main()
