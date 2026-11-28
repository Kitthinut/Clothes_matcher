#!/usr/bin/env python3
"""Build the ONE combined multi-attribute YOLO dataset (no training).

    python build_dataset.py                       # rebuild, then show an overview popup
    python build_dataset.py --cli                 # headless: no popup
    python build_dataset.py --data ../data --out multi-models --val 15 --test 15

Re-run it whenever labels change. The dataset is rebuilt from scratch under
``<out>/dataset/`` with ``images/{train,val,test}``, ``labels/{train,val,test}``,
``data.yaml`` and ``dataset_report.json``. The default split is 70% train /
15% validation / 15% test, assigned by a hash of the JSON stem so an image never
jumps between splits when the dataset grows. Override with --val / --test;
train gets the remainder (100 - val - test). The effective percentages are
recorded in ``dataset_report.json`` under "split_percent".

``train.py`` imports ``build_multi_dataset()`` from this module: this is the
single source of truth for the dataset. The CLI adds a Tk "Dataset overview"
popup that reports balance and, when a versioned (or legacy multi.pt) model is
present, an OVERFIT / UNDERFIT / GOOD FIT verdict from evaluate.py. Neither OK
nor Cancel deletes the generated dataset — Cancel only closes the window.

Expected input: any folder tree under --data containing <name>.json +
<name>.jpg/png/webp. JSON format: annotate/labeling_config/schema.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT.parent / "annotate" / "labeling_config" / "schema.yaml"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
SPLITS = ("train", "val", "test")

# Single source of truth for the default dataset split policy (70/15/15).
TRAIN_PCT = 70
VAL_PCT = 15
TEST_PCT = 15

BASE_MODEL = "yolo11n.pt"
VERSION_RE = re.compile(r"^model_v(\d+)\.pt$")


def validate_split(val_pct: int, test_pct: int) -> int:
    """Validate a split request and return the implied training percentage.

    Train always receives 100 - val - test, so val + test must stay below 100.
    Kept here so this builder, train.py and evaluate.py agree on what is valid.
    """
    if not isinstance(val_pct, int) or not isinstance(test_pct, int):
        raise ValueError("split percentages must be integers")
    if val_pct < 0 or test_pct < 0:
        raise ValueError(
            f"split percentages must be non-negative (got val={val_pct}, test={test_pct})"
        )
    if val_pct + test_pct >= 100:
        raise ValueError(
            f"--val + --test must be < 100 (got {val_pct} + {test_pct} = "
            f"{val_pct + test_pct}); the training split would be empty"
        )
    return 100 - val_pct - test_pct


def split_of(key: str, val_pct: int = VAL_PCT, test_pct: int = TEST_PCT) -> str:
    """Stable hash-based split assignment for a JSON stem."""
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def find_image(jp: Path):
    """Find the image sidecar for a JSON path, or None."""
    for ext in IMG_EXT:
        for cand in (jp.with_suffix(ext), jp.with_suffix(ext.upper())):
            if cand.exists():
                return cand
    return None


def schema_problems(o, schema):
    """List schema violations for one annotation object."""
    p = []
    if o.get("cls") not in schema["classes"]:
        p.append(f"unknown class '{o.get('cls')}'")
        return p
    allowed = schema["attributes"].get(o["cls"], {})
    for k, v in (o.get("attrs") or {}).items():
        if v is None:
            continue
        if k not in allowed:
            p.append(f"{o['cls']} has no attribute '{k}'")
        elif v not in allowed[k]:
            p.append(f"{o['cls']}.{k}='{v}' not in {allowed[k]}")
    b = o.get("box")
    if not (isinstance(b, list) and len(b) == 4
            and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in b)
            and b[2] > 0 and b[3] > 0):
        p.append(f"bad box {b}")
    return p


def load_schema() -> dict:
    """Read the annotation schema."""
    if not SCHEMA_PATH.is_file():
        sys.exit(f"Schema not found: {SCHEMA_PATH}")
    return yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))


def attribute_for_region(region: str) -> str:
    """sleeve_left / sleeve_right boxes store their value under `sleeve`."""
    for side in ("_left", "_right"):
        if region.endswith(side):
            return region[:-len(side)]
    return region


def class_names_for_object(obj, schema: dict) -> tuple[list[str], str | None]:
    """Return (class names, valid region) for one annotation object."""
    if not isinstance(obj, dict):
        return [], None
    cls, region = obj.get("cls"), obj.get("region")
    if cls not in schema["classes"]:
        return [], None
    if region not in schema["regions"].get(cls, []):
        return [], None
    if schema_problems(obj, schema):
        return [], None

    allowed = schema["attributes"].get(cls, {})
    attr = attribute_for_region(region)
    if attr not in allowed:
        return [], None
    value = (obj.get("attrs") or {}).get(attr)
    if value is None or value not in allowed[attr]:
        return [], None

    prefix = "" if cls == "top" else f"{cls}_"
    return [f"{prefix}{region}_{value}"], region


def expected_class_names(schema: dict) -> list[str]:
    """Every type the schema permits, used to spot missing / unlabeled types."""
    expected: set[str] = set()
    for cls in schema.get("classes", []):
        prefix = "" if cls == "top" else f"{cls}_"
        for region in schema.get("regions", {}).get(cls, []):
            attr = attribute_for_region(region)
            for value in schema.get("attributes", {}).get(cls, {}).get(attr, []):
                expected.add(f"{prefix}{region}_{value}")
    return sorted(expected)


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_config_pointer(out: Path, names: list[str], regions: list[str]) -> None:
    """Leave a config.json for evaluate.py when there is no trained version yet.

    train.py replaces this with full version metadata after training, so an
    existing config.json is never overwritten here.
    """
    cfg_path = out / "config.json"
    if cfg_path.exists():
        try:
            if isinstance(json.loads(cfg_path.read_text(encoding="utf-8")), dict):
                return
        except (OSError, ValueError):
            pass  # malformed: replace with a valid pointer below
    cfg = {
        "mode": "detect",
        "model": BASE_MODEL,
        "imgsz": 640,
        "classes": names,
        "regions": regions,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def build_multi_dataset(data_dir: Path, out: Path, schema: dict,
                        val_pct: int, test_pct: int):
    """Build the combined detection dataset. Returns (names, regions, report).

    ``out`` is the multi-models folder; the dataset itself lives in
    ``out / "dataset"``. The report includes split percentages, image counts,
    per-class counts on every split, per-garment totals and missing schema types
    so the overview popup can show imbalance.
    """
    train_pct = validate_split(val_pct, test_pct)
    out.mkdir(parents=True, exist_ok=True)
    ds = out / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    for split in SPLITS:
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect first so the class list is complete before label indices are assigned.
    collected: dict[str, dict] = {}
    names_set: set[str] = set()
    regions_set: set[str] = set()
    skipped = Counter()
    warns = []

    for jp in sorted(data_dir.rglob("*.json")):
        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:
            warns.append(f"{jp.name}: unreadable JSON ({exc})")
            continue
        if not isinstance(meta, dict) or not isinstance(meta.get("objects"), list):
            continue

        img = find_image(jp)
        if img is None:
            skipped["image file missing"] += 1
            continue

        boxes = []
        for obj in meta["objects"]:
            names, region = class_names_for_object(obj, schema)
            if not names:
                continue
            regions_set.add(region)
            for name in names:
                boxes.append((name, obj))
        if not boxes:
            skipped["no labeled region boxes"] += 1
            continue

        split = split_of(jp.stem, val_pct, test_pct)  # same hash split everywhere
        collected[jp.stem] = {"split": split, "image": img, "boxes": boxes}
        names_set.update(name for name, _ in boxes)

    names = sorted(names_set)
    index = {name: i for i, name in enumerate(names)}

    n_images = Counter()
    n_inst = defaultdict(Counter)
    garment_inst = Counter()
    garment_images = defaultdict(Counter)

    for stem, item in collected.items():
        split = item["split"]
        n_images[split] += 1
        dest = ds / "images" / split / (stem + item["image"].suffix.lower())
        shutil.copy2(item["image"], dest)

        lines = []
        seen_garments: set[str] = set()
        for name, obj in item["boxes"]:
            cx, cy, w, h = obj["box"]
            lines.append(f"{index[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_inst[split][name] += 1
            garment = str(obj.get("cls") or "unknown")
            garment_inst[garment] += 1
            seen_garments.add(garment)
        for garment in seen_garments:
            garment_images[garment][split] += 1
        (ds / "labels" / split / (stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    ds_yaml = ds / "data.yaml"
    ds_yaml.write_text(
        yaml.safe_dump({
            "path": str(ds.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": dict(enumerate(names)),
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = {
        "names": names,
        "regions": sorted(regions_set),
        "split_percent": {"train": train_pct, "val": val_pct, "test": test_pct},
        "images": {split: n_images[split] for split in SPLITS},
        # Flat train counts kept for backwards compatibility with evaluate.py.
        "instances": {name: n_inst["train"][name] for name in names},
        "instances_by_split": {
            split: {name: n_inst[split][name] for name in names} for split in SPLITS
        },
        "garments": {
            garment: {
                "instances": garment_inst[garment],
                "images": {split: garment_images[garment][split] for split in SPLITS},
            }
            for garment in sorted(garment_inst)
        },
        "missing_types": sorted(set(expected_class_names(schema)) - names_set),
        "skipped": dict(skipped),
    }
    (ds / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_config_pointer(out, names, report["regions"])

    print(f"Combined dataset -> {ds}")
    print(f"  classes: {len(names)}   regions: {len(report['regions'])}   "
          "images: " + ", ".join(f"{s}={n_images[s]}" for s in SPLITS))
    if skipped:
        print("  skipped:", dict(skipped))
    for warning in warns[:5]:
        print("  warning:", warning)
    return names, report["regions"], report


# ---------------------------------------------------------------------------
# CLI overview popup (data balance + optional latest-model verdict)
# ---------------------------------------------------------------------------

def find_latest_weights(out: Path) -> Path | None:
    """Newest model_vN.pt, else the legacy multi.pt, else None."""
    versions = []
    for path in out.glob("model_v*.pt"):
        match = VERSION_RE.match(path.name)
        if match:
            versions.append((int(match.group(1)), path))
    if versions:
        return max(versions, key=lambda item: item[0])[1]
    legacy = out / "multi.pt"
    return legacy if legacy.is_file() else None


def fit_label(verdict) -> str:
    """Map evaluate.py's verdict wording to OVERFIT / UNDERFIT / GOOD FIT."""
    raw = str(verdict or "")
    upper = raw.upper()
    if "OVERFIT" in upper:
        return "OVERFIT"
    if "UNDERFIT" in upper:
        return "UNDERFIT"
    if "N/A" in upper or not raw:
        return "N/A"
    if "GOOD" in upper:
        return "GOOD FIT"
    return raw


def evaluation_summary(out: Path, data_dir: Path, weights: Path,
                       names: list[str], regions: list[str]) -> dict:
    """Evaluate one checkpoint with evaluate.py (via train.py's bridge)."""
    meta = read_json(out / f"{weights.stem}.meta.json") \
        if VERSION_RE.match(weights.name) else read_json(out / "config.json")
    meta = dict(meta)
    meta.setdefault("classes", names)
    meta.setdefault("regions", regions)
    try:
        import train  # lazy: imports build_dataset, but not Ultralytics
        ev = train.evaluate_weights(weights, out, data_dir=data_dir, meta=meta)
    except Exception as exc:
        return {
            "fit": "N/A", "verdict": "n/a",
            "note": f"Evaluation unavailable: {exc}"[:300],
        }
    ev["fit"] = fit_label(ev.get("verdict"))
    return ev


def overview_text(names: list[str], regions: list[str], report: dict,
                  out: Path, data_dir: Path) -> str:
    """Plain-text body of the overview popup (also handy for --cli output)."""
    images = report.get("images") or {}
    pct = report.get("split_percent") or {}
    by_split = report.get("instances_by_split") or {}
    total = sum(int(images.get(s, 0)) for s in SPLITS)
    lines = [
        "Dataset overview",
        "",
        f"Total images: {total}",
        f"  train: {images.get('train', 0):>5}  ({pct.get('train', TRAIN_PCT)}%)",
        f"  val:   {images.get('val', 0):>5}  ({pct.get('val', VAL_PCT)}%)",
        f"  test:  {images.get('test', 0):>5}  ({pct.get('test', TEST_PCT)}%)",
        f"Classes: {len(names)}    Regions: {len(regions)}",
        "",
        "Instances per class (train / val / test):",
    ]
    width = max([len("class")] + [len(n) for n in names]) + 2
    lines.append(f"  {'class':<{width}}{'train':>7}{'val':>7}{'test':>7}")
    for name in names:
        counts = [int((by_split.get(s) or {}).get(name, 0)) for s in SPLITS]
        lines.append(f"  {name:<{width}}" + "".join(f"{c:>7}" for c in counts))
    lines.append("")

    garments = report.get("garments") or {}
    if garments:
        lines.append("Per garment (boxes / images carrying that garment):")
        for garment, info in sorted(garments.items()):
            gimgs = info.get("images") or {}
            lines.append(
                f"  {garment:<10} {info.get('instances', 0):>6} boxes   "
                f"{sum(int(gimgs.get(s, 0)) for s in SPLITS):>5} images"
            )
        lines.append("")

    train_counts = [int((by_split.get("train") or {}).get(n, 0)) for n in names]
    low = [n for n, c in zip(names, train_counts) if c < 30]
    missing = list(report.get("missing_types") or [])
    lines.append(f"Missing schema types ({len(missing)}): "
                 + (", ".join(missing) if missing else "none"))
    if low:
        lines.append(f"Low training counts (<30): {', '.join(low)}")
    if train_counts and min(train_counts) > 0 and max(train_counts) / min(train_counts) > 4:
        lines.append("Imbalance: busiest class has >4x the boxes of the rarest class.")
    lines.append("")

    weights = find_latest_weights(out)
    if weights is None:
        lines.append("Model: no model yet — data balance only")
    else:
        ev = evaluation_summary(out, data_dir, weights, names, regions)
        lines.append(f"Model: {weights.name}")
        lines.append(f"Fit verdict: {ev.get('fit', 'n/a')}   [raw: {ev.get('verdict', 'n/a')}]")
        metrics = [f"{key}={ev.get(key)}" for key in
                   ("map50", "map50_95", "f1", "consistency")
                   if ev.get(key) is not None]
        if metrics:
            lines.append("  " + "  ".join(metrics))
        if ev.get("note"):
            lines.append(f"  note: {ev['note']}")
        for reason in list(ev.get("reasons") or [])[:5]:
            lines.append(f"  - {reason}")
    return "\n".join(lines)


def overview_popup(text: str) -> None:
    """Show the overview. OK and Cancel both just close: nothing is deleted."""
    try:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
    except Exception as exc:
        print(f"Popup unavailable ({exc}); overview below.\n")
        print(text)
        return

    root.title("Dataset overview")
    root.geometry("760x620")
    root.minsize(560, 420)
    root.configure(bg="#eef1f5")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    box = tk.Text(frame, wrap="none", font=("Consolas", 10), bg="white",
                  relief="flat", highlightthickness=1, highlightbackground="#d9dfe8")
    scroll = ttk.Scrollbar(frame, command=box.yview)
    box.config(yscrollcommand=scroll.set)
    box.pack(side="left", fill="both", expand=True)
    scroll.pack(side="left", fill="y")
    box.insert("1.0", text)
    box.config(state="disabled")

    footer = ttk.Frame(root, padding=(14, 4, 14, 12))
    footer.pack(fill="x")
    ttk.Button(footer, text="Cancel", command=root.destroy).pack(side="right")
    ttk.Button(footer, text="OK", command=root.destroy).pack(side="right", padx=(0, 8))

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build the combined multi-attribute detection dataset "
                    "(no training), then show a dataset overview.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", default=str(ROOT.parent / "data"),
                    help="folder with images + JSON sidecars (searched recursively)")
    ap.add_argument("--out", default=str(ROOT / "multi-models"),
                    help="output folder for the dataset and model versions")
    ap.add_argument("--val", type=int, default=VAL_PCT,
                    help=f"percent of images for validation (default split "
                         f"{TRAIN_PCT}/{VAL_PCT}/{TEST_PCT})")
    ap.add_argument("--test", type=int, default=TEST_PCT,
                    help="percent of images for test")
    ap.add_argument("--seed", type=int, default=0,
                    help="CLI parity with train.py; the hash-based split ignores it")
    ap.add_argument("--cli", "--yes", action="store_true", dest="cli",
                    help="skip the overview popup (headless)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        validate_split(args.val, args.test)
    except ValueError as exc:
        sys.exit(f"Invalid split: {exc}")

    data_dir = Path(args.data).resolve()
    out = Path(args.out).resolve()
    if not data_dir.is_dir():
        sys.exit(f"Data folder not found: {data_dir} (override with --data).")
    schema = load_schema()

    names, regions, report = build_multi_dataset(data_dir, out, schema, args.val, args.test)
    if not names:
        sys.exit("No usable labeled region boxes found. Check --data and schema.yaml.")

    print("=" * 60)
    print("Dataset-only build complete (no YOLO training performed)")
    print(f"  dataset : {out / 'dataset'}")
    print(f"  data    : {out / 'dataset' / 'data.yaml'}")
    print(f"  report  : {out / 'dataset' / 'dataset_report.json'}")
    print("=" * 60)
    if not args.cli:
        overview_popup(overview_text(names, regions, report, out, data_dir))


if __name__ == "__main__":
    main()
