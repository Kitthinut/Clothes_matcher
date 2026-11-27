#!/usr/bin/env python3
"""
Train ONE YOLO detector for every labeled garment attribute region in a single pass.

    python train_multi.py                         # build dataset + train
    python train_multi.py --epochs 100 --batch 16
    python train_multi.py --model yolo11n.pt --imgsz 640 --device 0
    python train_multi.py --no-train              # only (re)build the dataset

The dataset split is 70% train / 15% validation / 15% test by default, using the same
hash-based assignment as build_dataset.py. Override with --val / --test; train receives
the remaining 100 - val - test percent (val + test must stay below 100). The effective
percentages are written to dataset/dataset_report.json under "split_percent".

Unlike train_individual.py (one small model per task in tasks.yaml), this script
builds a single combined detection dataset from every annotation sidecar found
under data/**/*.json and trains one model whose classes are `<region>_<value>`:

    neck_crew, sleeve_left_long, bottom_fit_skinny, ...

Outputs (all under train/multi-models/):
    dataset/images/{train,val,test}/     dataset/labels/{train,val,test}/
    dataset/data.yaml                    dataset/dataset_report.json
    config.json                          multi.pt   (+ best.pt copy)
    multi/weights/best.pt                Ultralytics run directory
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

from build_dataset import (
    TEST_PCT, TRAIN_PCT, VAL_PCT,
    find_image, schema_problems, split_of, validate_split,
)

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT.parent / "annotate" / "labeling_config" / "schema.yaml"
SPLITS = ("train", "val", "test")


def attribute_for_region(region: str) -> str:
    """sleeve_left / sleeve_right boxes store their value under `sleeve`."""
    for side in ("_left", "_right"):
        if region.endswith(side):
            return region[: -len(side)]
    return region


def class_names_for_object(obj, schema: dict) -> tuple[list[str], str | None]:
    """Return (class names, valid region) for one annotation object.

    Empty names mean the object must be skipped (unknown class/region/value,
    bad box or otherwise invalid according to schema.yaml).
    """
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

    # Contract examples: neck_crew, sleeve_left_long, bottom_fit_skinny.
    prefix = "" if cls == "top" else f"{cls}_"
    return [f"{prefix}{region}_{value}"], region


def build_multi_dataset(data_dir: Path, out: Path, schema: dict,
                        val_pct: int, test_pct: int):
    """Build the combined detection dataset. Returns (names, regions, report)."""
    train_pct = validate_split(val_pct, test_pct)
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
            meta = json.loads(jp.read_text("utf-8"))
        except Exception as exc:
            warns.append(f"{jp.name}: unreadable JSON ({exc})")
            continue
        if not isinstance(meta, dict) or not isinstance(meta.get("objects"), list):
            continue
        # Note: meta.get("done") is only annotation-tracker state. All objects
        # in the sidecar are used regardless of that flag.

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

        split = split_of(jp.stem, val_pct, test_pct)  # same hash split as build_dataset.py
        collected[jp.stem] = {"split": split, "image": img, "boxes": boxes}
        names_set.update(name for name, _ in boxes)

    names = sorted(names_set)
    index = {name: i for i, name in enumerate(names)}

    n_images = Counter()
    n_inst = defaultdict(Counter)
    for stem, item in collected.items():
        split = item["split"]
        n_images[split] += 1
        dest = ds / "images" / split / (stem + item["image"].suffix.lower())
        shutil.copy2(item["image"], dest)

        lines = []
        for name, obj in item["boxes"]:
            cx, cy, w, h = obj["box"]
            lines.append(f"{index[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_inst[split][name] += 1
        (ds / "labels" / split / (stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    (ds / "data.yaml").write_text(
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
        "split_percent": {"train": train_pct, "val": val_pct, "test": test_pct},
        "images": {split: n_images[split] for split in SPLITS},
        "instances": {name: n_inst["train"][name] for name in names},
    }
    (ds / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nCombined dataset -> {ds}")
    print(f"  classes: {len(names)}   images: " +
          ", ".join(f"{s}={n_images[s]}" for s in SPLITS))
    print("  train instances: " +
          ", ".join(f"{n}={n_inst['train'][n]}" for n in names))
    if skipped:
        print("  skipped:", dict(skipped))
    for warning in warns[:5]:
        print("  warning:", warning)

    if n_images["train"] == 0 or n_images["val"] == 0:
        print("  ERROR: train or val split is empty; check --data and your annotations.")
    return names, sorted(regions_set), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "data"),
                    help="folder with images + JSON sidecars (searched recursively)")
    ap.add_argument("--out", default=str(ROOT / "multi-models"),
                    help="output folder for dataset, config and weights")
    ap.add_argument("--val", type=int, default=VAL_PCT,
                    help=f"percent of images for validation (default split: "
                         f"{TRAIN_PCT}/{VAL_PCT}/{TEST_PCT} train/val/test)")
    ap.add_argument("--test", type=int, default=TEST_PCT,
                    help=f"percent of images for test (default split: "
                         f"{TRAIN_PCT}/{VAL_PCT}/{TEST_PCT} train/val/test)")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", help="0 for first GPU, cpu, mps ...")
    ap.add_argument("--patience", type=int, default=30,
                    help="early stop if val does not improve for N epochs")
    ap.add_argument("--square", action="store_true", help="force square training (rect=False)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-train", action="store_true", help="only build the dataset")
    args = ap.parse_args()

    try:
        train_pct = validate_split(args.val, args.test)
    except ValueError as exc:
        sys.exit(f"Invalid split: {exc}")

    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    data_dir = Path(args.data).resolve()
    # Ultralytics resolves a relative `project` against its runs dir; make it
    # absolute so Ultralytics weights and our final weights land in the same place.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    names, regions, report = build_multi_dataset(
        data_dir, out, schema, args.val, args.test
    )
    if not names:
        sys.exit("No usable labeled region boxes found. Check --data and schema.yaml.")
    print(f"Split: train={train_pct}%  val={args.val}%  test={args.test}%  (hash-based)")

    (out / "config.json").write_text(
        json.dumps({
            "model": args.model,
            "imgsz": args.imgsz,
            "classes": names,
            "regions": regions,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Config -> {out / 'config.json'}")

    if args.no_train:
        print("\nDataset-only run finished (--no-train). Next: python train_multi.py")
        return

    model_path = args.model
    if not Path(model_path).exists() and (ROOT.parent / model_path).exists():
        model_path = str(ROOT.parent / model_path)

    # A horizontal flip would swap left/right sleeve boxes.
    fliplr = 0.0 if any("left" in n or "right" in n for n in names) else 0.5
    use_rect = not args.square
    imgsz = args.imgsz

    kw = dict(
        data=str(out / "dataset" / "data.yaml"),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        patience=args.patience,
        fliplr=fliplr,
        rect=use_rect,
        project=str(out),
        name="multi",
        exist_ok=True,
        seed=args.seed,
    )
    if args.device:
        kw["device"] = args.device

    print(f"\n{'=' * 50}")
    print("Training ONE multi-attribute detector")
    print(f"  model     = {model_path}")
    print(f"  classes   = {len(names)}")
    print(f"  imgsz     = {imgsz}")
    print(f"  rect      = {use_rect}   (4:3 letterbox)" if use_rect
          else "  rect      = False (square)")
    print(f"  effective ≈ {imgsz} × {int(imgsz * 4 / 3)}  (portrait 4:3)" if use_rect else "")
    print(f"  fliplr    = {fliplr}")
    print(f"  epochs    = {args.epochs}  batch = {args.batch}  seed = {args.seed}")
    print()

    model = YOLO(model_path)
    model.train(**kw)

    run_dir = out / "multi"
    best = run_dir / "weights" / "best.pt"
    final = out / "multi.pt"
    if best.exists():
        shutil.copy2(best, final)
        shutil.copy2(best, out / "best.pt")
    else:
        # Fall back to the in-memory trained model (never a fresh pretrained one).
        model.save(final)
        shutil.copy2(final, out / "best.pt")

    print(f"\n{'=' * 50}")
    print(f"Done. Weights: {final}")
    print(f"Ultralytics run: {run_dir}")
    print("Next:  python evaluate.py --all   # or pick models with select_models.py")


if __name__ == "__main__":
    main()
