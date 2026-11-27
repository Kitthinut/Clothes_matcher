#!/usr/bin/env python3
"""Build the combined multi-object detection dataset without training YOLO.

Dataset-only companion to ``train_multi.py``. It calls the very same
``build_multi_dataset()`` function (imported lazily from ``train_multi``), so
class names, hash-based splits and ``dataset_report.json`` are identical to the
dataset that ``train_multi.py`` would build -- it just never loads a YOLO model
and never trains. The dataset directory is rebuilt from scratch on every run.

The default split is 70% train / 15% validation / 15% test, using the same hash-based
assignment as ``build_dataset.py`` and ``train_multi.py``. Override it with --val /
--test; train receives the remaining 100 - val - test percent (val + test must stay
below 100). The effective percentages are recorded in ``dataset_report.json`` under
"split_percent".

Usage (from the ``train/`` folder, or pass explicit paths):

    python build_dataset_multi.py                          # ../data -> multi-models/
    python build_dataset_multi.py --data ../data --out multi-models
    python build_dataset_multi.py --val 20 --test 10 --out ../tmp/multi-build

Outputs (all under ``<out>/dataset/``):

    images/{train,val,test}/    labels/{train,val,test}/
    data.yaml                   dataset_report.json

Next step -- train the single multi-attribute detector:

    python train_multi.py
"""

import argparse
import sys
from pathlib import Path

# build_dataset.py has no Ultralytics dependency, so importing the shared split
# constants here keeps --help lightweight and the policy in one place.
from build_dataset import TEST_PCT, TRAIN_PCT, VAL_PCT, validate_split

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    """Dataset-related flags, mirroring train_multi.py."""
    ap = argparse.ArgumentParser(
        description="Build the combined multi-attribute detection dataset (no training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--data", default=str(ROOT.parent / "data"),
        help="folder with images + JSON sidecars (searched recursively)",
    )
    ap.add_argument(
        "--out", default=str(ROOT / "multi-models"),
        help="output folder for the combined dataset",
    )
    ap.add_argument(
        "--val", type=int, default=VAL_PCT,
        help=f"percent of images for validation (default split: "
             f"{TRAIN_PCT}/{VAL_PCT}/{TEST_PCT} train/val/test)",
    )
    ap.add_argument(
        "--test", type=int, default=TEST_PCT,
        help=f"percent of images for test (default split: "
             f"{TRAIN_PCT}/{VAL_PCT}/{TEST_PCT} train/val/test)",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="CLI parity with train_multi.py; the hash-based split ignores it",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    try:
        train_pct = validate_split(args.val, args.test)
    except ValueError as exc:
        sys.exit(f"Invalid split: {exc}")

    # Import only after parse_args() so `--help` never pulls in Ultralytics.
    # train_multi.py imports YOLO at module level and cannot be edited here.
    try:
        import yaml
    except ImportError as exc:
        sys.exit(f"PyYAML is required to read schema.yaml: {exc}\n"
                 "Install dependencies: pip install -r requirements.txt")

    try:
        from train_multi import SCHEMA_PATH, build_multi_dataset
    except Exception as exc:
        sys.exit(
            "Could not import build_multi_dataset() from train_multi.py: "
            f"{exc}\n"
            "train_multi.py imports Ultralytics YOLO at module level: install "
            "the project dependencies (pip install -r requirements.txt) and "
            "run this script from the train/ folder."
        )

    data_dir = Path(args.data).resolve()
    out = Path(args.out).resolve()
    if not data_dir.is_dir():
        sys.exit(f"Data folder not found: {data_dir} (override with --data).")
    if not SCHEMA_PATH.is_file():
        sys.exit(f"Schema not found: {SCHEMA_PATH}.")
    out.mkdir(parents=True, exist_ok=True)

    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    names, regions, report = build_multi_dataset(
        data_dir, out, schema, args.val, args.test
    )
    if not names:
        sys.exit("No usable labeled region boxes found. Check --data and schema.yaml.")

    ds = out / "dataset"
    images = report["images"]
    print("\n" + "=" * 60)
    print("Dataset-only build complete (no YOLO training performed)")
    print(f"  data source : {data_dir}")
    print(f"  output dir  : {out}")
    print(f"  dataset dir : {ds}")
    print(f"  classes     : {len(names)}  ({len(regions)} regions)")
    print(f"  images      : train={images.get('train', 0)}  "
          f"val={images.get('val', 0)}  test={images.get('test', 0)}")
    print(f"  split       : train={train_pct}%  val={args.val}%  test={args.test}%  "
          f"(hash-based; --seed={args.seed} does not affect it)")
    print(f"  data.yaml   : {ds / 'data.yaml'}")
    print(f"  report      : {ds / 'dataset_report.json'}")
    if images.get("train", 0) == 0 or images.get("val", 0) == 0:
        print("  WARNING: train or val split is empty; check --data and annotations.")
    print("\nNext: train the single multi-attribute detector -> python train_multi.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
