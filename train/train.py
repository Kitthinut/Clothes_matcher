#!/usr/bin/env python3
"""
Train one task from tasks.yaml (after build_dataset.py).

    python train.py neck_crew
    python train.py neck_crew --epochs 150 --batch 8 --device 0
Results: runs/<task>/  (weights/best.pt, results.csv, plots)
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30, help="early stop if val does not improve for N epochs")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--model", help="override the model in tasks.yaml")
    ap.add_argument("--device", help="0 for first GPU, cpu, mps ...")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "tasks.yaml").read_text("utf-8"))[args.task]
    ds = ROOT / "datasets" / args.task
    if not ds.exists():
        sys.exit(f"Run first:  python build_dataset.py {args.task}")
    names = json.loads((ds / "dataset_report.json").read_text())["names"]

    # Horizontal flip would turn a left sleeve into a right sleeve -> disable when labels depend on side.
    fliplr = 0.0 if any(n.endswith(("_left", "_right")) for n in names) else 0.5

    kw = dict(
        data=str(ds / "data.yaml") if cfg["mode"] == "detect" else str(ds),
        epochs=args.epochs, imgsz=args.imgsz or cfg.get("imgsz", 640), batch=args.batch,
        patience=args.patience, fliplr=fliplr,
        project=str(ROOT / "runs"), name=args.task, exist_ok=True, seed=0,
    )
    if args.device:
        kw["device"] = args.device
    YOLO(args.model or cfg["model"]).train(**kw)
    print(f"\nDone. Next:  python evaluate.py {args.task}   |   python live_demo.py {args.task}")


if __name__ == "__main__":
    main()
