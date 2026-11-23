#!/usr/bin/env python3
"""
Real-time clothes detection using your laptop webcam.
Can run several trained models at once (neck + sleeve + bottom …).

    python live_demo.py
    python live_demo.py --weights ../train/runs/neck_type/weights/best.pt
    python live_demo.py --camera 1
    python live_demo.py --all

Press  q  to quit,  s  to save a snapshot.
On start: a popup lets you multi-select which models to load (unless --weights / --all).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / "train"
sys.path.insert(0, str(TRAIN))

from select_models import find_all_weights, load_models, pick_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help="one or more best.pt paths; if omitted, a multi-select popup opens",
    )
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--all", action="store_true", help="load every trained model (no popup)")
    args = ap.parse_args()

    if args.weights:
        weight_paths = [Path(w) for w in args.weights]
    elif args.all:
        weight_paths = find_all_weights(TRAIN / "runs")
    else:
        weight_paths = pick_models(title="Select models for live camera")

    weight_paths = [p for p in weight_paths if p.exists()]
    if not weight_paths:
        print("No weights selected or found under train/runs/*/weights/best.pt")
        return

    models = load_models(weight_paths)
    if not models:
        print("Failed to load any model.")
        return

    palette = [
        (0, 200, 0),
        (255, 140, 0),
        (0, 165, 255),
        (200, 0, 200),
        (0, 255, 255),
        (180, 105, 255),
    ]
    task_color = {
        name: palette[i % len(palette)] for i, (name, _) in enumerate(models)
    }

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    print("Live demo running. Press 'q' to quit, 's' to save snapshot.")
    print("Models:", ", ".join(n for n, _ in models))
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame.copy()
        for task_name, model in models:
            color = task_color[task_name]
            results = model.predict(
                frame, conf=args.conf, imgsz=args.imgsz, verbose=False
            )
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{task_name}: {model.names[cls_id]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                y0 = max(0, y1 - th - 6)
                cv2.rectangle(annotated, (x1, y0), (x1 + tw, y1), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(
            annotated,
            f"FPS {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("raja_fashion live", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = Path(f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(str(out), annotated)
            print(f"Saved {out}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
