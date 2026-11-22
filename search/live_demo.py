#!/usr/bin/env python3
"""
Real-time clothes detection using your laptop webcam.

    python live_demo.py
    python live_demo.py --weights ../train/runs/neck_crew/weights/best.pt
    python live_demo.py --camera 1          # if you have multiple cameras

Press  q  to quit,  s  to save a snapshot.
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else None
    if weights is None:
        for cand in [
            TRAIN / "runs" / "neck_crew" / "weights" / "best.pt",
            TRAIN / "runs" / "tops_full" / "weights" / "best.pt",
            TRAIN / "runs" / "top_length" / "weights" / "best.pt",
        ]:
            if cand.exists():
                weights = cand
                break
    if weights is None or not weights.exists():
        print("No weights found. Train a model first.")
        return

    print(f"Loading {weights}")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    print("Live demo running. Press 'q' to quit, 's' to save snapshot.")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated = results[0].plot()  # draws boxes + labels

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(annotated, f"FPS {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
