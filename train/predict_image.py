#!/usr/bin/env python3
"""
Test trained model(s) on a single photo (upload / file pick).

Can run several models at once — same picker as live_demo.

    python predict_image.py
    python predict_image.py photo.jpg
    python predict_image.py photo.jpg --weights runs/neck_type/weights/best.pt
    python predict_image.py photo.jpg --all --conf 0.25

Press any key in the image window to close.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


def _load_select_models():
    """Load select_models.py from the same folder (reliable on Windows)."""
    import importlib.util
    path = ROOT / "select_models.py"
    if not path.exists():
        sys.exit(
            f"Missing file: {path}\n"
            "Copy train/select_models.py into that folder, then try again."
        )
    spec = importlib.util.spec_from_file_location("select_models", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_sm = _load_select_models()
find_all_weights = _sm.find_all_weights
load_models = _sm.load_models
pick_models = _sm.pick_models


def pick_image() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose a photo to test",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def letterbox_4x3(img, long_side: int = 640):
    """
    Resize + letterbox to 4:3 portrait before inference (matches training).
    Returns the processed BGR image.
    """
    h, w = img.shape[:2]
    target_ratio = 4 / 3  # height / width

    # Pad to 4:3 first
    if h / w < target_ratio:
        new_h = int(round(w * target_ratio))
        new_w = w
    else:
        new_w = int(round(h / target_ratio))
        new_h = h
    canvas = cv2.copyMakeBorder(
        img,
        top=(new_h - h) // 2,
        bottom=(new_h - h) - (new_h - h) // 2,
        left=(new_w - w) // 2,
        right=(new_w - w) - (new_w - w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # Scale so the long side is `long_side` (height for portrait 4:3)
    ch, cw = canvas.shape[:2]
    scale = long_side / max(ch, cw)
    if abs(scale - 1.0) > 0.01:
        canvas = cv2.resize(
            canvas,
            (int(round(cw * scale)), int(round(ch * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Test model(s) on one uploaded photo")
    ap.add_argument("image", nargs="?", default=None, help="Path to image")
    ap.add_argument("--weights", nargs="+", default=None, help="one or more best.pt")
    ap.add_argument("--all", action="store_true", help="use every trained model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default=None, help="path to save annotated image")
    args = ap.parse_args()

    image_path = Path(args.image) if args.image else pick_image()
    if image_path is None or not image_path.exists():
        sys.exit("No image selected.")

    if args.weights:
        weight_paths = [Path(w) for w in args.weights]
    elif args.all:
        weight_paths = find_all_weights()
    else:
        weight_paths = pick_models(title="Select models for this photo")

    weight_paths = [p for p in weight_paths if p.exists()]
    if not weight_paths:
        sys.exit("No weights selected. Train first or pass --weights.")

    models = load_models(weight_paths)
    if not models:
        sys.exit("Failed to load models.")

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

    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")

    oh, ow = img.shape[:2]
    img = letterbox_4x3(img, long_side=args.imgsz)
    nh, nw = img.shape[:2]
    print(f"\nImage: {image_path}")
    print(f"Resize: {ow}x{oh} → {nw}x{nh} (4:3 letterbox, long side={args.imgsz})")
    print(f"Models: {', '.join(n for n, _ in models)}")
    print("Detections:")

    # Collect detections first (short labels on image, full list in summary)
    detections = []  # {task, cls, conf, xyxy, color}
    for task_name, model in models:
        color = task_color[task_name]
        results = model.predict(
            img, conf=args.conf, imgsz=args.imgsz, verbose=False
        )
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "task": task_name,
                "cls": cls_name,
                "conf": conf,
                "xyxy": (x1, y1, x2, y2),
                "color": color,
            })
            print(f"  {task_name}: {cls_name}  conf={conf:.2f}  box=[{x1},{y1},{x2},{y2}]")

    if not detections:
        print("  (none)")

    # Draw boxes with SHORT readable tags (1, 2, 3…) — full text goes to summary panel
    annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["xyxy"]
        color = d["color"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        tag = str(i)
        (tw, th), _ = cv2.getTextSize(tag, font, 0.9, 2)
        y0 = max(0, y1 - th - 10)
        cv2.rectangle(annotated, (x1, y0), (x1 + tw + 10, y1), color, -1)
        cv2.putText(
            annotated, tag, (x1 + 5, max(th + 2, y1 - 6)),
            font, 0.9, (0, 0, 0), 2, cv2.LINE_AA,
        )

    # Shrink photo for display, then attach summary panel on the right
    max_photo_h = 720
    ph, pw = annotated.shape[:2]
    if ph > max_photo_h:
        scale = max_photo_h / ph
        annotated = cv2.resize(annotated, (int(pw * scale), int(ph * scale)))
        ph, pw = annotated.shape[:2]

    panel_w = 340
    panel = _summary_panel(detections, panel_w, ph)
    canvas = cv2.hconcat([annotated, panel])

    # Save under project/predicts/ (not Downloads / source folder)
    predicts_dir = ROOT.parent / "predicts"
    predicts_dir.mkdir(parents=True, exist_ok=True)
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = predicts_dir / f"{image_path.stem}_pred{image_path.suffix.lower() or '.jpg'}"
        stem, suf = out.stem, out.suffix
        n = 1
        while out.exists():
            out = predicts_dir / f"{stem}_{n}{suf}"
            n += 1
    cv2.imwrite(str(out), canvas)
    print(f"\nSaved: {out}")

    summary_text = _print_summary_chart(detections)
    summary_text = f"Saved to:\n{out}\n\n" + summary_text

    cv2.imshow("raja_fashion predict", canvas)
    print("Press any key in the image window to close.")
    _show_summary_popup(summary_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _summary_panel(detections: list, width: int, height: int):
    """Right-side panel: numbered list + confidence bars."""
    import numpy as np

    panel = np.full((height, width, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "DETECTION SUMMARY", (16, 32), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (16, 42), (width - 16, 42), (80, 80, 80), 1)

    if not detections:
        cv2.putText(panel, "No detections", (16, 80), font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return panel

    # Group by task for a small chart
    by_task: dict[str, list] = {}
    for d in detections:
        by_task.setdefault(d["task"], []).append(d)

    y = 70
    cv2.putText(panel, f"Total: {len(detections)} box(es)", (16, y), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    y += 28

    for i, d in enumerate(detections, 1):
        if y > height - 80:
            cv2.putText(panel, "...", (16, y), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            break
        color = d["color"]
        # number badge
        cv2.rectangle(panel, (16, y - 16), (40, y + 4), color, -1)
        cv2.putText(panel, str(i), (22, y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # text
        line = f"{d['task']}  →  {d['cls']}"
        cv2.putText(panel, line, (50, y), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += 18
        # confidence bar
        bar_x0, bar_x1 = 50, width - 20
        bar_w = int((bar_x1 - bar_x0) * max(0.0, min(1.0, d["conf"])))
        cv2.rectangle(panel, (bar_x0, y - 6), (bar_x1, y + 2), (60, 60, 60), -1)
        cv2.rectangle(panel, (bar_x0, y - 6), (bar_x0 + bar_w, y + 2), color, -1)
        cv2.putText(panel, f"{d['conf']:.0%}", (bar_x1 - 42, y), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        y += 22

    # Mini chart: count per class
    y += 10
    if y < height - 40:
        cv2.line(panel, (16, y), (width - 16, y), (80, 80, 80), 1)
        y += 24
        cv2.putText(panel, "By label", (16, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
        counts: dict[str, int] = {}
        for d in detections:
            key = f"{d['task']}:{d['cls']}"
            counts[key] = counts.get(key, 0) + 1
        max_c = max(counts.values()) if counts else 1
        for key, c in sorted(counts.items(), key=lambda x: -x[1]):
            if y > height - 16:
                break
            cv2.putText(panel, f"{key}  x{c}", (16, y), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            bx0 = 16
            bw = int((width - 40) * (c / max_c))
            cv2.rectangle(panel, (bx0, y + 4), (bx0 + bw, y + 12), (100, 180, 255), -1)
            y += 28

    return panel


def _print_summary_chart(detections: list) -> str:
    lines = []
    lines.append("=" * 50)
    lines.append("  DETECTION SUMMARY")
    lines.append("=" * 50)
    if not detections:
        lines.append("  (none)")
    else:
        counts: dict[str, list] = {}
        for d in detections:
            key = f"{d['task']} / {d['cls']}"
            counts.setdefault(key, []).append(d["conf"])
        for i, d in enumerate(detections, 1):
            lines.append(
                f"  #{i}  {d['task']} → {d['cls']}   conf={d['conf']:.0%}"
            )
        lines.append("")
        lines.append("  By label:")
        for key, confs in sorted(counts.items()):
            avg = sum(confs) / len(confs)
            bar = "█" * max(1, int(avg * 20))
            lines.append(f"  {key:<28}  n={len(confs)}  conf≈{avg:.0%}  {bar}")
    lines.append("=" * 50)
    text = "\n".join(lines)
    print("\n" + text)
    return text


def _show_summary_popup(text: str):
    """Tkinter popup with the full detection summary (readable results)."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return

    win = tk.Tk()
    win.title("raja_fashion – Prediction result")
    win.geometry("520x420")
    win.minsize(400, 300)

    ttk.Label(win, text="Prediction result", font=("Segoe UI", 13, "bold")).pack(
        anchor="w", padx=16, pady=(14, 4)
    )
    frame = ttk.Frame(win)
    frame.pack(fill="both", expand=True, padx=12, pady=8)
    txt = tk.Text(frame, wrap="word", font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4")
    sb = ttk.Scrollbar(frame, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    txt.insert("1.0", text)
    txt.configure(state="disabled")

    bar = ttk.Frame(win, padding=12)
    bar.pack(fill="x")
    ttk.Button(bar, text="Close", command=win.destroy).pack(side="right")

    win.mainloop()


if __name__ == "__main__":
    main()
