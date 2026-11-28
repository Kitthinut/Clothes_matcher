#!/usr/bin/env python3
"""
Test trained model(s) on a single photo (upload / file pick).

No arguments opens a tkinter window: "Upload picture…" sits at the top-left, and
a model dropdown plus a multi-select picker (from select_models) choose which
individual / versioned / multi models to run. Choosing a picture runs those
models; boxes are drawn on the photo and labels, confidence and model weights
are listed in a clean results panel.

CLI mode is unchanged:

    python predict_image.py photo.jpg
    python predict_image.py photo.jpg --weights runs/neck_type/weights/best.pt
    python predict_image.py photo.jpg --all --conf 0.25

Press any key in the image window to close the CLI result view.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

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
model_label = _sm.model_label


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


PALETTE = [
    (0, 200, 0),
    (255, 140, 0),
    (0, 165, 255),
    (200, 0, 200),
    (0, 255, 255),
    (180, 105, 255),
]


def predict_detections(models, img, conf=0.25, imgsz=640, weight_paths=None) -> list[dict]:
    """Run every selected model on one image; return plain detection dicts.

    Detection models contribute boxes; classification models contribute their
    top-1 label without a box. No GUI / file I/O happens here, which keeps the
    assembly easy to test and shared by the CLI and the GUI.
    """
    paths = [Path(p) for p in (weight_paths or [])]
    task_color = {
        name: PALETTE[i % len(PALETTE)] for i, (name, _) in enumerate(models)
    }
    weights_by_task = {}
    for i, (name, _) in enumerate(models):
        weights_by_task[name] = paths[i].name if i < len(paths) else name

    detections = []
    for task, model in models:
        results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
        names = getattr(model, "names", {}) or {}
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) \
                        else names[cls_id]
                    xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                    detections.append({
                        "task": task, "cls": str(cls_name),
                        "conf": float(box.conf[0]), "xyxy": xyxy,
                        "color": task_color[task], "weights": weights_by_task.get(task, task),
                    })
                continue
            probs = getattr(result, "probs", None)
            if probs is not None:
                cls_id = int(probs.top1)
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) \
                    else names[cls_id]
                raw_conf = probs.top1conf
                conf_value = float(np.asarray(raw_conf).reshape(-1)[0])
                detections.append({
                    "task": task, "cls": str(cls_name),
                    "conf": conf_value,
                    "xyxy": None, "color": task_color[task],
                    "weights": weights_by_task.get(task, task),
                })
    return detections


def detection_rows(detections: list[dict]) -> list[tuple[str, ...]]:
    """Rows for both the Tk panel and the console: label, confidence, weights."""
    rows = []
    for i, d in enumerate(detections, 1):
        xyxy = d.get("xyxy")
        box = f"[{','.join(str(v) for v in xyxy)}]" if xyxy else "-"
        rows.append((str(i), d["task"], d["cls"], f"{d['conf']:.0%}",
                     d.get("weights", "-"), box))
    return rows


def detection_text(detections: list[dict]) -> str:
    """Readable detection summary (shared by the Tk panel and the CLI popup)."""
    lines = ["=" * 64, "  DETECTION SUMMARY", "=" * 64]
    if not detections:
        lines.append("  (none)")
    else:
        for row in detection_rows(detections):
            lines.append(
                f"  #{row[0]:<3} {row[1]} -> {row[2]}   {row[3]:>4}  "
                f"weights={row[4]}  box={row[5]}"
            )
        lines.append("")
        counts: dict[str, list[float]] = {}
        for d in detections:
            counts.setdefault(f"{d['task']} / {d['cls']}", []).append(d["conf"])
        lines.append("  By label:")
        for key, confs in sorted(counts.items()):
            avg = sum(confs) / len(confs)
            bar = "█" * max(1, int(avg * 20))
            lines.append(f"  {key:<28} n={len(confs)}  conf≈{avg:.0%}  {bar}")
    lines.append("=" * 64)
    return "\n".join(lines)


def draw_boxes(img, detections: list[dict]):
    """Draw numbered boxes on the photo; the numbers match the results panel."""
    annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, d in enumerate(detections, 1):
        if not d.get("xyxy"):
            continue
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
    return annotated


def compose_canvas(detections: list[dict], img, panel_w=340, max_photo_h=720):
    """Photo with boxes + right-side summary panel (BGR, ready to save/show)."""
    annotated = draw_boxes(img, detections)
    ph, pw = annotated.shape[:2]
    if ph > max_photo_h:
        scale = max_photo_h / ph
        annotated = cv2.resize(annotated, (int(pw * scale), int(ph * scale)))
        ph, pw = annotated.shape[:2]
    panel = _summary_panel(detections, panel_w, ph)
    return cv2.hconcat([annotated, panel])


def save_prediction(canvas, image_path: Path, save_path=None) -> Path:
    """Save under project/predicts/ (not Downloads / the source folder)."""
    predicts_dir = ROOT.parent / "predicts"
    predicts_dir.mkdir(parents=True, exist_ok=True)
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = predicts_dir / f"{image_path.stem}_pred{image_path.suffix.lower() or '.jpg'}"
        stem, suf = out.stem, out.suffix
        n = 1
        while out.exists():
            out = predicts_dir / f"{stem}_{n}{suf}"
            n += 1
    cv2.imwrite(str(out), canvas)
    return out


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


def _show_summary_popup(text: str):
    """Tkinter popup with the full detection summary (CLI mode only)."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return

    win = tk.Tk()
    win.title("raja_fashion – Prediction result")
    win.geometry("620x440")
    win.minsize(480, 320)

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


# ---- tkinter GUI --------------------------------------------------------------
def run_gui():
    """Main window: upload button (top-left), model picker, photo, result panel."""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import Image, ImageTk
    except ImportError as exc:
        sys.exit(
            f"GUI needs tkinter and Pillow ({exc}). "
            "Use CLI mode instead: python predict_image.py photo.jpg --weights ..."
        )

    found = find_all_weights()
    labels = [f"{model_label(w)} — {w.name}" for w in found]
    choices = ["All models"] + labels
    state = {"image": None, "weights": list(found), "busy": False, "photo": None}

    root = tk.Tk()
    root.title("raja_fashion – Test a photo")
    root.geometry("1180x760")
    root.minsize(900, 600)
    root.configure(bg="#eef1f5")
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass

    def status(text, color="#2f5d8a"):
        status_lbl.config(text=text, foreground=color)

    # Top row: the upload button is the left-most control.
    top = ttk.Frame(root, padding=(14, 12, 14, 6))
    top.pack(fill="x")
    ttk.Button(top, text="Upload picture…", command=lambda: on_upload()).pack(side="left")
    path_lbl = ttk.Label(top, text="No picture chosen", foreground="#667085")
    path_lbl.pack(side="left", padx=(12, 0))

    bar = ttk.Frame(root, padding=(14, 0, 14, 10))
    bar.pack(fill="x")
    ttk.Label(bar, text="Models:").pack(side="left")
    combo = ttk.Combobox(bar, values=choices, state="readonly", width=40)
    combo.current(0)
    combo.pack(side="left", padx=(6, 6))
    combo.bind("<<ComboboxSelected>>", lambda event=None: on_combo(event))
    ttk.Button(bar, text="Choose models…", command=lambda: open_picker()).pack(side="left")
    run_btn = ttk.Button(bar, text="Run prediction", command=lambda: on_run())
    run_btn.pack(side="left", padx=8)
    ttk.Label(bar, text="individual / versioned / multi", foreground="#667085").pack(side="left")

    pane = ttk.PanedWindow(root, orient="horizontal")
    pane.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    left = ttk.Frame(pane, padding=4)
    pane.add(left, weight=3)
    ttk.Label(left, text="Photo — numbered boxes match the list",
              font=("Segoe UI", 10, "bold")).pack(anchor="w")
    photo_canvas = tk.Canvas(left, bg="#f2f4f7", highlightthickness=1,
                             highlightbackground="#d9dfe8")
    photo_canvas.pack(fill="both", expand=True, pady=(6, 0))

    right = ttk.Frame(pane, padding=4)
    pane.add(right, weight=2)
    ttk.Label(right, text="Detections — label · confidence · weights",
              font=("Segoe UI", 10, "bold")).pack(anchor="w")
    tree_frame = ttk.Frame(right)
    tree_frame.pack(fill="both", expand=True, pady=(6, 0))
    tree = ttk.Treeview(
        tree_frame,
        columns=("n", "model", "label", "conf", "weights", "box"),
        show="headings", height=16,
    )
    for col, heading, width in (
        ("n", "#", 34), ("model", "Model", 110), ("label", "Class", 150),
        ("conf", "Conf.", 62), ("weights", "Weights", 110), ("box", "Box", 160),
    ):
        tree.heading(col, text=heading)
        tree.column(col, width=width, anchor="w")
    tree.tag_configure("odd", background="#f5f8fc")
    tree.tag_configure("none", foreground="#8a93a0")
    tree_sb = ttk.Scrollbar(tree_frame, command=tree.yview)
    tree.configure(yscrollcommand=tree_sb.set)
    tree.pack(side="left", fill="both", expand=True)
    tree_sb.pack(side="right", fill="y")

    status_lbl = ttk.Label(root, text="Upload a picture to run the selected model(s).",
                           padding=(14, 8), foreground="#2f5d8a")
    status_lbl.pack(fill="x")
    if not found:
        status("No trained weights found yet — run train.py first.", "#b00020")

    def show_photo(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cw = max(photo_canvas.winfo_width(), 560)
        ch = max(photo_canvas.winfo_height(), 420)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        pil.thumbnail((cw - 10, ch - 10), resample)
        state["photo"] = ImageTk.PhotoImage(pil)  # keep a reference alive
        photo_canvas.delete("all")
        photo_canvas.create_image(cw // 2, ch // 2, image=state["photo"], anchor="center")

    def finish(detections, canvas, out_path):
        state["busy"] = False
        run_btn.config(state="normal")
        for item in tree.get_children():
            tree.delete(item)
        rows = detection_rows(detections)
        if not rows:
            tree.insert("", "end", values=("-", "-", "no detections", "-", "-", "-"),
                        tags=("none",))
        for i, row in enumerate(rows):
            tree.insert("", "end", values=row, tags=("odd",) if i % 2 else ())
        show_photo(canvas)
        status(f"{len(detections)} detection(s) · saved {out_path}")

    def finish_error(exc):
        state["busy"] = False
        run_btn.config(state="normal")
        status(f"Prediction failed: {exc}", "#b00020")
        messagebox.showerror("Prediction failed", str(exc))

    def on_run():
        if state["busy"]:
            return
        if not state["image"]:
            messagebox.showinfo("Upload a picture", "Choose a picture first.")
            return
        if not state["weights"]:
            messagebox.showwarning("No models", "Choose at least one trained model.")
            return
        state["busy"] = True
        run_btn.config(state="disabled")
        status(f"Running {len(state['weights'])} model(s)…")

        image_path = state["image"]
        weights = list(state["weights"])

        def worker():
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise RuntimeError(f"Cannot read image: {image_path}")
                processed = letterbox_4x3(img, long_side=640)
                models = load_models(weights)
                if not models:
                    raise RuntimeError("Failed to load models")
                detections = predict_detections(models, processed, conf=0.25,
                                                imgsz=640, weight_paths=weights)
                canvas = compose_canvas(detections, processed)
                out_path = save_prediction(canvas, image_path)
                root.after(0, lambda: finish(detections, canvas, out_path))
            except Exception as exc:
                root.after(0, lambda e=exc: finish_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def on_upload():
        path = pick_image()
        if not path:
            return
        state["image"] = path
        path_lbl.config(text=str(path))
        status(f"Loaded {path.name}; running prediction…")
        on_run()  # a chosen picture immediately runs the selected models

    def on_combo(_event=None):
        value = combo.get()
        if value.startswith("Custom") or value not in choices:
            return
        idx = choices.index(value) - 1  # -1 = "All models"
        state["weights"] = list(found) if idx < 0 else [found[idx]]
        status(f"{len(state['weights'])} model(s) selected — upload a picture to run")

    def open_picker():
        if not found:
            messagebox.showwarning("No models", "Run train.py first.")
            return
        selected = pick_models(found, title="Select models for this photo",
                               preselect_all=False)
        if selected:
            state["weights"] = selected
            combo.set(f"Custom ({len(selected)} selected)")
            status(f"{len(selected)} model(s) selected — upload a picture to run")

    root.mainloop()


# ---- CLI mode (unchanged behaviour) -------------------------------------------
def run_cli(args):
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

    detections = predict_detections(models, img, conf=args.conf, imgsz=args.imgsz,
                                    weight_paths=weight_paths)
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"] or (0, 0, 0, 0)
        print(f"  {d['task']}: {d['cls']}  conf={d['conf']:.2f}  box=[{x1},{y1},{x2},{y2}]")
    if not detections:
        print("  (none)")

    canvas = compose_canvas(detections, img)
    out = save_prediction(canvas, image_path, save_path=args.save)
    print(f"\nSaved: {out}")

    summary = f"Saved to:\n{out}\n\n{detection_text(detections)}"
    cv2.imshow("raja_fashion predict", canvas)
    print("Press any key in the image window to close.")
    _show_summary_popup(summary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Test model(s) on one uploaded photo")
    ap.add_argument("image", nargs="?", default=None, help="Path to image")
    ap.add_argument("--weights", nargs="+", default=None, help="one or more best.pt")
    ap.add_argument("--all", action="store_true", help="use every trained model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default=None, help="path to save annotated image")
    ap.add_argument("--gui", action="store_true", help="open the GUI (default with no arguments)")
    args = ap.parse_args()

    if args.gui or (args.image is None and not args.weights and not args.all):
        run_gui()
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
