#!/usr/bin/env python3
"""
Train one or more tasks from tasks.yaml (after build_dataset.py).

    python train_individual.py                     # popup to choose task(s)
    python train_individual.py neck_crew
    python train_individual.py neck_crew top_length --epochs 150 --batch 8

Results: runs/<task>/  (weights/best.pt, results.csv, plots)

About image size & 4:3
----------------------
Your photos are 4:3 portrait. We keep that ratio during training.

Default setting (recommended):
  imgsz = 640          → YOLO uses multi-scale rectangular training
  rect  = True         → letterboxes to 4:3 instead of forcing square

Effective size becomes approximately:
  width  = 640
  height = 853          (640 × 4/3 ≈ 853)
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def pick_task_gui(tasks: dict, datasets_dir: Path) -> list[str] | None:
    """Popup to select one or more tasks that already have a built dataset."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return None

    choice = {"names": None}
    task_names = list(tasks.keys())

    root = tk.Tk()
    root.title("raja_fashion – Choose tasks to train")
    root.geometry("560x520")
    root.minsize(420, 400)
    root.configure(bg="#eef1f5")

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass

    header = ttk.Frame(root, padding=(20, 16, 20, 8))
    header.pack(fill="x")
    ttk.Label(header, text="Choose what to train", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(header, text="Select one or more tasks (Ctrl/Cmd+click, Shift+click)",
              foreground="#667085").pack(anchor="w", pady=(2, 0))

    body = ttk.Frame(root, padding=(20, 8, 20, 8))
    body.pack(fill="both", expand=True)

    lb_frame = ttk.Frame(body)
    lb_frame.pack(fill="both", expand=True)
    lb = tk.Listbox(lb_frame, font=("Segoe UI", 11), activestyle="none",
                    selectmode="extended", relief="flat", highlightthickness=1,
                    highlightbackground="#d9dfe8", bg="white", exportselection=False)
    sb = ttk.Scrollbar(lb_frame, command=lb.yview)
    lb.config(yscrollcommand=sb.set)
    lb.pack(side="left", fill="both", expand=True)
    sb.pack(side="left", fill="y")

    ready = []
    for name in task_names:
        cfg = tasks[name]
        mode = cfg.get("mode", "?")
        has_ds = (datasets_dir / name).exists()
        status = "ready" if has_ds else "no dataset"
        if "attribute" in cfg:
            desc = f"{name}   [{mode}]   {cfg.get('object','')}.{cfg.get('attribute','')}   ({status})"
        else:
            desc = f"{name}   [{mode}]   ({status})"
        lb.insert("end", desc)
        ready.append(has_ds)

    # Pre-select tasks that already have a dataset
    for i, ok in enumerate(ready):
        if ok:
            lb.selection_set(i)
            break
    if not any(ready) and task_names:
        lb.selection_set(0)

    detail = ttk.Label(body, text="", foreground="#667085", wraplength=500, justify="left")
    detail.pack(fill="x", pady=(8, 0))
    count_lbl = ttk.Label(body, text="", foreground="#2f5d8a")
    count_lbl.pack(anchor="w", pady=(4, 0))

    def on_select(_event=None):
        sel = lb.curselection()
        count_lbl.config(text=f"{len(sel)} selected" if sel else "0 selected")
        if not sel:
            detail.config(text="")
            return
        name = task_names[sel[-1]]
        cfg = tasks[name]
        has_ds = (datasets_dir / name).exists()
        lines = [f"Mode: {cfg.get('mode')}"]
        if "attribute" in cfg:
            lines.append(f"Object: {cfg.get('object')}  ·  Attribute: {cfg.get('attribute')}")
            lines.append("Classes: " + ", ".join(cfg.get("classes", {}).keys()))
        lines.append(f"Model: {cfg.get('model', '—')}  ·  imgsz: {cfg.get('imgsz', 640)}")
        lines.append(f"Dataset: {'found' if has_ds else 'MISSING – run build_dataset.py first'}")
        if len(sel) > 1:
            lines.append(f"\n(+ {len(sel) - 1} more task(s) selected)")
        detail.config(text="\n".join(lines))

    lb.bind("<<ListboxSelect>>", on_select)
    on_select()

    def select_all_ready():
        lb.selection_clear(0, "end")
        for i, ok in enumerate(ready):
            if ok:
                lb.selection_set(i)
        on_select()

    def confirm():
        sel = lb.curselection()
        if not sel:
            return
        choice["names"] = [task_names[i] for i in sel]
        root.destroy()

    def cancel():
        choice["names"] = None
        root.destroy()

    root.bind("<Return>", lambda e: confirm())
    root.bind("<Escape>", lambda e: cancel())

    footer = ttk.Frame(root, padding=(20, 8, 20, 16))
    footer.pack(fill="x")
    ttk.Button(footer, text="Select all ready", command=select_all_ready).pack(side="left")
    ttk.Button(footer, text="Cancel", command=cancel).pack(side="right")
    ttk.Button(footer, text="Train", command=confirm).pack(side="right", padx=(0, 8))

    root.mainloop()
    return choice["names"]


def train_one(task_name: str, tasks: dict, epochs: int, batch: int, patience: int,
              imgsz_override, model_override, device, force_square: bool) -> bool:
    cfg = tasks[task_name]
    ds = ROOT / "datasets" / task_name
    if not ds.exists():
        print(f"\nSKIP '{task_name}': no dataset. Run:  python build_dataset.py {task_name}")
        return False

    report_path = ds / "dataset_report.json"
    if not report_path.exists():
        print(f"\nSKIP '{task_name}': dataset_report.json missing. Rebuild the dataset.")
        return False
    names = json.loads(report_path.read_text())["names"]

    # Horizontal flip would swap left/right sleeves
    fliplr = 0.0 if any(str(n).endswith(("_left", "_right")) for n in names) else 0.5

    imgsz = imgsz_override or cfg.get("imgsz", 640)
    use_rect = cfg.get("rect", True) and not force_square

    kw = dict(
        data=str(ds / "data.yaml") if cfg["mode"] == "detect" else str(ds),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        fliplr=fliplr,
        rect=use_rect,
        project=str(ROOT / "runs"),
        name=task_name,
        exist_ok=True,
        seed=0,
    )
    if device:
        kw["device"] = device

    print(f"\n{'='*50}")
    print(f"Training '{task_name}'")
    print(f"  imgsz     = {imgsz}")
    print(f"  rect      = {use_rect}   (4:3 letterbox)" if use_rect else "  rect = False (square)")
    print(f"  effective ≈ {imgsz} × {int(imgsz * 4/3)}  (portrait 4:3)")
    print(f"  epochs    = {epochs}  batch = {batch}")
    print()

    YOLO(model_override or cfg["model"]).train(**kw)
    print(f"Done '{task_name}'. Next:  python evaluate.py {task_name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", default=None,
                    help="Task name(s) from tasks.yaml. If omitted, a popup lets you choose.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30,
                    help="early stop if val does not improve for N epochs")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="long side in pixels (default from tasks.yaml or 640)")
    ap.add_argument("--model", help="override the model in tasks.yaml")
    ap.add_argument("--device", help="0 for first GPU, cpu, mps ...")
    ap.add_argument("--square", action="store_true",
                    help="force square training (disables rect=True)")
    ap.add_argument("--cli", action="store_true", help="do not open the task picker popup")
    args = ap.parse_args()

    tasks = yaml.safe_load((ROOT / "tasks.yaml").read_text("utf-8"))

    selected = list(args.task) if args.task else None
    if not selected and not args.cli:
        selected = pick_task_gui(tasks, ROOT / "datasets")
        if selected is None:
            sys.exit("No task selected.")
    if not selected:
        sys.exit(f"Give task name(s), or run without --cli to pick from a list.\nAvailable: {', '.join(tasks)}")

    for name in selected:
        if name not in tasks:
            sys.exit(f"Unknown task '{name}'. Available: {', '.join(tasks)}")

    print(f"Will train {len(selected)} task(s): {', '.join(selected)}")

    ok_count = 0
    for name in selected:
        if train_one(name, tasks, args.epochs, args.batch, args.patience,
                     args.imgsz, args.model, args.device, args.square):
            ok_count += 1

    print(f"\n{'='*50}")
    print(f"Finished: {ok_count}/{len(selected)} task(s) trained successfully.")


if __name__ == "__main__":
    main()
