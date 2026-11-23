#!/usr/bin/env python3
"""
Shared multi-model picker for raja_fashion.

Used by:
  - live_demo.py  (webcam)
  - predict_image.py  (upload photo)
  - search/search.py  (closet search)

CLI:
    python select_models.py              # popup, print selected paths
    python select_models.py --cli        # list paths only, no GUI

Import:
    from select_models import pick_models, find_all_weights
    weights = pick_models()              # list[Path] or []
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def find_all_weights(runs_dir: Path | None = None) -> list[Path]:
    """Return every runs/<task>/weights/best.pt that exists."""
    base = runs_dir or RUNS
    if not base.exists():
        return []
    return sorted(base.glob("*/weights/best.pt"))


def task_name(weights: Path) -> str:
    """runs/neck_type/weights/best.pt -> neck_type"""
    return weights.parent.parent.name


def pick_models(
    weights: list[Path] | None = None,
    title: str = "Select models to run",
    preselect_all: bool = True,
) -> list[Path]:
    """
    Multi-select popup. Returns list of selected best.pt paths.
    Returns [] if cancelled or tkinter unavailable.
    """
    found = weights if weights is not None else find_all_weights()
    if not found:
        print("No weights found under train/runs/*/weights/best.pt")
        return []

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter not available – using all models")
        return found

    choice: dict = {"paths": None}

    root = tk.Tk()
    root.title("raja_fashion – " + title)
    root.geometry("560x480")
    root.minsize(420, 360)
    root.configure(bg="#eef1f5")

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass

    header = ttk.Frame(root, padding=(20, 16, 20, 8))
    header.pack(fill="x")
    ttk.Label(header, text=title, font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(
        header,
        text="Ctrl/Cmd+click or Shift+click for multiple  ·  these models run together",
        foreground="#667085",
    ).pack(anchor="w", pady=(2, 0))

    body = ttk.Frame(root, padding=(20, 8, 20, 8))
    body.pack(fill="both", expand=True)

    lb_frame = ttk.Frame(body)
    lb_frame.pack(fill="both", expand=True)
    lb = tk.Listbox(
        lb_frame,
        font=("Segoe UI", 11),
        activestyle="none",
        selectmode="extended",
        relief="flat",
        highlightthickness=1,
        highlightbackground="#d9dfe8",
        bg="white",
        exportselection=False,
    )
    sb = ttk.Scrollbar(lb_frame, command=lb.yview)
    lb.config(yscrollcommand=sb.set)
    lb.pack(side="left", fill="both", expand=True)
    sb.pack(side="left", fill="y")

    for w in found:
        lb.insert("end", f"{task_name(w):20s}  {w}")

    if preselect_all:
        lb.selection_set(0, "end")
    elif found:
        lb.selection_set(0)

    count_lbl = ttk.Label(body, text="", foreground="#2f5d8a")
    count_lbl.pack(anchor="w", pady=(8, 0))

    def update_count(_event=None):
        n = len(lb.curselection())
        count_lbl.config(text=f"{n} model(s) selected")

    lb.bind("<<ListboxSelect>>", update_count)
    update_count()

    def select_all():
        lb.selection_set(0, "end")
        update_count()

    def clear():
        lb.selection_clear(0, "end")
        update_count()

    def confirm():
        sel = lb.curselection()
        if not sel:
            return
        choice["paths"] = [found[i] for i in sel]
        root.destroy()

    def cancel():
        choice["paths"] = None
        root.destroy()

    root.bind("<Return>", lambda e: confirm())
    root.bind("<Escape>", lambda e: cancel())

    footer = ttk.Frame(root, padding=(20, 8, 20, 16))
    footer.pack(fill="x")
    ttk.Button(footer, text="Select all", command=select_all).pack(side="left")
    ttk.Button(footer, text="Clear", command=clear).pack(side="left", padx=6)
    ttk.Button(footer, text="Cancel", command=cancel).pack(side="right")
    ttk.Button(footer, text="OK – use selected", command=confirm).pack(side="right", padx=8)

    root.mainloop()
    return choice["paths"] if choice["paths"] is not None else []


def load_models(weight_paths: list[Path]):
    """Load YOLO models. Returns list of (task_name, model)."""
    from ultralytics import YOLO

    models = []
    for w in weight_paths:
        if not w.exists():
            print(f"Skip missing: {w}")
            continue
        print(f"Loading {w}")
        models.append((task_name(w), YOLO(str(w))))
    return models


def main():
    ap = argparse.ArgumentParser(description="Pick trained models")
    ap.add_argument("--cli", action="store_true", help="list all weights, no GUI")
    ap.add_argument("--runs", default=str(RUNS), help="runs directory")
    args = ap.parse_args()

    found = find_all_weights(Path(args.runs))
    if not found:
        print("No weights found.")
        sys.exit(1)

    if args.cli:
        for w in found:
            print(w)
        return

    selected = pick_models(found)
    if not selected:
        print("No models selected.")
        sys.exit(1)
    print("Selected:")
    for w in selected:
        print(f"  {task_name(w)}: {w}")


if __name__ == "__main__":
    main()
