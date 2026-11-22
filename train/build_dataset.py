#!/usr/bin/env python3
"""
Turn annotation JSONs (yours + friends') into a YOLO dataset for ONE task from tasks.yaml.

    python build_dataset.py neck_crew
    python build_dataset.py neck_crew --data ../data --val 15 --test 15

Re-run it any time you add or relabel data. The train/val/test split is decided by a hash of
the file name, so an image never jumps between splits when the dataset grows (no leakage,
and results stay comparable between runs).

Expected input: any folder tree under --data containing  <name>.json + <name>.jpg/png/webp.
JSON format: see annotate/labeling_config/schema.yaml
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT.parent / "annotate" / "labeling_config" / "schema.yaml"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
SPLITS = ("train", "val", "test")


def split_of(key: str, val_pct: int, test_pct: int) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


class Task:
    def __init__(self, name, cfg):
        self.name, self.cfg, self.mode = name, cfg, cfg["mode"]
        if "attribute" in cfg:
            self.object, self.attr = cfg["object"], cfg["attribute"]
            self.lookup = {v: c for c, vals in cfg["classes"].items() for v in vals}
            self.else_cls = cfg.get("else")
            self.names = list(cfg["classes"])
            if self.else_cls and self.else_cls not in self.names:
                self.names.append(self.else_cls)
        else:
            self.object, self.attr, self.else_cls = None, None, None
            self.objects = cfg["objects"]
            self.names = list(self.objects)

    def relevant(self, o):
        """True if this box belongs to the task's garment type."""
        if self.object is None:
            return o.get("cls") in self.objects
        return o.get("cls") == self.object

    def label(self, o):
        """Class name for this object, or None if it has not been labeled for this task."""
        if self.object is None:
            return o.get("cls")
        v = (o.get("attrs") or {}).get(self.attr)
        if v is None:
            return None
        return self.lookup.get(v, self.else_cls)

    def labeled_boxes(self, objects):
        """
        Return only boxes that have a valid label for this task.
        Important: an image has many region boxes (neck, button, sleeve…).
        We must NOT require every box to have the task attribute — only keep
        the ones that do (e.g. neck boxes for neck_type).
        """
        out = []
        for o in objects:
            if not self.relevant(o):
                continue
            l = self.label(o)
            if l is not None:
                out.append((o, l))
        return out


def schema_problems(o, schema):
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
    if not (isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in b)
            and b[2] > 0 and b[3] > 0):
        p.append(f"bad box {b}")
    return p


def find_image(jp: Path):
    for e in IMG_EXT:
        for cand in (jp.with_suffix(e), jp.with_suffix(e.upper())):
            if cand.exists():
                return cand
    return None


def pick_task_gui(tasks: dict) -> list[str] | None:
    """Popup window to select one or more tasks to build datasets for."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return None

    choice = {"names": None}

    root = tk.Tk()
    root.title("raja_fashion – Choose training tasks")
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

    task_names = list(tasks.keys())
    for name in task_names:
        cfg = tasks[name]
        mode = cfg.get("mode", "?")
        if "attribute" in cfg:
            desc = f"{name}   [{mode}]   {cfg.get('object', '')}.{cfg.get('attribute', '')}"
        else:
            desc = f"{name}   [{mode}]   objects={cfg.get('objects', [])}"
        lb.insert("end", desc)

    if task_names:
        lb.selection_set(0)
        lb.activate(0)

    detail = ttk.Label(body, text="", foreground="#667085", wraplength=500, justify="left")
    detail.pack(fill="x", pady=(8, 0))
    count_lbl = ttk.Label(body, text="1 selected", foreground="#2f5d8a")
    count_lbl.pack(anchor="w", pady=(4, 0))

    def on_select(_event=None):
        sel = lb.curselection()
        count_lbl.config(text=f"{len(sel)} selected" if sel else "0 selected")
        if not sel:
            detail.config(text="")
            return
        # Show details for the last clicked item
        name = task_names[sel[-1]]
        cfg = tasks[name]
        lines = [f"Mode: {cfg.get('mode')}"]
        if "attribute" in cfg:
            lines.append(f"Object: {cfg.get('object')}  ·  Attribute: {cfg.get('attribute')}")
            classes = cfg.get("classes", {})
            lines.append("Classes: " + ", ".join(classes.keys()))
            if cfg.get("else"):
                lines.append(f"Else class: {cfg['else']}")
        else:
            lines.append(f"Objects: {cfg.get('objects', [])}")
        lines.append(f"Model: {cfg.get('model', '—')}  ·  imgsz: {cfg.get('imgsz', 640)}")
        if len(sel) > 1:
            lines.append(f"\n(+ {len(sel) - 1} more task(s) selected)")
        detail.config(text="\n".join(lines))

    lb.bind("<<ListboxSelect>>", on_select)
    on_select()

    def select_all():
        lb.selection_set(0, "end")
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
    ttk.Button(footer, text="Select all", command=select_all).pack(side="left")
    ttk.Button(footer, text="Cancel", command=cancel).pack(side="right")
    ttk.Button(footer, text="Build dataset(s)", command=confirm).pack(side="right", padx=(0, 8))

    root.mainloop()
    return choice["names"]


def build_one(task_name: str, tasks: dict, schema: dict, data_dir: Path, out_root: Path,
              val_pct: int, test_pct: int, pad: float) -> bool:
    """Build dataset for a single task. Returns True if successful."""
    task = Task(task_name, tasks[task_name])
    out = out_root / task_name
    if out.exists():
        shutil.rmtree(out)
    for s in SPLITS:
        if task.mode == "detect":
            (out / "images" / s).mkdir(parents=True)
            (out / "labels" / s).mkdir(parents=True)
        else:
            for n in task.names:
                (out / s / n).mkdir(parents=True)

    skipped, warns = Counter(), []
    n_images = Counter()
    n_inst = defaultdict(Counter)

    for jp in sorted(data_dir.rglob("*.json")):
        try:
            meta = json.loads(jp.read_text("utf-8"))
        except Exception as e:
            warns.append(f"{jp.name}: unreadable JSON ({e})")
            continue
        if not isinstance(meta, dict) or "objects" not in meta:
            continue
        img = find_image(jp)
        if img is None:
            skipped["image file missing"] += 1
            continue

        problems = [f"{jp.name}: {p}" for o in meta["objects"] for p in schema_problems(o, schema)]
        if problems:
            skipped["schema errors"] += 1
            warns.extend(problems[:2])
            continue

        # Only boxes that actually have this task's attribute (e.g. neck boxes
        # for neck_type). Other regions on the same garment are ignored.
        labeled = task.labeled_boxes(meta["objects"])
        if not labeled:
            skipped["not labeled for this task yet"] += 1
            continue

        split = split_of(jp.stem, val_pct, test_pct)
        n_images[split] += 1

        if task.mode == "detect":
            shutil.copy2(img, out / "images" / split / (jp.stem + img.suffix.lower()))
            lines = []
            for o, l in labeled:
                cx, cy, w, h = o["box"]
                lines.append(f"{task.names.index(l)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                n_inst[split][l] += 1
            (out / "labels" / split / (jp.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
        else:
            with Image.open(img) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                W, H = im.size
                for i, (o, l) in enumerate(labeled):
                    cx, cy, w, h = o["box"]
                    x1 = max(0, (cx - w / 2 - pad * w) * W)
                    y1 = max(0, (cy - h / 2 - pad * h) * H)
                    x2 = min(W, (cx + w / 2 + pad * w) * W)
                    y2 = min(H, (cy + h / 2 + pad * h) * H)
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    im.crop((int(x1), int(y1), int(x2), int(y2))).save(
                        out / split / l / f"{jp.stem}_{i}.jpg", quality=92
                    )
                    n_inst[split][l] += 1

    if task.mode == "detect":
        (out / "data.yaml").write_text(yaml.safe_dump({
            "path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": dict(enumerate(task.names)),
        }, sort_keys=False))
    report = {"task": task.name, "mode": task.mode, "names": task.names,
              "images": dict(n_images), "instances": {s: dict(n_inst[s]) for s in SPLITS},
              "skipped": dict(skipped)}
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nTask '{task.name}' ({task.mode}) -> {out}")
    print(f"{'split':<7}{'images':>8}  " + "  ".join(f"{n:>12}" for n in task.names))
    for s in SPLITS:
        print(f"{s:<7}{n_images[s]:>8}  " + "  ".join(f"{n_inst[s][n]:>12}" for n in task.names))
    if skipped:
        print("Skipped:", dict(skipped))
    for w in warns[:5]:
        print("  warning:", w)

    ok = True
    if n_images["train"] == 0 or n_images["val"] == 0:
        print(f"  ERROR: train or val split is empty for '{task.name}'.")
        ok = False
    for s in ("val", "test"):
        if n_images[s] < 15:
            print(f"  ! only {n_images[s]} images in {s}: metrics will be noisy")
    tr = [n_inst["train"][n] for n in task.names]
    for n, c in zip(task.names, tr):
        if c < 30:
            print(f"  ! class '{n}' has only {c} training instances (aim for 50+)")
    if tr and min(tr) > 0 and max(tr) / min(tr) > 4:
        print("  ! classes are imbalanced (>4x)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", default=None,
                    help="Task name(s) from tasks.yaml. If omitted, a popup lets you choose.")
    ap.add_argument("--data", default=str(ROOT.parent / "data"),
                    help="folder with images + JSONs (searched recursively)")
    ap.add_argument("--out", default=str(ROOT / "datasets"))
    ap.add_argument("--val", type=int, default=15, help="percent of images for validation")
    ap.add_argument("--test", type=int, default=15, help="percent of images for test")
    ap.add_argument("--pad", type=float, default=0.08, help="crop padding for cls mode")
    ap.add_argument("--cli", action="store_true", help="do not open the task picker popup")
    args = ap.parse_args()

    tasks = yaml.safe_load((ROOT / "tasks.yaml").read_text("utf-8"))
    schema = yaml.safe_load(SCHEMA_PATH.read_text("utf-8"))

    selected = list(args.task) if args.task else None
    if not selected and not args.cli:
        selected = pick_task_gui(tasks)
        if selected is None:
            sys.exit("No task selected.")
    if not selected:
        sys.exit(f"Give task name(s), or run without --cli to pick from a list.\nAvailable: {', '.join(tasks)}")

    for name in selected:
        if name not in tasks:
            sys.exit(f"Unknown task '{name}'. Available: {', '.join(tasks)}")

    data_dir = Path(args.data)
    out_root = Path(args.out)
    print(f"Building {len(selected)} dataset(s): {', '.join(selected)}")

    all_ok = True
    for name in selected:
        ok = build_one(name, tasks, schema, data_dir, out_root, args.val, args.test, args.pad)
        all_ok = all_ok and ok

    print("\n" + "=" * 50)
    if all_ok:
        print(f"Done. Built: {', '.join(selected)}")
        print("Next:  python train.py <task>")
    else:
        print("Finished with errors on some tasks. Check the messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
