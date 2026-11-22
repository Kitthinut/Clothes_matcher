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
        return o["cls"] == self.object if self.object else o["cls"] in self.objects

    def label(self, o):
        """Class name for this object, or None if it has not been labeled for this task."""
        if self.object is None:
            return o["cls"]
        v = (o.get("attrs") or {}).get(self.attr)
        if v is None:
            return None
        return self.lookup.get(v, self.else_cls)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--data", default=str(ROOT.parent / "data"), help="folder with images + JSONs (searched recursively)")
    ap.add_argument("--out", default=str(ROOT / "datasets"))
    ap.add_argument("--val", type=int, default=15, help="percent of images for validation")
    ap.add_argument("--test", type=int, default=15, help="percent of images for test")
    ap.add_argument("--pad", type=float, default=0.08, help="crop padding for cls mode (fraction of box size)")
    args = ap.parse_args()

    tasks = yaml.safe_load((ROOT / "tasks.yaml").read_text("utf-8"))
    if args.task not in tasks:
        sys.exit(f"Unknown task '{args.task}'. Available: {', '.join(tasks)}")
    task = Task(args.task, tasks[args.task])
    schema = yaml.safe_load(SCHEMA_PATH.read_text("utf-8"))

    out = Path(args.out) / args.task
    if out.exists():
        shutil.rmtree(out)  # always rebuild from the JSONs: the JSONs are the truth
    for s in SPLITS:
        if task.mode == "detect":
            (out / "images" / s).mkdir(parents=True)
            (out / "labels" / s).mkdir(parents=True)
        else:
            for n in task.names:
                (out / s / n).mkdir(parents=True)

    skipped, warns = Counter(), []
    n_images = Counter()
    n_inst = defaultdict(Counter)  # split -> class -> count

    for jp in sorted(Path(args.data).rglob("*.json")):
        try:
            meta = json.loads(jp.read_text("utf-8"))
        except Exception as e:
            warns.append(f"{jp.name}: unreadable JSON ({e})")
            continue
        if not isinstance(meta, dict) or "objects" not in meta:
            continue  # e.g. contributor.json
        img = find_image(jp)
        if img is None:
            skipped["image file missing"] += 1
            continue

        problems = [f"{jp.name}: {p}" for o in meta["objects"] for p in schema_problems(o, schema)]
        if problems:
            skipped["schema errors"] += 1
            warns.extend(problems[:2])
            continue

        mine = [o for o in meta["objects"] if task.relevant(o)]
        labels = [task.label(o) for o in mine]
        if any(l is None for l in labels):
            # An unlabeled box in a detect dataset would be trained as "background" -> skip the whole image.
            skipped["not labeled for this task yet"] += 1
            continue

        split = split_of(jp.stem, args.val, args.test)
        n_images[split] += 1

        if task.mode == "detect":
            shutil.copy2(img, out / "images" / split / (jp.stem + img.suffix.lower()))
            lines = []
            for o, l in zip(mine, labels):
                cx, cy, w, h = o["box"]
                lines.append(f"{task.names.index(l)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                n_inst[split][l] += 1
            (out / "labels" / split / (jp.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        else:
            with Image.open(img) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                W, H = im.size
                for i, (o, l) in enumerate(zip(mine, labels)):
                    cx, cy, w, h = o["box"]
                    x1 = max(0, (cx - w / 2 - args.pad * w) * W)
                    y1 = max(0, (cy - h / 2 - args.pad * h) * H)
                    x2 = min(W, (cx + w / 2 + args.pad * w) * W)
                    y2 = min(H, (cy + h / 2 + args.pad * h) * H)
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    im.crop((int(x1), int(y1), int(x2), int(y2))).save(out / split / l / f"{jp.stem}_{i}.jpg", quality=92)
                    n_inst[split][l] += 1

    # ---- dataset descriptor + report ----
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
        print("\nSkipped:", dict(skipped))
    for w in warns[:10]:
        print("  warning:", w)

    # ---- sanity warnings ----
    if n_images["train"] == 0 or n_images["val"] == 0:
        sys.exit("\nERROR: train or val split is empty. Label more images (or lower --val/--test).")
    for s in ("val", "test"):
        if n_images[s] < 15:
            print(f"\n  ! only {n_images[s]} images in {s}: metrics will be very noisy")
    tr = [n_inst["train"][n] for n in task.names]
    for n, c in zip(task.names, tr):
        if c < 30:
            print(f"  ! class '{n}' has only {c} training instances (aim for 50+, ideally 200+)")
    if min(tr) > 0 and max(tr) / min(tr) > 4:
        print("  ! classes are imbalanced (>4x); consider labeling more of the rare class")


if __name__ == "__main__":
    main()
