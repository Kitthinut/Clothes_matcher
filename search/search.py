#!/usr/bin/env python3
"""
Search your closet for clothes that match a reference photo.

Workflow
  1. Run the trained YOLO model on the reference image → get attribute tags
     (neck=crew, length=regular, sleeve=short, …) + dominant colors.
  2. Look up the local closet database (JSON or folder of labeled images).
  3. Rank by attribute match score + color similarity.
  4. Show the top matches.

Usage
    python search.py path/to/reference.jpg
    python search.py path/to/reference.jpg --db ../data/closet --top 8
    python search.py path/to/reference.jpg --weights ../train/runs/neck_crew/weights/best.pt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

from color_detect import dominant_colors

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / "train"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_closet(db_path: Path):
    """
    Closet can be:
      - a folder of images + sidecar .json (same format as annotation)
      - a single closet.json list of items

    Returns list of dicts:
      {path, cls, tags: {attr: value}, colors: [{name, hex, pct}, ...]}
    """
    items = []

    # Option 1: single closet.json
    closet_json = db_path / "closet.json"
    if closet_json.exists():
        raw = json.loads(closet_json.read_text("utf-8"))
        for it in raw:
            if "colors" not in it and "path" in it:
                it["colors"] = dominant_colors(it["path"], top_k=3)
            items.append(it)
        return items

    # Option 2: folder of image + .json pairs
    for jp in sorted(db_path.rglob("*.json")):
        if jp.name in ("contributor.json", "closet.json"):
            continue
        try:
            meta = json.loads(jp.read_text("utf-8"))
        except Exception:
            continue

        img = None
        for e in IMG_EXT:
            cand = jp.with_suffix(e)
            if cand.exists():
                img = cand
                break
            cand = jp.with_suffix(e.upper())
            if cand.exists():
                img = cand
                break
        if img is None:
            continue

        # Aggregate attributes from all region boxes of the same garment
        tags = {}
        cls = None
        for o in meta.get("objects", []):
            c = o.get("cls")
            if c in ("top", "bottom"):
                cls = c
            for k, v in (o.get("attrs") or {}).items():
                if v:
                    tags[k] = v          # last value wins (normally only one)

        colors = dominant_colors(img, top_k=3)
        items.append({
            "path": str(img),
            "cls": cls,
            "tags": tags,
            "colors": colors,
        })
    return items


def detect_attributes(model, image_path: Path, conf: float = 0.25):
    """
    Run YOLO and return:
      - tags: dict of detected class names → confidence
      - boxes: list of {name, conf, xyxy}
    """
    results = model.predict(str(image_path), conf=conf, verbose=False)
    tags = {}
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls)
            name = names[cls_id]
            conf_score = float(b.conf)
            # keep highest confidence per class name
            if name not in tags or conf_score > tags[name]:
                tags[name] = conf_score
            boxes.append({
                "name": name,
                "conf": conf_score,
                "xyxy": [round(x, 1) for x in b.xyxy[0].tolist()],
            })
    return tags, boxes


def color_similarity(c1, c2):
    """Overlap of named colors weighted by percentage."""
    if not c1 or not c2:
        return 0.0
    n1 = {c["name"]: c["pct"] for c in c1}
    n2 = {c["name"]: c["pct"] for c in c2}
    common = set(n1) & set(n2)
    if not common:
        return 0.0
    return sum(min(n1[n], n2[n]) for n in common)


def attribute_score(query_tags: dict, item_tags: dict) -> float:
    """
    How well the item matches the detected tags.
    Works with both:
      - progressive models that output "crew" / "not_crew"
      - full models that output attribute values
    """
    if not query_tags:
        return 0.5

    hits = 0
    total = 0
    for qname, qconf in query_tags.items():
        total += 1
        # direct match on attribute value stored in item
        if qname in item_tags.values():
            hits += 1
            continue
        # match on key (e.g. model said "crew" and item has neck=crew)
        if qname in item_tags:
            hits += 1
            continue
        # progressive "not_xxx" handling
        if qname.startswith("not_"):
            real = qname[4:]
            if real not in item_tags.values() and real not in item_tags:
                hits += 0.6          # partial credit for correct rejection
    return hits / max(total, 1)


def rank(items, query_tags, query_colors, cls_filter=None):
    scored = []
    for it in items:
        if cls_filter and it.get("cls") and it["cls"] != cls_filter:
            continue
        a_score = attribute_score(query_tags, it.get("tags") or {})
        c_score = color_similarity(query_colors, it.get("colors") or [])
        # weighted combination – tweak these numbers if you want
        score = 0.65 * a_score + 0.35 * c_score
        scored.append((score, a_score, c_score, it))
    scored.sort(key=lambda x: -x[0])
    return scored


def find_default_weights():
    candidates = [
        TRAIN / "runs" / "neck_crew" / "weights" / "best.pt",
        TRAIN / "runs" / "tops_full" / "weights" / "best.pt",
        TRAIN / "runs" / "top_length" / "weights" / "best.pt",
        TRAIN / "runs" / "bottom_fit" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="Find similar clothes in your closet")
    ap.add_argument("image", help="Reference fashion photo")
    ap.add_argument("--db", default=str(ROOT.parent / "data" / "closet"),
                    help="Folder of labeled closet items or closet.json")
    ap.add_argument("--weights", default=None,
                    help="Path to YOLO weights")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cls", choices=["top", "bottom"], default=None,
                    help="Only search tops or bottoms")
    args = ap.parse_args()

    img = Path(args.image)
    if not img.exists():
        sys.exit(f"Image not found: {img}")

    weights = Path(args.weights) if args.weights else find_default_weights()
    if weights is None or not weights.exists():
        print("No trained weights found. Train a model first:")
        print("  cd train")
        print("  python build_dataset.py neck_crew --data ../data")
        print("  python train.py neck_crew")
        sys.exit(1)

    print(f"Model   : {weights}")
    model = YOLO(str(weights))

    print(f"Image   : {img.name}")
    tags, boxes = detect_attributes(model, img, conf=args.conf)
    colors = dominant_colors(img, top_k=3)

    print("Tags    :", {k: f"{v:.2f}" for k, v in tags.items()} or "(none)")
    print("Colors  :", [f"{c['name']}({c['pct']:.0%})" for c in colors] or "(none)")
    if boxes:
        print("Boxes   :", len(boxes))

    db = Path(args.db)
    if not db.exists():
        print(f"\nCloset database not found at: {db}")
        print("Put labeled images (or a closet.json) there and try again.")
        sys.exit(0)

    items = load_closet(db)
    print(f"Closet  : {len(items)} items")

    if not items:
        print("No items loaded. Check that JSON files sit next to images.")
        sys.exit(0)

    ranked = rank(items, tags, colors, cls_filter=args.cls)

    print(f"\nTop {min(args.top, len(ranked))} matches")
    print("-" * 64)
    for i, (score, a, c, it) in enumerate(ranked[:args.top], 1):
        print(f"{i:2}.  score={score:.3f}   attr={a:.2f}  color={c:.2f}")
        print(f"    {it['path']}")
        print(f"    tags  = {it.get('tags')}")
        print(f"    colors= {[x['name'] for x in it.get('colors', [])]}")
        print()


if __name__ == "__main__":
    main()
