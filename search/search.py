#!/usr/bin/env python3
"""
Find clothes in your local database that match a reference photo.

Database layout (required)
--------------------------
    search/database/
        top/       ← top images (+ optional .json labels)
        bottom/    ← bottom images (+ optional .json labels)

How ranking works
-----------------
1. Run your selected YOLO model(s) on the reference image → set of labels
   e.g. {crew, short, skinny, long}
2. For each image in database/top and database/bottom:
     - use sidecar .json attributes if present, else run the same models
3. Score = how many labels match (plus optional color boost)
4. Show the highest-scoring images

Usage
-----
    python search.py path/to/reference.jpg
    python search.py path/to/reference.jpg --db database --top 8
    python search.py path/to/reference.jpg --weights ../train/runs/neck_type/weights/best.pt
    python search.py path/to/reference.jpg --all --cls top
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

from color_detect import dominant_colors

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent / "train"
DEFAULT_DB = ROOT / "database"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── database loading ──────────────────────────────────────────────


def find_image_for_json(jp: Path) -> Path | None:
    for e in IMG_EXT:
        for cand in (jp.with_suffix(e), jp.with_suffix(e.upper())):
            if cand.exists():
                return cand
    return None


def tags_from_json(meta: dict) -> tuple[str | None, dict]:
    """Aggregate attrs from annotation JSON → (cls, tags)."""
    tags = {}
    cls = None
    for o in meta.get("objects", []):
        c = o.get("cls")
        if c in ("top", "bottom"):
            cls = c
        for k, v in (o.get("attrs") or {}).items():
            if v:
                tags[k] = str(v)
                tags[str(v)] = str(v)  # also index by value for YOLO class-name match
    return cls, tags


def tags_from_models(models: list, image_path: Path, conf: float) -> dict:
    """Run models on an image; return flat set of class names → conf."""
    tags = {}
    for model in models:
        results = model.predict(str(image_path), conf=conf, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                name = r.names[int(b.cls)]
                score = float(b.conf)
                if name not in tags or score > tags[name]:
                    tags[name] = score
    return tags


def load_database(db_path: Path, models: list | None, conf: float, run_models_on_db: bool):
    """
    Load items from database/top and database/bottom.

    Each item:
      {path, cls, tags: {label: score_or_1.0}, colors: [...]}
    """
    items = []
    for sub in ("top", "bottom"):
        folder = db_path / sub
        if not folder.is_dir():
            continue
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            jp = img.with_suffix(".json")
            cls = sub
            tags = {}
            if jp.exists():
                try:
                    meta = json.loads(jp.read_text("utf-8"))
                    jcls, tags = tags_from_json(meta)
                    if jcls:
                        cls = jcls
                except Exception:
                    pass
            # Fill / override with live model predictions if requested or no tags
            if models and (run_models_on_db or not tags):
                pred = tags_from_models(models, img, conf)
                for k, v in pred.items():
                    tags[k] = v
            # Normalize: presence tags for scoring (value names like crew, short)
            flat = {}
            for k, v in tags.items():
                flat[str(k)] = float(v) if isinstance(v, (int, float)) else 1.0
                if not isinstance(v, (int, float)) and str(v) not in flat:
                    flat[str(v)] = 1.0
            try:
                colors = dominant_colors(img, top_k=3)
            except Exception:
                colors = []
            items.append({
                "path": str(img.resolve()),
                "cls": cls,
                "tags": flat,
                "colors": colors,
            })
    return items


# ── scoring ───────────────────────────────────────────────────────


def detect_reference(models: list, image_path: Path, conf: float):
    tags = {}
    boxes = []
    for model in models:
        results = model.predict(str(image_path), conf=conf, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                name = r.names[int(b.cls)]
                score = float(b.conf)
                if name not in tags or score > tags[name]:
                    tags[name] = score
                boxes.append({
                    "name": name,
                    "conf": score,
                    "xyxy": [round(x, 1) for x in b.xyxy[0].tolist()],
                })
    return tags, boxes


def color_similarity(c1, c2) -> float:
    if not c1 or not c2:
        return 0.0
    n1 = {c["name"]: c["pct"] for c in c1}
    n2 = {c["name"]: c["pct"] for c in c2}
    common = set(n1) & set(n2)
    if not common:
        return 0.0
    return sum(min(n1[n], n2[n]) for n in common)


def rank_items(query_tags: dict, query_colors: list, items: list, cls_filter: str | None):
    """
    Score by number of shared labels (main) + color overlap (secondary).

    match_count = how many query labels appear on the item
    score = 0.75 * (match_count / max_labels) + 0.25 * color_sim
    """
    q_labels = set(query_tags.keys())
    if not q_labels:
        return []

    max_labels = len(q_labels)
    scored = []
    for it in items:
        if cls_filter and it.get("cls") != cls_filter:
            continue
        item_labels = set(it.get("tags") or {})
        matched = q_labels & item_labels
        match_count = len(matched)
        if match_count == 0:
            continue
        a_score = match_count / max_labels
        c_score = color_similarity(query_colors, it.get("colors") or [])
        score = 0.75 * a_score + 0.25 * c_score
        scored.append({
            "score": score,
            "match_count": match_count,
            "matched": sorted(matched),
            "a_score": a_score,
            "c_score": c_score,
            "item": it,
        })
    scored.sort(key=lambda x: (-x["match_count"], -x["score"]))
    return scored


# ── weights / models ──────────────────────────────────────────────


def resolve_weights(cli_weights, use_all: bool = False):
    sys.path.insert(0, str(TRAIN))
    from select_models import find_all_weights, pick_models

    if cli_weights:
        return [Path(w) for w in cli_weights]
    if use_all:
        return find_all_weights(TRAIN / "runs")
    selected = pick_models(title="Select models for closet search")
    return selected or []


# ── display ───────────────────────────────────────────────────────


def show_results(ranked: list, top_n: int, query_tags: dict):
    print("\n" + "=" * 60)
    print("  SEARCH RESULTS")
    print("=" * 60)
    print(f"  Query labels : {', '.join(sorted(query_tags.keys())) or '(none)'}")
    if not ranked:
        print("\n  No matches. Add more images to database/top and database/bottom,")
        print("  or lower --conf, or check that models detect labels on the reference.")
        print("=" * 60)
        return

    print(f"  Showing top {min(top_n, len(ranked))} of {len(ranked)} matches\n")
    for i, row in enumerate(ranked[:top_n], 1):
        it = row["item"]
        print(f"  #{i}  score={row['score']:.2f}  labels_matched={row['match_count']}")
        print(f"      file     : {it['path']}")
        print(f"      type     : {it.get('cls')}")
        print(f"      matched  : {', '.join(row['matched'])}")
        cols = it.get("colors") or []
        if cols:
            print(f"      colors   : {', '.join(c['name'] for c in cols[:3])}")
        print()
    print("=" * 60)


def try_open_images(ranked: list, top_n: int):
    """Optional OpenCV grid of top matches."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    tiles = []
    for row in ranked[:top_n]:
        img = cv2.imread(row["item"]["path"])
        if img is None:
            continue
        img = cv2.resize(img, (200, 260))
        label = f"{row['match_count']} labels"
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(img)
    if not tiles:
        return

    # pad to rectangle
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(tiles[r * cols:(r + 1) * cols]))
    grid = np.vstack(grid_rows)
    cv2.imshow("raja_fashion search – top matches", grid)
    print("Press any key in the results window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── main ──────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Search database/top & database/bottom for similar clothes")
    ap.add_argument("image", help="Reference fashion photo")
    ap.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Database root (must contain top/ and bottom/ subfolders)",
    )
    ap.add_argument("--weights", nargs="+", default=None, help="One or more best.pt")
    ap.add_argument("--all", action="store_true", help="Use every trained model")
    ap.add_argument("--top", type=int, default=8, help="How many results to show")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cls", choices=["top", "bottom"], default=None, help="Only search that subfolder")
    ap.add_argument(
        "--scan-db",
        action="store_true",
        help="Also run models on database images (slower; use if DB has no JSON labels)",
    )
    ap.add_argument("--no-show", action="store_true", help="Text results only, no image window")
    args = ap.parse_args()

    img = Path(args.image)
    if not img.exists():
        sys.exit(f"Image not found: {img}")

    db = Path(args.db)
    top_dir, bottom_dir = db / "top", db / "bottom"
    if not top_dir.is_dir() and not bottom_dir.is_dir():
        print(f"Database folders not found under: {db}")
        print("Create:")
        print(f"  {db}/top/")
        print(f"  {db}/bottom/")
        print("Put clothing photos in those folders (optional matching .json labels).")
        # create empty structure for the user
        top_dir.mkdir(parents=True, exist_ok=True)
        bottom_dir.mkdir(parents=True, exist_ok=True)
        sys.exit(1)

    weight_paths = [p for p in resolve_weights(args.weights, args.all) if Path(p).exists()]
    if not weight_paths:
        print("No trained weights found. Train models first, then re-run search.")
        sys.exit(1)

    print("Models:")
    models = []
    for w in weight_paths:
        print(f"  {w}")
        models.append(YOLO(str(w)))

    print(f"\nReference: {img}")
    query_tags, boxes = detect_reference(models, img, conf=args.conf)
    query_colors = dominant_colors(img, top_k=3)

    print("Detected labels:", ", ".join(f"{k}({v:.2f})" for k, v in sorted(query_tags.items())) or "(none)")
    print("Colors:", ", ".join(c["name"] for c in query_colors) or "(none)")
    if not query_tags:
        print("\nNo labels detected on the reference image. Try lower --conf or different models.")
        sys.exit(1)

    print(f"\nScanning database: {db}")
    items = load_database(db, models if args.scan_db else None, args.conf, args.scan_db)
    # If DB images have no tags and user didn't pass --scan-db, auto-scan
    if items and all(not it["tags"] for it in items):
        print("Database items have no labels – running models on them (--scan-db).")
        items = load_database(db, models, args.conf, run_models_on_db=True)

    print(f"  items indexed: {len(items)}  (top={sum(1 for i in items if i['cls']=='top')}, "
          f"bottom={sum(1 for i in items if i['cls']=='bottom')})")

    ranked = rank_items(query_tags, query_colors, items, args.cls)
    show_results(ranked, args.top, query_tags)

    if not args.no_show and ranked:
        try_open_images(ranked, args.top)


if __name__ == "__main__":
    main()
