"""Closet database manager (closet/add_to_closet.py).

Builds and maintains the user's personal clothing index `closet/closet.json`
(project spec section 6). Every added image is processed with the TRAINED
model of its garment category:

  1. load the image
  2. detect its clothing category  (--garment; the classifiers are per-garment)
  3. extract the relevant bounding box (GrabCut auto-box, or the full image)
  4. classify its attributes
  5. extract dominant colours (HSV pipeline, shared/colors.py)
  6. generate a visual feature embedding (penultimate backbone features)
  7. store everything in the closet index

Embeddings live as separate .npy files under closet/embeddings/ (JSON stays
a fast searchable index); each item references its embedding file. Images
are copied into closet/images/ so the closet is self-contained. Exact
duplicates (same MD5) are never added twice, and unchanged files are not
reprocessed (content-hash cache).

Usage:
  python closet/add_to_closet.py --add photo.jpg more.jpg --garment top
  python closet/add_to_closet.py --add "D:/clothes_shirts" --garment top
  python closet/add_to_closet.py --list
  python closet/add_to_closet.py --remove top-3f2a11
  python closet/add_to_closet.py --rebuild            # re-run model on all images
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.colors import auto_box, detect_colors  # noqa: E402
from shared.data import EXTS  # noqa: E402

CLOSET = ROOT / "closet"
CLOSET_JSON = CLOSET / "closet.json"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# index I/O
# ---------------------------------------------------------------------------

def load_closet():
    if CLOSET_JSON.exists():
        data = json.loads(CLOSET_JSON.read_text(encoding="utf-8"))
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("items", [])
        return data
    return {"schema_version": SCHEMA_VERSION, "items": []}


def save_closet(data):
    CLOSET_JSON.parent.mkdir(parents=True, exist_ok=True)
    CLOSET_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def file_hash(path, chunk=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# per-item processing (spec section 6.1)
# ---------------------------------------------------------------------------

def backend(backend_name, garment):
    """Load the trained model for a garment (same checkpoints training made)."""
    if backend_name == "pytorch":
        sys.path.insert(0, str(ROOT / "2_train" / "pytorch"))
        import inference as inf  # noqa: E402
        weights = ROOT / "weights" / f"{garment}_pytorch.pt"
    elif backend_name == "tensorflow":
        sys.path.insert(0, str(ROOT / "2_train" / "tensorflow"))
        import inference as inf  # noqa: E402
        weights = ROOT / "weights" / f"{garment}_tensorflow.keras"
    else:
        raise SystemExit("backend must be 'pytorch' or 'tensorflow'")
    if not weights.exists():
        raise SystemExit(f"No trained model at {weights}.\n"
                         f"Train first: python 2_train/{backend_name}/train.py "
                         f"--garment {garment}")
    bundle, spec = inf.load_model(weights)
    return inf, bundle, spec, weights


def process_image(path, garment, inf, bundle, spec, weights_path,
                  backend_name="pytorch", auto_box_on=True):
    """One image -> closet item dict (+ embedding vector). Never raises:
    problems are returned as {"error": ...} entries."""
    path = Path(path)
    try:
        img = Image.open(path).convert("RGB")
    except Exception as ex:
        return {"error": f"unreadable: {ex}"}

    # 3) bounding box: GrabCut auto-box, falling back to the full image
    box = auto_box(img) if auto_box_on else None

    # 4) attribute classification + confidence
    try:
        probs, labels = inf.predict_pil(bundle, spec, img, box=box)
        confidence = {k: round(max(p.values()), 3) for k, p in probs.items()}
    except Exception as ex:
        return {"error": f"prediction failed: {ex}"}

    # 5) dominant colours (HSV)
    try:
        crop = img.crop(tuple(box)) if box else img
        colors = detect_colors(crop)
    except Exception as ex:
        colors = []
        print(f"  (colour detection skipped: {ex})")

    # 6) visual embedding
    try:
        emb = inf.embed_pil(bundle, img, box=box)
    except Exception as ex:
        return {"error": f"embedding failed: {ex}"}

    source_hash = file_hash(path)
    item_id = f"{garment}-{source_hash[:8]}"
    return {
        "item": {
            "item_id": item_id,
            "category": garment,
            "source_hash": source_hash,
            "box": box,
            "attributes": labels,
            "confidence": confidence,
            "colors": colors,
            "image": {"width": img.width, "height": img.height},
            "embedding": {"dim": int(np.asarray(emb).shape[0]),
                          "backend": backend_name,
                          "dtype": "float32"},
            "model": {"weights": weights_path.name,
                      "spec_keys": sorted(spec)},
            "added": str(date.today()),
        },
        "embedding": np.asarray(emb, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def add_paths(paths, garment, backend_name, auto_box_on=True):
    inf, bundle, spec, weights_path = backend(backend_name, garment)
    closet = load_closet()
    known_hashes = {i["source_hash"] for i in closet["items"]}
    emb_dir = CLOSET / "embeddings"
    img_dir = CLOSET / "images"
    emb_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in EXTS]
        elif p.suffix.lower() in EXTS:
            files.append(p)
        else:
            print(f"skip {p}: not a supported image")
    if not files:
        raise SystemExit("No images found to add.")

    added = skipped = failed = 0
    for f in files:
        fhash = file_hash(f)
        if fhash in known_hashes:
            print(f"= {f.name}: already in the closet (same content)")
            skipped += 1
            continue
        result = process_image(f, garment, inf, bundle, spec, weights_path,
                               backend_name, auto_box_on)
        if "error" in result:
            print(f"! {f.name}: {result['error']}")
            failed += 1
            continue
        item, emb = result["item"], result["embedding"]

        # keep a copy of the image inside the closet (self-contained index)
        dst = img_dir / f"{item['item_id']}{f.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(f, dst)
        item["file"] = f"images/{dst.name}"

        np.save(emb_dir / f"{item['item_id']}.npy", emb)
        item["embedding"]["file"] = f"embeddings/{item['item_id']}.npy"

        closet["items"].append(item)
        known_hashes.add(fhash)
        added += 1
        attrs = ", ".join(f"{k}={v}" for k, v in item["attributes"].items())
        cols = ", ".join(c["name"] for c in item["colors"][:2]) or "-"
        print(f"+ {f.name}: {attrs} | [{cols}] | emb {item['embedding']['dim']}d "
              f"-> {item['item_id']}")

    save_closet(closet)
    print(f"\nAdded {added}, skipped {skipped}, failed {failed}. "
          f"Index: {CLOSET_JSON} ({len(closet['items'])} items total)")


def remove_item(item_id):
    closet = load_closet()
    before = len(closet["items"])
    closet["items"] = [i for i in closet["items"] if i["item_id"] != item_id]
    if len(closet["items"]) == before:
        raise SystemExit(f"No item with id {item_id}")
    save_closet(closet)
    # remove the orphaned embedding + image copy
    for rel in (f"embeddings/{item_id}.npy",):
        p = CLOSET / rel
        if p.exists():
            p.unlink()
    for p in (CLOSET / "images").glob(f"{item_id}.*"):
        p.unlink()
    print(f"Removed {item_id}. {len(closet['items'])} items remain.")


def list_items():
    closet = load_closet()
    if not closet["items"]:
        print("Closet is empty. Add items with: python closet/add_to_closet.py "
              "--add <image|folder> --garment top")
        return
    for i in closet["items"]:
        attrs = ", ".join(f"{k}={v}" for k, v in i["attributes"].items())
        cols = ", ".join(c["name"] for c in i["colors"][:2])
        print(f"{i['item_id']:<16} {i['category']:<7} {attrs} | [{cols}] | {i['file']}")
    print(f"\n{len(closet['items'])} items")


def rebuild(backend_name, auto_box_on=True):
    """Re-run the CURRENT model over every stored closet image (after retraining)."""
    closet = load_closet()
    if not closet["items"]:
        raise SystemExit("Closet is empty.")
    garments = {i["category"] for i in closet["items"]}
    for garment in sorted(garments):
        items = [i for i in closet["items"] if i["category"] == garment]
        inf, bundle, spec, weights_path = backend(backend_name, garment)
        print(f"Re-processing {len(items)} {garment} items with {weights_path.name} ...")
        for item in items:
            src = CLOSET / item["file"]
            if not src.exists():
                print(f"  ! missing image for {item['item_id']} - kept as-is")
                continue
            result = process_image(src, garment, inf, bundle, spec, weights_path,
                                   backend_name, auto_box_on)
            if "error" in result:
                print(f"  ! {item['item_id']}: {result['error']} - kept as-is")
                continue
            new_item, emb = result["item"], result["embedding"]
            item.update({k: new_item[k] for k in
                         ("box", "attributes", "confidence", "colors",
                          "embedding", "model")})
            np.save(CLOSET / "embeddings" / f"{item['item_id']}.npy", emb)
        save_closet(closet)
    print("Rebuild complete (item ids and images unchanged).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=None,
                    help="image file(s) and/or folder(s) to add")
    ap.add_argument("--garment", choices=("top", "bottom"), default="top",
                    help="which trained model processes the added images")
    ap.add_argument("--backend", choices=("pytorch", "tensorflow"),
                    default="pytorch")
    ap.add_argument("--no-auto-box", action="store_true",
                    help="use the whole image instead of the GrabCut garment box")
    ap.add_argument("--remove", default=None, help="remove one item by id")
    ap.add_argument("--list", action="store_true", help="list all items")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-run the current model on every stored image")
    a = ap.parse_args()

    if a.list:
        list_items()
    elif a.remove:
        remove_item(a.remove)
    elif a.rebuild:
        rebuild(a.backend, not a.no_auto_box)
    elif a.add is not None:
        add_paths(a.add, a.garment, a.backend, not a.no_auto_box)
    else:
        ap.error("nothing to do: use --add, --list, --remove or --rebuild")


if __name__ == "__main__":
    main()
