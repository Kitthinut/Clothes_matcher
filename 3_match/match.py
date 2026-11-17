"""Weighted closet search (3_match/match.py).

Pipeline (the whole idea of the project):
  reference outfit image
    -> garment box (GrabCut auto-box, falls back to the full image)
    -> model predicts attribute labels + visual embedding
    -> HSV detects dominant colours
    -> score EVERY item in your closet (closet/closet.json) with a
       weighted mix of  attribute similarity + colour overlap + embedding
       cosine similarity
    -> ranked "wear this with it" list.

Weights (spec section 7 — weighted search):
  * component weights:  --w-attr  --w-color  --w-emb  (defaults 0.5/0.2/0.3)
  * per-attribute weights come from the schema (optional "weight" field in
    shared/attributes_*.json, default 1.0) and can be raised per run, e.g.
    --w-neck 2.0 makes neck style matter twice as much.
  Multi-select attributes (pattern) score by set overlap, not equality, and
  missing information on either side simply drops out of the weighted sum
  (weights renormalise over what is known).

Used as a module by search_app.py, or directly:
  python 3_match/match.py reference.jpg --garment top --backend pytorch
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.colors import auto_box, detect_colors  # noqa: E402
from shared.schema import weights as schema_weights  # noqa: E402

CLOSET = ROOT / "closet"
# component defaults: attributes dominate, embedding breaks ties, colour ties in
DEFAULT_W = {"attr": 0.5, "color": 0.2, "emb": 0.3}


def backend_module(backend, garment):
    """Import the right inference module per backend (lazy: only one loads)."""
    if backend == "pytorch":
        sys.path.insert(0, str(ROOT / "2_train" / "pytorch"))
        import inference as inf  # noqa: E402
        weights = ROOT / "weights" / f"{garment}_pytorch.pt"
    elif backend == "tensorflow":
        sys.path.insert(0, str(ROOT / "2_train" / "tensorflow"))
        import inference as inf  # noqa: E402
        weights = ROOT / "weights" / f"{garment}_tensorflow.keras"
    else:
        raise SystemExit("backend must be 'pytorch' or 'tensorflow'")
    if not weights.exists():
        raise SystemExit(f"No trained model at {weights}.\n"
                         f"Train first:  python 2_train/{backend}/train.py "
                         f"--garment {garment}")
    bundle, spec = inf.load_model(weights)
    return inf, bundle, spec


# ---------------------------------------------------------------------------
# similarity pieces
# ---------------------------------------------------------------------------

def _attr_piece(ref_v, item_v):
    """One attribute: 1.0 exact / set overlap for multi-select / None unknown."""
    if ref_v is None or item_v is None:
        return None
    if isinstance(ref_v, list) or isinstance(item_v, list):
        a = set(ref_v if isinstance(ref_v, list) else [ref_v])
        b = set(item_v if isinstance(item_v, list) else [item_v])
        if not a or not b:
            return None
        return len(a & b) / len(a | b)  # Jaccard overlap
    return 1.0 if ref_v == item_v else 0.0


def attr_score(ref, item, garment, overrides=None):
    """Weighted attribute similarity (0..1) — per-attribute weights from the
    schema, optionally boosted by `overrides` {key: weight}."""
    weights = schema_weights(garment)
    overrides = overrides or {}
    num = den = 0.0
    for key, w in weights.items():
        w *= overrides.get(key, 1.0)
        piece = _attr_piece(ref["attributes"].get(key),
                            item["attributes"].get(key))
        if piece is None:
            continue  # unknown on either side -> out of the weighted sum
        num += w * piece
        den += w
    return num / den if den else 0.0


def color_score(ref_colors, item_colors, top=2):
    """Overlap of the top-N dominant colour names / N  (0..1)."""
    a = {c["name"] for c in ref_colors[:top]}
    b = {c["name"] for c in item_colors[:top]}
    return len(a & b) / max(len(a | b), 1)


def emb_score(ref_emb, item_emb):
    """Cosine similarity mapped to 0..1. Embeddings are only comparable when
    they come from the same model family (dims must agree)."""
    if ref_emb is None or item_emb is None:
        return None
    a, b = np.asarray(ref_emb, np.float32), np.asarray(item_emb, np.float32)
    if a.shape != b.shape:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return None
    return float((np.dot(a, b) / denom + 1.0) / 2.0)


def combine(parts):
    """{name: (weight, score-or-None)} -> weighted mean over known pieces."""
    num = den = 0.0
    for w, s in parts.values():
        if s is None:
            continue
        num += w * s
        den += w
    return num / den if den else 0.0


def score_match(ref, item, garment, w=None, overrides=None):
    """Full weighted score. `ref` carries 'attributes'/'colors'/'embedding';
    `item` is a closet.json entry (+ its loaded embedding)."""
    w = {**DEFAULT_W, **(w or {})}
    attr = attr_score(ref, item, garment, overrides)
    color = color_score(ref["colors"], item.get("colors", []))
    emb = emb_score(ref.get("embedding"), item.get("_emb"))
    parts = {"attr": (w["attr"], attr), "color": (w["color"], color),
             "emb": (w["emb"], emb)}
    return combine(parts), attr, color, emb


# ---------------------------------------------------------------------------
# reference analysis + search
# ---------------------------------------------------------------------------

def analyze_reference(reference_path, backend, garment):
    """Reference image -> {'attributes', 'colors', 'embedding', 'box'}."""
    inf, bundle, spec = backend_module(backend, garment)
    img = Image.open(reference_path).convert("RGB")
    box = auto_box(img)  # outfit photos: crop to the garment of interest
    probs, labels = inf.predict_pil(bundle, spec, img, box=box)
    emb = inf.embed_pil(bundle, img, box=box)
    return {"attributes": labels, "colors": detect_colors(img.crop(tuple(box))
            if box else img), "embedding": emb, "box": box}


def search(ref, garment, closet_dir=None, w=None, overrides=None, top_n=20):
    """Rank every closet item of this garment against the reference."""
    closet_dir = Path(closet_dir) if closet_dir else CLOSET
    closet_json = closet_dir / "closet.json"
    if not closet_json.exists():
        raise SystemExit(f"No closet at {closet_json}.\n"
                         "Build it first:  python closet/add_to_closet.py "
                         "--add <folder> --garment " + garment)
    closet = json.loads(closet_json.read_text(encoding="utf-8"))

    scored = []
    for item in closet["items"]:
        if item["category"] != garment:
            continue
        emb = None
        emb_file = item.get("embedding", {}).get("file")
        if emb_file and (closet_dir / emb_file).exists():
            emb = np.load(closet_dir / emb_file)
        scored.append({**item, "_emb": emb})
    if not scored:
        raise SystemExit(f"Closet has no '{garment}' items yet.")

    results = []
    for item in scored:
        s, attr, color, emb = score_match(ref, item, garment, w, overrides)
        results.append({**item, "score": s, "attr_score": attr,
                        "color_score": color, "emb_score": emb})
    results.sort(key=lambda d: -d["score"])
    return results[:top_n]


def find_matches(reference_path, garment, backend, closet_dir=None,
                 w=None, overrides=None, top_n=20):
    """Full pipeline: reference image -> ranked closet items (printed)."""
    ref = analyze_reference(reference_path, backend, garment)
    print("Reference labels:", ref["attributes"])
    print("Reference colors:", [c["name"] for c in ref["colors"]],
          "| box:", ref["box"] or "full image")

    results = search(ref, garment, closet_dir, w, overrides, top_n)
    print(f"\nTop {len(results)} matches from your closet:")
    for it in results:
        labels = ", ".join(f"{k}={v}" for k, v in it["attributes"].items())
        cols = ", ".join(c["name"] for c in it["colors"][:2])
        emb_txt = f"{it['emb_score']:.2f}" if it["emb_score"] is not None else " n/a"
        print(f"  {it['score']:.2f} | attr {it['attr_score']:.2f} | "
              f"color {it['color_score']:.2f} | emb {emb_txt} | "
              f"{it['file']} | {labels} | [{cols}]")
    return ref, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", nargs="?", default=None,
                    help="reference outfit image")
    ap.add_argument("--garment", choices=("top", "bottom"), default="top")
    ap.add_argument("--backend", choices=("pytorch", "tensorflow"),
                    default="pytorch")
    ap.add_argument("--closet", default=None, help="default: closet/")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--w-attr", type=float, default=DEFAULT_W["attr"])
    ap.add_argument("--w-color", type=float, default=DEFAULT_W["color"])
    ap.add_argument("--w-emb", type=float, default=DEFAULT_W["emb"])
    ap.add_argument("--show-board", action="store_true",
                    help="open an outfit-board window (reference + matches)")
    ap.add_argument("--save-board", default=None, metavar="DIR",
                    help="save outfit_board.png into this folder")
    a, unknown = ap.parse_known_args()
    if not a.reference:
        ap.error("give a reference image, e.g.  python 3_match/match.py ref.jpg")

    # any extra flag --w-<attribute> boosts that attribute, e.g. --w-neck 2
    overrides, rest = {}, iter(unknown)
    for tok in rest:
        if tok.startswith("--w-"):
            overrides[tok[4:]] = float(next(rest, 1.0))
        else:
            ap.error(f"unknown argument: {tok}")

    w = {"attr": a.w_attr, "color": a.w_color, "emb": a.w_emb}
    ref, results = find_matches(a.reference, a.garment, a.backend, a.closet,
                                w, overrides, a.top_n)

    if a.save_board or a.show_board:
        from board import build_board, save_board, show_board
        closet_dir = Path(a.closet) if a.closet else CLOSET
        board = build_board(Image.open(a.reference).convert("RGB"),
                            ref["attributes"], ref["colors"], results,
                            closet_dir, title=f"Outfit suggestions — {a.garment}")
        if a.save_board:
            print("Board ->", save_board(board, a.save_board))
        if a.show_board:
            show_board(board)


if __name__ == "__main__":
    main()
