"""Stratified 70/15/15 split (shared/split.py).

Theory: "Stratified sampling" - instead of shuffling blindly, samples are
grouped by their label combination ("buckets"); each bucket is then split
70/15/15 so rare attribute combos still land in every split.

Reads annotations through shared/data.py (legacy v1/v2 and region-based v3),
then writes dataset/<garment>/splits/{train,valid,test}.json with RELATIVE
image paths (portable - no absolute
paths baked in), and stamps the split name back into the master
annotations.json so the split travels with the annotation metadata.

CLI:  python shared/split.py --garment top
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path(__file__).resolve().parent
for p in (ROOT, SHARED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data import load_annotations, SCHEMA_VERSION  # noqa: E402
from schema import load_spec  # noqa: E402

RATIOS = (0.70, 0.15, 0.15)  # train / valid / test
SPLIT_NAMES = ("train", "valid", "test")


def _hashable(value):
    return tuple(value) if isinstance(value, list) else value


def bucket_key(attributes):
    return tuple(sorted((k, _hashable(v)) for k, v in attributes.items()))


def split_samples(samples, seed=42):
    """Stratified 70/15/15 -> dict {split_name: [sample, ...]}."""
    rng = random.Random(seed)

    # bucket key = the label tuple; rare buckets (< 3 samples) merge together
    buckets = defaultdict(list)
    usage = Counter(bucket_key(s["attributes"]) for s in samples)
    for s in samples:
        key = bucket_key(s["attributes"])
        buckets[key if usage[key] >= 3 else ""].append(s)

    train, valid, test = [], [], []
    for members in buckets.values():
        members = members[:]
        rng.shuffle(members)
        n = len(members)
        n_tr = round(n * RATIOS[0])
        n_va = round(n * RATIOS[1]) if n >= 3 else (1 if n == 2 else 0)
        n_tr = min(n_tr, n - n_va)  # never steal from an empty valid set
        train += members[:n_tr]
        valid += members[n_tr:n_tr + n_va]
        test += members[n_tr + n_va:]

    for part in (train, valid, test):
        rng.shuffle(part)
    return dict(zip(SPLIT_NAMES, (train, valid, test)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment", choices=("top", "bottom"), required=True)
    ap.add_argument("--dataset", default=ROOT / "dataset")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    spec = load_spec(a.garment)
    base = Path(a.dataset) / a.garment
    ann_path = base / "annotations.json"
    samples = load_annotations(ann_path)
    if not samples:
        raise SystemExit(f"No labelled samples in {ann_path}")

    parts = split_samples(samples, a.seed)
    image_root = json.loads(ann_path.read_text(encoding="utf-8")).get("image_root", "images")
    out_dir = base / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in parts.items():
        # store rel paths only -> split files stay portable between machines
        for s in items:
            s["path"] = f"{image_root}/{s['rel']}".replace("\\", "/")
        (out_dir / f"{name}.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "garment": a.garment,
                        "split": name, "image_root": image_root,
                        "samples": items}, indent=2),
            encoding="utf-8",
        )
        print(f"{name}: {len(items)} samples -> {out_dir / f'{name}.json'}")

    # stamp the split back into the master annotations (annotation metadata)
    master = json.loads(ann_path.read_text(encoding="utf-8"))
    assigned = {s["rel"]: name for name, items in parts.items() for s in items}
    for rel, info in master.get("images", {}).items():
        info["split"] = assigned.get(rel)
    ann_path.write_text(json.dumps(master, indent=2), encoding="utf-8")

    # quick label-balance report so you can spot classes with almost no data
    for key, cat in spec.items():
        tally = Counter()
        for s in samples:
            v = s["attributes"].get(key)
            for val in (v if isinstance(v, list) else [v] if v else []):
                tally[val] += 1
        print(f"{key}: " + ", ".join(f"{o}={tally.get(o, 0)}" for o in cat["options"]))


if __name__ == "__main__":
    main()
