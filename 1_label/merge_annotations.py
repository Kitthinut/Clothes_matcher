"""Validate and merge annotation packages (1_label/merge_annotations.py).

Replaces merge_kits.py for schema-v3 packages (project spec §12.4). Reads
dataset/friend_kits/*/ (or any package folders passed on the command line),
validates each one, and merges them into dataset/<garment>/annotations.json.

Validation (a failing package is skipped, never silently merged):
  1. JSON parses and matches the schema-v3 structure (v2 remains supported)
  2. every referenced image exists in the package
  3. bounding boxes are inside the stored image dimensions
  4. image dimensions present and sane
  5. attribute labels are valid values from attributes_<garment>.json
  6. duplicate image_ids (within or across packages)
  7. duplicate images by MD5 file hash (same photo sent twice)
  8. conflicting annotations for the same image id -> REVIEW, not overwrite

Outputs:
  - merged dataset/<garment>/annotations.json (annotator kept per garment)
  - dataset/<garment>/merge_report.json (per-package results, conflicts,
    duplicates) + a console summary.

Usage:  python 1_label/merge_annotations.py                  # all kits
        python 1_label/merge_annotations.py --name kawfang   # one kit
        python 1_label/merge_annotations.py D:/some_package  # ad-hoc folder
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.data import SCHEMA_VERSION, new_image_id, validate_annotations  # noqa: E402
from shared.schema import load_spec  # noqa: E402

KITS = ROOT / "dataset" / "friend_kits"


def file_hash(path, chunk=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_image(pkg_dir, image_root, rel):
    """Resolve an image path inside a package (with a filename fallback)."""
    base = pkg_dir / image_root
    direct = base / rel
    if direct.is_file():
        return direct
    matches = list(base.rglob(Path(rel).name))
    return matches[0] if matches else None


def validate_package(pkg_dir, garment):
    """Structural validation of one package -> (data, errors)."""
    errors = []
    ann_path = pkg_dir / "annotations.json"
    if not ann_path.exists():
        return None, [f"no annotations.json"]
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as ex:
        return None, [f"annotations.json is not valid JSON: {ex}"]

    if data.get("schema_version") not in (None, 1, 2, 3):
        errors.append(f"unsupported schema_version {data.get('schema_version')!r}")
    if data.get("schema_version", 2) < 2:
        errors.append("legacy schema (v1) — re-label or upgrade the package "
                      "with 1_label/annotate.py before merging")
    pkg_garment = data.get("garment", garment)
    if pkg_garment != garment:
        errors.append(f"package is for garment '{pkg_garment}', expected '{garment}'")
        return data, errors

    err, warn = validate_annotations(data, garment)
    errors.extend(err)
    warnings = list(warn)

    # every referenced image must exist inside the package
    for rel in data.get("images", {}):
        if not find_image(pkg_dir, data.get("image_root", "images"), rel):
            errors.append(f"referenced image missing: {rel}")

    return data, errors + warnings


def merge_package(pkg_dir, garment, master, seen_ids, seen_hashes, report):
    """Merge one validated package into `master`. Returns (merged, conflicts)."""
    pkg_dir = Path(pkg_dir)
    data, problems = validate_package(pkg_dir, garment)
    if data is None or any("legacy schema" in p or "expected" in p for p in problems):
        report["packages"][pkg_dir.name] = {"status": "skipped", "problems": problems}
        print(f"SKIP {pkg_dir.name}: " + "; ".join(problems or ["no annotations.json"]))
        return 0, []
    if problems:
        report["packages"][pkg_dir.name] = {"status": "skipped", "problems": problems}
        print(f"SKIP {pkg_dir.name} (validation failed):")
        for p in problems:
            print(f"  - {p}")
        return 0, []

    annotator = data.get("annotator", pkg_dir.name)
    prefix = pkg_dir.name.replace(" ", "_")
    dst_img_dir = ROOT / "dataset" / garment / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    # image_id -> master key, for idempotent re-merges and conflict checks
    id_to_master_key = {info.get("image_id"): rel for rel, info in master["images"].items()}

    merged, conflicts = 0, []
    for rel, info in data.get("images", {}).items():
        src = find_image(pkg_dir, data.get("image_root", "images"), rel)
        fhash = file_hash(src)

        # duplicate photo (same bytes) under a different name?
        if fhash in seen_hashes and seen_hashes[fhash] != rel:
            report["duplicates"].append({
                "package": pkg_dir.name, "file": rel,
                "duplicate_of": seen_hashes[fhash], "action": "skipped"})
            print(f"  - duplicate image skipped: {rel} (same file as {seen_hashes[fhash]})")
            continue

        image_id = info.get("image_id") or new_image_id(rel)
        existing_key = id_to_master_key.get(image_id)
        if existing_key:
            existing = master["images"][existing_key]
            existing_annotations = existing.get("regions", existing.get("garments"))
            incoming_annotations = info.get("regions", info.get("garments"))
            if existing_annotations == incoming_annotations:
                print(f"  - already merged, skipped: {rel}")
                continue
            # same image id, different annotations -> conflict, never overwrite
            conflicts.append({"package": pkg_dir.name, "file": rel,
                              "image_id": image_id,
                              "conflicts_with": {"master_key": existing_key},
                              "action": "kept existing — review manually"})
            print(f"  ! CONFLICT for {rel}: annotations differ from {existing_key} "
                  f"— keeping existing, see report")
            continue

        new_name = f"{prefix}_{Path(rel).name}"
        if not (dst_img_dir / new_name).exists():
            shutil.copy2(src, dst_img_dir / new_name)

        info = dict(info)
        if info.get("regions"):
            info.setdefault("target_size", data.get("target_size"))
            info.setdefault("annotator", annotator)
        for g in info.get("garments", []):
            g.setdefault("annotator", annotator)
        info.setdefault("image_id", image_id)
        master["images"][new_name] = info
        seen_ids[image_id] = (pkg_dir.name, new_name)
        seen_hashes[fhash] = new_name
        id_to_master_key[image_id] = new_name
        merged += 1

    report["packages"][pkg_dir.name] = {"status": "merged", "merged": merged,
                                        "conflicts": conflicts}
    print(f"MERGED {pkg_dir.name}: {merged} images -> {garment}")
    return merged, conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="merge one kit only")
    ap.add_argument("--garment", choices=("top", "bottom"), default=None,
                    help="required for ad-hoc package folders")
    ap.add_argument("packages", nargs="*", help="ad-hoc package folders (optional)")
    a = ap.parse_args()

    if a.packages:
        if not a.garment:
            raise SystemExit("Pass --garment with ad-hoc package folders.")
        kits = [Path(p) for p in a.packages]
        garment = a.garment
    else:
        if not KITS.is_dir():
            raise SystemExit(f"No kits found in {KITS}")
        kits = [KITS / a.name] if a.name else sorted(
            d for d in KITS.iterdir() if d.is_dir())
        garments = set()
        for k in kits:
            ann = k / "annotations.json"
            if ann.exists():
                garments.add(json.loads(ann.read_text(encoding="utf-8")).get("garment", "top"))
        if len(garments) > 1:
            raise SystemExit(f"Kits contain mixed garments {garments}; merge per garment.")
        garment = a.garment or (garments.pop() if garments else "top")

    dataset_dir = ROOT / "dataset" / garment
    master_path = dataset_dir / "annotations.json"
    master = json.loads(master_path.read_text(encoding="utf-8")) if master_path.exists() else {
        "schema_version": SCHEMA_VERSION, "garment": garment, "image_root": "images",
        "images": {}}
    master["schema_version"] = SCHEMA_VERSION
    master.setdefault("images", {})

    # existing master content is "seen" so re-merging a kit is idempotent
    seen_ids = {(info.get("image_id") or new_image_id(rel)): ("<master>", rel)
                for rel, info in master["images"].items()}
    seen_hashes = {}

    report = {"garment": garment, "date": str(date.today()),
              "packages": {}, "duplicates": [], "conflicts": []}
    total = 0
    for kit in kits:
        merged, conflicts = merge_package(kit, garment, master, seen_ids, seen_hashes, report)
        total += merged
        report["conflicts"].extend(conflicts)

    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_text(json.dumps(master, indent=2), encoding="utf-8")
    report_path = dataset_dir / "merge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDone. {total} images merged. Master: {master_path}")
    print(f"Merge report: {report_path}")
    if report["conflicts"] or report["duplicates"]:
        print("NOTE: conflicts/duplicates were recorded, never overwritten — "
              "review merge_report.json.")
    print("Next: python shared/split.py --garment " + garment)


if __name__ == "__main__":
    main()
