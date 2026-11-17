"""Shared annotation helpers (shared/data.py) — annotation schema v3.

One annotations.json per dataset folder / friend kit. Structure:

{
  "schema_version": 3,
  "garment": "top" | "bottom",
  "image_root": "images",
  "target_size": [288, 384],                 # box coordinate size (W, H)
  "images": {
    "photo1.jpg": {
      "width": 600, "height": 800,           # original image dimensions
      "regions": [
        {
          "category": "garment",
          "box": [x1, y1, x2, y2],           # px in target_size coordinates
          "attributes": {"length": "cropped", "pattern": ["stripe"]}
        },
        {
          "category": "sleeves", "side": "left",
          "box": [x1, y1, x2, y2],
          "attributes": {"sleeves": "long"}
        }
      ]
    }
  }
}

Design rules:
- Region boxes use target_size coordinates and are scaled back to original
  image pixels by load_annotations() for the existing trainer.
- New attributes can be added to attributes_<garment>.json without touching
  this format; loaders simply ignore keys they do not know.
- Legacy flat-format files (schema v1: image-level "attributes"/"colors")
  are auto-upgraded in memory by load_annotations() so old scripts keep
  working, but v1 files are NOT written anywhere anymore.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path(__file__).resolve().parent
for p in (ROOT, SHARED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import load_spec  # noqa: E402  (sibling module, path set above)

SCHEMA_VERSION = 3
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def new_image_id(rel_name):
    """Stable, filename-derived id: '<stem>-<first 6 hex of md5(path)>'."""
    rel_name = str(rel_name)
    h = hashlib.md5(rel_name.lower().encode("utf-8")).hexdigest()[:6]
    return f"{Path(rel_name).stem}-{h}"


def normalize_attributes(attrs, spec):
    """Ensure single categories are str and multi categories are sorted lists.
    Values not in the schema are dropped and reported in `dropped`
    ({key: [invalid values]})."""
    clean, dropped = {}, {}
    for key, value in (attrs or {}).items():
        if key not in spec:
            dropped[key] = value if isinstance(value, list) else [value]
            continue
        opts = set(spec[key]["options"])
        if spec[key]["mode"] == "multi":
            vals = value if isinstance(value, list) else [value]
            keep = sorted(set(v for v in vals if v in opts))
            bad = [v for v in vals if v not in opts]
            if keep:
                clean[key] = keep
            elif isinstance(value, list) and not value:
                clean[key] = []
            if bad:
                dropped.setdefault(key, []).extend(bad)
        else:
            if value in opts:
                clean[key] = value
            else:
                dropped.setdefault(key, []).append(value)
    return clean, dropped


def upgrade_legacy_image(info, garment):
    """v1 flat image entry -> v2 entry (in memory). Returns None if empty."""
    attrs = info.get("attributes") or {}
    if not attrs and not info.get("garments"):
        return None
    entry = dict(info)
    if info.get("garments"):
        return entry  # already v2
    colors = info.get("colors") or info.get("analysis", {}).get("colors")
    g = {"category": garment, "box": None, "attributes": attrs, "attribute_crops": {}}
    if colors:
        g["colors"] = colors
    entry["garments"] = [g]
    return entry


def load_annotations(ann_path, image_root=None, spec_override=None):
    """annotations.json -> one training sample per image/garment.

    Each sample: {rel, path, image_id, category, box (or None),
                  attributes, attribute_crops, colors, width, height, annotator}.
    Region-format annotations are combined into the same sample shape expected
    by both trainers. Missing image paths fall back to an rglob search.
    """
    ann_path = Path(ann_path)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    garment = data.get("garment", "top")
    spec = spec_override or load_spec(garment)
    root = Path(image_root) if image_root else ann_path.parent / data.get("image_root", "images")

    samples = []
    for rel, info in data.get("images", {}).items():
        path = root / rel
        if not path.is_file():
            matches = list(root.rglob(Path(rel).name))
            if not matches:
                continue
            path = matches[0]

        if info.get("regions"):
            target_size = info.get("target_size") or data.get("target_size")
            original_size = (info.get("width"), info.get("height"))
            if not all(original_size):
                from PIL import Image

                with Image.open(path) as image:
                    original_size = image.size

            def to_original(box):
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    return None
                if not target_size or len(target_size) != 2:
                    return list(box)
                target_w, target_h = target_size
                width, height = original_size
                if not target_w or not target_h:
                    return list(box)
                return [
                    round(box[0] * width / target_w),
                    round(box[1] * height / target_h),
                    round(box[2] * width / target_w),
                    round(box[3] * height / target_h),
                ]

            attributes, attribute_crops = {}, {}
            box, colors = None, []
            used_sleeve_sides = set()
            crop_keys = {
                key for key, cat in spec.items() if cat.get("crop_boxes")
            }
            for region in info["regions"]:
                region_attrs = region.get("attributes") or {}
                category = region.get("category")
                crop_key = (
                    category
                    if category in crop_keys
                    else next(
                        (key for key in crop_keys if key in region_attrs),
                        None,
                    )
                )
                region_box = to_original(region.get("box"))
                if crop_key:
                    if crop_key in region_attrs:
                        attributes[crop_key] = region_attrs[crop_key]
                    if region_box:
                        if crop_key == "sleeves":
                            side = region.get("side")
                            if side not in ("left", "right"):
                                side = next(
                                    (candidate for candidate in ("left", "right")
                                     if candidate not in used_sleeve_sides),
                                    "right",
                                )
                            attribute_crops.setdefault("sleeves", {})[side] = region_box
                            used_sleeve_sides.add(side)
                        else:
                            attribute_crops[crop_key] = region_box
                    continue

                attributes.update(region_attrs)
                if box is None:
                    box = region_box
                colors = region.get("shirt_colors") or region.get("colors") or colors

            attrs, _ = normalize_attributes(attributes, spec)
            if attrs or box:
                samples.append({
                    "rel": rel,
                    "path": str(path),
                    "image_id": info.get("image_id") or new_image_id(rel),
                    "category": garment,
                    "box": box,
                    "attributes": attrs,
                    "attribute_crops": attribute_crops,
                    "colors": colors,
                    "width": original_size[0],
                    "height": original_size[1],
                    "annotator": info.get("annotator", data.get("annotator", "")),
                })
            continue

        entry = upgrade_legacy_image(info, garment) if not info.get("garments") else info
        if not entry or not entry.get("garments"):
            continue
        for g in entry["garments"]:
            attrs, _ = normalize_attributes(g.get("attributes") or {}, spec)
            if not attrs and not g.get("box"):
                continue
            samples.append({
                "rel": rel,
                "path": str(path),
                "image_id": entry.get("image_id") or new_image_id(rel),
                "category": g.get("category", garment),
                "box": g.get("box"),
                "attributes": attrs,
                "attribute_crops": g.get("attribute_crops") or {},
                "colors": g.get("colors") or [],
                "width": entry.get("width"),
                "height": entry.get("height"),
                "annotator": g.get("annotator", data.get("package_id", "")),
            })
    return samples


def load_split(path):
    """Load a split file written by shared/split.py -> list of samples.
    Split files store paths relative to dataset/<garment>/ (portable), so they
    are resolved against that folder when the relative path does not exist."""
    path = Path(path)
    out = []
    for s in json.loads(path.read_text(encoding="utf-8")).get("samples", []):
        if not s.get("attributes"):
            continue
        s = dict(s)
        p = Path(s["path"])
        if not p.is_absolute() and not p.exists():
            candidate = path.parent.parent / s["path"]  # splits/*.json -> dataset/<garment>/
            if candidate.exists():
                p = candidate
        s["path"] = str(p)
        out.append(s)
    return out


def encode_labels(attrs, spec):
    """Attributes -> per-head targets for the loss.

    single: class index (int); missing: -1 (ignored in the loss).
    multi:  list of class indices; empty list = explicit "none of these"
            (a valid all-zero target, NOT missing). Missing: -1 (int).
    """
    labels = {}
    for key, cat in spec.items():
        value = attrs.get(key)
        if cat["mode"] == "multi":
            if value is None:
                labels[key] = -1
            else:
                vals = value if isinstance(value, list) else [value]
                labels[key] = [cat["options"].index(v) for v in vals if v in cat["options"]]
        else:
            labels[key] = cat["options"].index(value) if value in cat["options"] else -1
    return labels


def counts_per_option(samples, spec):
    """How many samples carry each option - useful to spot missing labels."""
    counts = {key: {o: 0 for o in cat["options"]} for key, cat in spec.items()}
    for s in samples:
        for key, value in s["attributes"].items():
            if key not in counts:
                continue
            for v in (value if isinstance(value, list) else [value]):
                if v in counts[key]:
                    counts[key][v] += 1
    return counts


# ---------------------------------------------------------------------------
# Validation (used by merge_annotations.py and the pre-check tooling)
# ---------------------------------------------------------------------------

def validate_garment_annotation(g, spec, width, height):
    """Validate one garment entry -> list of error strings (empty = valid)."""
    errors = []
    box = g.get("box")
    if box is not None:
        if len(box) != 4:
            errors.append(f"box must have 4 coordinates, got {box}")
        elif width and height:
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append(f"box {box} outside/out of order for image {width}x{height}")
    for key, value in (g.get("attributes") or {}).items():
        if key not in spec:
            errors.append(f"unknown attribute key '{key}'")
            continue
        opts = spec[key]["options"]
        vals = value if isinstance(value, list) else [value]
        for v in vals:
            if v not in opts:
                errors.append(f"invalid value '{v}' for '{key}' (valid: {opts})")
    def is_box(value):
        return isinstance(value, (list, tuple)) and len(value) == 4

    for key, crop in (g.get("attribute_crops") or {}).items():
        if key not in spec:
            errors.append(f"unknown attribute-crop key '{key}'")
        elif spec[key].get("crop_boxes") == 2:
            if is_box(crop):
                continue  # legacy single sleeve box
            if not isinstance(crop, dict) or not crop:
                errors.append(f"attribute crop for '{key}' must contain left/right boxes")
                continue
            for side, box in crop.items():
                if side not in ("left", "right"):
                    errors.append(f"attribute crop for '{key}' has unknown side '{side}'")
                elif not is_box(box):
                    errors.append(f"attribute crop for '{key}.{side}' must have 4 coordinates")
        elif not is_box(crop):
            errors.append(f"attribute crop for '{key}' must have 4 coordinates")
    return errors


def validate_region_annotation(region, spec, width, height):
    """Validate one target-size region entry."""
    errors = []
    category = region.get("category")
    attrs = region.get("attributes") or {}
    crop_keys = {key for key, cat in spec.items() if cat.get("crop_boxes")}
    if category not in (None, "garment", *crop_keys):
        errors.append(f"unknown region category {category!r}")
    if not isinstance(attrs, dict):
        errors.append("attributes must be an object")
        attrs = {}
    if not attrs:
        errors.append("region has no attributes")

    for key, value in attrs.items():
        if key not in spec:
            errors.append(f"unknown attribute key '{key}'")
            continue
        vals = value if isinstance(value, list) else [value]
        for val in vals:
            if val not in spec[key]["options"]:
                errors.append(
                    f"invalid value '{val}' for '{key}' "
                    f"(valid: {spec[key]['options']})"
                )

    crop_labels = [key for key in crop_keys if key in attrs]
    inferred_crop = crop_labels[0] if crop_labels else None
    if len(crop_labels) > 1:
        errors.append("a region can contain only one neck/sleeves type label")
    if category == "garment" and crop_labels:
        errors.append(f"{crop_labels[0]} labels must use a separate region")
    effective_category = category or inferred_crop or "garment"
    if effective_category in crop_keys and effective_category not in attrs:
        errors.append(f"{effective_category} region must include its type label")
    if effective_category == "sleeves" and region.get("side") not in (None, "left", "right"):
        errors.append(f"unknown sleeve side {region.get('side')!r}")
    if effective_category != "sleeves" and region.get("side") is not None:
        errors.append("side is only valid for a sleeves region")

    box = region.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        errors.append(f"box must have 4 coordinates, got {box}")
    elif not all(isinstance(value, (int, float)) for value in box):
        errors.append(f"box coordinates must be numbers, got {box}")
    elif width and height:
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors.append(f"box {box} outside/out of order for image {width}x{height}")
    return errors


def validate_annotations(data, garment):
    """Validate a parsed annotations.json dict -> (errors, warnings).
    Never raises; merge_annotations.py aggregates the results."""
    errors, warnings = [], []
    spec = load_spec(garment)
    if data.get("schema_version") not in (None, 1, 2, 3):
        warnings.append(f"unknown schema_version {data.get('schema_version')!r}")
    for rel, info in data.get("images", {}).items():
        if "regions" in info:
            regions = info.get("regions")
            if not isinstance(regions, list) or not regions:
                errors.append(f"{rel}: no garment annotations")
                continue
            image_w, image_h = info.get("width"), info.get("height")
            if (image_w is None) != (image_h is None):
                errors.append(f"{rel}: width/height must both be set or both absent")
            elif image_w is not None and not (
                0 < image_w <= 100000 and 0 < image_h <= 100000
            ):
                errors.append(f"{rel}: suspicious image size {image_w}x{image_h}")
            target_size = info.get("target_size") or data.get("target_size")
            if target_size is not None:
                if (
                    not isinstance(target_size, (list, tuple))
                    or len(target_size) != 2
                    or any(not isinstance(value, (int, float)) or value <= 0
                           for value in target_size)
                ):
                    errors.append(f"{rel}: invalid target_size {target_size!r}")
                    target_w = target_h = None
                else:
                    target_w, target_h = target_size
            else:
                target_w, target_h = image_w, image_h
            if target_w is None and any(
                isinstance(region, dict) and region.get("box") for region in regions
            ):
                errors.append(f"{rel}: target size missing; cannot verify region boxes")
            categories, sleeve_sides = set(), set()
            for index, region in enumerate(regions):
                if not isinstance(region, dict):
                    errors.append(f"{rel}: region[{index}] must be an object")
                    continue
                for error in validate_region_annotation(
                    region, spec, target_w, target_h
                ):
                    errors.append(f"{rel}: region[{index}]: {error}")
                region_attrs = region.get("attributes") or {}
                category = region.get("category") or next(
                    (key for key in spec if spec[key].get("crop_boxes")
                     and key in region_attrs),
                    "garment",
                )
                if category in ("garment", "neck"):
                    if category in categories:
                        errors.append(f"{rel}: more than one {category} region")
                    categories.add(category)
                if category == "sleeves":
                    side = region.get("side")
                    if side in ("left", "right"):
                        if side in sleeve_sides:
                            errors.append(f"{rel}: duplicate {side} sleeve region")
                        sleeve_sides.add(side)
                    elif len(sleeve_sides) >= 2:
                        errors.append(f"{rel}: more than two sleeve regions")
                    else:
                        sleeve_sides.add(f"unspecified-{len(sleeve_sides)}")
            continue

        entry = upgrade_legacy_image(info, garment) or info
        w, h = entry.get("width"), entry.get("height")
        if not entry.get("garments"):
            errors.append(f"{rel}: no garment annotations")
            continue
        if (w is None) != (h is None):
            errors.append(f"{rel}: width/height must both be set or both absent")
        elif w is not None and not (0 < w <= 100000 and 0 < h <= 100000):
            errors.append(f"{rel}: suspicious image size {w}x{h}")
        elif w is None and any(g.get("box") for g in entry["garments"]):
            errors.append(f"{rel}: image dimensions missing; "
                          f"cannot verify garment boxes")
        for i, g in enumerate(entry["garments"]):
            if g.get("category") not in ("top", "bottom"):
                errors.append(f"{rel}: garment[{i}] bad category {g.get('category')!r}")
            for e in validate_garment_annotation(g, spec, w, h):
                errors.append(f"{rel}: garment[{i}]: {e}")
    return errors, warnings
