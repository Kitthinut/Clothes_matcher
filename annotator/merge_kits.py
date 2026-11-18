"""Validate and merge annotation kits returned by label helpers.

A returned kit contains::

    kit_<name>/
      images/*.jpg              your friends' own pictures
      annotations.json          ONE file with all region boxes
      README.txt                who labelled it, when, notes (optional)

Run::

    python -m annotator.merge_kits <kit_folder> --check      # validate only
    python -m annotator.merge_kits <kit_folder>              # copy into project
    python -m annotator.merge_kits <kit_folder> --force      # overwrite conflicts
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from annotator.store import DATA_FILENAME, IMAGE_SUFFIXES, AnnotationStore, ImageAnnotation, resolve_annotation_dirs
from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.schema import Schema, load_schema

LOG = LOGGER.getChild("annotator.merge_kits")


def _subdir(kit: Path, name: str) -> Path:
    candidate = kit / name
    return candidate if candidate.is_dir() else kit


def _data_path(kit: Path) -> Path:
    for candidate in (kit / DATA_FILENAME, kit / "annotations" / DATA_FILENAME):
        if candidate.is_file():
            return candidate
    return kit / DATA_FILENAME


def inspect_kit(kit: str | Path, schema: Schema | None = None) -> dict[str, Any]:
    """Validate a returned kit without copying anything."""
    sc = schema or load_schema()
    root = Path(kit).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kit folder not found: {root}")

    image_dir = _subdir(root, "images")
    data_path = _data_path(root)
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    report: dict[str, Any] = {
        "kit": str(root), "data_file": str(data_path), "images_found": len(images),
        "valid": [], "missing_json": [], "empty_annotations": [],
        "problems": [], "extra_labels": [],
    }
    if not data_path.is_file():
        report["missing_json"] = [p.name for p in images]
        report["problems"].append(f"No {DATA_FILENAME} found in the kit")
        return report

    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        entries = dict(payload.get("images") or {})
    except (ValueError, OSError, TypeError) as exc:
        report["problems"].append(f"Could not read {DATA_FILENAME}: {exc}")
        report["missing_json"] = [p.name for p in images]
        return report

    for image in images:
        if image.name not in entries:
            report["missing_json"].append(image.name)
            continue
        try:
            ann = ImageAnnotation.from_dict(entries[image.name])
            ann.validate(sc)
        except (ValueError, KeyError, TypeError) as exc:
            report["problems"].append(f"{image.name}: {exc}")
            continue
        for garment in sc.garment_names:
            if not ann.boxes_for(garment):
                continue
            if not ann.garment_box(garment):
                report["problems"].append(f"{image.name}: missing {garment}_garment box")
            for region in sc.regions(garment):
                for side in region.sides:
                    found = [b for b in ann.boxes
                             if b.garment == garment and b.region == region.name and b.side == side]
                    if not found:
                        report["problems"].append(f"{image.name}: missing {garment}.{region.name} ({side})")
        if not ann.boxes:
            report["empty_annotations"].append(image.name)
        report["valid"].append(image.name)

    image_names = {p.name for p in images}
    report["extra_labels"] = sorted(name for name in entries if name not in image_names)
    return report


def _same_bytes(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def merge_kit(
    kit: str | Path,
    config: Config | None = None,
    schema: Schema | None = None,
    *,
    images_dir: str | Path | None = None,
    annotations_dir: str | Path | None = None,
    force: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    """Validate *kit* and (unless ``check_only``) copy it into the project."""
    cfg = config or load_config()
    sc = schema or load_schema()
    report = inspect_kit(kit, sc)
    if check_only:
        report["copied_images"] = 0
        report["copied_labels"] = 0
        return report

    root = Path(kit).resolve()
    image_src = _subdir(root, "images")
    data_src = Path(report["data_file"])
    images_dest, annotations_dest = resolve_annotation_dirs(cfg, images_dir, annotations_dir)
    images_dest.mkdir(parents=True, exist_ok=True)
    annotations_dest.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0
    skipped: list[str] = []
    conflicts: list[str] = []

    for name in report["valid"]:
        src = next(
            p for p in image_src.iterdir()
            if p.name == name and p.suffix.lower() in IMAGE_SUFFIXES
        )
        dest = images_dest / src.name
        if dest.exists() and not _same_bytes(src, dest):
            conflicts.append(src.name)
            if not force:
                skipped.append(src.name)
                continue
        if not _same_bytes(src, dest):
            shutil.copy2(src, dest)
            copied_images += 1

    # Copy the single annotations.json (or merge its entries into an existing one).
    data_dest = annotations_dest / DATA_FILENAME
    if data_src.is_file():
        incoming = json.loads(data_src.read_text(encoding="utf-8")).get("images") or {}
        changed = False
        if data_dest.exists():
            existing = json.loads(data_dest.read_text(encoding="utf-8")).get("images") or {}
            for key, value in incoming.items():
                if key in existing and existing[key] != value:
                    conflicts.append(key)
                    if not force:
                        skipped.append(key)
                        continue
                    existing[key] = value
                    changed = True
                elif key not in existing:
                    existing[key] = value
                    changed = True
            if changed:
                merged = {"schema_version": 2, "images": existing}
                data_dest.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            shutil.copy2(data_src, data_dest)
            changed = True
        copied_labels = 1 if changed else 0

    readme = root / "README.txt"
    if readme.exists():
        target = annotations_dest / f"{root.name}_README.txt"
        if force or not target.exists():
            shutil.copy2(readme, target)

    report.update({
        "copied_images": copied_images,
        "copied_labels": copied_labels,
        "skipped": sorted(set(skipped)),
        "conflicts": sorted(set(conflicts)),
        "dest_images": str(images_dest),
        "dest_annotations": str(annotations_dest),
    })
    LOG.info("Merged kit %s: %d images, annotations.json", root, copied_images)
    return report


def format_report(report: dict[str, Any]) -> str:
    """Return a compact human-readable summary."""
    lines = [
        f"Kit: {report['kit']}",
        f"Images found: {report['images_found']}  valid: {len(report['valid'])}",
    ]
    for key, label in (
        ("missing_json", "Missing from annotations.json"),
        ("empty_annotations", "Empty annotations"),
        ("extra_labels", "Annotations without an image"),
        ("problems", "Problems"),
    ):
        values = report.get(key) or []
        if values:
            lines.append(f"  {label}: {', '.join(str(v) for v in values[:20])}")
    if "copied_images" in report:
        lines.append(
            f"Copied {report['copied_images']} images and annotations.json into "
            f"{report.get('dest_images')} / {report.get('dest_annotations')}"
        )
        if report.get("conflicts"):
            lines.append("  Conflicts (kept existing unless --force): " + ", ".join(report["conflicts"][:20]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and merge a returned annotation kit")
    parser.add_argument("kit", help="Folder containing the returned kit")
    parser.add_argument("--check", action="store_true", help="Validate only; do not copy")
    parser.add_argument("--force", action="store_true", help="Overwrite conflicting files")
    parser.add_argument("--images", type=str, default=None, help="Destination image folder")
    parser.add_argument("--annotations", type=str, default=None, help="Destination annotation folder")
    args = parser.parse_args(argv)

    report = merge_kit(
        args.kit, force=args.force, check_only=args.check,
        images_dir=args.images, annotations_dir=args.annotations,
    )
    print(format_report(report))
    if report["missing_json"] or report["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
