"""Convert schema-v1 annotations (one garment box + attribute dict) to v2.

Run once after upgrading::

    python -m annotator.migrate_v1_to_v2

The original ``.json`` files are copied to ``*.v1.bak`` before being replaced.
Each old garment box becomes:
  - one ``<garment>_garment`` box,
  - one region box per attribute that had a value,
  - left/right sleeve boxes (the old box is split vertically),
  - an HSV-detected colour box (editable later in the annotator).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from annotator.store import AnnotationStore, Box, ImageAnnotation, IMAGE_SUFFIXES, resolve_annotation_dirs
from common.colors import dominant_color_name
from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.schema import Schema, load_schema
from PIL import Image

LOG = LOGGER.getChild("annotator.migrate")


def convert_annotation(data: dict[str, Any], sc: Schema, image_path: Path) -> ImageAnnotation:
    """Convert one v1 annotation dict to a v2 :class:`ImageAnnotation`."""
    with Image.open(image_path) as src:
        image = src.convert("RGB")
    boxes: list[Box] = []
    for old in data.get("boxes", []):
        garment = str(old.get("label", ""))
        if garment not in sc.garment_names:
            LOG.warning("Skipping unknown garment %r in %s", garment, image_path.name)
            continue
        xyxy = tuple(int(round(float(v))) for v in old["xyxy"])
        boxes.append(Box.create(garment, "garment", "center", garment, xyxy))

        x1, y1, x2, y2 = xyxy
        mid = (x1 + x2) // 2
        values = dict(old.get("attributes") or {})
        for region in sc.regions(garment):
            value = values.get(region.name)
            if value not in region.classes:
                LOG.warning("%s: %s.%s=%r invalid; skipping", image_path.name, garment, region.name, value)
                continue
            for side in region.sides:
                if region.paired and side == "right":
                    region_xyxy = (mid, y1, x2, y2)
                elif region.paired and side == "left":
                    region_xyxy = (x1, y1, mid, y2)
                elif region.name == "neck":
                    w = x2 - x1
                    region_xyxy = (x1 + w // 4, y1, x1 + 3 * w // 4, y1 + max(8, (y2 - y1) // 4))
                else:
                    region_xyxy = xyxy
                boxes.append(Box.create(garment, region.name, side, value, region_xyxy))

        crop = image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))
        color = dominant_color_name(crop, set(sc.color_classes))
        if color is not None:
            boxes.append(Box.create(garment, "color", "center", color, xyxy))

    return ImageAnnotation(image=image_path.name, width=int(data.get("width", image.width)),
                           height=int(data.get("height", image.height)), boxes=boxes)


def migrate_v1_to_v2(
    config: Config | None = None,
    schema: Schema | None = None,
    *,
    images_dir: str | Path | None = None,
    annotations_dir: str | Path | None = None,
    backup: bool = True,
) -> dict[str, int]:
    """Convert every v1 annotation in place; return a small summary."""
    cfg = config or load_config()
    sc = schema or load_schema()
    images, annotations = resolve_annotation_dirs(cfg, images_dir, annotations_dir)
    store = AnnotationStore(images, annotations, sc)
    counts = {"converted": 0, "skipped": 0, "backed_up": 0}

    for name, data in list(store.images.items()):
        image_path = images / name
        if not image_path.exists():
            counts["skipped"] += 1
            continue
        if int(data.get("version", 1)) >= 2 or (data.get("boxes") and "garment" in data["boxes"][0]):
            counts["skipped"] += 1  # already v2
            continue
        if backup:
            backup_path = annotations / f"{Path(name).stem}.json.v1.bak"
            if not backup_path.exists():
                backup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                counts["backed_up"] += 1
        annotation = convert_annotation(data, sc, image_path)
        store.save(annotation)
        counts["converted"] += 1

    LOG.info("Migration finished: %s", counts)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Migrate v1 annotations to schema v2")
    parser.add_argument("--images", type=str, default=None)
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)
    print(migrate_v1_to_v2(images_dir=args.images, annotations_dir=args.annotations, backup=not args.no_backup))


if __name__ == "__main__":
    main()
