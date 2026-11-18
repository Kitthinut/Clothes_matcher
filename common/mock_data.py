"""Tiny synthetic v2 dataset for end-to-end smoke tests without real data.

Each image gets region boxes (one per attribute) plus a whole-garment box, like
the real annotator produces. Images may contain only a top, only a bottom, or
both.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from annotator.store import AnnotationStore, Box, ImageAnnotation
from common.colors import dominant_color_name
from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.paths import ensure_dir
from common.schema import Schema, load_schema

LOG = LOGGER.getChild("mock_data")


def _region_box(
    garment: str,
    region: str,
    side: str,
    value: str,
    base: tuple[int, int, int, int],
    rng: random.Random,
    schema: Schema,
) -> Box:
    """Place a region box inside the garment box (sleeves split left/right)."""
    x1, y1, x2, y2 = base
    width, height = x2 - x1, y2 - y1
    if region == "garment":
        return Box.create(garment, "garment", "center", garment, base)
    if region == "sleeves":
        mid = (x1 + x2) // 2
        if side == "left":
            return Box.create(garment, region, side, value, (x1, y1 + height // 4, mid, y2))
        return Box.create(garment, region, side, value, (mid, y1 + height // 4, x2, y2))
    if region == "neck":
        return Box.create(garment, region, side, value, (x1 + width // 4, y1, x1 + 3 * width // 4, y1 + height // 4))
    if region in {"length", "bottom_length", "waist_length"}:
        return Box.create(garment, region, side, value, (x1, y1 + height // 3, x2, y2))
    inset = max(2, width // 8)
    return Box.create(garment, region, side, value, (x1 + inset, y1 + inset, x2 - inset, y2 - inset))


def _draw_image(path: Path, rng: random.Random, schema: Schema) -> ImageAnnotation:
    """Draw one synthetic image with garment regions and save its annotation."""
    width, height = 320, 240
    image = Image.new("RGB", (width, height), (238, 238, 234))
    draw = ImageDraw.Draw(image)
    garments = [g for g in schema.garment_names if rng.random() > 0.2] or [rng.choice(schema.garment_names)]

    boxes: list[Box] = []
    colors = {"top": (60, 110, 190), "bottom": (45, 45, 60)}
    for garment in garments:
        if garment == "top":
            base = (rng.randint(60, 100), rng.randint(25, 45), rng.randint(205, 250), rng.randint(105, 130))
        else:
            base = (rng.randint(80, 115), rng.randint(130, 150), rng.randint(195, 235), rng.randint(200, 228))
        draw.rectangle(base, fill=colors[garment], outline=(20, 20, 20), width=2)
        for offset in range(8, base[2] - base[0], 18):
            draw.line([(base[0] + offset, base[1] + 4), (base[0] + offset, base[3] - 4)], fill=(255, 255, 255), width=1)
        boxes.append(Box.create(garment, "garment", "center", garment, base))

        crop = image.crop(base)
        allowed = set(schema.color_classes)
        detected = dominant_color_name(crop, allowed) or "black"
        for region in schema.regions(garment):
            for side in region.sides:
                if region.source == "hsv":
                    value = detected
                elif region.paired:
                    value = rng.choice(region.classes)
                else:
                    value = rng.choice(region.classes)
                boxes.append(_region_box(garment, region.name, side, value, base, rng, schema))

    image.save(path, quality=85)
    return ImageAnnotation(image=path.name, width=width, height=height, boxes=boxes)


def ensure_mock_dataset(
    n_images: int = 12,
    n_products: int = 8,
    *,
    config: Config | None = None,
    schema: Schema | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Create a synthetic raw dataset, annotations and a mock catalog.

    Idempotent: existing files are kept unless ``force=True``.
    """
    cfg = config or load_config()
    sc = schema or load_schema()
    image_dir = ensure_dir(cfg.path("paths.images_raw"))
    ann_dir = ensure_dir(cfg.path("paths.annotations"))
    catalog_dir = ensure_dir(cfg.path("paths.catalog"))
    products_path = catalog_dir / "products.json"
    store = AnnotationStore(image_dir, ann_dir, sc)

    existing = sorted(image_dir.glob("mock_*.jpg"))
    if existing and not force:
        LOG.info("Mock dataset already present (%d images); keeping it", len(existing))
    else:
        for i in range(n_images):
            rng = random.Random(1000 + i)
            name = f"mock_{i:04d}.jpg"
            store.save(_draw_image(image_dir / name, rng, sc))
        LOG.info("Generated %d synthetic images in %s", n_images, image_dir)

    if not products_path.exists() or force:
        products = []
        sources = sorted(image_dir.glob("mock_*.jpg"))[: max(1, n_products)]
        for i, src in enumerate(sources):
            dst = catalog_dir / f"product_{i:04d}{src.suffix}"
            shutil.copy2(src, dst)
            products.append({
                "id": f"prod_{i:04d}",
                "image_path": str(dst),
                "title": f"Synthetic outfit {i:04d}",
                "price": round(10.0 + i * 2.5, 2),
                "url": f"https://example.invalid/products/{i:04d}",
            })
        products_path.write_text(json.dumps(products, indent=2), encoding="utf-8")
        LOG.info("Generated %d mock catalog products in %s", len(products), catalog_dir)

    return {
        "images": image_dir,
        "annotations": ann_dir,
        "catalog": catalog_dir,
        "products": products_path,
    }
