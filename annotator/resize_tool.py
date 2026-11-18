"""Resize an image set to one target size with black letterbox padding.

Images are resized without distortion: the original is scaled to fit inside the
target rectangle (Lanczos resampling, high-quality JPEG save), then centred on
a black canvas. Boxes in the JSON annotations are scaled and offset by the same
transform so they still line up with the garment.

Run from the annotator's size-check dialog, not normally as a CLI.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from annotator.store import IMAGE_SUFFIXES, AnnotationStore, ImageAnnotation
from common.schema import Schema, load_schema


@dataclass
class ResizeReport:
    """Summary of a resize run."""

    resized: int = 0
    unchanged: int = 0
    backed_up: int = 0
    annotations_updated: int = 0
    target: tuple[int, int] = (384, 288)

    def message(self) -> str:
        return (
            f"Resized {self.resized} image(s) to {self.target[0]}x{self.target[1]} px "
            f"({self.unchanged} were already that size).\n"
            f"Originals backed up: {self.backed_up}.\n"
            f"Annotation boxes updated: {self.annotations_updated}."
        )


def parse_size(text: str) -> tuple[int, int]:
    """Parse '384x288', '384,288', '384 288' into a (width, height) tuple."""
    match = re.fullmatch(r"\s*(\d+)\s*[xX, ]\s*(\d+)\s*(?:px)?\s*", str(text))
    if not match:
        raise ValueError(f"Invalid size {text!r}; use for example 384x288")
    width, height = int(match.group(1)), int(match.group(2))
    if width < 16 or height < 16:
        raise ValueError("Size must be at least 16x16 pixels")
    return width, height


def _letterbox(image: Image.Image, target: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    """Return padded image plus (scale, offset_x, offset_y) for coordinate mapping."""
    tw, th = target
    scale = min(tw / image.width, th / image.height)
    new_w, new_h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    ox, oy = (tw - new_w) // 2, (th - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def resize_images(
    images_dir: str | Path,
    annotations_dir: str | Path,
    target: tuple[int, int] | str,
    *,
    backup_dir: str | Path | None = None,
    schema: Schema | None = None,
) -> ResizeReport:
    """Resize every image in *images_dir* in place and rescale its annotations.

    Existing originals are copied to *backup_dir* before being overwritten.
    Images already at the target size are left untouched.
    """
    size = parse_size(target) if isinstance(target, str) else (int(target[0]), int(target[1]))
    sc = schema or load_schema()
    images = Path(images_dir)
    annotations = Path(annotations_dir)
    backup = Path(backup_dir) if backup_dir is not None else images.parent / "images_original_backup"
    backup.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(images, annotations, sc)
    report = ResizeReport(target=size)

    for path in sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        with Image.open(path) as src:
            image = src.convert("RGB")
        if image.size == size:
            report.unchanged += 1
            continue

        original_backup = backup / path.name
        if not original_backup.exists():
            shutil.copy2(path, original_backup)
            report.backed_up += 1

        padded, scale, ox, oy = _letterbox(image, size)
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            padded.save(path, quality=95, subsampling=0, optimize=True)
        else:
            padded.save(path)

        if store.has_annotation(path):
            ann = store.load(path)
            for box in ann.boxes:
                x1, y1, x2, y2 = box.xyxy
                box.xyxy = (
                    max(0, min(size[0], round(x1 * scale + ox))),
                    max(0, min(size[1], round(y1 * scale + oy))),
                    max(0, min(size[0], round(x2 * scale + ox))),
                    max(0, min(size[1], round(y2 * scale + oy))),
                )
            ann.width, ann.height = size
            store.save(ann)
            report.annotations_updated += 1
        report.resized += 1

    return report
