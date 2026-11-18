"""Annotation storage: ONE ``annotations.json`` for the whole image folder.

Older versions used one ``<stem>.json`` per image. On first launch those files
are consolidated into ``annotations.json`` automatically and kept as
``*.json.bak``, so existing labels are never lost.

YOLO ``.txt`` files are not written here; ``annotator.export`` derives them
from this JSON when building the training dataset.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from common.logging_conf import LOGGER
from common.paths import ensure_dir
from common.schema import Schema, load_schema

LOG = LOGGER.getChild("annotator.store")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA_FILENAME = "annotations.json"
SCHEMA_VERSION = 2


def resolve_annotation_dirs(
    config: Any,
    images: str | Path | None = None,
    annotations: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve optional CLI folders, falling back to configured project paths."""
    images_dir = Path(images) if images is not None else config.path("paths.images_raw")
    annotations_dir = Path(annotations) if annotations is not None else config.path("paths.annotations")
    return images_dir.resolve(), annotations_dir.resolve()


@dataclass
class Box:
    """One region box (or a whole-garment box)."""

    id: str
    garment: str
    region: str
    side: str
    label: str
    xyxy: tuple[int, int, int, int]
    hidden: bool = False
    hsv: tuple[float, float, float] | None = None  # primary colour (legacy field)
    colors: list[dict[str, Any]] | None = None     # all detected colours, ranked

    @classmethod
    def create(
        cls,
        garment: str,
        region: str,
        side: str,
        label: str,
        xyxy: Iterable[int],
        *,
        hsv: tuple[float, float, float] | None = None,
        colors: list[dict[str, Any]] | None = None,
    ) -> "Box":
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        return cls(
            id=uuid.uuid4().hex[:10], garment=garment, region=region,
            side=side, label=label,
            xyxy=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            hsv=hsv,
            colors=colors,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Box":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            garment=str(data["garment"]),
            region=str(data["region"]),
            side=str(data.get("side", "center")),
            label=str(data["label"]),
            xyxy=tuple(int(v) for v in data["xyxy"]),  # type: ignore[arg-type]
            hidden=bool(data.get("hidden", False)),
            hsv=tuple(float(v) for v in data["hsv"]) if data.get("hsv") else None,  # type: ignore[assignment]
            colors=[dict(c) for c in data["colors"]] if data.get("colors") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id, "garment": self.garment, "region": self.region,
            "side": self.side, "label": self.label,
            "xyxy": list(self.xyxy), "hidden": self.hidden,
        }
        if self.hsv is not None:
            result["hsv"] = [round(float(v), 4) for v in self.hsv]
        if self.colors is not None:
            result["colors"] = self.colors
        return result

    def detection_name(self, schema: Schema) -> str:
        return schema.detection_name(self.garment, self.region, self.side, self.label)


@dataclass
class ImageAnnotation:
    """All region boxes for one image."""

    image: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    absent_garments: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageAnnotation":
        return cls(
            image=str(data["image"]),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            boxes=[Box.from_dict(b) for b in data.get("boxes", [])],
            absent_garments=list(data.get("absent_garments") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image, "width": self.width, "height": self.height,
            "boxes": [b.to_dict() for b in self.boxes],
            "absent_garments": list(self.absent_garments),
        }

    def validate(self, schema: Schema) -> None:
        for box in self.boxes:
            schema.validate_box(box.garment, box.region, box.side, box.label)

    def boxes_for(self, garment: str) -> list[Box]:
        return [b for b in self.boxes if b.garment == garment]

    def region_boxes(self, garment: str, region: str) -> list[Box]:
        return [b for b in self.boxes if b.garment == garment and b.region == region]

    def garment_box(self, garment: str) -> Box | None:
        return next((b for b in self.boxes if b.garment == garment and b.region == "garment"), None)

    def present_garments(self) -> list[str]:
        return sorted({b.garment for b in self.boxes})

    def yolo_lines(self, schema: Schema) -> list[str]:
        """YOLO detection lines (class cx cy w h, normalised) for export."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Image {self.image!r} has no recorded size")
        class_ids = schema.detection_class_ids()
        lines: list[str] = []
        for box in self.boxes:
            x1, y1, x2, y2 = box.xyxy
            cx = (x1 + x2) / (2 * self.width)
            cy = (y1 + y2) / (2 * self.height)
            w = (x2 - x1) / self.width
            h = (y2 - y1) / self.height
            lines.append(f"{class_ids[box.detection_name(schema)]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines


class AnnotationStore:
    """Read/write all annotations in one JSON file, plus resume state."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        schema: Schema | None = None,
    ) -> None:
        self.images_dir = ensure_dir(images_dir)
        self.annotations_dir = ensure_dir(annotations_dir)
        self.schema = schema or load_schema()
        self.data_path = self.annotations_dir / DATA_FILENAME
        self.session_path = self.annotations_dir / "_session.json"
        self.images: dict[str, dict[str, Any]] = {}
        self._load_data()

    # ------------------------------------------------------------- storage
    def _load_data(self) -> None:
        if self.data_path.exists():
            try:
                payload = json.loads(self.data_path.read_text(encoding="utf-8"))
                self.images = dict(payload.get("images") or {})
                return
            except (ValueError, OSError, TypeError) as exc:
                LOG.warning("Could not read %s: %s", self.data_path, exc)
        self._consolidate_legacy()

    def _consolidate_legacy(self) -> None:
        """Fold old one-file-per-image annotations into annotations.json."""
        by_stem = {p.stem: p for p in self.image_paths()}
        migrated: list[str] = []
        for path in sorted(self.annotations_dir.glob("*.json")):
            if path.name in {DATA_FILENAME, "_session.json"} or path.name.endswith(".bak"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                LOG.warning("Ignoring unreadable legacy annotation %s: %s", path, exc)
                continue
            name = by_stem.get(path.stem)
            image_name = name.name if name is not None else str(data.get("image") or f"{path.stem}.jpg")
            data["image"] = image_name
            self.images[image_name] = data
            migrated.append(path.name)
        if migrated:
            self._write_data()
            for old_name in migrated:
                old = self.annotations_dir / old_name
                backup = self.annotations_dir / f"{old_name}.bak"
                if not backup.exists():
                    shutil.copy2(old, backup)
                old.unlink()
            LOG.info("Consolidated %d per-image JSON files into %s", len(migrated), self.data_path)

    def _write_data(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "images": self.images}
        self.data_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ----------------------------------------------------------------- paths
    def image_paths(self) -> list[Path]:
        return sorted(
            p for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

    def has_annotation(self, image_path: str | Path) -> bool:
        return Path(image_path).name in self.images

    def all_annotations(self) -> dict[str, ImageAnnotation]:
        return {name: ImageAnnotation.from_dict(data) for name, data in self.images.items()}

    # ------------------------------------------------------------ load/save
    def load(self, image_path: str | Path) -> ImageAnnotation:
        from PIL import Image

        name = Path(image_path).name
        if name in self.images:
            annotation = ImageAnnotation.from_dict(self.images[name])
        else:
            with Image.open(image_path) as img:
                width, height = img.size
            annotation = ImageAnnotation(image=name, width=width, height=height)
        annotation.validate(self.schema)
        return annotation

    def save(self, annotation: ImageAnnotation) -> None:
        annotation.validate(self.schema)
        name = Path(annotation.image).name
        if name not in self.images and annotation.width <= 0:
            with Image.open(self.images_dir / name) as img:
                annotation.width, annotation.height = img.size
        self.images[name] = annotation.to_dict()
        self._write_data()
        LOG.debug("Saved %s into %s", name, self.data_path.name)

    # ------------------------------------------------------------- progress
    def done_images(self) -> set[str]:
        return {p.name for p in self.image_paths() if self.has_annotation(p)}

    def progress(self) -> tuple[int, int]:
        images = self.image_paths()
        done = sum(1 for p in images if self.has_annotation(p))
        return done, len(images)

    def load_session(self) -> dict[str, Any]:
        if self.session_path.exists():
            return json.loads(self.session_path.read_text(encoding="utf-8"))
        return {"last_index": 0, "skipped": [], "updated_at": None}

    def save_session(self, last_index: int, skipped: Iterable[str] = ()) -> None:
        payload = {
            "last_index": int(last_index),
            "skipped": sorted(set(skipped)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
