"""Schema loading and validation for schema v2 (multi-class detection).

``schema.yaml`` is the single source of truth. Every region of a garment gets
its own box and YOLO class ``<garment>_<region>_[<side>_]<value>``, e.g.
``top_neck_crew`` or ``top_sleeves_left_long``. Whole-item boxes are
``top_garment`` / ``bottom_garment``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from common.paths import project_path


@dataclass(frozen=True)
class Region:
    """One attribute region of one garment, e.g. ``top.sleeves``."""

    garment: str
    name: str
    title: str
    sides: tuple[str, ...]
    classes: tuple[str, ...]
    source: str = "manual"

    @property
    def key(self) -> str:
        """Dataset/model key, e.g. ``top_sleeves``."""
        return f"{self.garment}_{self.name}"

    @property
    def paired(self) -> bool:
        """Whether this region needs one box per side (left/right sleeves)."""
        return len(self.sides) > 1

    def class_id(self, label: str) -> int:
        """Return the integer class index of *label* inside this region."""
        return self.classes.index(label)


# Backwards-compatible alias used by training helpers.
Attribute = Region


@dataclass(frozen=True)
class Garment:
    """A garment type with its whole-item box and attribute regions."""

    name: str
    garment_box_class: str
    regions: Mapping[str, Region]


@dataclass(frozen=True)
class Schema:
    """Validated, immutable view of ``schema.yaml``."""

    version: int
    color_classes: tuple[str, ...]
    garments: Mapping[str, Garment]
    detection_classes: tuple[str, ...]

    @property
    def garment_names(self) -> tuple[str, ...]:
        return tuple(self.garments)

    def garment(self, name: str) -> Garment:
        try:
            return self.garments[name]
        except KeyError as exc:
            raise KeyError(f"Unknown garment {name!r}. Known: {self.garment_names}") from exc

    def regions(self, garment: str) -> tuple[Region, ...]:
        """Return all attribute regions of *garment* in schema order."""
        return tuple(self.garment(garment).regions.values())

    # Backwards-compatible name used by training/export/tests.
    def attributes(self, garment: str) -> tuple[Region, ...]:
        return self.regions(garment)

    def region(self, garment: str, name: str) -> Region:
        try:
            return self.garment(garment).regions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown region {garment}.{name}") from exc

    def classes(self, garment: str, name: str) -> tuple[str, ...]:
        return self.region(garment, name).classes

    # Backwards-compatible alias.
    def attribute(self, garment: str, name: str) -> Region:
        return self.region(garment, name)

    def classification_tasks(self, garment: str | None = None) -> tuple[Region, ...]:
        tasks = [region for g in self.garment_names for region in self.regions(g)]
        if garment is not None:
            self.garment(garment)
            tasks = [t for t in tasks if t.garment == garment]
        return tuple(tasks)

    def task_keys(self, garment: str | None = None) -> tuple[str, ...]:
        return tuple(t.key for t in self.classification_tasks(garment))

    def detection_class_ids(self) -> dict[str, int]:
        """Return detection class name -> id mapping."""
        return {name: idx for idx, name in enumerate(self.detection_classes)}

    def garment_box_class(self, garment: str) -> str:
        return self.garment(garment).garment_box_class

    def detection_name(self, garment: str, region: str, side: str, value: str) -> str:
        """Build the YOLO class name for one annotated region box."""
        if region == "garment":
            return self.garment_box_class(garment)
        reg = self.region(garment, region)
        if reg.paired:
            if side not in reg.sides:
                raise ValueError(f"Invalid side {side!r} for {garment}.{region}; choose from {reg.sides}")
            return f"{garment}_{region}_{side}_{value}"
        return f"{garment}_{region}_{value}"

    def parse_detection_name(self, name: str) -> tuple[str, str, str, str] | None:
        """Parse a YOLO class name into ``(garment, region, side, value)``."""
        for garment in self.garment_names:
            prefix = f"{garment}_"
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            if rest == "garment":
                return garment, "garment", "center", garment
            g = self.garment(garment)
            for region in sorted(g.regions, key=len, reverse=True):
                if not rest.startswith(f"{region}_"):
                    continue
                tail = rest[len(region) + 1:]
                reg = g.regions[region]
                if reg.paired:
                    parts = tail.split("_", 1)
                    if len(parts) != 2 or parts[0] not in reg.sides:
                        continue
                    side, value = parts
                else:
                    side, value = "center", tail
                if value in reg.classes:
                    return garment, region, side, value
        return None

    def validate_box(self, garment: str, region: str, side: str, value: str) -> None:
        """Raise ``ValueError`` if a region box is invalid for the schema."""
        self.garment(garment)
        if region == "garment":
            return
        reg = self.region(garment, region)
        if side not in reg.sides:
            raise ValueError(f"Invalid {garment}.{region} side {side!r}; choose from {reg.sides}")
        if value not in reg.classes:
            raise ValueError(f"Invalid {garment}.{region}={value!r}; choose from {reg.classes}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "color_classes": list(self.color_classes),
            "detection_classes": list(self.detection_classes),
            "garments": {
                g: {
                    "garment_box": self.garment(g).garment_box_class,
                    "regions": {
                        r.name: {
                            "title": r.title, "sides": list(r.sides),
                            "classes": list(r.classes), "source": r.source,
                        }
                        for r in self.regions(g)
                    },
                }
                for g in self.garment_names
            },
        }


def _generate_detection_classes(garments: Mapping[str, Garment]) -> tuple[str, ...]:
    classes: list[str] = []
    for name, garment in garments.items():
        classes.append(garment.garment_box_class)
        for region in garment.regions.values():
            for side in region.sides:
                for value in region.classes:
                    if region.paired:
                        classes.append(f"{name}_{region.name}_{side}_{value}")
                    else:
                        classes.append(f"{name}_{region.name}_{value}")
    if len(set(classes)) != len(classes):
        raise ValueError("Duplicate detection class names generated from schema")
    return tuple(classes)


@lru_cache(maxsize=4)
def _load_cached(raw_path: str) -> Schema:
    path = Path(raw_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Schema root must be a mapping: {path}")
    if int(data.get("version", 1)) != 2:
        raise ValueError(f"Unsupported schema version {data.get('version')!r}; expected 2")

    raw_garments: Mapping[str, Any] = data.get("garments") or {}
    if not raw_garments:
        raise ValueError(f"Schema has no garments: {path}")
    color_classes = tuple(str(c) for c in (data.get("color_classes") or []))
    if not color_classes:
        raise ValueError("Schema has no color_classes")

    garments: dict[str, Garment] = {}
    for g_name, g_data in raw_garments.items():
        g_data = g_data or {}
        box_class = str(g_data.get("garment_box") or f"{g_name}_garment")
        raw_regions = g_data.get("regions") or {}
        if not raw_regions:
            raise ValueError(f"Garment {g_name!r} has no regions")
        regions: dict[str, Region] = {}
        for r_name, r_data in raw_regions.items():
            r_data = r_data or {}
            raw_classes = r_data.get("classes") or []
            if not isinstance(raw_classes, (list, tuple)) or not raw_classes:
                raise ValueError(f"{g_name}.{r_name} must have a non-empty class list")
            classes = tuple(str(c) for c in raw_classes)
            if len(set(classes)) != len(classes):
                raise ValueError(f"Duplicate classes in {g_name}.{r_name}")
            sides = tuple(str(s) for s in (r_data.get("sides") or ["center"]))
            if not sides:
                raise ValueError(f"{g_name}.{r_name} has no sides")
            if r_data.get("source") == "hsv":
                classes = color_classes
            regions[str(r_name)] = Region(
                str(g_name), str(r_name), str(r_data.get("title", r_name)),
                sides, tuple(classes), str(r_data.get("source", "manual")),
            )
        garments[str(g_name)] = Garment(str(g_name), box_class, regions)

    return Schema(2, color_classes, garments, _generate_detection_classes(garments))


def load_schema(path: str | Path | None = None, *, reload: bool = False) -> Schema:
    """Load and cache ``schema.yaml``; use ``reload=True`` after edits."""
    resolved = Path(path) if path is not None else project_path("schema.yaml")
    if reload:
        _load_cached.cache_clear()
    return _load_cached(str(resolved.resolve()))
