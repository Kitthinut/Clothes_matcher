"""Validation report for v2 region annotations (imbalance + missing labels)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from annotator.store import AnnotationStore, ImageAnnotation, resolve_annotation_dirs
from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.schema import Schema, load_schema

LOG = LOGGER.getChild("annotator.validate")


def validate_annotations(
    config: Config | None = None,
    schema: Schema | None = None,
    *,
    min_samples: int = 3,
    images_dir: str | Path | None = None,
    annotations_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyse region annotations and return a structured report."""
    cfg = config or load_config()
    sc = schema or load_schema()
    images, annotations = resolve_annotation_dirs(cfg, images_dir, annotations_dir)
    store = AnnotationStore(images, annotations, sc)

    image_paths = store.image_paths()
    missing_labels = [p.name for p in image_paths if not store.has_annotation(p)]
    detection_counts: Counter[str] = Counter()
    region_counts: dict[str, Counter[str]] = {t.key: Counter() for t in sc.classification_tasks()}
    garment_counts: Counter[str] = Counter()
    issues: list[str] = []

    for path in image_paths:
        if not store.has_annotation(path):
            continue
        try:
            ann = store.load(path)
        except (ValueError, KeyError, OSError) as exc:
            issues.append(f"{path.name}: {exc}")
            continue

        present = set()
        for box in ann.boxes:
            class_name = box.detection_name(sc)
            detection_counts[class_name] += 1
            present.add(box.garment)
            if box.region != "garment":
                region_counts[f"{box.garment}_{box.region}"][box.label] += 1
        for garment in present:
            garment_counts[garment] += 1
        for garment in sc.garment_names:
            if garment not in present:
                continue  # an image may legitimately have only a top or only a bottom
            if not ann.garment_box(garment):
                issues.append(f"{path.name}: missing {garment}_garment box (needed for colour)")
            for region in sc.regions(garment):
                for side in region.sides:
                    found = [b for b in ann.boxes if b.garment == garment and b.region == region.name and b.side == side]
                    if not found:
                        issues.append(f"{path.name}: missing {garment}.{region.name} ({side}) box")

    imbalance: dict[str, dict[str, Any]] = {}
    for key, counter in region_counts.items():
        values = list(counter.values())
        max_count, min_count = (max(values), min(values)) if values else (0, 0)
        ratio = round(max_count / min_count, 2) if min_count else float("inf")
        under = sorted(c for c, n in counter.items() if n < min_samples)
        task = next((t for t in sc.classification_tasks() if t.key == key), None)
        empty = [c for c in task.classes if counter.get(c, 0) == 0] if task else []
        imbalance[key] = {
            "counts": dict(counter), "max": max_count, "min": min_count, "ratio": ratio,
            "underrepresented": under, "empty_classes": empty,
        }

    report = {
        "images": len(image_paths),
        "annotated": len(image_paths) - len(missing_labels),
        "missing_labels": missing_labels,
        "garment_counts": dict(garment_counts),
        "detection_counts": dict(detection_counts),
        "attribute_counts": {k: dict(v) for k, v in region_counts.items()},
        "imbalance": imbalance,
        "issues": issues,
        "warnings": [
            f"{key}: class imbalance ratio {info['ratio']} (min={info['min']}, max={info['max']})"
            for key, info in imbalance.items() if info["min"] and info["ratio"] >= 3
        ] + [
            f"{key}: empty classes {info['empty_classes']}"
            for key, info in imbalance.items() if info["empty_classes"]
        ],
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    """Return a compact human-readable version of *report*."""
    lines = [
        f"Images: {report['annotated']}/{report['images']} annotated",
        f"Missing labels: {', '.join(report['missing_labels']) or 'none'}",
        f"Garments present: {report['garment_counts']}",
        f"Region boxes: {sum(report['detection_counts'].values())}",
        "",
        "Per-region class counts:",
    ]
    for key, info in report["imbalance"].items():
        lines.append(f"  {key:<26} {info['counts']}  ratio={info['ratio']}")
    if report["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in report["warnings"]]
    if report["issues"]:
        lines += ["", "Issues:"] + [f"  - {i}" for i in report["issues"][:25]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate Raja-Fashion region annotations")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--images", type=str, default=None, help="Folder of source images")
    parser.add_argument("--annotations", type=str, default=None, help="Folder of annotation .json/.txt files")
    args = parser.parse_args(argv)
    report = validate_annotations(
        min_samples=args.min_samples, images_dir=args.images, annotations_dir=args.annotations,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
