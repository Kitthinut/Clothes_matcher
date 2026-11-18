"""Export raw annotations into YOLO detection and classification datasets.

Produces::

    datasets/detect/
      data.yaml
      images/train/*.jpg   labels/train/*.txt
      images/val/*.jpg     labels/val/*.txt
    datasets/cls/<garment>_<attribute>/
      train/<class>/*.jpg
      val/<class>/*.jpg

The train/val split is computed once per image (seeded) and reused for both
stages, so a crop and its source image always land on the same side.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from annotator.store import IMAGE_SUFFIXES, AnnotationStore, resolve_annotation_dirs
from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.schema import Schema, load_schema

LOG = LOGGER.getChild("annotator.export")


def split_image_names(names: list[str], val_split: float, seed: int = 42) -> tuple[set[str], set[str]]:
    """Deterministically split image file names into train and val sets."""
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1): {val_split}")
    ordered = sorted(names)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_split))
    if ordered and n_val == 0 and val_split > 0:
        n_val = 1
    val = set(shuffled[:n_val])
    train = set(ordered) - val
    return train, val


def export_datasets(
    config: Config | None = None,
    schema: Schema | None = None,
    *,
    out_dir: Path | None = None,
    val_split: float | None = None,
    seed: int = 42,
    clean: bool = True,
    images_dir: str | Path | None = None,
    annotations_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build detection and classification datasets from annotation JSON files.

    Args:
        config: Config override.
        schema: Schema override.
        out_dir: Destination root (defaults to ``config.path('paths.datasets')``).
        val_split: Validation fraction (defaults to ``config.train.val_split``).
        seed: Split seed.
        clean: Remove previously generated dataset folders first.
        images_dir: Optional source image folder (CLI ``--images``).
        annotations_dir: Optional source annotation folder (CLI ``--annotations``).

    Returns:
        Summary dict with image/box/crop counts and the output paths.
    """
    cfg = config or load_config()
    sc = schema or load_schema()
    images, annotations = resolve_annotation_dirs(cfg, images_dir, annotations_dir)
    store = AnnotationStore(images, annotations, sc)
    root = Path(out_dir) if out_dir is not None else cfg.path("paths.datasets")
    fraction = cfg.get("train.val_split", 0.2) if val_split is None else val_split

    detect_dir = root / "detect"
    cls_dir = root / "cls"
    if clean:
        for target in (detect_dir, cls_dir):
            if target.exists():
                shutil.rmtree(target)

    splits = {"train": detect_dir / "images" / "train", "val": detect_dir / "images" / "val"}
    label_dirs = {"train": detect_dir / "labels" / "train", "val": detect_dir / "labels" / "val"}
    for d in [*splits.values(), *label_dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)
    cls_dir.mkdir(parents=True, exist_ok=True)

    annotated = sorted(p for p in store.image_paths() if store.has_annotation(p))
    train_names, val_names = split_image_names([p.name for p in annotated], fraction, seed)

    counts: dict[str, Any] = {
        "images_train": 0, "images_val": 0, "boxes": 0, "crops": 0,
        "detection_by_class": defaultdict(int),
        "cls_by_task": defaultdict(lambda: defaultdict(int)),
        "skipped_invalid": [],
    }

    for image_path in annotated:
        ann = store.load(image_path)
        split = "val" if image_path.name in val_names else "train"
        shutil.copy2(image_path, splits[split] / image_path.name)
        counts[f"images_{split}"] += 1
        (label_dirs[split] / f"{image_path.stem}.txt").write_text(
            "\n".join(ann.yolo_lines(sc)), encoding="utf-8"
        )

        with Image.open(image_path) as src:
            image = src.convert("RGB")
        for box in ann.boxes:
            counts["boxes"] += 1
            class_name = box.detection_name(sc)
            counts["detection_by_class"][class_name] += 1
            if box.region == "garment":
                continue  # whole-item boxes are not classifier crops
            x1, y1, x2, y2 = box.xyxy
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 <= x1 or y2 <= y1:
                counts["skipped_invalid"].append(f"{image_path.name}:{box.id}")
                continue
            crop = image.crop((x1, y1, x2, y2))
            region = sc.region(box.garment, box.region)
            label = box.label
            if label not in region.classes:
                LOG.warning("Invalid %s=%r in %s; skipping crop", box.detection_name(sc), label, image_path.name)
                continue
            dest_dir = cls_dir / region.key / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            crop.save(dest_dir / f"{image_path.stem}_{box.id}_{box.side}.jpg", quality=90)
            counts["crops"] += 1
            counts["cls_by_task"][region.key][label] += 1

    # YOLO detection dataset metadata. Absolute path keeps training robust
    # regardless of the current working directory.
    data_yaml = {
        "path": str(detect_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": list(sc.detection_classes),
        "nc": len(sc.detection_classes),
    }
    (detect_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (cls_dir / "tasks.yaml").write_text(
        yaml.safe_dump(
            {t.key: list(t.classes) for t in sc.classification_tasks()},
            sort_keys=False, allow_unicode=True,
        ),
        encoding="utf-8",
    )

    summary = {
        "detect_dir": detect_dir,
        "cls_dir": cls_dir,
        "images_train": counts["images_train"],
        "images_val": counts["images_val"],
        "boxes": counts["boxes"],
        "crops": counts["crops"],
        "detection_by_class": dict(counts["detection_by_class"]),
        "cls_by_task": {k: dict(v) for k, v in counts["cls_by_task"].items()},
        "skipped_invalid": counts["skipped_invalid"],
    }
    LOG.info(
        "Exported %d train / %d val images, %d boxes, %d crops -> %s",
        counts["images_train"], counts["images_val"], counts["boxes"], counts["crops"], root,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export Raja-Fashion YOLO datasets")
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--images", type=str, default=None, help="Folder of source images")
    parser.add_argument("--annotations", type=str, default=None, help="Folder of annotation .json/.txt files")
    parser.add_argument("--out", type=str, default=None, help="Dataset output folder")
    args = parser.parse_args(argv)
    result = export_datasets(
        val_split=args.val_split, seed=args.seed, clean=not args.no_clean,
        images_dir=args.images, annotations_dir=args.annotations,
        out_dir=Path(args.out) if args.out else None,
    )
    print(f"Detection dataset: {result['detect_dir']}")
    print(f"Classification datasets: {result['cls_dir']}")
    print(f"Images: {result['images_train']} train / {result['images_val']} val")
    print(f"Boxes: {result['boxes']}  Crops: {result['crops']}")


if __name__ == "__main__":
    main()
