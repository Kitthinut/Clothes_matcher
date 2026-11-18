"""Shared training helpers: augmentation defaults, model registry, metrics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from common.config import Config
from common.logging_conf import LOGGER
from common.schema import Schema, Attribute

LOG = LOGGER.getChild("train.common")


def augmentation_kwargs(config: Config) -> dict[str, Any]:
    """Return fashion-friendly augmentation settings from config.

    Vertical flip is disabled (garments have a meaningful top/bottom), colour
    jitter is mild, and remaining geometric transforms stay conservative.
    """
    aug = config.get("train.augment") or {}
    return {
        "flipud": float(aug.get("flipud", 0.0)),
        "fliplr": float(aug.get("fliplr", 0.5)),
        "hsv_h": float(aug.get("hsv_h", 0.01)),
        "hsv_s": float(aug.get("hsv_s", 0.3)),
        "hsv_v": float(aug.get("hsv_v", 0.3)),
        "degrees": float(aug.get("degrees", 5.0)),
        "translate": float(aug.get("translate", 0.05)),
        "scale": float(aug.get("scale", 0.4)),
        "shear": float(aug.get("shear", 2.0)),
    }


def base_train_kwargs(config: Config, *, epochs: int | None, imgsz: int | None, batch: int | None, device: str | None) -> dict[str, Any]:
    """Merge CLI overrides with config defaults for YOLO ``train()``."""
    return {
        "epochs": int(epochs if epochs is not None else config.get("train.detect_epochs", 50)),
        "imgsz": int(imgsz if imgsz is not None else config.imgsz),
        "batch": int(batch if batch is not None else config.get("train.batch", 16)),
        "device": str(device if device is not None else config.device),
        "patience": 10,
        "verbose": False,
        "plots": False,
        **augmentation_kwargs(config),
    }


def count_cls_samples(task_dir: Path, task: Attribute) -> dict[str, int]:
    """Count training images per class for one classification task."""
    train_dir = task_dir / "train"
    return {label: len(list((train_dir / label).glob("*.*"))) for label in task.classes}


def should_train(counts: dict[str, int], min_total: int, min_per_class: int) -> tuple[bool, str]:
    """Decide whether a classifier has enough data, returning ``(ok, reason)``."""
    total = sum(counts.values())
    if total < min_total:
        return False, f"only {total} samples (need >= {min_total})"
    lacking = [label for label, n in counts.items() if n < min_per_class]
    if lacking:
        return False, f"classes below {min_per_class} samples: {lacking}"
    return True, "ok"


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_metadata(
    models_dir: Path,
    *,
    config: Config,
    schema: Schema,
    detection: dict[str, Any] | None = None,
    classifier: dict[str, Any] | None = None,
    detect_metrics: dict[str, Any] | None = None,
    cls_metrics: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Update ``models/labels.json``/``labels.txt`` and ``models/metrics.json``.

    Returns the paths of the labels JSON and metrics JSON files.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    labels_path = models_dir / "labels.json"
    metrics_path = models_dir / "metrics.json"

    labels = _read_json(labels_path)
    labels.setdefault("schema_version", schema.version)
    labels["model_family"] = config.model_family
    labels.setdefault("detect", {"weights": "detect/best.pt", "names": list(schema.detection_classes)})
    labels.setdefault("classifiers", {})
    if detection is not None:
        labels["detect"].update(detection)
    if classifier is not None:
        labels["classifiers"][classifier["key"]] = classifier
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    (models_dir / "labels.txt").write_text(
        "\n".join(schema.detection_classes) + "\n", encoding="utf-8"
    )

    metrics = _read_json(metrics_path)
    if detect_metrics is not None:
        metrics["detect"] = detect_metrics
    if cls_metrics is not None:
        metrics.setdefault("classifiers", {}).update(cls_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return labels_path, metrics_path


def save_best_weights(results: Any, destination: Path) -> Path:
    """Copy a YOLO run's ``best.pt`` to *destination* and return its path."""
    save_dir = Path(getattr(results, "save_dir", "."))
    source = save_dir / "weights" / "best.pt"
    if not source.exists():
        raise FileNotFoundError(f"No best.pt produced by training run in {save_dir}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    LOG.info("Saved weights to %s", destination)
    return destination
