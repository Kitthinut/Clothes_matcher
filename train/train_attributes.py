"""Train one YOLO classifier per garment attribute.

Each dataset lives at ``datasets/cls/<garment>_<attribute>/`` with
``train/<class>/*.jpg`` and ``val/<class>/*.jpg`` subfolders. Attributes with
too few samples are skipped with a warning instead of failing the run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.schema import load_schema
from train.common import (
    base_train_kwargs,
    count_cls_samples,
    save_best_weights,
    should_train,
    write_metadata,
)

LOG = LOGGER.getChild("train.attributes")


def train_attributes(
    config: Config | None = None,
    *,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    export_onnx: bool = False,
    mock: bool = False,
) -> dict[str, Any]:
    """Train all eligible attribute classifiers and save weights/metadata."""
    cfg = config or load_config()
    schema = load_schema()
    models_dir = cfg.path("paths.models")
    cls_root = cfg.path("paths.datasets") / "cls"
    min_total = int(cfg.get("train.min_total_samples", 20))
    min_per_class = int(cfg.get("train.min_samples_per_class", 3))

    summary: dict[str, Any] = {"trained": [], "skipped": [], "weights": {}, "top1": {}}

    if mock or not cls_root.exists():
        reason = "mock mode" if mock else f"no classification datasets at {cls_root}"
        LOG.warning("Skipping all attribute training: %s", reason)
        write_metadata(models_dir, config=cfg, schema=schema)
        summary["reason"] = reason
        return summary

    for task in schema.classification_tasks():
        task_dir = cls_root / task.key
        counts = count_cls_samples(task_dir, task)
        ok, reason = should_train(counts, min_total, min_per_class)
        if not ok:
            LOG.warning("SKIP %-24s %s", task.key, reason)
            summary["skipped"].append({"key": task.key, "reason": reason, "counts": counts})
            continue

        LOG.info("TRAIN %-24s counts=%s", task.key, counts)
        kwargs = base_train_kwargs(
            cfg,
            epochs=epochs if epochs is not None else cfg.get("train.cls_epochs", 30),
            imgsz=imgsz,
            batch=batch,
            device=device,
        )
        kwargs.pop("patience", None)  # small classification sets finish quickly
        from ultralytics import YOLO  # lazy import

        model = YOLO(cfg.cls_weights)
        results = model.train(data=str(task_dir), **kwargs)
        weights = save_best_weights(results, models_dir / "cls" / task.key / "best.pt")
        top1 = float(getattr(results, "top1", 0.0))

        write_metadata(
            models_dir, config=cfg, schema=schema,
            classifier={
                "key": task.key,
                "garment": task.garment,
                "attribute": task.name,
                "classes": list(task.classes),
                "weights": f"cls/{task.key}/best.pt",
                "samples": counts,
            },
            cls_metrics={task.key: {"top1": top1, "samples": counts}},
        )
        summary["trained"].append(task.key)
        summary["weights"][task.key] = str(weights)
        summary["top1"][task.key] = top1

        if export_onnx:
            exported = Path(model.export(format="onnx"))
            target = models_dir / "cls" / task.key / "best.onnx"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exported, target)

    LOG.info("Attribute training finished: %d trained, %d skipped", len(summary["trained"]), len(summary["skipped"]))
    write_metadata(models_dir, config=cfg, schema=schema)
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train one classifier per attribute")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Skip real training (metadata only)")
    args = parser.parse_args(argv)

    result = train_attributes(
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        export_onnx=args.export_onnx, mock=args.mock,
    )
    print(f"Trained: {result['trained']}")
    print(f"Skipped: {[s['key'] for s in result['skipped']]}")


if __name__ == "__main__":
    main()
