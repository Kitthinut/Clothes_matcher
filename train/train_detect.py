"""Train the stage-1 garment detector (classes: top, bottom).

Usage::

    python -m train.train_detect --epochs 80 --imgsz 640 --batch 16 --device 0
    python -m train.train_detect --mock      # skip training, write metadata only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.schema import load_schema
from train.common import base_train_kwargs, save_best_weights, write_metadata

LOG = LOGGER.getChild("train.detect")


def train_detect(
    config: Config | None = None,
    *,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    data: str | Path | None = None,
    export_onnx: bool = False,
    mock: bool = False,
) -> dict[str, Any]:
    """Train (or intentionally skip) the detection model.

    Returns a summary dict including status, weights path and mAP metrics.
    """
    cfg = config or load_config()
    schema = load_schema()
    models_dir = cfg.path("paths.models")
    data_yaml = Path(data) if data is not None else cfg.path("paths.datasets") / "detect" / "data.yaml"

    # Always make the label registry available, even in mock/skip mode.
    write_metadata(models_dir, config=cfg, schema=schema)

    if mock or not data_yaml.exists():
        reason = "mock mode" if mock else f"dataset not found at {data_yaml}"
        LOG.warning("Skipping detector training: %s", reason)
        return {"status": "skipped", "reason": reason, "data": str(data_yaml)}

    from ultralytics import YOLO  # imported lazily so metadata-only mode stays light

    kwargs = base_train_kwargs(cfg, epochs=epochs, imgsz=imgsz, batch=batch, device=device)
    kwargs["data"] = str(data_yaml)
    LOG.info("Training detector from %s with %s", cfg.detect_weights, kwargs)
    model = YOLO(cfg.detect_weights)
    results = model.train(**kwargs)

    weights = save_best_weights(results, models_dir / "detect" / "best.pt")
    raw_metrics = getattr(results, "results_dict", {}) or {}
    metrics = {
        "mAP50": float(raw_metrics.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(raw_metrics.get("metrics/mAP50-95(B)", 0.0)),
        "weights": "detect/best.pt",
        "epochs": kwargs["epochs"],
    }
    write_metadata(
        models_dir, config=cfg, schema=schema,
        detection={"weights": "detect/best.pt", "trained": True},
        detect_metrics=metrics,
    )

    onnx_path: str | None = None
    if export_onnx:
        exported = Path(model.export(format="onnx"))
        target = models_dir / "detect" / "best.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, target)
        onnx_path = str(target)

    LOG.info("Detector training complete: mAP50=%.4f", metrics["mAP50"])
    return {"status": "trained", "weights": str(weights), "metrics": metrics, "onnx": onnx_path}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the Raja-Fashion garment detector")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu, 0, cuda:0, ...")
    parser.add_argument("--data", type=str, default=None, help="Override detect data.yaml path")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Skip real training (metadata only)")
    args = parser.parse_args(argv)

    result = train_detect(
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        data=args.data, export_onnx=args.export_onnx, mock=args.mock,
    )
    print(result)


if __name__ == "__main__":
    main()
