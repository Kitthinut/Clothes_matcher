"""Tests for training helpers and metadata writing (no real training)."""

from __future__ import annotations

import json

from common.config import Config
from common.schema import load_schema
from train.common import (
    augmentation_kwargs,
    base_train_kwargs,
    count_cls_samples,
    should_train,
    write_metadata,
)
from train.train_attributes import train_attributes
from train.train_detect import train_detect


def test_augmentation_defaults(temp_config: Config) -> None:
    aug = augmentation_kwargs(temp_config)
    assert aug["flipud"] == 0.0  # no vertical flip for fashion
    assert aug["fliplr"] == 0.5
    assert 0 < aug["hsv_h"] < 0.05 and aug["hsv_v"] <= 0.3  # mild colour jitter


def test_base_train_kwargs_override(temp_config: Config) -> None:
    kwargs = base_train_kwargs(temp_config, epochs=3, imgsz=None, batch=8, device="cpu")
    assert kwargs["epochs"] == 3
    assert kwargs["batch"] == 8
    assert kwargs["imgsz"] == temp_config.imgsz
    assert kwargs["device"] == "cpu"
    assert kwargs["flipud"] == 0.0


def test_count_and_skip_logic(tmp_path) -> None:
    schema = load_schema()
    task = schema.attribute("top", "length")
    train_dir = tmp_path / "train"
    for label, n in {"cropped": 4, "regular": 2}.items():
        d = train_dir / label
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img_{i}.jpg").touch()
    counts = count_cls_samples(tmp_path, task)
    assert counts == {"cropped": 4, "regular": 2}

    ok, reason = should_train(counts, min_total=5, min_per_class=3)
    assert not ok and "regular" in reason
    ok, reason = should_train(counts, min_total=5, min_per_class=2)
    assert ok and reason == "ok"
    ok, reason = should_train({"cropped": 1}, min_total=20, min_per_class=3)
    assert not ok and "only 1 samples" in reason


def test_write_metadata(temp_config: Config, tmp_path) -> None:
    models_dir = tmp_path / "models"
    labels_path, metrics_path = write_metadata(
        models_dir,
        config=temp_config,
        schema=load_schema(),
        classifier={"key": "top_length", "classes": ["cropped", "regular"], "weights": "cls/top_length/best.pt"},
        cls_metrics={"top_length": {"top1": 0.8}},
    )
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels["model_family"] == "yolo26n"
    assert labels["detect"]["names"] == list(load_schema().detection_classes)
    assert "top_length" in labels["classifiers"]
    assert (models_dir / "labels.txt").read_text(encoding="utf-8").splitlines() == list(load_schema().detection_classes)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["classifiers"]["top_length"]["top1"] == 0.8


def test_mock_training_writes_metadata(temp_config: Config) -> None:
    detector = train_detect(temp_config, mock=True)
    assert detector["status"] == "skipped"
    attributes = train_attributes(temp_config, mock=True)
    assert attributes["trained"] == []
    assert attributes["skipped"] == []

    models_dir = temp_config.path("paths.models")
    assert (models_dir / "labels.json").exists()
    assert (models_dir / "metrics.json").exists()
