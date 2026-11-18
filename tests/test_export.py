"""Tests for v2 dataset export and annotation validation."""

from __future__ import annotations

import yaml

from annotator.export import export_datasets, split_image_names
from annotator.validate import format_report, validate_annotations
from common.config import Config
from common.mock_data import ensure_mock_dataset
from common.schema import load_schema


def _prepare(config: Config, n_images: int = 10) -> None:
    ensure_mock_dataset(n_images=n_images, n_products=3, config=config, schema=load_schema(), force=True)


def test_split_is_deterministic() -> None:
    names = [f"img_{i:03d}.jpg" for i in range(20)]
    a = split_image_names(names, 0.25, seed=7)
    b = split_image_names(names, 0.25, seed=7)
    c = split_image_names(names, 0.25, seed=8)
    assert a == b and a != c
    assert len(a[0]) + len(a[1]) == 20
    assert not (a[0] & a[1])


def test_export_datasets(temp_config: Config, tmp_path) -> None:
    _prepare(temp_config)
    out = tmp_path / "exported"
    summary = export_datasets(config=temp_config, out_dir=out, seed=7)

    schema = load_schema()
    assert summary["images_train"] + summary["images_val"] == 10
    assert summary["boxes"] > 0
    expected_crops = summary["boxes"] - summary["detection_by_class"].get("top_garment", 0) \
        - summary["detection_by_class"].get("bottom_garment", 0)
    assert summary["crops"] == expected_crops

    detect = out / "detect"
    data = yaml.safe_load((detect / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == list(schema.detection_classes)
    assert data["nc"] == len(schema.detection_classes)
    for split in ("train", "val"):
        assert list((detect / "images" / split).glob("*.jpg"))
        labels = list((detect / "labels" / split).glob("*.txt"))
        assert labels
        for label_file in labels:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                assert 0 <= int(line.split()[0]) < data["nc"]

    tasks_yaml = yaml.safe_load((out / "cls" / "tasks.yaml").read_text(encoding="utf-8"))
    assert set(tasks_yaml) == set(schema.task_keys())
    for task in schema.classification_tasks():
        task_dir = out / "cls" / task.key
        for split in ("train", "val"):
            for class_dir in (task_dir / split).iterdir():
                assert class_dir.name in task.classes


def test_export_is_reproducible(temp_config: Config, tmp_path) -> None:
    _prepare(temp_config)
    first = export_datasets(config=temp_config, out_dir=tmp_path / "a", seed=3)
    second = export_datasets(config=temp_config, out_dir=tmp_path / "b", seed=3)
    assert first["images_train"] == second["images_train"]
    assert first["images_val"] == second["images_val"]
    assert first["detection_by_class"] == second["detection_by_class"]


def test_validation_report(temp_config: Config, tmp_path) -> None:
    from PIL import Image

    _prepare(temp_config, n_images=6)
    report = validate_annotations(temp_config, min_samples=1)
    assert report["annotated"] == 6
    assert report["missing_labels"] == []
    assert set(report["detection_counts"]).issubset(set(load_schema().detection_classes))
    assert len(report["attribute_counts"]) == 11
    assert "Per-region class counts" in format_report(report)

    Image.new("RGB", (32, 32)).save(temp_config.path("paths.images_raw") / "unannotated.jpg")
    report2 = validate_annotations(temp_config)
    assert report2["missing_labels"] == ["unannotated.jpg"]
