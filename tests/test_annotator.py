"""Tests for the v2 annotation store and optional browser app construction."""

from __future__ import annotations

from annotator.store import AnnotationStore, Box, ImageAnnotation
from common.config import Config
from common.schema import load_schema


def _store(tmp_path, schema):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    return images, annotations, AnnotationStore(images, annotations, schema)


def test_box_and_yolo_lines() -> None:
    schema = load_schema()
    box = Box.create("top", "neck", "center", "crew", (200, 150, 100, 50))
    assert box.xyxy == (100, 50, 200, 150)
    assert box.detection_name(schema) == "top_neck_crew"
    sleeve = Box.create("top", "sleeves", "left", "long", (10, 20, 30, 80))
    assert sleeve.detection_name(schema) == "top_sleeves_left_long"

    ann = ImageAnnotation("a.jpg", 400, 200, [box, sleeve])
    ann.validate(schema)
    ids = schema.detection_class_ids()
    lines = ann.yolo_lines(schema)
    assert len(lines) == 2
    assert lines[0].startswith(str(ids["top_neck_crew"]) + " ")
    assert lines[1].startswith(str(ids["top_sleeves_left_long"]) + " ")


def test_store_roundtrip_and_progress(tmp_path) -> None:
    schema = load_schema()
    from PIL import Image

    images, annotations, store = _store(tmp_path, schema)
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (400, 200), "white").save(images / name)

    ann = ImageAnnotation("a.jpg", 400, 200, [
        Box.create("bottom", "garment", "center", "bottom", (10, 20, 110, 180)),
        Box.create("bottom", "fit_type", "center", "straight", (20, 30, 100, 170)),
        Box.create("bottom", "color", "center", "black", (10, 20, 110, 180)),
    ])
    store.save(ann)

    assert (annotations / "annotations.json").exists()
    assert "a.jpg" in __import__("json").loads((annotations / "annotations.json").read_text(encoding="utf-8"))["images"]
    assert not list(annotations.glob("*.txt"))  # JSON is the only saved format
    loaded = store.load(images / "a.jpg")
    assert loaded.garment_box("bottom") is not None
    assert loaded.region_boxes("bottom", "fit_type")[0].label == "straight"
    assert store.progress() == (1, 2)
    assert store.done_images() == {"a.jpg"}
    assert store.load(images / "b.jpg").boxes == []


def test_legacy_per_image_json_is_consolidated(tmp_path) -> None:
    import json

    from PIL import Image

    schema = load_schema()
    images, annotations, _ = _store(tmp_path, schema)
    Image.new("RGB", (100, 100), "white").save(images / "old.jpg")
    legacy = ImageAnnotation("old.jpg", 100, 100, [
        Box.create("top", "neck", "center", "crew", (10, 10, 40, 40)),
    ])
    (annotations / "old.json").write_text(
        json.dumps(legacy.to_dict()), encoding="utf-8"
    )

    store = AnnotationStore(images, annotations, schema)
    assert store.has_annotation(images / "old.jpg")
    assert (annotations / "annotations.json").exists()
    assert (annotations / "old.json.bak").exists()
    assert not (annotations / "old.json").exists()
    assert store.load(images / "old.jpg").boxes[0].label == "crew"


def test_box_hidden_roundtrip() -> None:
    schema = load_schema()
    box = Box.create("top", "neck", "center", "crew", (1, 2, 30, 40))
    box.hidden = True
    restored = Box.from_dict(box.to_dict())
    assert restored.hidden is True
    ann = ImageAnnotation("a.jpg", 100, 100, [restored])
    ann.validate(schema)
    assert ann.to_dict()["boxes"][0]["hidden"] is True


def test_session_resume(tmp_path) -> None:
    schema = load_schema()
    _, _, store = _store(tmp_path, schema)
    assert store.load_session()["last_index"] == 0
    store.save_session(4, ["skip.jpg"])
    session = store.load_session()
    assert session["last_index"] == 4
    assert session["skipped"] == ["skip.jpg"]


def test_make_app_constructs(tmp_path) -> None:
    import gradio as gr

    raw = {
        "project": "test",
        "model": {"family": "yolo26n", "imgsz": 320, "device": "cpu"},
        "paths": {
            "images_raw": str(tmp_path / "images"), "annotations": str(tmp_path / "annotations"),
            "catalog": str(tmp_path / "catalog"), "datasets": str(tmp_path / "datasets"),
            "models": str(tmp_path / "models"), "database": str(tmp_path / "db.sqlite3"),
        },
        "train": {}, "search": {},
    }
    demo = make_app_for_test(raw)
    assert isinstance(demo, gr.Blocks)


def make_app_for_test(raw: dict):
    from annotator.app_web import make_app

    return make_app(config=Config(raw, root=__import__("pathlib").Path(".")))
