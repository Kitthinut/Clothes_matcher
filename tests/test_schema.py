"""Tests for schema v2 loading, region model and generated detection classes."""

from __future__ import annotations

import pytest

from common.schema import load_schema


def test_schema_matches_spec() -> None:
    schema = load_schema(reload=True)
    assert schema.version == 2
    assert schema.garment_names == ("top", "bottom")
    assert schema.garment_box_class("top") == "top_garment"
    assert schema.garment_box_class("bottom") == "bottom_garment"

    assert schema.classes("top", "length") == ("cropped", "regular")
    assert schema.classes("top", "type") == ("no_button", "half_button", "button_down")
    assert schema.classes("top", "pattern") == ("dot", "stripe", "check", "graphic", "plain", "corner_only")
    assert schema.classes("top", "neck") == ("crew", "v", "square", "off_shoulder", "polo")
    assert schema.classes("top", "sleeves") == ("long", "short", "sleeveless", "camis")
    assert schema.region("top", "sleeves").sides == ("left", "right")
    assert schema.region("top", "color").source == "hsv"
    assert schema.classes("top", "color") == schema.color_classes

    assert schema.classes("bottom", "waist_length") == ("low", "high")
    assert schema.classes("bottom", "bottom_length") == ("long", "short", "cropped")
    assert schema.classes("bottom", "pattern") == ("graphic", "stripe", "check", "plain")
    assert schema.classes("bottom", "fit_type") == ("skinny", "straight", "baggy", "boot_cut")
    assert "plain" in schema.classes("top", "pattern") and "no_pattern" not in schema.classes("top", "pattern")


def test_detection_classes_and_parsing() -> None:
    schema = load_schema()
    classes = schema.detection_classes
    assert len(classes) == 65
    assert classes[0] == "top_garment"
    assert "top_neck_crew" in classes
    assert "top_sleeves_left_long" in classes
    assert "top_sleeves_right_camis" in classes
    assert "bottom_color_black" in classes
    assert len(set(classes)) == len(classes)

    assert schema.parse_detection_name("top_neck_crew") == ("top", "neck", "center", "crew")
    assert schema.parse_detection_name("top_sleeves_left_long") == ("top", "sleeves", "left", "long")
    assert schema.parse_detection_name("bottom_waist_length_high") == ("bottom", "waist_length", "center", "high")
    assert schema.parse_detection_name("top_garment") == ("top", "garment", "center", "top")
    assert schema.parse_detection_name("nonsense") is None

    for name in classes:
        parsed = schema.parse_detection_name(name)
        assert parsed is not None, name
        assert schema.detection_name(*parsed) == name


def test_tasks_and_validation() -> None:
    schema = load_schema()
    keys = schema.task_keys()
    assert len(keys) == 11
    assert "top_sleeves" in keys and "top_color" in keys and "bottom_color" in keys
    assert "top_garment" not in keys
    assert schema.detection_class_ids()["top_neck_crew"] == classes_index(schema, "top_neck_crew")

    schema.validate_box("top", "neck", "center", "crew")
    schema.validate_box("top", "sleeves", "left", "long")
    with pytest.raises(ValueError, match="Invalid top.sleeves side"):
        schema.validate_box("top", "sleeves", "middle", "long")
    with pytest.raises(ValueError, match="Invalid top.neck"):
        schema.validate_box("top", "neck", "center", "turtleneck")
    with pytest.raises(KeyError, match="Unknown garment"):
        schema.garment("shoes")


def classes_index(schema, name: str) -> int:
    return list(schema.detection_classes).index(name)


def test_rejects_invalid_schema(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\ngarments:\n  top:\n    regions:\n      x: {classes: [a]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported schema version"):
        load_schema(bad, reload=True)
    load_schema(reload=True)
