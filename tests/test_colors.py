"""Tests for multi-colour detection (stripes / mixed-colour garments)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from annotator.app import DesktopAnnotator
from annotator.store import AnnotationStore, Box, ImageAnnotation
from common.colors import analyze_color, analyze_colors
from common.config import Config
from common.schema import load_schema


ALLOWED = {"black", "white", "gray", "red", "orange", "brown", "yellow",
           "green", "teal", "blue", "navy", "purple", "pink"}


def _striped(path: Path) -> None:
    image = Image.new("RGB", (300, 200), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    for x in range(0, 300, 60):
        draw.rectangle((x, 0, x + 30, 199), fill=(12, 12, 12))
    image.save(path)


def test_analyze_colors_finds_two_colours(tmp_path) -> None:
    path = tmp_path / "stripes.jpg"
    _striped(path)
    colors = analyze_colors(Image.open(path), allowed=ALLOWED)
    names = [c.name for c in colors]
    assert {"black", "white"}.issubset(set(names))
    assert all(c.fraction >= 0.12 for c in colors)
    assert analyze_color(Image.open(path), allowed=ALLOWED).name in names


def test_box_stores_all_detected_colours(tmp_path) -> None:
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    store = AnnotationStore(images, annotations, load_schema())
    colors = analyze_colors(Image.new("RGB", (100, 100), "white"), allowed=ALLOWED)
    box = Box.create("top", "color", "center", colors[0].name, (10, 10, 90, 90),
                     colors=[c.to_dict() for c in colors])
    store.save(ImageAnnotation("a.jpg", 100, 100, [box]))

    reloaded = AnnotationStore(images, annotations, load_schema()).load(images / "a.jpg")
    assert reloaded.boxes[0].colors
    assert reloaded.boxes[0].colors[0]["name"] == colors[0].name


def test_app_draws_striped_colour_box_and_shows_swatches(temp_config: Config, tk_root) -> None:
    root = tk_root
    try:
        images = temp_config.path("paths.images_raw")
        images.mkdir(parents=True, exist_ok=True)
        _striped(images / "stripes.jpg")
        app = DesktopAnnotator(root, config=temp_config, image_dir=images,
                               annotations_dir=temp_config.path("paths.annotations"))
        app.set_category("top_color_center")
        p = app.to_canvas(30, 30)
        q = app.to_canvas(270, 170)
        ev = lambda pt: SimpleNamespace(x=pt[0], y=pt[1])
        app.on_press(ev(p)); app.on_drag(ev(q)); app.on_release(ev(q))

        box = next(b for b in app.boxes() if b.region == "color")
        assert box.colors and len(box.colors) >= 2
        names = {c["name"] for c in box.colors}
        assert {"black", "white"}.issubset(names)
        assert len(app.color_info[box.id]) >= 2
        assert "Colours" in app.status_var.get()
    finally:
        pass  # shared Tk root
