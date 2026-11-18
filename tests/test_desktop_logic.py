"""Logic tests for the Tkinter annotator: drawing, coordinates and colour.

A deterministic synthetic image is used so the colour assertions do not depend
on a user photo that may have been resized or edited.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from annotator.app import DesktopAnnotator
from common.config import Config


def _event(point: tuple[float, float]) -> SimpleNamespace:
    return SimpleNamespace(x=point[0], y=point[1])


def _drag(app: DesktopAnnotator, a: tuple[float, float], b: tuple[float, float]) -> None:
    """Simulate a left-drag between two ORIGINAL-IMAGE pixel points."""
    p, q = app.to_canvas(*a), app.to_canvas(*b)
    app.on_press(_event(p))
    app.on_drag(_event(q))
    app.on_release(_event(q))


def _sample_image(path: Path) -> None:
    image = Image.new("RGB", (420, 560), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 300, 280), fill=(250, 250, 245))   # white top
    draw.rectangle((90, 340, 250, 520), fill=(12, 12, 12))     # black bottom
    image.save(path)


def test_drawing_and_automatic_colour(temp_config: Config) -> None:
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
        root.withdraw()
    except tkinter.TclError:
        pytest.skip("No display available for tkinter")
    try:
        images = temp_config.path("paths.images_raw")
        images.mkdir(parents=True, exist_ok=True)
        _sample_image(images / "sample.jpg")
        app = DesktopAnnotator(
            root, config=temp_config, image_dir=images,
            annotations_dir=temp_config.path("paths.annotations"),
        )

        app.set_category("top_neck_center")
        assert app.active_choice() == ("top", "neck", "center")
        assert "Top > Neck type" in app.status_var.get()

        before = len(app.boxes())
        _drag(app, (150, 120), (230, 180))
        neck = [b for b in app.boxes() if b.garment == "top" and b.region == "neck"]
        assert len(neck) == 1
        assert neck[0].xyxy == (150, 120, 230, 180)
        assert neck[0].label == "crew"

        p = app.to_canvas(320, 300)
        app.on_press(_event(p))
        app.on_drag(_event((p[0] + 2, p[1] + 2)))
        app.on_release(_event((p[0] + 2, p[1] + 2)))
        assert len(app.boxes()) == before + 1  # tiny drag ignored

        app.select_box(neck[0].id)
        _drag(app, (180, 150), (200, 170))
        assert neck[0].xyxy == (170, 140, 250, 200)

        app.select_box(neck[0].id)
        corner = app.to_canvas(*neck[0].xyxy[2:])
        app.on_press(_event(corner))
        app.on_drag(_event((corner[0] + 20, corner[1] + 20)))
        app.on_release(_event((corner[0] + 20, corner[1] + 20)))
        assert neck[0].xyxy[2] > 250
        assert [b for b in app.store.load(app.current_path()).boxes if b.region == "neck"][0].xyxy == neck[0].xyxy

        app.set_category("top_color_center")
        _drag(app, (120, 120), (260, 260))
        top = next(b for b in app.boxes() if b.garment == "top" and b.region == "color")
        assert top.label == "white"
        assert top.hsv is not None and 0 <= top.hsv[0] <= 360
        assert app.color_info[top.id].friendly in {"white", "cream"}

        app.set_garment("bottom")
        app.set_category("bottom_color_center")
        _drag(app, (120, 380), (220, 500))
        bottom = next(b for b in app.boxes() if b.garment == "bottom" and b.region == "color")
        assert bottom.label == "black"
        assert app.color_info[bottom.id].friendly == "black"
    finally:
        root.destroy()
