"""Tests for the offer-to-resize workflow and coordinate-preserving resize."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from annotator.resize_tool import parse_size, resize_images
from annotator.store import AnnotationStore, Box, ImageAnnotation
from common.config import Config
from common.schema import load_schema


def _make_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (200, 120, 90)).save(path)


def test_parse_size() -> None:
    assert parse_size("384x288") == (384, 288)
    assert parse_size("384,288") == (384, 288)
    assert parse_size(" 400 X 300 px ") == (400, 300)
    with pytest.raises(ValueError):
        parse_size("banana")


def test_resize_preserves_aspect_with_padding_and_scales_boxes(tmp_path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    store = AnnotationStore(images, annotations, load_schema())
    _make_image(images / "a.jpg", (200, 100))
    _make_image(images / "b.jpg", (400, 400))
    store.save(ImageAnnotation("a.jpg", 200, 100, [
        Box.create("top", "neck", "center", "crew", (20, 10, 100, 50)),
    ]))
    store.save(ImageAnnotation("b.jpg", 400, 400, []))

    report = resize_images(images, annotations, "384x288",
                           backup_dir=tmp_path / "backup")
    assert report.resized == 2
    assert report.unchanged == 0
    assert report.backed_up == 2
    assert report.annotations_updated == 2
    for name in ("a.jpg", "b.jpg"):
        with Image.open(images / name) as img:
            assert img.size == (384, 288)
    with Image.open(tmp_path / "backup" / "a.jpg") as img:
        assert img.size == (200, 100)  # original preserved

    store = AnnotationStore(images, annotations, load_schema())  # reload combined file
    saved = store.load(images / "a.jpg")
    assert (saved.width, saved.height) == (384, 288)
    assert saved.boxes[0].xyxy == (38, 67, 192, 144)

    again = resize_images(images, annotations, (384, 288), backup_dir=tmp_path / "backup")
    assert again.resized == 0 and again.unchanged == 2
    assert not (tmp_path / "backup" / "a_resized").exists()


def test_app_asks_and_resizes(temp_config: Config, monkeypatch) -> None:
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
        root.withdraw()
    except tkinter.TclError:
        pytest.skip("No display available for tkinter")
    try:
        from annotator import app as app_module
        from annotator.app import DesktopAnnotator

        images = temp_config.path("paths.images_raw")
        images.mkdir(parents=True, exist_ok=True)
        _make_image(images / "a.jpg", (100, 200))
        _make_image(images / "b.jpg", (300, 150))

        monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: True)
        monkeypatch.setattr(app_module.simpledialog, "askstring",
                            lambda *a, **k: "384x288")
        shown = []
        def capture(kind):
            def _show(*args, **kwargs):
                text = kwargs.get("message") or (args[1] if len(args) > 1 else "")
                shown.append((kind, str(text)))
            return _show

        monkeypatch.setattr(app_module.messagebox, "showinfo", capture("info"))
        monkeypatch.setattr(app_module.messagebox, "showwarning", capture("warn"))

        app = DesktopAnnotator(root, config=temp_config,
                               image_dir=images,
                               annotations_dir=temp_config.path("paths.annotations"))
        app.offer_resize()

        for name in ("a.jpg", "b.jpg"):
            with Image.open(images / name) as img:
                assert img.size == (384, 288)
        assert app.image_report.uniform
        assert any("Resized 2 image" in message for kind, message in shown if kind == "info")
    finally:
        root.destroy()
