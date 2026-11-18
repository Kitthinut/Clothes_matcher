"""Tests for the startup image-size check and friend-kit zip builder."""

from __future__ import annotations

import zipfile

import gradio  # noqa: F401  (keeps import order consistent with other tests)
import yaml
from PIL import Image

from annotator.image_check import inspect_images, show_image_report
from annotator.make_friend_kit import FRIEND_FILES, make_friend_kit
from common.paths import PROJECT_ROOT


def _image(path, size):
    Image.new("RGB", size, "white").save(path)
    return path


def test_inspect_images_uniform_and_mixed(tmp_path) -> None:
    same = [_image(tmp_path / f"a{i}.jpg", (320, 240)) for i in range(3)]
    report = inspect_images(same)
    assert report.total == 3
    assert report.uniform
    assert report.ok
    assert "All images are the same size: 320x240" in report.message()

    mixed = same + [_image(tmp_path / "b.jpg", (640, 480))]
    mixed_report = inspect_images(mixed)
    assert not mixed_report.uniform
    assert not mixed_report.ok
    assert "NOT all the same size" in mixed_report.message()
    assert "640x480" in mixed_report.message()


def test_inspect_images_too_small(tmp_path) -> None:
    report = inspect_images([_image(tmp_path / "tiny.jpg", (100, 100))])
    assert not report.ok
    assert report.too_small == ["tiny.jpg"]
    assert "smaller than 224 px" in report.message()


def test_show_image_report_uses_info_or_warning(monkeypatch) -> None:
    from tkinter import messagebox

    calls = []
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: calls.append(("info", k.get("parent"))))
    monkeypatch.setattr(messagebox, "showwarning", lambda *a, **k: calls.append(("warn", k.get("parent"))))

    show_image_report("parent", inspect_images([]))  # empty -> warning
    assert calls[-1][0] == "warn"
    show_image_report("parent", inspect_images([__import__("pathlib").Path("x.jpg")]))  # unreadable -> warning
    assert calls[-1][0] == "warn"


def test_friend_kit_zip_has_only_needed_files(tmp_path) -> None:
    target = make_friend_kit(out=tmp_path / "lite.zip", root=PROJECT_ROOT)
    with zipfile.ZipFile(target) as archive:
        names = {n.replace("\\", "/") for n in archive.namelist()}
        guide = archive.read("Raja-Fashion/SHARE_WITH_FRIENDS.md").decode("utf-8")
        friend_cfg = yaml.safe_load(archive.read("Raja-Fashion/config.yaml").decode("utf-8"))
    expected = {f"Raja-Fashion/{name}" for name in FRIEND_FILES}
    assert expected.issubset(names)
    assert not any("dataset" in n.lower() or n.endswith(".pt") for n in names)
    assert not any(n.startswith("Raja-Fashion/search/") or n.startswith("Raja-Fashion/train/") for n in names)
    assert "no `.txt` file" in guide
    assert "Raja-Fashion/annotator/resize_tool.py" in names
    assert friend_cfg["paths"]["images_raw"] == "images"
    assert friend_cfg["paths"]["annotations"] == "."
