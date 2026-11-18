"""Shared pytest fixtures: an isolated config pointing at temporary folders."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import Config


@pytest.fixture(scope="session")
def tk_root():
    """One hidden Tk root shared by GUI tests (Tk dislikes repeated roots)."""
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
        root.withdraw()
    except tkinter.TclError:
        pytest.skip("No display available for tkinter")
    yield root
    root.destroy()


@pytest.fixture
def temp_config(tmp_path: Path) -> Config:
    """Return a Config whose every path lives in the test's tmp directory."""
    raw = {
        "project": "Raja-Fashion-test",
        "model": {
            "family": "yolo26n",
            "imgsz": 320,
            "device": "cpu",
            "conf": 0.25,
            "iou": 0.45,
        },
        "paths": {
            "images_raw": str(tmp_path / "images_raw"),
            "annotations": str(tmp_path / "annotations"),
            "catalog": str(tmp_path / "catalog"),
            "datasets": str(tmp_path / "datasets"),
            "models": str(tmp_path / "models"),
            "database": str(tmp_path / "db" / "test.sqlite3"),
        },
        "train": {
            "detect_epochs": 1,
            "cls_epochs": 1,
            "batch": 4,
            "val_split": 0.2,
            "min_samples_per_class": 1,
            "min_total_samples": 4,
            "export_onnx": False,
            "augment": {
                "flipud": 0.0, "fliplr": 0.5,
                "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.3,
                "degrees": 5.0, "translate": 0.05, "scale": 0.4, "shear": 2.0,
            },
        },
        "search": {"default_top_k": 5, "embedding_weight": 1.0, "embedding_backend": "mock"},
    }
    return Config(raw, root=Path("."))
