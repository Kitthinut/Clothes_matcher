"""Tests for kit validation/merging and custom --images / --annotations folders."""

from __future__ import annotations

import shutil

from annotator.export import export_datasets
from annotator.merge_kits import inspect_kit, merge_kit
from annotator.store import DATA_FILENAME, AnnotationStore
from annotator.validate import validate_annotations
from common.config import Config
from common.mock_data import ensure_mock_dataset
from common.schema import load_schema


def _make_kit(temp_config: Config, root, n: int = 3):
    """Create a returned-kit folder from the synthetic annotations."""
    ensure_mock_dataset(n_images=n, n_products=2, config=temp_config, schema=load_schema(), force=True)
    src_images = temp_config.path("paths.images_raw")
    src_ann = temp_config.path("paths.annotations")
    store = AnnotationStore(src_images, src_ann, load_schema())
    for i in range(n):
        store.save(store.load(src_images / f"mock_{i:04d}.jpg"))  # writes .json only
    kit = root / "kit_friend"
    (kit / "images").mkdir(parents=True)
    (kit / "annotations").mkdir(parents=True)
    for i in range(n):
        stem = f"mock_{i:04d}"
        shutil.copy2(src_images / f"{stem}.jpg", kit / "images" / f"{stem}.jpg")
    shutil.copy2(src_ann / DATA_FILENAME, kit / DATA_FILENAME)
    (kit / "README.txt").write_text("Annotator: friend\nDate: 2026-11-18\n", encoding="utf-8")
    return kit


def test_inspect_and_merge_kit(temp_config: Config, tmp_path) -> None:
    kit = _make_kit(temp_config, tmp_path)
    report = inspect_kit(kit)
    assert report["images_found"] == 3
    assert len(report["valid"]) == 3
    assert not report["missing_json"]
    assert not report["problems"]

    dest_images = tmp_path / "merged_images"
    dest_ann = tmp_path / "merged_annotations"
    merged = merge_kit(kit, temp_config, images_dir=dest_images, annotations_dir=dest_ann)
    assert merged["copied_images"] == 3
    assert merged["copied_labels"] == 1
    assert len(list(dest_images.glob("*.jpg"))) == 3
    assert (dest_ann / DATA_FILENAME).exists()
    assert (dest_ann / "kit_friend_README.txt").exists()

    again = merge_kit(kit, temp_config, images_dir=dest_images, annotations_dir=dest_ann)
    assert again["copied_images"] == 0 and again["copied_labels"] == 0


def test_kit_problems_and_conflicts(temp_config: Config, tmp_path) -> None:
    import json

    kit = _make_kit(temp_config, tmp_path, n=2)
    data_path = kit / DATA_FILENAME
    data = json.loads(data_path.read_text(encoding="utf-8"))
    removed = data["images"].pop("mock_0001.jpg")
    data_path.write_text(json.dumps(data), encoding="utf-8")
    report = inspect_kit(kit)
    assert report["missing_json"] == ["mock_0001.jpg"]

    # Restore, then create a conflicting annotation and verify force behaviour.
    data["images"]["mock_0001.jpg"] = removed
    data_path.write_text(json.dumps(data), encoding="utf-8")
    dest_images = tmp_path / "dest_images"
    dest_ann = tmp_path / "dest_ann"
    merge_kit(kit, temp_config, images_dir=dest_images, annotations_dir=dest_ann)

    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["images"]["mock_0000.jpg"]["note"] = "friend changed this"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    skipped = merge_kit(kit, temp_config, images_dir=dest_images, annotations_dir=dest_ann)
    assert "mock_0000.jpg" in skipped["skipped"]

    forced = merge_kit(kit, temp_config, images_dir=dest_images, annotations_dir=dest_ann, force=True)
    assert "mock_0000.jpg" in forced["conflicts"]
    merged_data = json.loads((dest_ann / DATA_FILENAME).read_text(encoding="utf-8"))
    assert merged_data["images"]["mock_0000.jpg"]["note"] == "friend changed this"


def test_custom_dirs_for_validate_and_export(temp_config: Config, tmp_path) -> None:
    kit = _make_kit(temp_config, tmp_path, n=2)
    images = kit / "images"
    annotations = kit
    report = validate_annotations(temp_config, images_dir=images, annotations_dir=annotations)
    assert report["annotated"] == 2
    assert report["missing_labels"] == []

    summary = export_datasets(
        temp_config, images_dir=images, annotations_dir=annotations,
        out_dir=tmp_path / "exported", seed=5,
    )
    assert summary["images_train"] + summary["images_val"] == 2


def test_custom_dirs_for_ui_construction(temp_config: Config, tmp_path) -> None:
    kit = _make_kit(temp_config, tmp_path, n=1)
    images, annotations = kit / "images", kit / "annotations"

    from annotator.app_web import make_app

    assert make_app(config=temp_config, image_dir=images, annotations_dir=annotations) is not None

    tkinter = __import__("tkinter")
    try:
        root = tkinter.Tk()
        root.withdraw()
    except tkinter.TclError:
        return
    try:
        from annotator.app import DesktopAnnotator

        app = DesktopAnnotator(root, config=temp_config, image_dir=images, annotations_dir=annotations)
        assert app.current_path().parent.resolve() == images.resolve()
    finally:
        root.destroy()
