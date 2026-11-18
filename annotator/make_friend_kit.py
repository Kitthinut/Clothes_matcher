"""Build a minimal zip for friends who help with annotation.

The package contains ONLY the desktop annotator and its lightweight helpers —
no YOLO weights, no training/search code, no datasets and no user photos.

Run::

    python -m annotator.make_friend_kit
    python -m annotator.make_friend_kit --out D:\\friend.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common.paths import PROJECT_ROOT

FRIEND_FILES: tuple[str, ...] = (
    "schema.yaml",
    "config.yaml",
    "requirements.txt",
    "SHARE_WITH_FRIENDS.md",
    "annotator/__init__.py",
    "annotator/app.py",
    "annotator/canvas.py",
    "annotator/store.py",
    "annotator/ui_utils.py",
    "annotator/image_check.py",
    "annotator/resize_tool.py",
    "common/__init__.py",
    "common/config.py",
    "common/colors.py",
    "common/logging_conf.py",
    "common/mock_data.py",
    "common/paths.py",
    "common/schema.py",
)


def make_friend_kit(out: str | Path | None = None, root: Path | None = None) -> Path:
    """Create the zip and return its path."""
    base = Path(root) if root is not None else PROJECT_ROOT
    target = Path(out) if out is not None else base / "Raja-Fashion-lite.zip"
    if not target.is_absolute():
        target = base / target
    target.parent.mkdir(parents=True, exist_ok=True)

    missing = [name for name in FRIEND_FILES if not (base / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files for friend kit: {missing}")

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FRIEND_FILES:
            archive.write(base / name, arcname=f"Raja-Fashion/{name}")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the minimal annotator zip for friends")
    parser.add_argument("--out", type=str, default=None, help="Output zip path")
    args = parser.parse_args(argv)
    target = make_friend_kit(args.out)
    print(f"Created {target} with {len(FRIEND_FILES)} files")


if __name__ == "__main__":
    main()
