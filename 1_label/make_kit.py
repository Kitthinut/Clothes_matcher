"""Prepare dataset/<garment>/ for annotation (1_label/make_kit.py).

With no arguments, this creates dataset/top/images and initializes
dataset/top/annotations.json if it does not already exist. Existing annotations
and images are preserved.

Usage:
    python 1_label/make_kit.py
    python 1_label/make_kit.py --garment bottom --from "D:/some_folder"
"""

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.data import EXTS, SCHEMA_VERSION  # noqa: E402


def copy_images(src, image_dir):
    """Copy source images without overwriting existing files."""
    src = Path(src)
    if not src.is_dir():
        raise SystemExit(f"Image source folder does not exist: {src}")

    photos = sorted(
        path for path in src.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTS
    )
    conflicts = []
    for photo in photos:
        destination = image_dir / photo.relative_to(src)
        if destination.exists() and not filecmp.cmp(
            photo, destination, shallow=False
        ):
            conflicts.append(destination)

    if conflicts:
        names = "\n".join(f"  {path}" for path in conflicts)
        raise SystemExit(
            "Refusing to overwrite images with different contents:\n" + names
        )

    copied = 0
    for photo in photos:
        destination = image_dir / photo.relative_to(src)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(photo, destination)
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--garment", choices=("top", "bottom"), default="top")
    parser.add_argument(
        "--from", dest="src", default=None, help="optional folder of images to copy"
    )
    args = parser.parse_args()

    dataset_dir = ROOT / "dataset" / args.garment
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = dataset_dir / "annotations.json"
    if annotations_path.exists():
        try:
            annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as ex:
            raise SystemExit(f"Cannot read {annotations_path}: {ex}") from ex
        if not isinstance(annotations, dict):
            raise SystemExit(f"{annotations_path} must contain a JSON object.")
        if annotations.get("garment", args.garment) != args.garment:
            raise SystemExit(
                f"{annotations_path} contains annotations for "
                f"{annotations['garment']}, not {args.garment}."
            )
        initialized = False
    else:
        initialized = True

    copied = copy_images(args.src, image_dir) if args.src else 0
    if initialized:
        annotations_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "garment": args.garment,
                    "image_root": "images",
                    "target_size": [288, 384],
                    "images": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    status = "initialized" if initialized else "preserved existing"
    print(
        f"Dataset ready: {dataset_dir} "
        f"(annotations {status}; {copied} images copied)"
    )


if __name__ == "__main__":
    main()
