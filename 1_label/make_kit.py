"""Create a labelling kit for a friend (1_label/make_kit.py).

A kit is a self-contained annotation package (project spec §12.1):

    dataset/friend_kits/<name>/
    ├── images/            photos to label
    ├── annotations.json   empty schema-v3 annotations
    ├── schema.json        copy of the attribute schema (no script edits needed)
    ├── LABELS.md          cheat sheet of the labels
    └── README.txt         package id + annotator id + instructions

The friend labels it with 1_label/annotate.py, then you merge it with
1_label/merge_annotations.py.

Usage:  python 1_label/make_kit.py --name kawfang --garment bottom \
        --from "D:/some_folder_with_photos"
"""

import argparse
import json
import shutil
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.data import EXTS, SCHEMA_VERSION  # noqa: E402
from shared.schema import load_full, load_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="kit / friend name")
    ap.add_argument("--garment", choices=("top", "bottom"), required=True)
    ap.add_argument("--from", dest="src", default=None, help="folder with photos to copy in")
    a = ap.parse_args()

    kit = ROOT / "dataset" / "friend_kits" / a.name
    if kit.exists() and any(kit.iterdir()):
        raise SystemExit(f"Kit already exists and is not empty: {kit}")

    (kit / "images").mkdir(parents=True)
    package_id = f"{a.name}-{date.today():%Y%m%d}-{uuid.uuid4().hex[:4]}"

    (kit / "annotations.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "package_id": package_id,
        "annotator": a.name,
        "garment": a.garment,
        "image_root": "images",
        "target_size": [288, 384],
        "images": {},
    }, indent=2), encoding="utf-8")

    # schema snapshot: friends see the exact labels without touching any code
    shutil.copy2(ROOT / "shared" / f"attributes_{a.garment}.json",
                 kit / "schema.json")

    copied = 0
    if a.src:
        src = Path(a.src)
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in EXTS:
                shutil.copy2(p, kit / "images" / p.name)
                copied += 1

    # cheat sheet of the exact labels this kit expects
    lines = [f"# {a.garment.upper()} labels for '{a.name}'", ""]
    for cat in load_full(a.garment)["categories"]:
        names = " / ".join(o["name"] for o in cat["options"])
        mode = "(pick ONE)" if cat.get("mode", "single") == "single" \
            else "(tick ALL that apply, or none)"
        lines.append(f"- {cat['title']} {mode}: {names}")
    lines += [
        "",
        "How to label:",
        "1. python -m pip install pillow opencv-python   (once)",
        "2. python 1_label/annotate.py <this kit folder> --garment " + a.garment,
        "3. Add a Garment region with its labels, then add separate typed",
        "   Neck and left/right Sleeve regions when those categories exist.",
        "   Draw each region and press Enter to save it.",
        "4. When done, send the whole folder back.",
    ]
    (kit / "LABELS.md").write_text("\n".join(lines), encoding="utf-8")

    (kit / "README.txt").write_text(
        f"Annotation package {package_id}\n"
        f"Annotator: {a.name}\n"
        f"Garment: {a.garment}\n"
        f"Schema version: {SCHEMA_VERSION}\n"
        f"Created: {date.today():%Y-%m-%d}\n\n"
        f"Label with 1_label/annotate.py (see LABELS.md).\n"
        f"Merge back with 1_label/merge_annotations.py --name {a.name}\n",
        encoding="utf-8")

    print(f"Kit ready: {kit}  ({copied} images copied, package {package_id})")


if __name__ == "__main__":
    main()
