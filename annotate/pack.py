#!/usr/bin/env python3
"""
Pack your labeled photos into one small zip to send back.

    python pack.py --input my_labeled_folder --contributor ann

What it does:
  1. runs the same checks as validate.py (refuses to pack if there are errors)
  2. fixes phone rotation, REMOVES EXIF (GPS location, camera info), shrinks to max 1024 px
  3. renames files to <contributor>_<number>_<hash>.jpg so merging never collides
  4. writes the final JSONs (with contributor, schema version, image hash)
  5. zips everything into  pack_<contributor>.zip
"""
import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from validate import check_folder, find_image, load_schema

MAX_SIDE = 1024
JPEG_QUALITY = 85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--contributor", required=True, help="short name, letters/numbers only, e.g. ann")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    who = "".join(ch for ch in args.contributor.lower() if ch.isalnum())
    if not who:
        sys.exit("contributor must contain letters or numbers")
    schema = load_schema()
    ok, errors = check_folder(Path(args.input), schema)
    if errors:
        print("Fix these first (run validate.py for the full list):")
        for e in errors[:15]:
            print("  -", e)
        sys.exit(1)

    out_zip = Path(args.out) / f"pack_{who}.zip"
    seen, n = set(), 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for jp in ok:
            meta = json.loads(jp.read_text("utf-8"))
            with Image.open(find_image(jp)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")  # bakes rotation in, drops EXIF
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
            data = buf.getvalue()
            sha = hashlib.sha1(data).hexdigest()
            if sha in seen:
                print("  skipped duplicate photo:", jp.name)
                continue
            seen.add(sha)
            n += 1
            stem = f"{who}_{n:04d}_{sha[:8]}"
            meta.update({"image": stem + ".jpg", "contributor": who, "schema_version": schema["version"],
                         "sha1": sha})
            meta.setdefault("worn", "on_body")
            z.writestr(f"{stem}.jpg", data)
            z.writestr(f"{stem}.json", json.dumps(meta, indent=1))
        z.writestr("contributor.json", json.dumps(
            {"contributor": who, "images": n, "schema_version": schema["version"],
             "packed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"\nPacked {n} images -> {out_zip}  ({out_zip.stat().st_size / 1e6:.1f} MB). Send that file back.")


if __name__ == "__main__":
    main()
