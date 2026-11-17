"""Image pre-check / dataset quality control (image_precheck.py).

Validates images BEFORE they enter annotation packages (project spec §13).
Original images are never modified — resizing/padding happens later in model
preprocessing; images with a non-3:4 aspect ratio are flagged for review, not
rejected.

Checks:
  - readability / corrupted files
  - file format whitelist
  - width/height bounds (> 1000 and < 3000 px per dimension)
  - aspect ratio close to 3:4 (W x H)   -> flag for review if far off
  - exact duplicates (MD5 file hash)
  - visual duplicates (8x8 average-hash, Hamming distance)

Outputs (in --report-dir, default <folder>/precheck_report):
  precheck_report.csv   one row per image: status + reasons
  valid_images.txt      images ready for annotation packages
  review_images.txt     usable but need a human look (aspect/duplicates)
  rejected_images.txt   unusable
  summary.json          dataset statistics

Usage:
  python image_precheck.py dataset/top/images
  python image_precheck.py "D:/photos" --report-dir reports/top
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_DIM = 1000   # spec: both dimensions strictly greater than 1000 px
MAX_DIM = 3000   # spec: both dimensions strictly less than 3000 px
TARGET_ASPECT = 3 / 4          # width / height
ASPECT_TOLERANCE = 0.10        # ±10% around 3:4 counts as on-target
# max total Hamming distance (of 192 bits, 3 channels) for "likely same photo".
# Calibrated on test data: re-scaled re-uploads land at 0-7, unrelated photos
# at 20+. Matches are flagged for human review, never auto-rejected.
VISUAL_DUP_DISTANCE = 10


def file_hash(path, chunk=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash_channel(channel):
    """One RGB channel as a flat 9x8 row-major list -> 64-bit difference hash
    (each bit: pixel brighter than its right neighbour)."""
    v = 0
    for r in range(8):
        for c in range(8):
            if channel[r * 9 + c] > channel[r * 9 + c + 1]:
                v |= 1 << (r * 8 + c)
    return v


def image_hash(img):
    """Colour-aware 9x8 difference-hash (dHash) -> tuple of 3 x 64-bit ints,
    one per RGB channel. Luminance-only hashes cannot tell a red shirt from a
    blue shirt, so each channel is hashed separately."""
    rgb = img.convert("RGB").resize((9, 8), Image.Resampling.LANCZOS)
    data = rgb.tobytes()  # row-major, 3 bytes per pixel
    return tuple(dhash_channel(list(data[k::3])) for k in range(3))


def image_hash_distance(a, b):
    """Total Hamming distance across the three channel hashes."""
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def hash_informative(h):
    """Flat images hash to (nearly) the same value on every channel and would
    match everything; require a minimum number of set bits overall."""
    return sum(bin(x).count("1") for x in h) >= 8


def precheck_folder(img_dir, min_dim=MIN_DIM, max_dim=MAX_DIM,
                    target_aspect=TARGET_ASPECT, tolerance=ASPECT_TOLERANCE):
    """Run all checks. Returns (rows, summary).

    rows: list of dicts {file, width, height, status, reasons, duplicate_of}
    status ∈ {valid, review, rejected}.
    """
    img_dir = Path(img_dir)
    files = sorted(p for p in img_dir.rglob("*") if p.is_file())
    rows, hashes, vhashes = [], {}, {}

    for p in files:
        reasons, duplicate_of = [], None
        suffix = p.suffix.lower()
        rel = p.relative_to(img_dir).as_posix()

        if suffix not in FORMATS:
            rows.append({"file": rel, "width": "", "height": "", "status": "rejected",
                         "reasons": f"unsupported format '{suffix}'", "duplicate_of": ""})
            continue

        # --- readability / corruption -------------------------------------
        try:
            with Image.open(p) as im:
                im.verify()  # structural check (catches truncated files)
            with Image.open(p) as im:
                im.load()    # full decode (catches corrupt pixel data)
                w, h = im.size
                sig = image_hash(im)
        except Exception as ex:
            rows.append({"file": rel, "width": "", "height": "", "status": "rejected",
                         "reasons": f"unreadable/corrupted: {ex}", "duplicate_of": ""})
            continue

        # --- exact duplicates ---------------------------------------------
        fhash = file_hash(p)
        if fhash in hashes:
            rows.append({"file": rel, "width": w, "height": h, "status": "rejected",
                         "reasons": "exact duplicate (same file hash)",
                         "duplicate_of": hashes[fhash]})
            continue
        hashes[fhash] = rel

        # --- visual duplicates ---------------------------------------------
        # skip flat/uninformative hashes (solid-colour images all hash the
        # same) so plain backgrounds don't trigger false duplicate matches
        informative = hash_informative(sig)
        dup = next(((f, d) for f, d in vhashes.items()
                    if d is not None and image_hash_distance(sig, d) <= VISUAL_DUP_DISTANCE),
                   None) if informative else None
        if dup:
            duplicate_of = dup[0]
            reasons.append(f"visual duplicate of {dup[0]} "
                           f"(hash distance {image_hash_distance(sig, dup[1])})")

        # --- dimensions ------------------------------------------------------
        if not (w > min_dim and h > min_dim):
            reasons.append(f"too small ({w}x{h}, need > {min_dim}px per dimension)")
        if not (w < max_dim and h < max_dim):
            reasons.append(f"too large ({w}x{h}, need < {max_dim}px per dimension)")

        # --- aspect ratio (flag, do not reject) -------------------------------
        aspect = w / h
        if abs(aspect - target_aspect) / target_aspect > tolerance:
            reasons.append(f"aspect {aspect:.3f} off target 3:4 "
                           f"(allowed ±{tolerance:.0%}) — can be resized/padded")

        status = "review" if reasons else "valid"
        rows.append({"file": rel, "width": w, "height": h, "status": status,
                     "reasons": "; ".join(reasons), "duplicate_of": duplicate_of or ""})
        vhashes[rel] = sig if informative else None

    summary = {
        "folder": str(img_dir),
        "total": len(rows),
        "valid": sum(1 for r in rows if r["status"] == "valid"),
        "review": sum(1 for r in rows if r["status"] == "review"),
        "rejected": sum(1 for r in rows if r["status"] == "rejected"),
        "visual_duplicates": sum(1 for r in rows if "visual duplicate" in r["reasons"]),
        "aspect_flagged": sum(1 for r in rows if "aspect" in r["reasons"]),
        "sizes": {},
    }
    for r in rows:
        if r["width"]:
            summary["sizes"][f"{r['width']}x{r['height']}"] = \
                summary["sizes"].get(f"{r['width']}x{r['height']}", 0) + 1
    return rows, summary


def write_reports(rows, summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # utf-8-sig (BOM) so Excel opens the ± and arrows in reasons correctly
    with open(out_dir / "precheck_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["file", "width", "height", "status",
                                          "reasons", "duplicate_of"])
        w.writeheader()
        w.writerows(rows)

    for status in ("valid", "review", "rejected"):
        names = [r["file"] for r in rows if r["status"] == status]
        (out_dir / f"{status}_images.txt").write_text("\n".join(names), encoding="utf-8")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of images to check")
    ap.add_argument("--report-dir", default=None,
                    help="output folder (default: <folder>/precheck_report)")
    ap.add_argument("--min-dim", type=int, default=MIN_DIM)
    ap.add_argument("--max-dim", type=int, default=MAX_DIM)
    ap.add_argument("--tolerance", type=float, default=ASPECT_TOLERANCE,
                    help="aspect-ratio tolerance around 3:4 (default 0.10)")
    a = ap.parse_args()

    rows, summary = precheck_folder(a.folder, a.min_dim, a.max_dim,
                                    TARGET_ASPECT, a.tolerance)
    out_dir = write_reports(rows, summary, a.report_dir or (Path(a.folder) / "precheck_report"))

    print(f"Checked {summary['total']} images in {summary['folder']}")
    print(f"  valid:    {summary['valid']}  (ready for annotation packages)")
    print(f"  review:   {summary['review']}  (aspect flagged / visual duplicates)")
    print(f"  rejected: {summary['rejected']}")
    if summary["sizes"]:
        print("  dimensions: " + ", ".join(f"{k}: {v}" for k, v in
                                           sorted(summary["sizes"].items(), key=lambda x: -x[1])))
    print(f"Reports -> {out_dir}")


if __name__ == "__main__":
    main()
