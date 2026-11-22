#!/usr/bin/env python3
"""
Reliable dominant-color extraction using HSV + a curated named-color palette.

Palette inspired by:
  - https://www.w3schools.com/colors/colors_groups.asp
  - CSS Color Module Level 4 named colors

Usage:
    from color_detect import dominant_colors
    colors = dominant_colors("photo.jpg", top_k=3)
    # -> [{"name": "navy", "hex": "#000080", "pct": 0.42}, ...]

    python color_detect.py path/to/photo.jpg
"""

from collections import Counter
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image, ImageOps

# H in OpenCV range 0-179, S/V 0-255
PALETTE = [
    # Neutrals
    {"name": "black",      "hex": "#000000", "h": (0, 180), "s": (0, 55),   "v": (0, 45)},
    {"name": "white",      "hex": "#FFFFFF", "h": (0, 180), "s": (0, 35),   "v": (200, 255)},
    {"name": "gray",       "hex": "#808080", "h": (0, 180), "s": (0, 45),   "v": (45, 190)},
    {"name": "beige",      "hex": "#F5F5DC", "h": (18, 42), "s": (10, 70),  "v": (170, 255)},
    {"name": "cream",      "hex": "#FFFDD0", "h": (25, 50), "s": (5, 45),   "v": (210, 255)},
    # Blues
    {"name": "navy",       "hex": "#000080", "h": (100, 130), "s": (80, 255), "v": (15, 95)},
    {"name": "blue",       "hex": "#0000FF", "h": (100, 130), "s": (90, 255), "v": (70, 220)},
    {"name": "light_blue", "hex": "#ADD8E6", "h": (90, 120),  "s": (25, 130), "v": (150, 255)},
    # Reds / Pinks
    {"name": "red",        "hex": "#FF0000", "h": (0, 12),   "s": (110, 255), "v": (70, 255)},
    {"name": "burgundy",   "hex": "#800020", "h": (160, 180),"s": (90, 255),  "v": (25, 110)},
    {"name": "pink",       "hex": "#FFC0CB", "h": (145, 175),"s": (35, 160),  "v": (150, 255)},
    # Greens
    {"name": "green",      "hex": "#008000", "h": (40, 85),  "s": (70, 255),  "v": (35, 180)},
    {"name": "olive",      "hex": "#808000", "h": (28, 55),  "s": (50, 200),  "v": (35, 140)},
    # Yellows / Oranges
    {"name": "yellow",     "hex": "#FFFF00", "h": (20, 38),  "s": (90, 255),  "v": (140, 255)},
    {"name": "orange",     "hex": "#FFA500", "h": (8, 25),   "s": (110, 255), "v": (110, 255)},
    # Browns
    {"name": "brown",      "hex": "#8B4513", "h": (8, 28),   "s": (70, 255),  "v": (25, 115)},
    {"name": "tan",        "hex": "#D2B48C", "h": (15, 38),  "s": (35, 130),  "v": (110, 200)},
    # Purple
    {"name": "purple",     "hex": "#800080", "h": (125, 160),"s": (70, 255),  "v": (35, 180)},
]


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (N,3) → OpenCV-style HSV (H 0-179, S/V 0-255)."""
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-6
    idx = (mx == r) & mask
    h[idx] = (60.0 * ((g[idx] - b[idx]) / diff[idx]) + 360.0) % 360.0
    idx = (mx == g) & mask
    h[idx] = 60.0 * ((b[idx] - r[idx]) / diff[idx]) + 120.0
    idx = (mx == b) & mask
    h[idx] = 60.0 * ((r[idx] - g[idx]) / diff[idx]) + 240.0

    s = np.where(mx > 1e-6, diff / mx, 0.0)
    v = mx
    return np.stack([
        (h / 2.0).astype(np.uint8),
        (s * 255.0).astype(np.uint8),
        (v * 255.0).astype(np.uint8),
    ], axis=1)


def _match_palette(h: int, s: int, v: int) -> str:
    best_name, best_dist = "unknown", 1e9
    for c in PALETTE:
        h0, h1 = c["h"]
        s0, s1 = c["s"]
        v0, v1 = c["v"]
        # handle red wrap-around
        in_h = (h0 <= h <= h1) if h0 <= h1 else (h >= h0 or h <= h1)
        if not (in_h and s0 <= s <= s1 and v0 <= v <= v1):
            continue
        hc = (h0 + h1) / 2.0 if h0 <= h1 else ((h0 + h1 + 180) % 180)
        dh = min(abs(h - hc), 180 - abs(h - hc))
        ds = abs(s - (s0 + s1) / 2.0)
        dv = abs(v - (v0 + v1) / 2.0)
        dist = dh * 2.0 + ds * 0.5 + dv * 0.3
        if dist < best_dist:
            best_dist = dist
            best_name = c["name"]
    return best_name


def dominant_colors(image_path: str | Path, top_k: int = 3, sample: int = 25000) -> List[Dict]:
    """
    Return the top-k dominant named colors of an image.

    Each item: {"name": str, "hex": str, "pct": float}
    """
    path = Path(image_path)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((450, 450))
        arr = np.asarray(im).reshape(-1, 3)

    if len(arr) > sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(arr), sample, replace=False)
        arr = arr[idx]

    hsv = _rgb_to_hsv(arr)
    names = [_match_palette(int(h), int(s), int(v)) for h, s, v in hsv]

    cnt = Counter(names)
    total = sum(cnt.values()) or 1
    name_to_hex = {c["name"]: c["hex"] for c in PALETTE}

    result = []
    for name, n in cnt.most_common(top_k + 2):
        if name == "unknown":
            continue
        result.append({
            "name": name,
            "hex": name_to_hex.get(name, "#888888"),
            "pct": round(n / total, 3),
        })
        if len(result) >= top_k:
            break
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python color_detect.py <image> [top_k]")
        sys.exit(1)
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for c in dominant_colors(sys.argv[1], top_k=k):
        print(f"{c['name']:12} {c['hex']}  {c['pct']:.1%}")
