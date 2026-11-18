"""Automatic garment-colour detection from a Colour box (HSV + k-means).

The annotator no longer asks the user to pick a colour: they draw a box over a
representative patch of the garment and this module names its main colours
(a striped garment returns both colours with their pixel shares).

Robustness measures:
- trim the box border (sloppy boxes often include background),
- ignore near-black/near-white/low-saturation pixels **only when** enough other
  pixels remain (a white or black garment must still be detected),
- downweight skin-like pixels and favour the centre of the box,
- cluster in a circular hue representation (cos/sin) so red near 0/360 wraps
  correctly, then pick the largest cluster.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ColorInfo:
    """Result of automatic colour detection for one box."""

    name: str          # label stored in the annotation (must be a schema class)
    friendly: str      # human-readable name, e.g. "cream"
    hsv: tuple[float, float, float]  # H 0-360, S 0-1, V 0-1
    hex: str
    fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "friendly": self.friendly,
            "hsv": [round(float(v), 3) for v in self.hsv],
            "hex": self.hex, "fraction": round(self.fraction, 3),
        }


# Friendly names that are not necessarily valid schema class labels.
_SNAP = {
    "cream": "white", "offwhite": "white", "ivory": "white", "beige": "white",
    "tan": "brown", "khaki": "brown", "grey": "gray", "darkgray": "gray",
    "maroon": "red", "burgundy": "red", "cyan": "teal", "lightblue": "blue",
    "olive": "green", "mustard": "yellow", "pink": "pink",
}


def name_from_hsv(h: float, s: float, v: float) -> str:
    """Human-readable colour name for (hue 0-360, saturation 0-1, value 0-1)."""
    if s < 0.10:
        if v < 0.16:
            return "black"
        if v > 0.86:
            return "white"
        if v > 0.68:
            return "cream" if h < 60 or h >= 330 else "lightgray"
        if v < 0.35:
            return "darkgray"
        return "gray"
    if v < 0.18:
        return "black"
    if s < 0.20 and v > 0.70:
        return "beige" if h < 60 or h >= 330 else "lightgray"
    if h < 12 or h >= 348:
        return "red" if s > 0.45 else "pink"
    if h < 25:
        return "brown" if v < 0.6 else "orange"
    if h < 45:
        return "orange" if s > 0.5 else "beige"
    if h < 70:
        return "yellow"
    if h < 160:
        return "green"
    if h < 200:
        return "teal"
    if h < 250:
        return "navy" if v < 0.5 else "blue"
    if h < 290:
        return "purple"
    return "pink"


def snap_to_schema(name: str, allowed: set[str] | None) -> str:
    """Map a friendly colour name to a class label present in *allowed*."""
    if not allowed:
        return name
    if name in allowed:
        return name
    for candidate in (_SNAP.get(name), "white" if name in {"cream", "beige"} else None,
                      "gray", "black", "white", "brown", "blue", "green", "red"):
        if candidate and candidate in allowed:
            return candidate
    return sorted(allowed)[0]


def _rgb_from_hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, float(np.clip(s, 0, 1)), float(np.clip(v, 0, 1)))
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _cluster_colors(
    image: Image.Image | np.ndarray, *, k: int = 3, max_pixels: int = 12000
) -> list[ColorInfo]:
    """Cluster crop pixels in HSV and return ranked, de-duplicated colours."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        rgb = arr[:, :, ::-1] if arr.ndim == 3 and arr.shape[2] == 3 else arr
    if rgb.ndim != 3 or rgb.size == 0 or min(rgb.shape[:2]) < 2:
        return None

    h, w = rgb.shape[:2]
    # Trim a sloppy border before deciding anything about colour.
    trim_x, trim_y = int(w * 0.08), int(h * 0.08)
    if w - 2 * trim_x >= 3 and h - 2 * trim_y >= 3:
        rgb = rgb[trim_y:h - trim_y, trim_x:w - trim_x]
    h, w = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hh = hsv[:, :, 0] * 2.0            # OpenCV H 0-180 -> 0-360
    ss = hsv[:, :, 1] / 255.0
    vv = hsv[:, :, 2] / 255.0

    # Shadow/highlight/low-sat filtering is conditional: a white/black garment
    # must keep its pixels when that is all the box contains.
    artefacts = (vv < 0.12) | (vv > 0.95) | (ss < 0.07)
    keep = ~artefacts
    if keep.sum() < 0.25 * keep.size:
        keep = np.ones_like(keep)

    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    sigma = max(6.0, 0.5 * max(w, h))
    dist = np.sqrt(((xs - cx) / sigma) ** 2 + ((ys - cy) / sigma) ** 2)
    weights = np.clip(1.0 - dist, 0.12, 1.0) * keep
    # Downweight skin-like pixels without removing them entirely.
    skin = (hh >= 5) & (hh <= 28) & (ss >= 0.15) & (ss <= 0.55) & (vv >= 0.35) & (vv <= 0.92)
    weights[skin] *= 0.45

    flat_w = weights.ravel()
    total = float(flat_w.sum())
    if total <= 0:
        flat_w = np.ones(h * w, dtype=np.float32)
        total = float(flat_w.sum())
    flat_w = flat_w / total
    n = min(max_pixels, h * w)
    # Sampling WITH replacement keeps this safe when many pixels were filtered
    # to zero weight (e.g. an all-white garment).
    idx = np.random.default_rng(0).choice(h * w, size=n, replace=True, p=flat_w)

    sel_h, sel_s, sel_v = hh.ravel()[idx], ss.ravel()[idx], vv.ravel()[idx]
    rad = np.deg2rad(sel_h)
    # Circular hue features; multiply by S so low-saturation hue noise is weak.
    features = np.stack([np.cos(rad) * sel_s, np.sin(rad) * sel_s,
                         sel_s * 1.5, sel_v * 1.5], axis=1).astype(np.float32)
    k = max(1, min(k, n))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(features, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    total = float(counts.sum()) or 1.0

    # Merge clusters that landed on the same colour name (k-means often splits
    # one colour into two similar shades). Use a circular mean for hue.
    grouped: dict[str, dict[str, float]] = {}
    for (co, si, sat, val), count in zip(centers, counts):
        hue = float(np.rad2deg(np.arctan2(float(si), float(co))) % 360.0)
        sat, val = float(np.clip(sat, 0, 1)), float(np.clip(val, 0, 1))
        friendly = name_from_hsv(hue, sat, val)
        entry = grouped.setdefault(friendly, {"n": 0.0, "cos": 0.0, "sin": 0.0, "s": 0.0, "v": 0.0})
        rad = np.deg2rad(hue)
        entry["n"] += float(count)
        entry["cos"] += float(count) * np.cos(rad)
        entry["sin"] += float(count) * np.sin(rad)
        entry["s"] += float(count) * sat
        entry["v"] += float(count) * val

    results: list[ColorInfo] = []
    for friendly, entry in sorted(grouped.items(), key=lambda kv: -kv[1]["n"]):
        n = entry["n"] or 1.0
        hue = float(np.rad2deg(np.arctan2(entry["sin"], entry["cos"])) % 360.0)
        sat, val = float(np.clip(entry["s"] / n, 0, 1)), float(np.clip(entry["v"] / n, 0, 1))
        rgb = _rgb_from_hsv(hue, sat, val)
        results.append(ColorInfo(
            name=friendly, friendly=friendly, hsv=(hue, sat, val),
            hex="#%02x%02x%02x" % rgb, fraction=entry["n"] / total,
        ))
    return results


def analyze_colors(
    image: Image.Image | np.ndarray, *, k: int = 3,
    allowed: set[str] | None = None, max_colors: int = 3,
    min_fraction: float = 0.12, max_pixels: int = 12000,
) -> list[ColorInfo]:
    """Return up to *max_colors* dominant colours of a garment crop.

    Small stray clusters (background/skin in a slightly sloppy box) are
    ignored; a striped or multi-colour garment returns its main colours ranked
    by pixel share. Names are snapped to *allowed* (the schema colour classes).
    """
    clusters = _cluster_colors(image, k=k, max_pixels=max_pixels)
    if not clusters:
        return []
    results: list[ColorInfo] = []
    seen: set[str] = set()
    for cluster in clusters:
        if results and cluster.fraction < min_fraction:
            continue
        name = snap_to_schema(cluster.friendly, allowed)
        if name in seen:
            continue
        seen.add(name)
        results.append(ColorInfo(
            name=name, friendly=cluster.friendly, hsv=cluster.hsv,
            hex=cluster.hex, fraction=cluster.fraction,
        ))
        if len(results) >= max_colors:
            break
    return results


def analyze_color(image: Image.Image | np.ndarray, *, k: int = 3,
                  allowed: set[str] | None = None, max_pixels: int = 12000) -> ColorInfo | None:
    """Return only the dominant colour (backwards-compatible helper)."""
    colors = analyze_colors(image, k=k, allowed=allowed, max_colors=1, max_pixels=max_pixels)
    return colors[0] if colors else None


def detect_colors(image: Image.Image | np.ndarray, k: int = 3,
                  allowed: set[str] | None = None) -> list[dict[str, Any]]:
    """List-of-dicts form; now includes all significant colours."""
    return [info.to_dict() for info in analyze_colors(image, k=k, allowed=allowed)]


def dominant_color_name(image: Image.Image | np.ndarray, allowed: set[str] | None = None) -> str | None:
    """Return the primary colour name (backwards-compatible helper)."""
    colors = analyze_colors(image, allowed=allowed, max_colors=1)
    return colors[0].name if colors else None
