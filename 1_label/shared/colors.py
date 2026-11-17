"""HSV colour detection (shared/colors.py).

Theory:
- GrabCut (OpenCV): iterative graph-cut segmentation that separates the
  garment (foreground) from the background using colour models.
- k-means clustering: groups the garment pixels into a few dominant colours.
- HSV colour space: Hue (the colour itself) / Saturation (how vivid) /
  Value (how bright). Naming colours in HSV is more robust to lighting
  than naming them in RGB, because brightness lives only in V.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def segment_garment(bgr, max_side=400):
    """Return uint8 mask (1 = garment) same size as the input image."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))  # GrabCut is slow on huge images
    small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))))
    sh, sw = small.shape[:2]

    # init: middle = probable foreground, thin frame = sure background
    m = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    m[int(sh * 0.06):int(sh * 0.94), int(sw * 0.06):int(sw * 0.94)] = cv2.GC_PR_FGD
    m[int(sh * 0.50):int(sh * 0.70), int(sw * 0.40):int(sw * 0.60)] = cv2.GC_FGD
    c = max(2, int(min(sh, sw) * 0.04))
    for ys in (slice(0, c), slice(sh - c, sh)):
        for xs in (slice(0, c), slice(sw - c, sw)):
            m[ys, xs] = cv2.GC_BGD

    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(small, m, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        fg = np.ones((sh, sw), np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:  # keep only the biggest blob
        fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def auto_box(pil_image, min_fraction=0.05, pad=2):
    """Garment bounding box [x1, y1, x2, y2] from GrabCut segmentation
    (traditional CV - no extra model). Returns None when the foreground is
    too small to trust, in which case callers should use the full image."""
    rgb = np.array(pil_image.convert("RGB"))
    mask = segment_garment(rgb[:, :, ::-1].copy())
    ys, xs = np.where(mask > 0)
    if len(xs) < min_fraction * mask.size:
        return None
    h, w = mask.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)]


def name_from_hsv(h, s, v):
    """(hue 0-360, sat 0-1, val 0-1) -> plain English colour name."""
    if s < 0.12:
        if v < 0.18:
            return "black"
        if v > 0.85:
            return "white"
        return "gray"
    if v < 0.22:
        return "black"
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "brown" if v < 0.55 else "orange"
    if h < 65:
        return "mustard" if v < 0.6 else "yellow"
    if h < 160:
        return "olive" if s < 0.35 else "green"
    if h < 200:
        return "teal"
    if h < 255:
        return "navy" if v < 0.5 else "blue"
    if h < 290:
        return "purple"
    return "pink"


def dominant_colors_hsv(bgr, mask, k=3):
    """K-means in HSV inside the mask -> [{name, hsv, hex, fraction}] sorted."""
    px = bgr[mask > 0]
    if len(px) == 0:
        px = bgr.reshape(-1, 3)
    if len(px) > 20000:  # k-means doesn't need every pixel
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]

    # BGR -> RGB -> HSV. OpenCV returns H 0-180, S/V 0-255; normalise both.
    # reshape: cvtColor wants an (1, N, 3) "image", not a flat pixel list
    hsv = cv2.cvtColor(px[:, ::-1].reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
    hsv = hsv.astype(np.float32)
    hsv[:, 0] *= 2.0          # H: 0-180 -> 0-360
    hsv[:, 1:] /= 255.0       # S, V: 0-255 -> 0-1
    k = max(1, min(k, len(hsv)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, centers = cv2.kmeans(hsv, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lab.ravel(), minlength=k)

    merged = {}
    for (h, s, v), n in zip(centers, counts):
        name = name_from_hsv(h, s, v)
        d = merged.setdefault(name, {"n": 0, "hsv": [h, s, v]})
        d["n"] += int(n)

    total = sum(d["n"] for d in merged.values()) or 1
    out = []
    for name, d in merged.items():
        h, s, v = [float(x) for x in d["hsv"]]  # plain floats -> JSON-safe
        rgb = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[(h / 2.0), s * 255, v * 255]]]), cv2.COLOR_HSV2RGB
        )[0, 0])
        out.append({
            "name": name,
            "hsv": [round(h), round(s, 2), round(v, 2)],
            "hex": "#%02x%02x%02x" % rgb,
            "fraction": round(d["n"] / total, 3),
        })
    return [c for c in sorted(out, key=lambda c: -c["fraction"]) if c["fraction"] >= 0.03]


def detect_colors(pil_image):
    """PIL image (any mode) -> dominant colour list using the HSV pipeline."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    return dominant_colors_hsv(bgr, segment_garment(bgr))
