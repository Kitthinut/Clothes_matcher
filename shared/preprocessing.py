"""Consistent preprocessing for training AND inference (shared/preprocessing.py).

Every image (garment-box crop) goes through the exact same steps on both
backends:

  1. crop the annotated/predicted garment box (None = whole image)
  2. resize to the model input 288 x 384 (W x H, spec section 3.3)
  3. scale to [0, 1] and normalise with ImageNet statistics

Keeping this in ONE module is what guarantees the "consistent preprocessing
between training and inference" requirement — the backends only differ in
which tensor library executes it.
"""

INPUT_W, INPUT_H = 288, 384          # spec: 288 x 384 (W x H)
MEAN = [0.485, 0.456, 0.406]         # ImageNet statistics
STD = [0.229, 0.224, 0.225]


def garment_crop(pil_img, box=None):
    """PIL image + box [x1, y1, x2, y2] in original pixels -> garment crop.
    box=None uses the whole image. Boxes outside the image are clamped."""
    if not box:
        return pil_img
    x1, y1, x2, y2 = box
    return pil_img.crop((max(0, int(x1)), max(0, int(y1)),
                         min(pil_img.width, int(x2)), min(pil_img.height, int(y2))))


def normalize_box(box, width, height):
    """Pixel box -> normalised [y1, x1, y2, x2] in 0..1 (crop_and_resize order).
    None / bad box -> the full-image box."""
    if not box or any(v is None for v in box):
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = box
    return [max(0.0, y1 / height), max(0.0, x1 / width),
            min(1.0, y2 / height), min(1.0, x2 / width)]


def config(garment, arch, extra=None):
    """Preprocessing/config block saved with every checkpoint (spec section
    5.3) so inference code can always rebuild the exact training settings."""
    import platform
    import sys

    base = {
        "garment": garment,
        "arch": arch,
        "input_size": [INPUT_W, INPUT_H],
        "preprocessing": {
            "crop": "garment box (original px) -> resize to input size",
            "resize": "bilinear (torch) / area (tf.image.resize)",
            "scale": "1/255",
            "mean": MEAN,
            "std": STD,
        },
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    base.update(extra or {})
    return base
