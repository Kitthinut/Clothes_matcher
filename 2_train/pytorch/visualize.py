"""Prediction + Grad-CAM popup (2_train/pytorch/visualize.py).

Theory: see gradcam.py. Every attribute gets a box (where the model looked)
plus its predicted label + confidence, shown in one popup window
(Left/Right to browse several images, Esc to close).

Usage (cmd):
  python 2_train\pytorch\visualize.py img1.jpg img2.jpg --garment top
  python 2_train\pytorch\visualize.py img.jpg --save out_folder   (PNGs instead of popup)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 2_train/pytorch/ -> project root
PY = Path(__file__).resolve().parent
for p in (ROOT, PY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from PIL import Image

from gradcam import GradCAM
from inference import load_model, predict_pil
from shared.preprocessing import INPUT_H, INPUT_W, garment_crop
from shared.visualize import draw_results, remap_box, show_or_save


def analyze(path, bundle, spec, cam, box=None):
    """One image path -> (PIL image, results {key: {label, prob, box}}).
    `box` = annotated garment box; None predicts on the whole image.
    Grad-CAM boxes are mapped from model-input px back to original px."""
    model, tf, dev = bundle
    img = Image.open(path).convert("RGB")
    probs, labels = predict_pil(bundle, spec, img, box=box)
    x = tf(garment_crop(img, box)).unsqueeze(0).to(dev)
    cam_boxes = cam.boxes(x)
    results = {k: {"label": labels[k], "prob": max(probs[k].values()),
                   "box": remap_box(cam_boxes.get(k), box, (INPUT_W, INPUT_H))}
               for k in spec}
    return img, results


def analyze_images(paths, garment, weights=None, boxes=None):
    """List of paths -> [(title, annotated PIL)], model loaded once."""
    weights = weights or ROOT / "weights" / f"{garment}_pytorch.pt"
    bundle, spec = load_model(weights)
    cam = GradCAM(bundle[0])
    items = []
    for i, p in enumerate(paths):
        img, results = analyze(p, bundle, spec, cam,
                               box=boxes[i] if boxes else None)
        items.append((Path(p).name, draw_results(img, results)))
    return items


def popup_from_samples(samples, garment, save_dir=None):
    """shared-data samples ({path, box, ...}) -> popup (used by train.py --show).
    Predicts on the same garment-box crop the model was trained on."""
    paths = [s["path"] for s in samples]
    boxes = [s.get("box") for s in samples]
    items = analyze_images(paths, garment, boxes=boxes)
    show_or_save(items, save_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--garment", choices=("top", "bottom"), default="top")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--save", default=None, help="folder for PNGs instead of popup")
    a = ap.parse_args()
    items = analyze_images(a.images, a.garment, a.weights)
    show_or_save(items, a.save)


if __name__ == "__main__":
    main()
