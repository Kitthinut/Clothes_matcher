"""Inference - PyTorch (2_train/pytorch/inference.py).

Loads a trained checkpoint and predicts attribute probabilities for one
PIL image (+ optional garment box, as stored in schema-v2 annotations).
Preprocessing is identical to training (shared/preprocessing.py).

Used by 3_match/match.py, 3_match/search_app.py and the visualizers.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # 2_train/pytorch/ -> project root
PY = Path(__file__).resolve().parent
for p in (ROOT, PY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import EVAL_TF  # noqa: E402
from model import MultiHeadNet, decode, labels_from_probs  # noqa: E402
from shared.preprocessing import garment_crop  # noqa: E402


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, dev=None):
    """Checkpoint -> (bundle, spec). bundle = (model, transform, device)."""
    dev = dev or get_device()
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = MultiHeadNet(ck["spec"], pretrained=False)
    model.load_state_dict(ck["state"])
    model.eval().to(dev)
    return (model, EVAL_TF, dev), ck["spec"]


@torch.no_grad()
def predict_pil(bundle, spec, img, box=None):
    """PIL image + optional garment box ->
    (probs {key: {option: prob}}, labels {key: str | list[str]})."""
    model, tf, dev = bundle
    x = tf(garment_crop(img, box)).unsqueeze(0).to(dev)
    probs = decode(model(x), spec)
    labels = labels_from_probs(probs, spec)
    return probs, labels


@torch.no_grad()
def embed_pil(bundle, img, box=None):
    """PIL image + optional garment box -> (dim,) numpy visual embedding
    (penultimate backbone features of the loaded checkpoint)."""
    model, tf, dev = bundle
    x = tf(garment_crop(img, box)).unsqueeze(0).to(dev)
    return model.embed(x).squeeze(0).cpu().numpy()
