"""Inference - TensorFlow (2_train/tensorflow/inference.py).

Loads the saved Keras model + spec sidecar and predicts attribute
probabilities for one PIL image (+ optional garment box, as stored in
schema-v2 annotations). Preprocessing matches training
(shared/preprocessing.py). Used by 3_match scripts.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.preprocessing import INPUT_H, INPUT_W, MEAN, STD, garment_crop  # noqa: E402


def _prep(img, box=None):
    """PIL image (+ optional garment box) -> normalised (1, H, W, 3) float."""
    crop = garment_crop(img.convert("RGB"), box)
    x = np.array(crop.resize((INPUT_W, INPUT_H))).astype(np.float32) / 255.0
    return (x - MEAN) / STD


def load_model(path, dev=None):
    """Checkpoint path -> (bundle, spec). bundle = (model, prep fn)."""
    path = Path(path)
    side = path.with_suffix(".json")  # spec sidecar written by train.py
    if not side.exists():
        raise FileNotFoundError(f"Missing spec sidecar {side} (train first)")
    payload = json.loads(side.read_text(encoding="utf-8"))
    spec = payload["spec"]
    model = tf.keras.models.load_model(path)
    return (model, _prep), spec


def predict_pil(bundle, spec, img, box=None):
    """PIL image + optional garment box ->
    (probs {key: {option: prob}}, labels {key: str | list[str]})."""
    model, prep = bundle
    x = tf.constant(prep(img, box)[None, ...])
    outs = model.predict(x, verbose=0)
    names = model.output_names
    probs, labels = {}, {}
    for i, k in enumerate(names):
        arr = outs[i] if isinstance(outs, list) else outs[k]
        probs[k] = {opt: float(p) for opt, p in zip(spec[k]["options"], arr[0])}
        if spec[k].get("mode") == "multi":
            picked = [o for o, v in probs[k].items() if v >= 0.5]
            labels[k] = picked or [max(probs[k], key=probs[k].get)]
        else:
            labels[k] = max(probs[k], key=probs[k].get)
    return probs, labels


def _embedding_model(model):
    """Wrap the MobileNetV2 base (avg-pooled, 1280-d) as its own model so we
    can read the penultimate embedding the heads consume."""
    for layer in model.layers:
        out = getattr(layer, "output", None)
        if out is not None and len(out.shape) == 2 and out.shape[-1] == 1280:
            return tf.keras.models.Model(model.inputs, out)
    raise RuntimeError("could not locate the MobileNetV2 pooling layer for embeddings")


def embed_pil(bundle, img, box=None):
    """PIL image + optional garment box -> (1280,) numpy visual embedding."""
    model, prep = bundle
    x = tf.constant(prep(img, box)[None, ...])
    emb_model = getattr(model, "_embedding_model", None)
    if emb_model is None:
        emb_model = _embedding_model(model)
        model._embedding_model = emb_model  # build once, reuse
    return emb_model.predict(x, verbose=0)[0]
