"""tf.data pipeline - TensorFlow (2_train/tensorflow/dataset.py).

Theory:
- tf.data = lazy parallel graph (decode -> augment -> batch -> prefetch)
  so the GPU never waits on disk.
- Input = garment-box crop: schema v2 stores the box in original pixels;
  tf.image.crop_and_resize cuts and resizes it to the shared 288x384 input
  entirely inside the graph (no python decode step, stays fast).
- Missing labels: Keras losses have no ignore_index like PyTorch, so a
  missing label is masked with a per-head sample_weight of 0 (weight 0 =
  zero contribution to loss AND metrics). The weights are baked into the
  dataset as part of the tensor structure so they survive shuffling.

Targets:
  single attribute -> int32 class index; weight (n,) with 0 for missing
  multi attribute  -> float32 multi-hot (-1 row = missing); weight (n, n_opts)
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.preprocessing import INPUT_H, INPUT_W, MEAN, STD, normalize_box  # noqa: E402
from shared.schema import load_spec  # noqa: E402
from shared.data import encode_labels  # noqa: E402


def _decode(path, box, labels, weights, train):
    x = tf.io.read_file(path)
    x = tf.io.decode_image(x, channels=3, expand_animations=False)
    x = tf.image.crop_and_resize(tf.expand_dims(tf.cast(x, tf.float32), 0),
                                 tf.reshape(box, (1, 4)), [0],
                                 crop_size=(INPUT_H, INPUT_W))[0]
    if train:  # augmentation: flip + mild colour jitter
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.1)
        x = tf.image.random_contrast(x, 0.9, 1.1)
    x = x / 255.0
    x = (x - MEAN) / STD
    return x, labels, weights


def make_dataset(samples, garment, batch=16, train=True):
    """Sample list (schema v2, with box + width/height) ->
    (tf.data.Dataset yielding (image, labels, weights), n)."""
    spec = load_spec(garment)
    paths, boxes, labels, weights = [], [], {}, {}
    for s in samples:
        paths.append(s["path"])
        w = s.get("width") or 0
        h = s.get("height") or 0
        # missing width/height: fall back to the full-image box
        boxes.append(normalize_box(s.get("box"), w or 10**6, h or 10**6)
                     if w and h else [0.0, 0.0, 1.0, 1.0])
    for k, cat in spec.items():
        enc = [encode_labels(s["attributes"], spec)[k] for s in samples]
        if cat["mode"] == "multi":
            n_opt = len(cat["options"])
            rows = np.zeros((len(samples), n_opt), np.float32)
            mask = np.zeros((len(samples), n_opt), np.float32)
            for i, v in enumerate(enc):
                if isinstance(v, list):          # present (may be empty = none)
                    rows[i, v] = 1.0
                    mask[i, :] = 1.0
                # else: -1 -> missing, target stays all-zero with zero weight
            labels[k] = rows
            weights[k] = mask
        else:
            y = np.array([v if isinstance(v, int) else -1 for v in enc], np.int32)
            missing = y < 0
            # Keras sparse CE rejects -1 even with zero weight, so remap the
            # missing ones to class 0 and zero their sample_weight instead
            labels[k] = np.where(missing, 0, y)
            weights[k] = (~missing).astype(np.float32)
    n = len(samples)

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, np.array(boxes, np.float32), labels, weights))
    if train:
        ds = ds.shuffle(min(n, 2000))
    ds = ds.map(_decode_train if train else _decode_eval,
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, n


# _decode wants (path, box, labels, weights, train) — bind `train` per split
def _decode_train(p, b, y, w):
    return _decode(p, b, y, w, True)


def _decode_eval(p, b, y, w):
    return _decode(p, b, y, w, False)
