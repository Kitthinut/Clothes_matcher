# TensorFlow backend guide (2_train/tensorflow)

Same project as the PyTorch version, but built on **Keras 3**
(`tf.keras`). Read the main `../README.md` first — this file only covers
the TensorFlow-specific parts.

---

## 1. Install

**Windows (native, CPU) — cmd:**

```cmd
REM inside the outfit-ai folder
python -m venv .venv-tf
.venv-tf\Scripts\activate
pip install tensorflow pillow numpy
```

> TensorFlow ≥ 2.11 has **no GPU on native Windows** — a Google decision,
> not a driver problem. For GPU see section 3.

**Windows (GPU via WSL2, recommended):**

```bash
# inside WSL2 Ubuntu
cd /mnt/d/FIBO_Year_3/FRA362_Studio/outfit-ai
python3 -m venv .venv-tf && source .venv-tf/bin/activate
pip install "tensorflow[and-cuda]" pillow numpy
python check_gpu.py        # should list your NVIDIA GPU
```

**Old stack (DirectML, discontinued):** only if you must —
Python 3.10 + `pip install tensorflow-cpu==2.10 tensorflow-directml-plugin`.
Note the plugin expects Keras 2 style code, while this folder is written
for Keras 3.

Check everything: `python check_gpu.py` (project root).

---

## 2. Usage

Exactly the same commands as PyTorch, just from this folder:

```cmd
python 2_train\tensorflow\train.py --garment top                REM train
python 2_train\tensorflow\train.py --garment top --test-only    REM test split
python 3_match\match.py ref.jpg --garment top --backend tensorflow
python run_all.py --backend tensorflow                          REM full pipeline
```

Outputs land in `weights/`:

| File | What it is |
|---|---|
| `top_tensorflow.keras` | trained model (Keras native format) |
| `top_tensorflow.json` | spec + config sidecar — `inference.py` reads this to rebuild the heads (input size, preprocessing, versions) |
| `top_tensorflow_metrics.json` | final test report (accuracy/precision/recall/F1 per attribute) |

> **Note:** the PyTorch side is the primary tested path (smoke-trained
> end-to-end on this machine). The TF code mirrors it and is
> syntax/compile-checked, but running it requires a TF environment —
> see section 1 (native Windows venv is Python 3.11, so use WSL2).

---

## 3. How the TF version differs from PyTorch

| Topic | PyTorch (`../pytorch/`) | TensorFlow (this folder) |
|---|---|---|
| Backbone | ResNet18 (torchvision) | MobileNetV2 (Keras Applications — Keras has no ResNet18) |
| Training loop | explicit loop + AMP + cosine LR | `model.fit`, two-phase fine-tuning |
| Phase 1 | — | frozen backbone, heads only, full LR |
| Phase 2 | — | backbone unfrozen, LR x 0.1 (classic transfer-learning recipe) |
| Missing labels (single) | `-1` rows dropped via mask in CrossEntropy | Keras sparse CE **rejects** -1, so a missing label becomes class 0 with `sample_weight = 0` (zero contribution to loss *and* metrics) |
| Missing labels (multi/pattern) | BCEWithLogits with `-1`-row sentinel masking | sigmoid + BCE with sample-weight masking |
| Augmentation | torchvision transforms | `tf.image` ops inside the `tf.data` map |
| Early stopping | manual patience counter | `EarlyStopping(patience=4, restore_best_weights=True)` |

Implementation notes (why the code looks the way it does):

- **`tf.data` pipeline** (`dataset.py`): decode → augment → batch → prefetch,
  a lazy parallel graph so the GPU/CPU never waits on disk.
- **Sample weights live *inside* the dataset** as the third tuple element
  `(image, labels, weights)`. If you passed them as arrays to `model.fit`,
  they would not follow the shuffle and would mask the wrong samples.
- **Layer names are flattened in Keras 3**, so `unfreeze_backbone()` walks
  `model.layers` instead of `get_layer("mobilenetv2_1.00_224")`.
- Head names (`length`, `neck`, ...) become Keras output names; each gets its
  own `SparseCategoricalCrossentropy` loss and accuracy metric in `compile()`.

---

## 4. Troubleshooting (TF-specific)

| Problem | Fix |
|---|---|
| `GPUs = none` in `check_gpu.py` | expected on native Windows TF ≥ 2.11 — use WSL2 for GPU, or train on CPU (this dataset trains in minutes) |
| `No such layer: mobilenetv2_1.00_224` | you are running old code — Keras 3 flattens layers; `unfreeze_backbone()` handles it |
| `label value of -1 outside valid range` | fixed already: missing labels are remapped to class 0 + weight 0 in `dataset.py` |
| `shuffle=True was passed, but will be ignored` | harmless warning — shuffling happens inside `tf.data` (`dataset.py`) |
| `Saved model refuses to load` | the `.json` spec sidecar must sit next to the `.keras` file |
| Keras 2 vs 3 errors (e.g. `AdamW` missing) | TF 2.10/DirectML ships Keras 2; this folder targets Keras 3 (TF ≥ 2.16) |

---

## 5. Grad-CAM boxes (what the model looks at)

```cmd
python 2_train\tensorflow\visualize.py ref.jpg --garment top          REM popup
python 2_train\tensorflow\visualize.py ref.jpg --save weights\visual_debug_tf
python 2_train\tensorflow\train.py --garment top --test-only --show 4
```

One coloured box + label + confidence per attribute, in a popup
(Left/Right browse, Esc close). See main README section 5 for the theory.
Implementation: `gradcam.py` builds a grad model from MobileNetV2's
`out_relu` conv map + one `GradientTape(persistent=True)` forward; the
heatmap is thresholded to a box in `shared\visualize.py`.
