# PyTorch backend guide (2_train/pytorch)

Same project as the TensorFlow version, built on **PyTorch + torchvision**.
Read the main `../README.md` first — this file covers only the
PyTorch-specific parts.

## Install (native CUDA works out of the box)

```cmd
cd D:\FIBO_Year_3\FRA362_Studio\outfit-ai
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision pillow numpy opencv-python
python check_gpu.py   REM should say: CUDA OK -> NVIDIA GeForce RTX 4060...
```

## Usage

```cmd
python 2_train\pytorch\train.py --garment top                REM train
python 2_train\pytorch\train.py --garment top --test-only    REM test split
python 3_match\match.py ref.jpg --garment top --backend pytorch
python run_all.py --backend pytorch                          REM full pipeline
```

Outputs in `weights/`: `top_pytorch.pt` (weights + schema + full config in
one checkpoint), `top_pytorch.csv` (per-epoch loss/accuracy log) and
`top_pytorch_metrics.json` (final test report: accuracy/precision/recall/F1
per attribute).

## How it differs from TensorFlow

| Topic | PyTorch (this folder) | TensorFlow (`../tensorflow/`) |
|---|---|---|
| Backbone | ResNet18 (torchvision) | MobileNetV2 (Keras has no ResNet18) |
| Training | explicit loop: AMP mixed precision, cosine LR, manual early stopping | `model.fit` two-phase fine-tuning |
| Missing labels (single) | `-1` target rows dropped inside CrossEntropy via a mask | remap to class 0 + sample_weight 0 |
| Missing labels (multi/pattern) | BCEWithLogits with `-1`-row sentinel masking | sigmoid + BCE with sample-weight masking |
| Multi-select pattern | sigmoid head + 0.5 threshold (fallback top-1) | same |
| Device | CUDA (native Windows) | CPU on native Windows, CUDA only via WSL2 |
| Head keys | prefixed `h_` in the ModuleDict (the plain key `type` collides with `nn.Module.type()`) | plain names as Keras output names |

Theory notes are in the file headers: transfer learning (ImageNet ResNet18),
multi-task learning (one softmax head per attribute), LR split
(backbone 10x smaller than heads).

## Grad-CAM boxes (what the model looks at)

```cmd
python 2_train\pytorch\visualize.py ref.jpg --garment top             REM popup
python 2_train\pytorch\visualize.py ref.jpg --save weights\visual_debug
python 2_train\pytorch\train.py --garment top --test-only --show 4
```

One coloured box + label + confidence per attribute, in a popup
(Left/Right browse, Esc close). See main README section 5 for the theory.
Implementation: `gradcam.py` hooks ResNet18's layer4 conv map and does one
backward per head; the heatmap is thresholded to a box in
`shared\visualize.py`.
