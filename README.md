# Outfit AI — top & bottom matcher (PyTorch + TensorFlow)

Give the AI a **reference outfit image**, it predicts the fashion attributes
below plus the **HSV colours**, then ranks the clothes in your `closet/`
that fit best — completely free and offline.

| Garment | Categories |
|---|---|
| **Top** | Length (cropped/regular/long), Type (t-shirt/shirt/blouse/sweater/cardigan), Neck (crew/V/square/off shoulder/polo), Pattern — multi-select (solid/dotted/striped/checked/floral/graphic), Sleeves (sleeveless/short/long/puff/camisole) |
| **Bottom** | Waist (low/mid/high-rise), Length (mini/knee-length/cropped/full-length/shorts), Pattern — multi-select (solid/dotted/striped/checked/floral/graphic), Fit (skinny/straight/wide-leg/relaxed/flare) |

Both PyTorch and TensorFlow versions are included and interchangeable.

> **Note (2026-11):** annotations use schema v3 regions. Length, type, and
> pattern labels share one garment region; neck and each sleeve are separately
> boxed and typed. The loader converts these regions to original-image
> coordinates and continues to accept legacy v1/v2 annotations.
> Old trained weights were removed because they predict the outdated labels —
> retrain after labelling: label → merge → split → train (workflow below).

---

## 1. How it works (the theory, in 4 lines)

1. **Transfer learning** — an ImageNet-pretrained backbone (ResNet18 for
   PyTorch, MobileNetV2 for TensorFlow) already knows shapes/textures.
2. **Multi-task learning** — one shared backbone, one small classifier head
   per attribute (softmax for single-choice, sigmoid for multi-select
   pattern); all heads train together on the same garment-crop features.
3. **HSV colour detection** — OpenCV GrabCut cuts the garment out, k-means
   groups the pixels in HSV space, and each cluster gets a colour name
   (HSV keeps brightness separate from hue, so it survives lighting changes).
4. **Weighted search** — a reference image is scored against every closet
   item with a weighted mix: attribute similarity (per-attribute weights,
   set-overlap for multi-select patterns) + dominant-colour overlap +
   cosine similarity of the visual embeddings. No deep model at match time.

Data split: **70 / 15 / 15 train / valid / test**, stratified
(rare label combinations still appear in every split).

---

## 2. Install (VS Code)

```cmd
cd D:\FIBO_Year_3\FRA362_Studio\outfit-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# TensorFlow on Windows only:  pip install tensorflow==2.10.*
```

All commands in this README are written for **cmd** (they work in
PowerShell too, except PowerShell may block venv activation — fix with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`).

If TF and PyTorch fight over package versions, make a second venv
(`.venv-tf`) with tensorflow only and run the TF scripts from that one.

## 3. Folder map

```
outfit-ai/
├─ 1_label/            labelling tools
│  ├─ annotate.py        GUI labeller (garment + typed neck/sleeve regions)
│  ├─ make_kit.py        create a kit for a friend
│  └─ merge_annotations.py  validate & merge kits into the dataset
├─ 2_train/
│  ├─ pytorch/           model.py dataset.py train.py inference.py
│  │                     gradcam.py visualize.py (+ README.md)
│  └─ tensorflow/        model.py dataset.py train.py inference.py
│                        gradcam.py visualize.py (+ README.md)
├─ 3_match/
│  ├─ match.py           weighted closet search (CLI: --w-attr/--w-neck/...)
│  ├─ board.py           outfit board PNG (reference + top matches)
│  └─ search_app.py      same matching, with a window
├─ closet/               YOUR WARDROBE (add_to_closet.py builds it)
│  ├─ closet.json          searchable index: attributes, colours, boxes
│  ├─ images/              self-contained copies of your clothes photos
│  └─ embeddings/          per-item .npy visual features (similarity search)
├─ evaluation/           metrics.py + pytorch_vs_tensorflow.py comparison
├─ image_precheck.py     quality-check images before labelling (CSV report)
├─ shared/              schema JSONs, HSV colours, data loading, split,
│                       preprocessing (one source of truth for both backends)
├─ check_gpu.py         what does each framework see? + fix hints
├─ dataset/
│  ├─ top/    images/ + annotations.json + splits/   (training data)
│  ├─ bottom/ same
│  └─ friend_kits/<name>/  <- GIVE THIS FOLDER TO FRIENDS TO LABEL
└─ weights/              trained model files
```

## 4. Typical workflow

```cmd
REM 0. quality-check new pictures before labelling (optional but recommended)
python image_precheck.py dataset\top\images

REM 1. label your own pictures (or make a friend kit first, see below)
python 1_label\annotate.py --garment top

REM 2. merge friend kits when they send them back
python 1_label\merge_annotations.py

REM 3. split 70/15/15
python shared\split.py --garment top

REM 4. train (either backend)
python 2_train\pytorch\train.py --garment top
python 2_train\pytorch\train.py --garment top --test-only      REM final check
python 2_train\tensorflow\train.py --garment top               REM TF alternative

REM 4b. compare backends on the same test split (needs both installed)
python evaluation\pytorch_vs_tensorflow.py --garment top

REM 5. build your closet from photos of your own clothes (uses the trained model)
python closet\add_to_closet.py --add "D:\my_clothes_photos" --garment top
python closet\add_to_closet.py --list
python closet\add_to_closet.py --rebuild           REM after retraining

REM 6. match a reference outfit image against YOUR closet
python 3_match\match.py my_reference.jpg --garment top
REM tweak the weighting: attribute vs colour vs embedding, or boost one
REM attribute (any schema key works, e.g. --w-pattern 2)
python 3_match\match.py my_reference.jpg --w-attr 0.4 --w-color 0.1 --w-emb 0.5
python 3_match\match.py my_reference.jpg --w-neck 3

REM or the GUI way (searches the closet, double-click to open a match):
python 3_match\search_app.py
```

## 5. Seeing what the AI looks at (Grad-CAM boxes)

The model has no box labels - but at test time Grad-CAM
(Selvaraju et al. 2017) can show WHERE the model looked for each attribute:
for every head, gradients of the predicted class are pushed back to the last
conv layer; the ReLU'd weighted feature map is a heatmap, thresholded to a
box. The popup draws one coloured box + label + confidence per attribute,
e.g. `sleeves: sleeveless 97%`.

```cmd
REM popup window (Left/Right to browse, Esc to close)
python 2_train\pytorch\visualize.py ref.jpg --garment top
python 2_train\tensorflow\visualize.py ref.jpg --garment top

REM save PNGs instead of popup
python 2_train\pytorch\visualize.py ref.jpg --save weights\visual_debug

REM automatic after test / training:
python 2_train\pytorch\train.py --garment top --test-only --show 4
python 2_train\tensorflow\train.py --garment top --test-only --show 4
python 2_train\pytorch\train.py --garment top --show 4 --save-visual weights\visual_debug
```

Boxes are evidence, not ground truth - the heatmap highlights the region
that influenced the decision, which is usually the right area (neckline for
`neck`, the side seams for `sleeves`) but can bleed into the background.

### 5b. The outfit board (one-picture answer)

`match.py --show-board` / `--save-board DIR` renders a single PNG: your
reference outfit on the left with its predicted labels, and the best closet
matches as tiles with score, attributes and colours. The GUI's
"Outfit board (PNG)" button does the same for the current results.

```cmd
python 3_match\match.py my_reference.jpg --garment top --save-board results
```

## 6. Friend kits (labelling without installing everything)

```cmd
python 1_label\make_kit.py --name kawfang --garment bottom --from "D:\photos"
```

- Sends `dataset/friend_kits/kawfang/` to your friend (images + empty
  annotations + `LABELS.md` cheat sheet).
- Friend follows `dataset/friend_kits/HOW_TO_LABEL.md`
  (needs only `pip install pillow opencv-python`).
- You run `python 1_label/merge_annotations.py` — images get the kit name as a
  prefix, annotations merge into the master file (invalid kits, duplicate
  images and conflicting labels are reported, never silently overwritten).

## 7. Backend choice — TF or PyTorch?

| | PyTorch (default) | TensorFlow |
|---|---|---|
| Install on Windows | easy wheels | native wheels stop at TF 2.10 (else WSL2) |
| Model | ResNet18 | MobileNetV2 (Keras has no ResNet18) |
| Training loop | explicit, good for learning | `model.fit` two-phase fine-tuning |
| Code style | identical logic in both | identical logic in both |

Both produce the same predictions format and both work with the same
index/search tools — pick by `--backend`.

## 8. GPU (CUDA) support

Check what each framework sees at any time:

```cmd
python check_gpu.py
```

| Backend | Your CUDA (RTX 4060) | How |
|---|---|---|
| **PyTorch** | works natively | install the normal `torch` wheel - it already ships with CUDA. Training logs show `Device: CUDA (NVIDIA GeForce RTX 4060...)` |
| **TensorFlow** | **not on native Windows** | Google dropped native-Windows GPU after TF 2.10. It is not a driver issue. |

TensorFlow options, free and in order of recommendation:

1. **WSL2 (real CUDA, recommended)** - you already have WSL2. Install an Ubuntu
   distro, then inside WSL (bash uses normal Linux commands):
   ```bash
   cd /mnt/d/FIBO_Year_3/FRA362_Studio/outfit-ai
   python3 -m venv .venv-tf && source .venv-tf/bin/activate
   pip install "tensorflow[and-cuda]"
   python check_gpu.py   # should list your 4060
   ```
   All project files are reachable in WSL at `/mnt/d/...`, so you train with
   the same commands as above and the weights land in the same `weights/` folder.
2. **DirectML plugin (old stack)** - `pip install tensorflow-cpu==2.10
   tensorflow-directml-plugin` with Python 3.10. Works but the plugin is
   discontinued by Microsoft and needs the older Keras 2 code style.
3. **CPU** - TF works fine on CPU for this dataset size (a few minutes per
   training); use PyTorch when you want GPU speed.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `No data. Run split` | label first: `python 1_label/annotate.py --garment top`, then `python shared/split.py --garment top` |
| `No closet at ...` / `no 'top' items yet` | build it: `python closet/add_to_closet.py --add <folder> --garment top` |
| Missing weights error | train once, or point `run_all.py --skip-train` only after a model exists |
| Colours look wrong | plain background, one garment per picture; check with `python -c "from shared.colors import detect_colors; from PIL import Image; print(detect_colors(Image.open('x.jpg')))"` |
| Class accuracy near 0 | that option probably has <30 labelled examples — label more (a friend kit helps) |
