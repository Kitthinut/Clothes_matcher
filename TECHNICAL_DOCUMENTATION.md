# Raja-Fashion — Technical Documentation

Generated: 2026-11-19  
Project root: `D:\FIBO_Year_3\FRA362_Studio\Raja-Fashion`

This document explains **the platform**, **the system architecture**, **how each
file is used**, **what controls what**, and **the theories/formulas behind the
implementation**, ending with the exact tools/libraries used.

---

## 1. Platform and runtime

| Item | Value / role |
|---|---|
| Operating system | Windows (developed and tested here); path logic uses `pathlib`, so the code is portable |
| Language | Python 3.10+ (tested on 3.12), type hints throughout |
| Desktop GUI | `tkinter` / `ttk` — built into Python; no browser or web server needed to annotate |
| Optional web GUI | Gradio (annotator fallback + search UI) |
| API | FastAPI + Uvicorn (`POST /search`) |
| Database | SQLite through the standard-library `sqlite3` module |
| ML framework | Ultralytics YOLO26 (`yolo26n`, `yolo26n-cls`) for detection/classification |
| Image/array maths | Pillow, NumPy, OpenCV (colour detection) |
| Config/data format | YAML (`schema.yaml`, `config.yaml`), JSON (`annotations.json`, model registry) |
| Tests | `pytest` |

Two dependency tiers:

- **`requirements.txt`** — lightweight annotator only: Pillow, NumPy, opencv-python, PyYAML, pytest.
- **`requirements-full.txt`** — adds Ultralytics, Gradio, FastAPI, Uvicorn, python-multipart for training/search.

No GPU, account, cloud service or model download is required for annotation.
GPU is optional and only helps training.

---

## 2. System architecture (big picture)

```
schema.yaml ──► common/schema.py ──► every module (classes, regions, labels)
config.yaml ──► common/config.py ──► paths, model, augment, search defaults

  [ Desktop annotator ]                 [ Optional browser annotator ]
        annotator/app.py                       annotator/app_web.py
        annotator/canvas.py                    annotator/ui/editor_v2.js
              │                                       │
              └──────────────► annotator/store.py ◄────┘
                              annotations.json (ONE file)

   annotations.json ──► annotator/export.py ──► datasets/detect + datasets/cls
                                  │
                                  ▼
                       train/train_detect.py  ──► models/detect/best.pt
                       train/train_attributes.py ─► models/cls/<task>/best.pt
                                  │
                                  ▼
                       models/labels.json + metrics.json
                                  │
                                  ▼
   image ──► search/analyze.py ──► search/ingest.py ──► search/db.py (SQLite)
                                  │
                                  ▼
        search/scoring.py ──► search/app.py (Gradio) / search/api.py (FastAPI)
```

Data-flow principles:

1. **Schema-first:** `schema.yaml` is the single source of truth. Adding a
   region/class changes dropdowns, the YOLO class list, datasets, classifiers,
   filters and tests without code edits.
2. **JSON as the only annotation file:** all images live as keys in one
   `annotations.json`. YOLO `.txt` labels are *generated* by export, never
   hand-maintained.
3. **Two-stage ML:** detector finds attribute-region boxes; one classifier per
   region refines probabilities on each crop.
4. **Loose coupling:** the GUI, exporter, trainer and searcher communicate only
   through files (`annotations.json`, `datasets/`, `models/`).

---

## 3. Control map — what controls what

| Controller | What it controls | Consumers |
|---|---|---|
| `schema.yaml` | Garments, regions, sides, class values, colour names, generated 65 detection classes | all modules |
| `config.yaml` → `paths.*` | Where images, annotations, datasets, models and DB live | annotator, export, validate, ingest, search |
| `config.yaml` → `model.family` | The single weights family name (`yolo26n`); derives `yolo26n.pt` / `yolo26n-cls.pt` | train, analyze, embeddings |
| `config.yaml` → `train.*` | Epochs, batch, imgsz, val split, augmentation, skip thresholds | train scripts |
| `config.yaml` → `search.*` | Embedding backend, weight, top-K default | scoring / search UI |
| `annotator/app.py` | UI state: active garment, active region chip, selected box, undo/redo, autosave, resize prompt | canvas, store |
| `annotator/store.py` | Persistence, progress, session resume, legacy consolidation, JSON validity | app, export, validate, resize, analyze |
| `annotator/export.py` | Deterministic train/val split, YOLO labels, classifier crops | train scripts |
| `models/labels.json` | Runtime class lists + weight paths for inference | analyze |
| `search/db.py` | Catalog schema, FK cascade, idempotent inserts | ingest, scoring, API |
| `search/scoring.py` | Lock/weight/ignore semantics, final ranking | search UI + API |

CLI flags override config for a run: `--images`, `--annotations`, `--out`,
`--epochs`, `--imgsz`, `--batch`, `--device`, `--check`, `--force`.

---

## 4. File-by-file reference

### 4.1 Root files

| File | Purpose / how it is used |
|---|---|
| `.gitignore` | Excludes Python caches, virtualenvs, personal `sample_data/`, generated `datasets/`, `models/`, `db/`, backups, friend zips, PID/log files and editor/OS files. Keeps personal photos and labels out of git. |
| `README.md` | User guide: setup, annotation workflow, training commands, search, sharing, folder layout. |
| `SHARE_WITH_FRIENDS.md` | The guide sent to friends: what is in `Raja-Fashion-lite.zip`, where to put their photos, which file to run, how to label, and exactly what to send back (`images/` + one `annotations.json` + notes). |
| `TECHNICAL_DOCUMENTATION.md` | This document. |
| `schema.yaml` | Schema v2. Declares `color_classes`, per-garment `garment_box`, and `regions` with `title`, `sides`, `classes`, optional `source: hsv`. |
| `config.yaml` | Runtime defaults (paths, model, training, search). |
| `requirements.txt` | Lightweight annotator dependencies. |
| `requirements-full.txt` | Adds ML/web stack for training and search. |
| `pytest.ini` | Test discovery in `tests/`, quiet reporting. |

### 4.2 `common/` — shared foundations

| File | Purpose / how it is used |
|---|---|
| `__init__.py` | Re-exports `Schema` and `load_schema` for convenience. |
| `paths.py` | `PROJECT_ROOT` and `project_path()` / `ensure_dir()`; all relative paths resolve from project root. |
| `schema.py` | Loads/validates `schema.yaml`; `Region`, `Garment`, `Schema` dataclasses; generates 65 detection class names (`top_neck_crew`, `top_sleeves_left_long`, …); `detection_name()` and `parse_detection_name()` map annotation boxes ↔ YOLO classes. |
| `config.py` | Loads `config.yaml` with dot-path `get()`, path resolution and derived weights names. |
| `logging_conf.py` | One consistent logger (`raja_fashion.*`) with stdout handler and optional file handler. |
| `colors.py` | HSV colour engine. `analyze_colors()` returns up to three ranked colours with HSV/hex/percentage; `analyze_color()`/`dominant_color_name()` are compatibility wrappers. Uses conditional filtering, centre weighting, circular hue k-means and name snapping. |
| `mock_data.py` | Generates a tiny deterministic synthetic dataset for smoke tests: images, region-box annotations and a catalog manifest. |

### 4.3 `annotator/` — labelling and dataset preparation

| File | Purpose / how it is used |
|---|---|
| `__init__.py` | Exposes store types. |
| `app.py` | **Main desktop annotator.** Builds the Tk window, TOP/BOTTOM toggle, attribute chips, label buttons, colour cards, box list, progress bar, undo/redo, autosave, F1 help and the startup resize prompt. Owns UI state. |
| `canvas.py` | `AnnotationCanvas(tk.Canvas)`: image display, fit/zoom/pan, geometry hit-testing, live rubber-band rectangle, box select/move/resize, 5 px tiny-box rejection. All coordinates are converted between display and original-image pixels. |
| `ui_utils.py` | `ToolTip` + `tip()` helper for hover hints. |
| `store.py` | `Box`, `ImageAnnotation`, `AnnotationStore`. Reads/writes **one** `annotations.json`; auto-consolidates old per-image JSONs into it with `.bak` backups; progress/session helpers; YOLO label generation for export. |
| `image_check.py` | Inspects every image size, groups by dimensions, checks the 224 px minimum, produces the pop-up report text. |
| `resize_tool.py` | Letterbox resizing in place with black padding, high-quality JPEG save, backup folder, and rescaling of annotation box coordinates by the same transform. |
| `export.py` | Builds `datasets/detect/{images,labels}/{train,val}` + `data.yaml` and `datasets/cls/<task>/{train,val}/<class>/*.jpg` from `annotations.json`. Deterministic seeded split. |
| `validate.py` | Reports missing labels, class imbalance, empty classes and missing required region boxes. |
| `merge_kits.py` | Validates and merges a friend's returned kit (`images/` + one `annotations.json`), with conflict detection and `--check`/`--force`. |
| `migrate_v1_to_v2.py` | Converts legacy one-box-per-garment annotations to v2 region boxes (kept for old data). |
| `make_friend_kit.py` | Builds `Raja-Fashion-lite.zip` with only the annotator, schema, lightweight requirements, friend guide and a friend-specific config (`images/`, root `annotations.json`). |
| `app_web.py` | Optional Gradio browser annotator. Same store/format; custom HTML/CSS/JS canvas bridge. |
| `ui/editor.css` | Styling for the browser fallback. |
| `ui/editor_v2.js` | Browser fallback canvas logic: payload sync, drawing, selection and saving. |

### 4.4 `train/` — model training

| File | Purpose / how it is used |
|---|---|
| `__init__.py` | Package marker. |
| `common.py` | Augmentation defaults, CLI/config merge, sample counting and skip decisions, model registry (`labels.json`, `labels.txt`) and metrics writing, `best.pt` copy helper. |
| `train_detect.py` | Trains the 65-class region detector from `yolo26n.pt`; saves `models/detect/best.pt`, mAP metrics, optional ONNX. |
| `train_attributes.py` | Loops over all schema regions; trains one YOLO classifier per region from `yolo26n-cls.pt`; skips under-populated tasks with a warning; saves `models/cls/<key>/best.pt`, top-1 metrics, optional ONNX. |

### 4.5 `search/` — inference, database and ranking

| File | Purpose / how it is used |
|---|---|
| `__init__.py` | Exports result types. |
| `types.py` | `AttributePrediction`, `GarmentResult`, `AnalysisResult` — the public `analyze()` contract. |
| `analyze.py` | Two-stage inference. Detector parses region class names into `(garment, region, side, value)`; classifiers refine probabilities; HSV fallback adds multi-colour probabilities; groups regions into garments (top-only, bottom-only or both). Mock mode reads `annotations.json`. |
| `embeddings.py` | Swappable `EmbeddingBackend`: YOLO penultimate pooled layer (default), CLIP-style optional, deterministic mock fallback; cosine similarity and L2 normalisation. |
| `db.py` | SQLite schema (`products`, `garments`, `garment_attributes`, `garment_embeddings`), connection helper, inserts and candidate reconstruction with embeddings as float BLOBs. |
| `ingest.py` | Analyses every catalog image and stores results; idempotent/resumable by product id. |
| `scoring.py` | Lock/soft/ignore filters, soft-match weighted average, optional embedding term, top-K ranking and per-attribute explanations. |
| `app.py` | Gradio search UI with schema-generated attribute filters and result gallery. |
| `api.py` | FastAPI app: `/health`, `/stats`, `POST /search` (multipart image + filters JSON). |

### 4.6 `tests/`

| File | What it verifies |
|---|---|
| `conftest.py` | Shared `temp_config` (isolated folders) and one hidden Tk root. |
| `test_schema.py` | v2 loading, 65 generated classes, name ↔ class parsing, validation. |
| `test_annotator.py` | Store round-trip, legacy JSON consolidation, YOLO line generation, browser app construction. |
| `test_colors.py` | Multi-colour detection on stripes, colour list persistence, app swatches. |
| `test_desktop_logic.py` | Drawing, tiny-box rejection, move/resize, coordinate correctness, white/black colour detection. |
| `test_export.py` | Deterministic split, dataset structure, class balance report. |
| `test_image_check.py` | Size reporting and friend-kit zip contents/config. |
| `test_merge_kits.py` | Kit validation, JSON-only merging, conflicts, custom folders. |
| `test_resize.py` | Letterbox transform, box rescaling, backup, app resize dialog. |
| `test_train.py` | Augmentation defaults, skip logic, metadata, mock training. |
| `test_scoring.py` | Soft match, weighted average, lock exclusion, embedding blend, top-K. |
| `test_search_pipeline.py` | Analyze → ingest → search → FastAPI/Gradio smoke tests. |

Generated directories (not code): `datasets/`, `models/`, `db/`, `sample_data/`,
`__pycache__/`, `.pytest_cache/`.

---

## 5. Execution flows

### 5.1 Annotation flow

1. `python -m annotator.app` → `main()` creates `tk.Tk()` and `DesktopAnnotator`.
2. `AnnotationStore` loads `annotations.json`; if absent, legacy per-image JSONs
   are consolidated and backed up.
3. `image_check.inspect_images()` runs; `offer_resize()` shows the pop-up and can
   call `resize_images()`.
4. User picks TOP/BOTTOM + attribute chip; `AnnotationCanvas` converts pointer
   coordinates to original-image pixels; on release a `Box` is created/updated.
5. Colour boxes call `analyze_colors()`; swatches and `colors` field update live.
6. `store.save()` rewrites the single JSON (and `_session.json`).
7. Undo/redo restore snapshots from a 50-step stack.

### 5.2 Training flow

```
annotations.json → export.py → datasets/detect/data.yaml + labels/*.txt
                              datasets/cls/<task>/train|val/<class>/*.jpg
                 → train_detect.py → models/detect/best.pt + metrics
                 → train_attributes.py → models/cls/<key>/best.pt + metrics
                 → models/labels.json (class names + weight paths)
```

### 5.3 Search flow

```
reference image → Analyzer.analyze() → GarmentResult(garments[…], probs, embedding)
catalog images  → ingest.py → SQLite garments/attributes/embeddings
query + filters → score_garment() → ranked results → Gradio / FastAPI
```

---

## 6. Theories and formulas

### 6.1 Schema v2 detection classes

Each region box becomes one YOLO class:

```
class = <garment>_<region>_[<side>_]<value>
examples: top_neck_crew, top_sleeves_left_long, bottom_color_black
```

Total classes = garment-box classes + Σ(regions × sides × values) = **65**.

### 6.2 YOLO label encoding

For a box `(x1,y1,x2,y2)` in an image of size `(W,H)`:

```
cx = (x1 + x2) / (2W)      cy = (y1 + y2) / (2H)
w  = (x2 - x1) / W         h  = (y2 - y1) / H
line = "class_id cx cy w h"
```

This is exactly what `ImageAnnotation.yolo_lines()` produces for
`datasets/detect/labels/`.

### 6.3 Detection and classification learning

YOLO predicts a class probability and a box. Training minimises a combined
loss (classification + box localisation + objectness):

```
L = L_cls + L_box + L_obj
```

Commonly `L_cls` is binary cross-entropy over class logits and `L_box` uses
CIoU; YOLO26 can also use NMS-free end-to-end inference. Validation reports
**mAP50** and **mAP50-95** (mean Average Precision at IoU 0.5 and averaged over
0.5:0.05:0.95).

For each classifier, softmax converts logits `z_k` to probabilities:

```
p_k = exp(z_k) / Σ_j exp(z_j)
L_cls = -Σ_k y_k log(p_k)
```

Top-1 accuracy = fraction of crops where `argmax(p)` equals the true class.

### 6.4 Fashion augmentation rationale

`flipud = 0.0` because a garment has a meaningful top/bottom; `fliplr = 0.5`
mirrors left/right sleeves correctly; HSV jitter is mild (`h=0.01`, `s/v=0.3`);
rotation 5°, translation 0.05, scale 0.4, shear 2°. Early stopping uses
`patience = 10`.

### 6.5 Train/val split

Seeded shuffle, then:

```
n_val = max(1, round(n * val_split))   # val_split = 0.2
train = all images - val
```

The same assignment is used for detection images and classifier crops, so a
crop never appears on the opposite side from its source image.

### 6.6 Colour detection (HSV + k-means)

RGB is converted to HSV `(H∈[0,360), S,V∈[0,1])`. To handle hue wrap-around
(red near 0 and 360), clustering uses circular features:

```
x = [cos(H)·S, sin(H)·S, 1.5·S, 1.5·V]
```

Weighted k-means minimises:

```
J = Σ_i w_i · ||x_i - μ_{c_i}||²
```

`w_i` downweights border pixels, near-black/near-white/low-saturation artefacts
(only when enough other pixels remain), and skin-like pixels. Clusters with the
same name are merged with a circular hue mean. Each cluster's share is
`fraction = n_cluster / Σ n`. Friendly names are snapped to the schema colour
classes (`cream → white`, `grey → gray`, etc.). Up to three clusters above
12% are returned; the largest is the single training label.

### 6.7 Letterbox resize and coordinate transform

Target `(W_t,H_t)`; source `(w,h)`:

```
s  = min(W_t / w, H_t / h)
nw = round(w·s), nh = round(h·s)
ox = (W_t - nw) / 2, oy = (H_t - nh) / 2
x' = round(x·s + ox), y' = round(y·s + oy)
```

The source is resized with Lanczos and pasted onto a black canvas — no
distortion or cropping. Annotation boxes use the same `(s,ox,oy)` transform.

### 6.8 Image-size check

Uniform means exactly one distinct `(width,height)` pair. Acceptable means
uniform **and** `min(width,height) ≥ 224`. Otherwise a warning lists each size
and count.

### 6.9 Embeddings and cosine similarity

The default YOLO embedding is the L2-normalised pooled penultimate classifier
feature vector:

```
v̂ = v / ||v||₂
sim(a,b) = (a·b) / (||a||₂ · ||b||₂)   ∈ [-1, 1]
```

The mock backend uses a normalised concatenation of mean RGB plus an 8×8
greyscale thumbnail (67 dimensions).

### 6.10 Search scoring

For reference `r` and candidate `c`, with active attributes `A`:

```
match_a = P_r(candidate_label_a)          # soft match from reference probs
base    = Σ_{a∈A} weight_a · match_a / Σ_{a∈A} weight_a
score   = (base + λ · sim(r,c)) / (1 + λ)  where λ = embedding_weight
```

`lock` is a hard filter: if `c.label_a != target` the candidate is excluded.
`ignore` or `weight = 0` removes the attribute from both sums. Final score is
clamped to `[0,1]`; results are sorted descending and limited to top-K.

### 6.11 Persistence and idempotency

- `annotations.json`: `{schema_version, images: {filename: {width, height, boxes, absent_garments}}}`.
- `merge_kit` merges by filename key; a conflicting key is skipped unless `--force`.
- `ingest` skips products already in SQLite; re-running with `--force` replaces them.
- SQLite PK/FK: `products.id`, `garments.id`, `garment_attributes(garment_id, attribute)`, `garment_embeddings.garment_id`; cascading deletes keep tables consistent.

### 6.12 Canvas geometry

With displayed scale `s` and origin `(ox,oy)`:

```
image_point = (canvas_x - ox) / s , (canvas_y - oy) / s
canvas_point = ox + image_x·s , oy + image_y·s
```

A drag is a click if movement < 5 screen px; a new box is rejected if either
side is < 5 screen px. Handles belong to the selected box and take priority in
hit-testing.

---

## 7. Tools and libraries actually used

| Tool / library | Used for | Where |
|---|---|---|
| Python 3.10+/3.12 | Language/runtime | everything |
| `tkinter`, `ttk` | Desktop window, widgets, canvas, dialogs | `annotator/app.py`, `canvas.py`, `ui_utils.py`, `image_check.py` |
| Pillow | Image open/resize/letterbox/thumbnail/save | `common/colors.py`, `common/mock_data.py`, `annotator/resize_tool.py`, `search/*` |
| NumPy | Arrays, normalisation, weighted sampling, thumbnail vectors | `common/colors.py`, `search/embeddings.py`, `search/analyze.py` |
| OpenCV (`cv2`) | RGB→HSV, k-means, colour naming | `common/colors.py` |
| PyYAML | Read/write `schema.yaml`, `config.yaml`, exported `data.yaml`, friend config | `common/schema.py`, `common/config.py`, `annotator/export.py`, `annotator/make_friend_kit.py` |
| Ultralytics YOLO26 | Detection and classification train/infer, ONNX export | `train/*`, `search/analyze.py`, `search/embeddings.py` |
| SQLite (`sqlite3`) | Catalog database | `search/db.py`, `search/ingest.py`, `search/api.py` |
| Gradio | Optional browser annotator + search UI | `annotator/app_web.py`, `search/app.py` |
| FastAPI + Uvicorn | `POST /search` HTTP API | `search/api.py` |
| python-multipart | Multipart image upload in FastAPI | `search/api.py` |
| argparse | CLI flags | all CLI entry points |
| dataclasses / json / pathlib / logging | Type-safe state, persistence, paths, logs | all modules |
| pytest | 47 automated tests | `tests/` |

License note: Ultralytics YOLO is **AGPL-3.0**. The annotator itself ships no
YOLO weights and needs no Ultralytics install.

---

## 8. Quick command reference

```bash
# Annotate (main desktop app)
python -m annotator.app

# Validate / export dataset
python -m annotator.validate --min-samples 3
python -m annotator.export

# Train
python -m train.train_detect --epochs 80 --imgsz 640 --batch 16 --device 0
python -m train.train_attributes --epochs 30 --imgsz 224 --batch 32 --device 0

# Search
python -m search.ingest
python -m search.app                          # Gradio UI
uvicorn search.api:app --reload --port 8000   # FastAPI

# Sharing
python -m annotator.make_friend_kit
python -m annotator.merge_kits <kit> --check
python -m annotator.merge_kits <kit>

# Tests
pytest -q
```
