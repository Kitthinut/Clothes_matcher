# Raja-Fashion

End-to-end clothing **attribute detection** + **similarity search** built on the
latest Ultralytics YOLO release. Two-stage pipeline:

1. **Stage 1 — detection:** one YOLO detect model finds `top` / `bottom` boxes.
2. **Stage 2 — classification:** one YOLO classification model **per attribute**
   runs on each cropped garment.

All garments, attributes and class labels live in [`schema.yaml`](schema.yaml),
which is the single source of truth for the annotator, dataset export, training,
database, scoring, UI and tests. Adding an attribute needs no code changes.

---

## 1. Model choice (verified 2026-11-18)

- Installed with `pip install -U ultralytics` → `ultralytics.__version__ = 8.4.215`.
- Official docs (<https://docs.ultralytics.com/models/>) list **YOLO26** as the
  newest **stable** family (released Jan 2026, all seven tasks including
  classification). YOLO27 is "coming soon" with no weights; YOLOv8 is legacy.
- The installed package ships `cfg/models/26/yolo26.yaml` and
  `cfg/models/26/yolo26-cls.yaml`.
- The chosen family is stored in **one config variable**:
  `model.family: yolo26n` in [`config.yaml`](config.yaml). Detection weights
  (`yolo26n.pt`) and classifier weights (`yolo26n-cls.pt`) are derived from it.

### License notice (AGPL-3.0)

Ultralytics YOLO is distributed under the **AGPL-3.0** license. If you modify
Ultralytics code or link it into your own application and distribute that
application (including as a hosted network service), AGPL-3.0 generally requires
the complete corresponding source to be made available to users under AGPL-3.0.
Training weights and datasets may carry additional terms. Review the license at
<https://github.com/ultralytics/ultralytics/blob/main/LICENSE> and with your
legal counsel before commercial or hosted deployment. This project's own code
is not intended to override that obligation.

---

## 2. Setup

Python 3.10+ (tested on 3.12):

```bash
# Desktop annotator only — lightweight, no YOLO/GPU/account needed
pip install -r requirements.txt

# Only if you want training / search / the optional browser annotator:
pip install -r requirements-full.txt
```

Verify:

```bash
python -c "import ultralytics; print(ultralytics.__version__)";  # 8.4.x
pytest -q
```

All commands below assume the project root as the working directory.

### Mock/sample data (no real data or weights needed)

A tiny synthetic dataset and catalog are included and were used for the smoke
test. Regenerate at any time:

```bash
python -c "from common.mock_data import ensure_mock_dataset; ensure_mock_dataset(force=True)"
```

This writes `sample_data/images/*.jpg`, one `sample_data/annotations/annotations.json`
and `sample_data/catalog/products.json` (+ product images).

### Where to put your own data

| Folder | What goes there | Written by |
|---|---|---|
| `sample_data/images/` | **Your own photos to label** (`.jpg`, `.png`, `.webp`) — replace or add to the synthetic samples | You |
| `sample_data/annotations/` | `annotations.json` (all boxes + attributes — the source of truth), `_session.json` (resume state) | The annotator |
| `sample_data/catalog/` | Product images + `products.json` used by the search database | You |
| `datasets/detect/` | Generated `data.yaml`, `images/{train,val}`, `labels/{train,val}` | `annotator.export` — never edit by hand |
| `datasets/cls/` | Generated per-attribute crops `train/<class>/*.jpg` | `annotator.export` — never edit by hand |
| `models/`, `db/` | Weights, labels/metrics, SQLite database | Training / ingest |

Typical flow with your own photos:

1. Copy your photos into `sample_data/images/` (or point the tools at another
   folder — see below). Remove `mock_*.jpg` if you do not want the samples.
2. `python -m annotator.app` — label them.
3. `python -m annotator.validate --min-samples 3` — check imbalance/missing labels.
4. `python -m annotator.export` — build `datasets/`.
5. Train (section 4) and ingest your catalog (section 5).

To keep photos outside the project, all four tools accept the same flags:

```bash
python -m annotator.app        --images "D:\my_photos" --annotations "D:\my_labels"
python -m annotator.validate   --images "D:\my_photos" --annotations "D:\my_labels"
python -m annotator.export     --images "D:\my_photos" --annotations "D:\my_labels" --out datasets
python -m annotator.merge_kits "D:\returned_kit" --images "D:\my_photos" --annotations "D:\my_labels"
```

Without flags, the folders come from `paths.images_raw` and `paths.annotations`
in [`config.yaml`](config.yaml). Editing those two lines changes the defaults
for every command.

**Sharing the labelling work:** run
`python -m annotator.make_friend_kit` to build `Raja-Fashion-lite.zip` — a small
package with only the annotator, schema, config, lightweight requirements and
[`SHARE_WITH_FRIENDS.md`](SHARE_WITH_FRIENDS.md). Friends do not need the full
project, YOLO, a GPU or an account. They send back images + `*.json` only;
merge with `python -m annotator.merge_kits <kit>`.

---

## 3. Annotation workflow (desktop pop-up window)

```bash
python -m annotator.app        # opens a real Tkinter window (no browser, no server)
```

**Schema v2 = multi-class object detection.** Each attribute region gets its own
box and its own YOLO class (`top_neck_crew`, `top_sleeves_left_long`,
`bottom_color_black`, …). 65 detection classes are generated from `schema.yaml`.

- The right panel has one compact **TOP / BOTTOM toggle** and a row of attribute
  chips: Top → Length, Clothes type, Pattern, Neck type, Sleeves L, Sleeves R,
  Colour; Bottom → Waist length, Bottom length, Pattern, Fit type, Colour.
  Click a chip (or press **Tab / Shift+Tab**, **T / B**), then **drag on the
  image** to draw that attribute's box. The highlighted chip and the bottom hint
  bar always show the real active state. Annotated chips show `✓ value`.
- Each box gets a **distinct colour per attribute** and a small text tag. Click
  a box to select it, drag it to move, drag its corner handles to resize, and
  press **Delete** to remove it. Boxes are stored in original-image pixels.
- Choosing a box shows large **label buttons** with the existing option values
  (e.g. `cropped / regular`); click one or press **1–9**. The first option is
  used until you choose; if no box is selected, `1–9` moves between chips.
- **Sleeves use two boxes:** left arm and right arm are separate regions/classes
  (`top_sleeves_left_long`, `top_sleeves_right_long`).
- **Colour is fully automatic:** draw a *Colour* box over a representative patch
  of the garment. HSV + k-means names the dominant colour (shadows/highlights
  and skin/background are downweighted; red wraps around 0/360). A read-only
  swatch with the hex, HSV values and friendly name updates live as you move or
  resize that box. There is no manual colour picker. Top and bottom colours are
  stored separately, including their HSV values.
- An image can contain **only a top, only a bottom, or both**. Tick **top absent**
  / **bottom absent** to grey out that garment's chips.
- **All annotations live in ONE file**: `sample_data/annotations/annotations.json`
  (or `--annotations`). It maps each image name to its size, region boxes
  (`garment`, `region`, `side`, `label`, `xyxy`, optional `hsv`) and
  `absent_garments`. No per-image `.json` or `.txt` files are written;
  `annotator.export` derives the YOLO `datasets/detect/labels/*.txt` from it.
  If old per-image `*.json` files exist, they are merged into `annotations.json`
  on first launch and kept as `*.json.bak`.
- When the app opens it checks every image and shows a pop-up reporting whether
  they are all the same size and large enough (>= 224 px per side). If sizes
  differ, it asks whether to resize **all** images to one target size (default
  **384x288 px**, 4:3): each image is scaled with Lanczos and centred on a black
  canvas so nothing is distorted or cropped; originals are copied to
  `images_original_backup/`, and existing annotation boxes are scaled/offset to
  the new coordinates. The **Check sizes** button reopens the dialog.
- Resume state (`_session.json`) remembers the last image and skips. Wheel zooms,
  middle-drag or Space+drag pans.

Shortcuts: **T/B** garment, **Tab** chips, **1–9** label/chip, **←/→** navigate,
**Enter** save + next, **Delete** remove, **Ctrl+Z / Ctrl+Y** undo/redo,
**Ctrl+S** save, **Ctrl+N** next unlabelled, **F1** help.

Upgrading from the old one-box-per-garment format? Run once (keeps `.v1.bak` backups):

```bash
python -m annotator.migrate_v1_to_v2
```

Optional browser version (same files/format): `python -m annotator.app_web`
→ http://127.0.0.1:7860.

Validate and export:

```bash
python -m annotator.validate --min-samples 3   # imbalance + missing labels report
python -m annotator.export                     # build datasets/ from annotations
```

The export produces:

```
datasets/detect/{data.yaml,images/{train,val},labels/{train,val}}
datasets/cls/<garment>_<attribute>/{train,val}/<class>/*.jpg
datasets/cls/tasks.yaml
```

The train/val split is seeded and shared between both stages. Crops come from
the annotated boxes; an invalid or missing attribute value falls back to the
first class in the schema with a warning (check the validation report first).

---

## 4. Training

```bash
# Stage 1 — detector. Weights -> models/detect/best.pt
python -m train.train_detect --epochs 80 --imgsz 640 --batch 16 --device 0

# Stage 2 — one classifier per attribute. Weights -> models/cls/<key>/best.pt
python -m train.train_attributes --epochs 30 --imgsz 224 --batch 32 --device 0
```

Shared CLI options: `--epochs`, `--imgsz`, `--batch`, `--device` (all optional;
defaults come from `config.yaml`). Add `--export-onnx` to also write
`best.onnx` beside each checkpoint.

Augmentation defaults are fashion-aware: **no vertical flip** (`flipud: 0.0`),
mild HSV jitter, small rotations/translation/shear. Attributes with too few
samples (`train.min_total_samples`, `train.min_samples_per_class`) are **skipped
with a warning** rather than failing the run.

Both scripts write:

- `models/labels.json` — detection names + per-classifier class lists/weights.
- `models/labels.txt` — plain detection class list.
- `models/metrics.json` — mAP50 / mAP50-95 for detection, top-1 per classifier.

Mock metadata-only mode (no training, no downloads):

```bash
python -m train.train_detect --mock
python -m train.train_attributes --mock
```

> First real training downloads the `yolo26n*.pt` weights from Ultralytics.
> **Schema v2 changed the detector from 2 classes to 65 region classes** — any
> detector/classifiers trained with the old schema must be retrained before
> search uses them; until then the pipeline runs in annotation-backed mock mode.

---

## 5. Search & user database

SQLite (`db/raja_fashion.sqlite3`) with plain `sqlite3`: `products`, `garments`,
`garment_attributes`, `garment_embeddings`. Catalog images live in
`sample_data/catalog/`.

```bash
python -m search.ingest            # analyse + store catalog (idempotent, resumable)
python -m search.ingest --force    # re-analyse existing products
python -m search.app               # Gradio UI  -> http://127.0.0.1:7861
uvicorn search.api:app --reload --port 8000     # FastAPI -> /docs
```

The Gradio UI offers, for every attribute from the schema: **ignore**, a
**weight slider 0–5** (soft preference), or **lock** (hard filter), plus a
garment-type choice (`top` / `bottom` / `both`), a visual-similarity weight and
top-K. Results show images, scores and per-attribute explanations.

Score:

```
base = Σ(weight_a × match_a) / Σ(weight_a)
match_a = P_reference(candidate_label)          # soft match
score = (base + embedding_weight × cosine_sim) / (1 + embedding_weight)
```

A locked attribute that does not match excludes the candidate.

FastAPI example:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -F "image=@sample_data/images/mock_0000.jpg" \
  -F 'garment_type=top' \
  -F 'filters={"length":{"mode":"lock","weight":3},"pattern":{"mode":"soft","weight":2}}' \
  -F 'embedding_weight=1' -F 'top_k=10'
```

Other endpoints: `GET /health`, `GET /stats`.

### Embeddings (swappable)

`search/embeddings.py` exposes `EmbeddingBackend`. Default (`yolo`) uses the
pooled penultimate layer of a trained YOLO classifier — no extra dependencies.
`clip` (optional `open-clip-torch`) and `mock` (deterministic colour/thumbnail
features) are also implemented. Select via `search.embedding_backend` in
`config.yaml`. When no trained checkpoint exists the pipeline automatically
falls back to mock embeddings, which is why the sample catalog can be searched
immediately.

### Mock analysis

With no trained weights, `Analyzer(mode="auto")` uses annotation-backed boxes
and attribute probabilities for images that have annotation JSON files (and a
deterministic central box otherwise). After real weights are present in
`models/`, the same API switches to YOLO inference automatically.

---

## 6. Project structure

```
Raja-Fashion/
├─ schema.yaml  config.yaml  requirements.txt  requirements-full.txt  pytest.ini
├─ README.md  SHARE_WITH_FRIENDS.md
├─ common/      config, schema v2 loader, HSV colours, paths, logging, mock data
├─ annotator/   desktop Tk window (app.py), canvas/box widget, image-size check + resize,
│               friend-kit zip builder, JSON store, export, validate,
│               merge_kits, v1→v2 migration, optional Gradio UI (app_web.py)
├─ train/       train_detect.py, train_attributes.py, shared helpers
├─ search/      analyze, embeddings, db, ingest, scoring, Gradio app, FastAPI
├─ models/      weights, labels.json, labels.txt, metrics.json
├─ datasets/    detect/ and cls/ exports
├─ db/          raja_fashion.sqlite3
├─ sample_data/ synthetic images, annotations and catalog
└─ tests/       schema, annotator, export, train, scoring, full pipeline
```

## 7. Adding a new attribute

1. Add `attribute_name: [class_a, class_b, ...]` under a garment in `schema.yaml`.
2. Run `python -m annotator.validate` and `python -m annotator.export`.
3. Re-run training: a new classifier dataset and model appear automatically.
4. The search UI/API gains the new filter with no code change.

## 8. Tests

```bash
pytest -q           # 37 tests: schema, store/UI, export, training helpers,
                    #           scoring, analyze/ingest/search/API/UI smoke,
                    #           desktop logic, kit merge, custom folders
```
