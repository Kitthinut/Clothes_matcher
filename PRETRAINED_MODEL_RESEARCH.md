# Pre-trained Model Research (PRETRAINED_MODEL_RESEARCH.md)

Investigation of existing pre-trained models for the Outfit AI project, per the
project spec (section 11). Goal: pick the **simplest combination** that achieves
the required results — do not use every model.

Date: 2026-11-17. License/pricing details below are as-published by the
respective projects; re-check before redistribution.

---

## 1. Tasks this project needs

| Task | Needed? | Notes |
|---|---|---|
| Garment detection (bounding box) | Partially | Reference outfits can contain Top + Bottom together; closet images are usually single-garment. Our hybrid annotator also produces boxes for training data. |
| Top / Bottom classification | Yes | One softmax head — trivial with transfer learning. |
| Attribute classification (length, type, neck, pattern, sleeves, waist, fit) | Yes | Core of the project; 9 heads over 2 garment types. |
| Colour extraction | Yes | Solved with traditional CV (`shared/colors.py`: GrabCut + k-means in HSV). No deep model needed. |
| Visual embedding (similarity search) | Yes, later | Closet search should compare embeddings, not just labels. |
| Segmentation | Nice-to-have | Improves colour extraction on busy backgrounds. GrabCut already adequate for plain backgrounds. |

---

## 2. Candidate comparison

### Object detection

| Model | Task | Public weights | Clothing attributes? | Fine-tunable | Bboxes | Size / inference | License | Framework |
|---|---|---|---|---|---|---|---|---|
| **YOLOv8/v11** (Ultralytics) | Detection + segmentation | Yes (COCO) | No (COCO has no garment attributes) | Yes, custom dataset needed | Yes | ~6–50 M params, real-time | **AGPL-3.0** (copyleft — restrictive for this coursework) | PyTorch |
| **Faster R-CNN** (torchvision) | Detection | Yes (COCO) | No | Yes | Yes | ~40 M params, slower | BSD (torchvision) | PyTorch |
| **DeepFashion2 trained Mask R-CNN** | Detection + segmentation of 13 garment categories | Community reproductions only (no single official checkpoint) | Category-level only (not our attributes) | Yes | Yes | ~45 M params | Dataset is **CC BY-NC 4.0 (non-commercial)** | PyTorch |

Verdict: detection models bring no attribute recognition and our images are
mostly one garment per photo. The hybrid annotator gives us boxes for free.
**Not used for now**; revisit only if "reference outfit with person wearing
top+bottom" detection becomes unreliable.

### Classification backbones (attribute heads)

| Model | Public weights | Fine-tunable | Size | License | Framework | Notes |
|---|---|---|---|---|---|---|
| **ResNet-18** | Yes (ImageNet, torchvision) | Yes | ~11 M | BSD | PyTorch (+TF copies) | Current PyTorch default. Fast, small, enough for 9 small heads. |
| **ResNet-50** | Yes | Yes | ~25 M | BSD | Both | AI-pytorch used it (270 MB checkpoint); accuracy gain over R18 was modest for this data size. |
| **MobileNetV2** | Yes (Keras) | Yes | ~3.5 M | Apache 2.0 | TensorFlow | Current TF default; Keras has no ResNet-18. |
| **EfficientNet-B0/B3** | Yes (both frameworks) | Yes | 5–12 M | Apache 2.0 | Both | Viable alternative; more augmentation-sensitive. |

Verdict: **keep transfer-learned ResNet-18 (PyTorch) / MobileNetV2 (TensorFlow)**
with per-attribute softmax heads. Already implemented in `2_train/`, meets the
multi-task requirement (spec §5.1 approach 3), and the licence is unrestricted.

### Embedding models (closet similarity)

| Model | Public weights | Output | Size | License | Framework | Notes |
|---|---|---|---|---|---|---|
| **CLIP** (image encoder) | Yes | 512-d aligned image/text embedding | ~150 M | MIT | PyTorch (open_clip for TF-ish use) | Could also zero-shot classify attributes; overkill for training our own heads but excellent as a general similarity embedding. |
| **DINOv2** (ViT-S/B) | Yes | Strong self-supervised embedding | 22–86 M | Apache 2.0 | PyTorch | Best "universal" visual embedding; heavier than ResNet-18, no TF port. |
| ResNet-18 penultimate features | Yes (ours) | 512-d embedding | free | — | Both | Reuse the trained backbone as the embedding source — zero extra model. |

Verdict: **Phase 3/4 uses the trained backbone's penultimate layer** as the
embedding (simplest, consistent between closet indexing and reference
processing). DINOv2/CLIP are listed as an upgrade path if cosine similarity on
our backbone proves weak — they would be an *addition* to the closet index, not
a replacement pipeline.

### Segmentation

| Model | Public weights | Size | License | Notes |
|---|---|---|---|---|
| **SAM / SAM 2** | Yes (ViT-H/B/L) | 300 M+ (big) | Apache 2.0 | Promptable "everything" masks; heavy, needs prompt; overkill for flat-lay photos. |
| GrabCut (OpenCV, traditional) | — | — | — | Already implemented in `shared/colors.py`; adequate on plain backgrounds. |

Verdict: **keep GrabCut**. SAM only if closet photos with cluttered scenes
become a requirement (it is also a large download for annotation machines).

### Datasets worth knowing about

| Dataset | Content | Licence | Usable here? |
|---|---|---|---|
| DeepFashion2 | 491k images, 13 categories, boxes/masks/landmarks | CC BY-NC 4.0 | Reference/pre-training only (non-commercial) |
| Fashionpedia | 48k images, 27 categories + 294 attributes | CC BY 4.0 | **Attribute taxonomy overlap** — useful inspiration for labels; fine-tuning possible but heavy |
| DeepFashion | 800k images, category/attribute/landmark labels | Research-only | Non-commercial; old |
| H&M (Kaggle) | Transactions + product images/text | Competition terms | Not a CV attribute dataset; useful as product-taxonomy reference only |

Our own annotated data remains the correct training source because our label
vocabulary (spec §3.1/3.2) differs from all of the above.

---

## 3. Recommended combination (simplest that works)

1. **Detection / Top-Bottom classification**: not a separate model.
   Closet items are single-garment; the hybrid annotator supplies the garment
   box for training and a light classifier head handles Top/Bottom.
2. **Attribute classification**: transfer-learned **ResNet-18 (PyTorch)** and
   **MobileNetV2 (TensorFlow)** backbones, one softmax head per attribute
   (already in `2_train/`), garment-box crop as input.
3. **Colour**: existing traditional CV pipeline (`shared/colors.py`).
4. **Shape/length/fit**: learned from the classification heads on the garment
   crop + simple geometry (box aspect ratio) as a hint feature — no extra model.
5. **Embeddings**: penultimate features of the same trained backbone
   (upgrade path: DINOv2 or CLIP, documented above).

No new third-party model dependencies are introduced in Phases 1–2. This
satisfies "prefer the simplest combination that achieves the required results".
