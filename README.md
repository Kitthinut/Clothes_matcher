# FRA362-3 Group13 || Raja_fashion

**Find the same (or almost the same) clothes from your own closet given any fashion reference photo.**

A complete open-source pipeline:

1. **Annotate** – label tops & bottoms with detailed attributes (and left/right sleeves)
2. **Train** – progressive YOLO training (start with one attribute, then expand)
3. **Search** – detect attributes + dominant color → rank your closet items

---

## Why this design?

| Goal | Decision |
|------|----------|
| Easy progressive training | Attribute-specific boxes + `tasks.yaml` so you can train “crew vs not-crew” first |
| Friends can help label | Clean JSON sidecar + strict photo checker + detailed HOW_TO_LABEL.md |
| Low storage | Only images + small JSON (no huge annotation databases) |
| Real-world usable | Color matching (HSV + curated palette) + attribute matching |
| Familiar stack | Pure YOLO (Ultralytics) – detection or classification heads |

---

## Project structure

```
raja_fashion/
├── annotate/                  # Labeling tools (helpers will get this folder)
│   ├── check_photos.py        # Size + 4:3 ratio gate
│   ├── label.py               # Main labeling GUI
│   ├── validate.py            # Schema checker before you merge data
│   ├── HOW_TO_LABEL.md        # Detailed guide with examples
│   └── labeling_config/
│       └── schema.yaml        # Allowed classes & attributes
│
├── train/                     # Dataset building + YOLO training + evaluation
│   ├── tasks.yaml             # Progressive tasks (neck_crew, top_length, …)
│   ├── build_dataset.py       # JSON → YOLO dataset (detect or cls)
│   ├── train.py               # Thin wrapper around ultralytics
│   ├── evaluate.py            # Overfit / underfit diagnosis + metrics
│   ├── datasets/              # Generated (git-ignored)
│   └── runs/                  # Training outputs (git-ignored)
│
├── search/                    # Inference + closet search + live demo
│   ├── search.py              # Reference photo → ranked closet matches
│   ├── live_demo.py           # Webcam real-time detection
│   └── color_detect.py        # HSV dominant-color extraction
│
├── data/                      # Your photos + closet database (not in git)
│   └── closet/                # Put labeled closet items here
│
├── requirements.txt
└── README.md                  # ← you are here
```

---

## How the pieces work together

```mermaid
flowchart TD
    A[Raw photos] --> B[check_photos.py]
    B -->|size & 4:3 OK| C[label.py]
    C --> D[photo.json sidecars]
    D --> E[validate.py]
    E --> F[build_dataset.py + tasks.yaml]
    F --> G[YOLO dataset]
    G --> H[train.py]
    H --> I[best.pt weights]
    I --> J[evaluate.py]
    J --> K{Good fit?}
    K -->|No| C
    K -->|Yes| L[search.py / live_demo.py]
    L --> M[Closet matches / real-time boxes]
```

---

## Quick start

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Check photos before labeling
cd annotate
python check_photos.py ../data/my_photos

# 3. Label
python label.py ../data/my_photos

# 4. Validate
python validate.py ../data/my_photos

# 5. Build a first task (crew-neck detector)
cd ../train
python build_dataset.py neck_crew --data ../data

# 6. Train
python train.py neck_crew --epochs 80 --batch 16

# 7. Diagnose
python evaluate.py neck_crew

# 8. Search your closet
cd ../search
python search.py path/to/reference.jpg --db ../data/closet

# 9. Live webcam test
python live_demo.py
```

---

## Labeling philosophy (important)

We do **not** put every attribute into one giant class name.  
Instead each visual region gets its own box:

| Region          | Attribute stored |
|-----------------|------------------|
| length          | regular / cropped |
| button          | none / half / down |
| neck            | crew / v / polo / … |
| pattern         | plain / graphic / … |
| sleeve_left     | short / long / … |
| sleeve_right    | short / long / … |

This makes progressive training trivial: a task only looks at the boxes that contain the attribute it cares about.

Left / right sleeves are **explicitly chosen by the labeler** (never guessed by position), so horizontal flip augmentation is automatically disabled when sleeve side matters.

---

## Tools & versions used

| Tool | Purpose | Docs |
|------|---------|------|
| **Ultralytics YOLO** (YOLOv8 / YOLO11 / YOLO26 when released) | Detection & classification | https://docs.ultralytics.com |
| **Pillow** | Image I/O, EXIF, padding | https://pillow.readthedocs.io |
| **OpenCV** | Webcam + drawing | https://docs.opencv.org |
| **PyYAML** | schema.yaml & tasks.yaml | https://pyyaml.org |
| **Matplotlib + Pandas** | Training curves & evaluation plots | – |
| **tkinter** | Zero-dependency labeling GUI | built into Python |

We intentionally stay on pure open-source packages so friends can run everything without accounts or paid services.

---

## Color detection formula

We convert RGB → HSV (OpenCV ranges: H ∈ [0,179], S,V ∈ [0,255]) then match against a curated palette of fashion-relevant named colors (navy, burgundy, beige, olive…).

For a pixel $(R,G,B)$:

$$
\begin{align*}
M &= \max(R,G,B),\quad m = \min(R,G,B)\\
V &= M\\
S &= \begin{cases}
0 & \text{if } M=0\\
\frac{M-m}{M} & \text{otherwise}
\end{cases}\\
H &= \begin{cases}
0° & \text{if } M=m\\
60° \times \frac{G-B}{M-m} \bmod 360° & \text{if } M=R\\
60° \times \frac{B-R}{M-m}+120° & \text{if } M=G\\
60° \times \frac{R-G}{M-m}+240° & \text{if } M=B
\end{cases}
\end{align*}
$$

Then H is scaled to $[0,179]$.  
A pixel is assigned to the palette entry whose HSV box contains it; the final dominant colors are the top-k names by pixel count.

Palette inspired by the widely-used [W3Schools color groups](https://www.w3schools.com/colors/colors_groups.asp) and CSS Color Module Level 4 named colors.

---

## Evaluation formulas (evaluate.py)

### Detection (mAP)

- **Precision** $= \frac{TP}{TP+FP}$
- **Recall** $= \frac{TP}{TP+FN}$
- **AP** = area under the precision-recall curve
- **mAP50-95** = mean AP over IoU thresholds 0.50 … 0.95 (step 0.05)

### Classification

- **Accuracy** $= \frac{\text{correct}}{N}$
- **95 % Wilson score interval** for accuracy $k/n$:

$$
\hat{p} = \frac{k}{n},\quad
z = 1.96,\quad
c = \frac{\hat{p}+\frac{z^2}{2n}}{1+\frac{z^2}{n}},\quad
h = \frac{z\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}
$$

Interval $= [c-h,\ c+h]$

### Overfitting / underfitting rules of thumb

| Condition | Verdict |
|-----------|---------|
| train_score − val_score > 0.15 (detect) or > 0.08 (cls) | OVERFITTING |
| both train & val scores low | UNDERFITTING |
| val loss bottoms early then rises while train loss keeps falling | OVERFITTING |
| otherwise | GOOD FIT |

These are heuristics, not statistical tests; always look at the sample sizes printed by `evaluate.py`.

---

## Progressive training workflow

1. Label only the **neck** region on ~200 images.
2. `python build_dataset.py neck_crew`
3. `python train.py neck_crew`
4. `python evaluate.py neck_crew`
5. If the model is healthy, expand labeling to length, sleeves, etc. and add new tasks in `tasks.yaml`.

Because every attribute lives in its own box, adding a new attribute never invalidates old labels.

---

## Sending data to the project owner

Friends should only send:

```
friend_name/
├── img001.jpg
├── img001.json
├── img002.jpg
├── img002.json
└── …
```

Zip it and share. The owner runs `validate.py` then merges into the central `data/` folder.

---

## License & credits

- Code: MIT (or whatever the group decides)
- YOLO: Ultralytics AGPL-3.0 (or Enterprise)
- Color palette: public domain / CSS named colors

Happy labeling & searching!
