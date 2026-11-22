# How to Label Clothes (raja_fashion)

Thank you for helping! This guide explains exactly how to draw boxes and choose attributes so the model learns correctly.

**Before you start**
1. Run the size/ratio checker:
   ```bash
   python check_photos.py path/to/your_photos
   ```
   Only keep photos that are **> 1000 px and < 3000 px** on both sides and roughly **4:3 portrait**.
2. Then open the label tool:
   ```bash
   python label.py path/to/your_photos
   ```

---

## Core idea

- One **box per visual region / attribute**.
- For a single shirt you will normally draw **several boxes**:
  - Length box (around the whole torso)
  - Neck box (around the collar / neckline)
  - Button box (around the button placket if any)
  - Pattern box (around the main patterned area)
  - Sleeve LEFT box + Sleeve RIGHT box (around each arm)

The tool forces you to pick the region first, then draw, then choose the value.

---

## 1. Tops

### 1.1 Length

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **regular** | Hem sits at or below the waist / hip line | Draw a box from the shoulders down to the hem. The box should cover the whole torso. Do **not** include skin below the shirt. | ![regular](examples/top_length_regular.jpg) |
| **cropped** | Hem is clearly above the waist and shows midriff | Draw a box from the shoulders to the hem. The box **must include the shirt + the visible midriff/skin** between the top and the bottoms so the model can see it is cropped. | ![cropped](examples/top_length_cropped.jpg) |

---

### 1.2 Button type

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **none** (No-button) | Pullover, hoodie, t-shirt – no buttons or zipper on the front | Draw a small box on the center of the chest where buttons would normally be. Even if there are none, still draw the region. | ![none](examples/top_button_none.jpg) |
| **half** (Half-button) | 2–5 buttons only near the neck (polo, henley) | Draw a tight vertical box that covers only the short button placket (from neck down to the last button). | ![half](examples/top_button_half.jpg) |
| **down** (Button-down) | Full button placket from neck all the way to the hem | Draw a long vertical box that follows the entire button line from collar to bottom hem. | ![down](examples/top_button_down.jpg) |

---

### 1.3 Neck type

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **crew** | Round, close-fitting neckline that sits at the base of the neck | Draw a box tightly around the neckline / collar area only (not the whole shirt). | ![crew](examples/top_neck_crew.jpg) |
| **v** | Clear V-shaped neckline | Draw a box that follows the V shape – include both sides of the V and the bottom point. | ![v](examples/top_neck_v.jpg) |
| **polo** | Collared shirt with a short button placket | Draw a box that covers the collar + the short button area together. | ![polo](examples/top_neck_polo.jpg) |
| **square** | Straight horizontal neckline (boat neck, square neck) | Draw a wide horizontal box along the straight neckline. | ![square](examples/top_neck_square.jpg) |
| **off_shoulder** | One or both shoulders are exposed | Draw a box that covers the neckline **and** the exposed shoulder(s). | ![off](examples/top_neck_off.jpg) |

---

### 1.4 Pattern

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **plain** | Solid color, no print or pattern | Draw a box on the largest plain area of the garment (usually the chest or back). | ![plain](examples/top_pattern_plain.jpg) |
| **graphic** | Large print, logo, illustration, big text | Draw a box that fully covers the graphic / print. | ![graphic](examples/top_pattern_graphic.jpg) |
| **stripe** | Vertical, horizontal or diagonal stripes | Draw a box that contains several stripes so the pattern is obvious. | ![stripe](examples/top_pattern_stripe.jpg) |
| **check** | Gingham, plaid, windowpane | Draw a box that shows the repeating check pattern clearly. | ![check](examples/top_pattern_check.jpg) |
| **dot** | Polka dots or small repeated circles | Draw a box that contains multiple dots. | ![dot](examples/top_pattern_dot.jpg) |
| **corner_only** | Small logo or motif only in one corner (chest pocket, hem…) | Draw a tight box only around that small logo / motif. | ![corner](examples/top_pattern_corner.jpg) |

A shirt can have **more than one pattern box** if it has both a large graphic and a small corner logo.

---

### 1.5 Sleeves (LEFT and RIGHT)

Always label **both arms** when they are visible.

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **short** | Sleeve ends above the elbow | Draw a box from the shoulder seam down to the end of the sleeve. Include a little bit of the arm if the sleeve is short. | ![short](examples/top_sleeve_short.jpg) |
| **long** | Sleeve reaches the wrist or longer | Draw a box from the shoulder all the way to the wrist / cuff. | ![long](examples/top_sleeve_long.jpg) |
| **camis** | Very thin strap (camisole / spaghetti strap) | Draw a thin vertical box that follows the strap from shoulder to the top of the garment. | ![camis](examples/top_sleeve_camis.jpg) |
| **sleeveless** | No sleeve at all (tank top, muscle tee) | Draw a box on the shoulder / armhole area to show there is no sleeve fabric. | ![sleeveless](examples/top_sleeve_sleeveless.jpg) |

**Important side rule**
- **Sleeve — LEFT arm** = the wearer’s left arm (appears on the **right side** of the photo when the person is facing you).
- **Sleeve — RIGHT arm** = the wearer’s right arm (appears on the **left side** of the photo).
- If one arm is completely hidden, still label the visible one. The tool will warn you if a side is missing.

---

## 2. Bottoms

Bottoms only have three attributes:

### 2.1 Length

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **long** | Reaches the ankle or the floor | Draw a box from the waistband all the way down to the hem near the shoes. | ![long](examples/bottom_length_long.jpg) |
| **short** | Ends above the knee | Draw a box from the waistband to the hem. The box should end above the knee and show the thigh. | ![short](examples/bottom_length_short.jpg) |

---

### 2.2 Pattern

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **plain** | Solid color | Box on the largest plain area (usually the thigh). | ![plain](examples/bottom_pattern_plain.jpg) |
| **graphic** | Large print or text on the bottoms | Box that fully covers the graphic. | ![graphic](examples/bottom_pattern_graphic.jpg) |
| **stripe** | Stripes | Box that shows several stripes. | ![stripe](examples/bottom_pattern_stripe.jpg) |
| **check** | Plaid / gingham | Box that shows the repeating check. | ![check](examples/bottom_pattern_check.jpg) |

---

### 2.3 Fit type

| Value | When to choose | How to label | Visual example |
|-------|----------------|--------------|----------------|
| **skinny** | Tight from hip all the way to the ankle | Draw a box that follows the leg closely. The fabric should look stretched against the leg. | ![skinny](examples/bottom_fit_skinny.jpg) |
| **straight** | Same width from knee to ankle (classic straight-leg) | Draw a box down the leg. The width at the knee and at the ankle should look similar. | ![straight](examples/bottom_fit_straight.jpg) |
| **baggy** | Very loose, lots of extra fabric and volume | Draw a generous box. You should see lots of empty space / folds inside the box. | ![baggy](examples/bottom_fit_baggy.jpg) |

---

## 3. On-body vs Flat / Hanging

At the top of the right panel choose:

- **On body** – person is wearing the clothes
- **Flat / hanging** – clothes laid flat or on a hanger

This helps the model later distinguish real-world photos from product shots.

---

## 4. What to send back

After you finish labeling a folder, please send:

```
your_name_photos/
├── photo1.jpg
├── photo1.json
├── photo2.jpg
├── photo2.json
└── ...
```

Only the images + the `.json` files that sit next to them.  
Do **not** send the `delete/` or `originals_before_4x3/` folders.

You can zip the whole folder:

```bash
zip -r my_labels.zip your_name_photos/
```

---

## 5. Quick keyboard shortcuts

| Key | Action |
|-----|--------|
| `1` | Switch to TOP mode |
| `2` | Switch to BOTTOM mode |
| `N` | Force-draw a new box even on top of existing ones |
| `←` `→` | Previous / next photo |
| `Delete` | Remove selected box |
| `Ctrl+Z` | Undo |

---

## 6. Quality checklist (the tool already shows this)

- Every drawn region has a value chosen
- Both left and right sleeves are present when both arms are visible
- No duplicate boxes for the same region (except pattern)
- Boxes are reasonably tight but still cover the whole region
- For **cropped** tops: the box includes the midriff

Thank you — clean labels make the model much stronger!
