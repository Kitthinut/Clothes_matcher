# Helping label clothes — quick guide

Thanks for helping! I do **not** need to send you the whole project. You only
need a small zip containing the annotator, and you use **your own pictures**.
No account, no browser, no GPU and no model download are needed.

---

## 1. What I send you

A zip called `Raja-Fashion-lite.zip` containing only:

```
Raja-Fashion/
├─ schema.yaml
├─ config.yaml
├─ requirements.txt
├─ SHARE_WITH_FRIENDS.md        ← this file
├─ annotator/
│  ├─ __init__.py  app.py  canvas.py
│  ├─ store.py  ui_utils.py  image_check.py  resize_tool.py
└─ common/
   ├─ __init__.py  config.py  colors.py
   ├─ logging_conf.py  mock_data.py  paths.py  schema.py
```

If you received the full project instead, you can ignore everything except
those folders/files.

---

## 2. Set it up once

1. Unzip the folder on your computer.
2. Install Python 3.10 or newer if you don't have it.
3. Open a terminal in the `Raja-Fashion` folder and run:

```bat
pip install -r requirements.txt
```

That's it — four lightweight packages (Pillow, numpy, opencv-python, PyYAML).

---

## 3. Where to put your pictures

**Easy way:** put your photos in a folder called `images` next to `annotator`:

```
Raja-Fashion\images\
```

**Or use any folder you already have** (for example `D:\my_outfit_photos`):

```bat
python -m annotator.app --images "D:\my_outfit_photos" --annotations "D:\my_labels"
```

If you leave out the flags, the app uses `images` and `annotations` folders.

**Important:**
- Use `.jpg`, `.jpeg`, or `.png` pictures.
- Do **not** rename or delete pictures after you start labelling — each label
  file is named after its picture.
- When the window opens, a pop-up tells you whether all your pictures are the
  same size and big enough. If they are not, it asks whether to resize them all
  to one size (default **384x288 px**). Originals are copied to
  `images_original_backup/` first. If a photo is not 4:3, black bars are added
  on the spare sides so nothing is squashed or cropped. Any labels already
  drawn are rescaled to the new size automatically. You can say no and just
  tell me in your notes.
- Keep the `annotations` folder that the app creates. That is your work.

---

## 4. Which file to run

In the `Raja-Fashion` folder, run:

```bat
python -m annotator.app
```

A **desktop window** opens (no website, no server). The size check appears
first; click OK to start. You can re-open it any time with **Check sizes**.

---

## 5. How to label a picture

1. In the right panel, click **TOP** or **BOTTOM**, then click an **attribute
   chip** (Length, Clothes type, Pattern, Neck type, Sleeves L, Sleeves R,
   Colour for tops; Waist length, Bottom length, Pattern, Fit type, Colour for
   bottoms). The active chip is highlighted.
2. **Drag on the picture** to draw a box for that attribute. Each attribute gets
   its own box (multi-class object detection).
3. For **Colour**, draw a box over a clean patch of the garment (for example the
   plain part of a sleeve or trouser leg). The colour is detected automatically
   from the pixels inside that box; a swatch shows the result. You cannot pick
   the colour by hand — just redraw or resize the Colour box if it is wrong.
4. **Sleeves need two boxes** — `Sleeves L` is the left arm and `Sleeves R` is
   the right arm.
5. Click a box to select it; drag corner handles to resize or drag inside to
   move it. Large **label buttons** show the valid options for that attribute
   (e.g. crew / v / square for neck); click one or press `1`–`9`.
6. A picture may contain **only a top, only a bottom, or both** — just label
   what you can see. Tick `top absent` / `bottom absent` to grey out a garment.
7. Press **Submit** or `Enter` to save and go to the next picture. Your changes
   are also saved automatically as you edit.

Helpful keys: `T`/`B` garment · `Tab` next chip · `1`–`9` label or chip ·
`Delete` remove a box · `Ctrl+Z` undo · `Ctrl+Y` redo · `Ctrl+S` save ·
`←`/`→` change picture · `Ctrl+N` jump to the next unlabelled picture · `F1` help.

---

## 6. What to send back

You will have:

- **Your original pictures** — unchanged, in the images folder.
- **One `annotations.json` file** with all your labelled pictures in it.

### Send me back (zip these together)

```
<your_name>_labels.zip
├─ images\                      ← all your pictures
│   ├─ IMG_0001.jpg
│   └─ ...
├─ annotations.json              ← ONE file with all your boxes
└─ README.txt                    ← your name, date, any notes
```

There is **no `.txt` file** to send — `annotations.json` is the only label file,
and I generate the YOLO training files myself.

The `README.txt` can be a plain text file like:

```
Name: Alex
Date: 2026-11-18
Notes: picture 7 has two tops; I boxed both. Some images are 640x480, others 500x400.
```

**Checklist before zipping:**
- [ ] Every labelled picture appears by name inside `annotations.json`.
- [ ] Each garment you labelled has its region boxes (both sleeve arms).
- [ ] You didn't rename pictures after labelling.
- [ ] `README.txt` has your name, the date and any size/quality notes.
- [ ] Both folders are inside the zip (not just the pictures).

That's all — no need to send the annotator back.

---

*Ultralytics YOLO is AGPL-3.0 licensed. The annotator itself does not include
YOLO weights, so you do not need to install Ultralytics.*
