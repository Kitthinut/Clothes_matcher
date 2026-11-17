# Clothing labeling guide

This guide explains how to label the clothing photos and return the results.

## Before you start

- Extract the project folder somewhere on your computer.
- Install Python 3 if it is not already installed.
- Open PowerShell in the project folder (the folder containing `1_label`).
- Install the image library the annotator uses:

  ```powershell
  py -m pip install Pillow
  ```

If `py` is not recognized, try `python` instead.

## Choose what to label

Run one of these commands from the project folder:

```powershell
py .\1_label\annotate.py --garment top
```

```powershell
py .\1_label\annotate.py --garment bottom
```

The first command labels tops; the second labels bottoms. If you leave off
`--garment`, it defaults to tops. Keep tops and bottoms in their separate
folders and use the matching command.

The photos should be in `dataset\top\images` or `dataset\bottom\images`,
depending on what you are labeling. If you were given a folder of photos that
isn't already in the project, you can copy it into the right place with:

```powershell
py .\1_label\make_kit.py --garment top --from "C:\path\to\your\photos"
```

Use `bottom` instead of `top` for bottom photos. Keep the original photos and
filenames unchanged.

## What you'll see and do

1. A start dialog shows how many photos there are and how many have labels.
   Choose where to start and click **Start**.
2. The photo appears on the left. The controls are on the right.
3. Drag on the photo to draw a box around the clothing item. Choose its
   attributes from the options on the right, then click **Add region** or press
   **Enter**.
4. When labeling tops, **Neck type** and **Sleeves** are separate regions:
   choose that region type and draw its box before adding it. Other clothing
   attributes belong to the **Garment** region. Bottoms use the **Garment**
   region.
5. Use **Next >** and **< Prev** to move between photos. Labels are saved as
   you add or update them, navigate, click **Save**, or close the window.

Useful shortcuts: **Ctrl+S** saves, **Ctrl+N** jumps to the next unlabelled
photo, and the **left/right arrow keys** move between photos. Avoid **Delete**
and **Delete this picture** unless you intend to remove a photo from the kit.

## Send the labels back

When you're finished, send back the matching file:

- Tops: `dataset\top\annotations.json`
- Bottoms: `dataset\bottom\annotations.json`

Please send the `annotations.json` file, not a screenshot. Don't rename it or
edit it by hand. If both tops and bottoms were assigned, return both files.
