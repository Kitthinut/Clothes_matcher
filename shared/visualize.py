"""Prediction viewer (shared/visualize.py).

Draws one box + label per attribute on the image and shows it in a tkinter
popup with Left/Right to step through several images.

Theory: the boxes come from Grad-CAM (see 2_train/*/gradcam.py) - this file
only does the drawing. bbox_from_heat turns a heatmap into a box: all pixels
with activation >= 40% of the max are "inside the looked-at region"; the
bounding rectangle of those pixels is the box. No box labels are needed.
"""

HEAT_THRESHOLD = 0.4  # fraction of the heatmap max that counts as "inside"

COLORS = ("#00e5ff", "#ff9100", "#00e676", "#ff1744", "#b388ff", "#ffea00", "#69f0ae")

import numpy as np
from PIL import Image, ImageDraw, ImageTk


def bbox_from_heat(heat, thr=HEAT_THRESHOLD):
    """2-D numpy heatmap -> (x1, y1, x2, y2) pixel box, or None if empty."""
    m = heat >= thr * heat.max()
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def remap_box(box, crop_box, input_size):
    """Grad-CAM box in model-input pixels -> original image pixels.

    crop_box:   garment box [x1,y1,x2,y2] in original px (None = full image)
    input_size: (W, H) the model input (the crop resized to this)
    """
    if box is None:
        return None
    if crop_box:
        x1, y1, x2, y2 = crop_box
    else:
        x1, y1 = 0, 0
        x2, y2 = input_size  # placeholder, scale factors become 1 below
    w_in, h_in = input_size
    sx = (x2 - x1) / w_in
    sy = (y2 - y1) / h_in
    return (int(x1 + box[0] * sx), int(y1 + box[1] * sy),
            int(x1 + box[2] * sx), int(y1 + box[3] * sy))


def _label_text(label):
    """Multi-label heads predict lists — render them readably."""
    if isinstance(label, (list, tuple)):
        return "+".join(str(v) for v in label) if label else "(none)"
    return str(label)


def draw_results(img, results):
    """PIL image + {key: {label, prob, box:(x1,y1,x2,y2) px or None}}
    -> annotated PIL image (one coloured box + text per attribute)."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, (k, r) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        text = f"{k}: {_label_text(r['label'])} {r.get('prob', 0):.0%}"
        box = r.get("box")
        if box:
            d.rectangle(box, outline=color, width=3)
            d.text((box[0] + 4, box[1] + 4), text, fill=color)
        else:
            d.text((8, 8 + i * 14), f"{text} (no region found)", fill=color)
    return im


class Viewer:
    """tkinter popup: shows the annotated images one by one, Left/Right to browse."""

    def __init__(self, items):
        self.items = items  # [(title, annotated PIL image)]
        self.i = 0
        import tkinter as tk
        self.win = tk.Tk()
        self.win.title("Grad-CAM viewer  -  Left/Right: browse, Esc: close")
        self.canvas = tk.Label(bg="#222")
        self.canvas.pack(fill="both", expand=True)
        self.bar = tk.Label(self.win, anchor="w", bg="#111", fg="white")
        self.bar.pack(fill="x")
        self.win.bind("<Left>", lambda e: self.step(-1))
        self.win.bind("<Right>", lambda e: self.step(1))
        self.win.bind("<Escape>", lambda e: self.win.destroy())
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)
        self.show()
        self.win.mainloop()

    def show(self):
        title, im = self.items[self.i]
        scale = min(900 / im.width, 700 / im.height, 1.0)
        disp = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.config(image=self.photo)
        self.bar.config(text=f"[{self.i + 1}/{len(self.items)}] {title}")

    def step(self, d):
        self.i = (self.i + d) % len(self.items)
        self.show()


def show_or_save(items, save_dir=None):
    """items: [(title, annotated PIL)] -> popup window, or save PNGs if save_dir."""
    if save_dir:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for j, (title, im) in enumerate(items, 1):
            path = save_dir / f"{j:02d}_{Path(title).stem}.png"
            im.save(path)
            print(f"saved {path}")
        return
    try:
        Viewer(items)
    except Exception as e:
        print(f"popup failed ({e}); saving PNGs to weights\\visual_debug instead")
        show_or_save(items, save_dir="weights/visual_debug")
