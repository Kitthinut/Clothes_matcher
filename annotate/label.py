#!/usr/bin/env python3
"""
raja_fashion label tool: draw boxes around clothes, click the attributes.

    python label.py                      # choose your photo folder in a window
    python label.py my_photos            # open a folder directly
    python label.py my_photos --only neck    # first experiment: only ask for the neck type

How it works
  1. Pick TOP or BOTTOM. Choose an attribute card on the right, then draw its box.
  2. Choose that attribute value in the selected-box panel on the right.
  3. Use Next to go to the next photo. Everything is saved automatically next to the photo
     as <photo>.json, in the format described in labeling_config/schema.yaml.

Needs: Python 3, Pillow, PyYAML (pip install -r requirements.txt). The window uses tkinter.
Run check_photos.py first so all photos have the right size and 4:3 shape.
"""

import argparse
import copy
import json
import webbrowser
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import yaml
from PIL import Image, ImageOps, ImageTk

from check_photos import (
    DELETE_DIR,
    IMG_EXT,
    ORIG_DIR,
    display_size,
    move_to_delete,
    problems,
)

HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "labeling_config" / "schema.yaml"
GUIDE = HERE / "HOW_TO_LABEL.md"

COLORS = {
    "top": "#1e88e5",
    "bottom": "#2e7d32",
    "sleeve_left": "#ef6c00",
    "sleeve_right": "#8e24aa",
}
NICE = {
    "top": "TOP",
    "bottom": "BOTTOM",
    "sleeve_left": "SLEEVE: wearer's LEFT arm",
    "sleeve_right": "SLEEVE: wearer's RIGHT arm",
}
MODES = [("top", "1  TOP"), ("bottom", "2  BOTTOM")]
MODE_COLOR = {"top": COLORS["top"], "bottom": COLORS["bottom"]}
ATTR_TITLE = {
    "length": "Length",
    "button": "Button type",
    "neck": "Neck type",
    "pattern": "Pattern",
    "sleeve": "Sleeve type",
    "fit": "Fit type",
}

# Each drawn box represents one visual region/attribute. The parent class
# remains TOP or BOTTOM; sleeve side is explicitly selected, never guessed.
REGIONS = {
    "top": [
        ("length", "Length", "length"),
        ("button", "Button type", "button"),
        ("neck", "Neck type", "neck"),
        ("pattern", "Pattern", "pattern"),
        ("sleeve_left", "Sleeve — LEFT arm", "sleeve"),
        ("sleeve_right", "Sleeve — RIGHT arm", "sleeve"),
    ],
    "bottom": [
        ("length", "Length", "length"),
        ("pattern", "Pattern", "pattern"),
        ("fit", "Fit type", "fit"),
    ],
}
REGION_TITLES = {
    (category, region): title
    for category, entries in REGIONS.items()
    for region, title, _attribute in entries
}
REGION_ATTRS = {
    (category, region): attribute
    for category, entries in REGIONS.items()
    for region, _title, attribute in entries
}
DISPLAY_VALUES = {
    "none": "No-button",
    "half": "Half-button",
    "down": "Button-down",
    "off_shoulder": "Off shoulders",
}
SHORT_TITLE = {
    "length": "Length", "button": "Button", "neck": "Neck", "pattern": "Pattern",
    "sleeve_left": "Sleeve L", "sleeve_right": "Sleeve R",
    "fit": "Fit",
}

def display_value(value):
    return DISPLAY_VALUES.get(value, value.replace("_", " "))
ICON = {"done": "✔", "partial": "◐", "todo": "○"}


def list_images(folder: Path):
    skips = [folder / DELETE_DIR, folder / ORIG_DIR]
    return [
        p
        for p in sorted(folder.rglob("*"))
        if p.suffix.lower() in IMG_EXT and not any(s in p.parents for s in skips)
    ]


class App:
    def __init__(self, root, folder, only):
        self.root = root
        self.schema = yaml.safe_load(SCHEMA.read_text("utf-8"))
        self.attr_on = {
            (c, a): tk.BooleanVar(value=True)
            for c, d in self.schema["attributes"].items()
            for a in d
        }
        if only:
            for (c, a), v in self.attr_on.items():
                v.set(a in only or f"{c}.{a}" in only)

        self.folder, self.files, self.status = None, [], {}
        self.cur, self.jp, self.pil = None, None, None
        self.meta, self.objs, self.extra = {}, [], []
        self.sel, self.undo, self.drag = None, [], None
        self.done_forced = set()   # photos manually marked done in the task tracker
        self.geom, self._photo, self._cache_key = (0, 0, 1, 1), None, None
        self.force_draw = False   # next drag draws a new box even on top of an existing one
        self._lock = False

        base = tkfont.nametofont("TkDefaultFont")
        self.f_big = base.copy()
        self.f_big.configure(size=base.cget("size") + 2)
        self.f_bold = base.copy()
        self.f_bold.configure(weight="bold", size=base.cget("size") + 1)
        self.f_title = base.copy()
        self.f_title.configure(weight="bold", size=base.cget("size") + 3)
        style = ttk.Style()
        style.configure("Mode.Toolbutton", font=self.f_bold, padding=(10, 5))
        style.configure("Nav.TButton", font=self.f_bold, padding=(8, 6))
        # ttk.Button has no per-widget -anchor option on many Tk builds; set it on a style.
        style.configure("Card.TButton", anchor="w")

        self.mode = tk.StringVar(value="top")
        self.region_var = tk.StringVar(value="length")
        self.worn = tk.StringVar(value="on_body")
        self.progress = tk.StringVar()
        self.folder_var = tk.StringVar(value="No folder chosen")
        self.line = tk.StringVar()
        self.warn = tk.StringVar()

        self.build_ui()
        self.bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if folder:
            root.after(100, lambda: self.open_folder(Path(folder)))
        else:
            root.after(150, self.choose_folder)

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        r = self.root
        r.columnconfigure(1, weight=1)
        r.rowconfigure(0, weight=1)

        # left: photo list
        left = ttk.Frame(r, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        ttk.Button(
            left, text="Open folder…", command=self.choose_folder, style="Nav.TButton"
        ).pack(fill="x")
        ttk.Label(left, textvariable=self.folder_var, wraplength=250).pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Label(left, textvariable=self.progress, font=self.f_bold).pack(
            anchor="w", pady=6
        )
        lf = ttk.Frame(left)
        lf.pack(fill="both", expand=True)
        self.lb = tk.Listbox(
            lf, width=32, activestyle="none", exportselection=False, font=self.f_big
        )
        sb = ttk.Scrollbar(lf, command=self.lb.yview)
        self.lb.config(yscrollcommand=sb.set)
        self.lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.lb.bind("<<ListboxSelect>>", self.on_list_select)
        ttk.Label(
            left,
            text="✔ done    ◐ some attributes empty    ○ not labeled",
            foreground="#555",
        ).pack(anchor="w", pady=4)
        nav = ttk.Frame(left)
        nav.pack(fill="x")
        ttk.Button(
            nav, text="◀ Previous", command=lambda: self.goto(-1), style="Nav.TButton"
        ).pack(side="left", expand=True, fill="x")
        ttk.Button(
            nav, text="Next ▶", command=lambda: self.goto(1), style="Nav.TButton"
        ).pack(side="left", expand=True, fill="x", padx=(6, 0))
        ttk.Button(
            left,
            text="Next photo not done ⏭",
            command=self.next_todo,
            style="Nav.TButton",
        ).pack(fill="x", pady=(6, 0))
        ttk.Button(
            left,
            text="✔ Done with this photo (toggle)",
            command=self.toggle_done,
            style="Nav.TButton",
        ).pack(fill="x", pady=(6, 0))
        ttk.Button(
            left,
            text=f"Bad photo? Move it to '{DELETE_DIR}'",
            command=self.delete_photo,
        ).pack(fill="x", pady=(14, 0))

        # centre: tools + canvas
        mid = ttk.Frame(r, padding=(0, 8))
        mid.grid(row=0, column=1, sticky="nsew")
        # Keep the image viewport stable when the right-side category cards change.
        mid.configure(width=840)
        mid.grid_propagate(False)
        mid.rowconfigure(1, weight=1)
        mid.columnconfigure(0, weight=1)
        tools = ttk.Frame(mid)
        tools.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(tools, text="Draw a box around:", font=self.f_bold).pack(
            side="left", padx=(0, 8)
        )
        for val, txt in MODES:
            tk.Label(tools, width=1, bg=MODE_COLOR[val]).pack(
                side="left", fill="y", padx=(4, 0)
            )
            ttk.Radiobutton(
                tools,
                text=txt,
                value=val,
                variable=self.mode,
                style="Mode.Toolbutton",
                command=self.update_mode,
            ).pack(side="left")
        self.update_mode()
        ttk.Button(tools, text="Undo", command=self.do_undo).pack(side="right")
        ttk.Button(tools, text="Delete selected box", command=self.delete_box).pack(
            side="right", padx=6
        )
        self.new_box_btn = ttk.Button(
            tools, text="＋ New box (N)", command=self.toggle_new_box
        )
        self.new_box_btn.pack(side="right", padx=6)
        self.canvas = tk.Canvas(
            mid, bg="#2b2b2b", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda *args: (args, self.render())[1])
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Keep the photo/canvas stationary while annotating.
        # Consume mouse-wheel and touchpad scroll events over the canvas
        # so they cannot scroll or otherwise move the annotation view.
        for wheel_event in (
            "<MouseWheel>", "<Shift-MouseWheel>",
            "<Button-4>", "<Button-5>",
        ):
            self.canvas.bind(wheel_event, self.ignore_canvas_scroll)
            mid.bind(wheel_event, self.ignore_canvas_scroll)
            r.bind(wheel_event, self.ignore_canvas_scroll)
        ttk.Label(
            mid, textvariable=self.warn, foreground="#b00020", wraplength=800
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(mid, textvariable=self.line, font=self.f_bold).grid(
            row=3, column=0, sticky="w", pady=(2, 0)
        )

        # Right column: fixed outer size with its own scroll area, so selecting a
        # box (or switching TOP/BOTTOM) never resizes the window and shifts the
        # photo/boxes on the canvas.
        right = ttk.Frame(r, width=360, padding=(8, 8, 2, 8))
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        rcanvas = tk.Canvas(right, highlightthickness=0, bd=0)
        rscroll = ttk.Scrollbar(right, orient="vertical", command=rcanvas.yview)
        rcanvas.configure(yscrollcommand=rscroll.set)
        rcanvas.grid(row=0, column=0, sticky="nsew")
        rscroll.grid(row=0, column=1, sticky="ns")
        self.right_inner = ttk.Frame(rcanvas)
        self._rwin = rcanvas.create_window((0, 0), window=self.right_inner, anchor="nw")
        self.right_inner.bind("<Configure>", lambda e: rcanvas.configure(scrollregion=rcanvas.bbox("all")))
        rcanvas.bind("<Configure>", lambda e: rcanvas.itemconfigure(self._rwin, width=e.width))

        f1 = ttk.LabelFrame(self.right_inner, text=" This photo ", padding=6)
        f1.pack(fill="x")
        ttk.Radiobutton(
            f1, text="On body", value="on_body", variable=self.worn,
            command=self.save,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            f1, text="Flat / hanging", value="flat", variable=self.worn,
            command=self.save,
        ).pack(side="left")

        self.region_cards = ttk.LabelFrame(self.right_inner, text=" Draw a box for each category ", padding=6)
        self.region_cards.pack(fill="x", pady=(8, 0))
        self.region_cards.columnconfigure(0, weight=1)
        self.region_cards.columnconfigure(1, weight=1)
        self.region_card_buttons = {}
        self._build_region_cards()

        self.box_frame = ttk.LabelFrame(
            self.right_inner, text=" Selected box — choose its value ", padding=8
        )
        self.box_frame.pack(fill="x", pady=(8, 0))
        self.check_frame = ttk.LabelFrame(self.right_inner, text=" Label checklist ", padding=6)
        self.check_frame.pack(fill="x", pady=(8, 0))

        bottom = ttk.Frame(right, padding=(0, 6, 0, 0))
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(bottom, text="Open label guide", command=self.open_guide).pack(fill="x")
        ttk.Label(
            bottom, foreground="#555", justify="left",
            text="1/2 TOP/BOTTOM · N new box · ← → photos · Del remove · Ctrl+Z undo",
        ).pack(anchor="w", pady=(4, 0))
        self.build_panel()
        self.refresh_checklist()
        self.update_line()

    def bind_keys(self):
        r = self.root
        for k, m in (("1", "top"), ("2", "bottom")):
            r.bind(
                k, lambda *args, m=m: (args, self.mode.set(m), self.update_mode())[1]
            )
        r.bind("<Left>", lambda *args: (args, self.goto(-1))[1])
        r.bind("<Right>", lambda *args: (args, self.goto(1))[1])
        r.bind("<Delete>", lambda *args: (args, self.delete_box())[1])
        r.bind("<BackSpace>", lambda *args: (args, self.delete_box())[1])
        r.bind("<Control-z>", lambda *args: (args, self.do_undo())[1])
        r.bind("<Command-z>", lambda *args: (args, self.do_undo())[1])
        r.bind("n", lambda *args: (args, self.toggle_new_box())[1])
        r.bind("N", lambda *args: (args, self.toggle_new_box())[1])

    # ------------------------------------------------------------- folder / list
    def choose_folder(self):
        d = filedialog.askdirectory(title="Choose the folder with your photos")
        if d:
            self.open_folder(Path(d))

    def open_folder(self, folder: Path):
        self.save()
        files = list_images(folder)
        if not files:
            messagebox.showinfo("No photos", f"No photos found in:\n{folder}")
            return
        self.folder, self.files = folder, files
        self.done_forced.clear()
        self.folder_var.set(str(folder))
        self.cur = None
        self.recompute_status()
        self.fill_list()
        todo = [i for i, f in enumerate(files) if self.status[f] != "done"]
        self.open_image(todo[0] if todo else 0)

    def read_objs(self, jp: Path):
        try:
            objects = [
                obj
                for obj in json.loads(jp.read_text("utf-8")).get("objects", [])
                if obj.get("cls") in COLORS
            ]
            for obj in objects:
                if obj.get("cls") in ("sleeve_left", "sleeve_right"):
                    obj["region"] = obj["cls"]
                    obj["cls"] = "top"
            return objects
        except Exception:
            return []

    def _read_meta(self, jp: Path):
        try:
            data = json.loads(jp.read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def recompute_status(self):
        self.done_forced.clear()
        for f in self.files:
            meta = self._read_meta(f.with_suffix(".json"))
            if meta.get("done"):
                self.done_forced.add(f)
            self.status[f] = (
                "done" if f in self.done_forced
                else self.status_of(self.read_objs(f.with_suffix(".json")))
            )

    def region_attribute(self, obj):
        cls = obj.get("cls")
        region = obj.get("region")
        if cls in REGIONS and (cls, region) in REGION_ATTRS:
            return REGION_ATTRS[(cls, region)]
        return None

    def focus_attrs(self, cls, region=None):
        if region is not None and (cls, region) in REGION_ATTRS:
            attr = REGION_ATTRS[(cls, region)]
            enabled = self.attr_on.get((cls, attr))
            return [attr] if enabled is None or enabled.get() else []
        return [
            a for a in self.schema["attributes"].get(cls, {})
            if (self.attr_on.get((cls, a)) is None or self.attr_on[(cls, a)].get())
        ]

    def status_of(self, objs):
        relevant = [o for o in objs if o.get("cls") in ("top", "bottom")]
        if not relevant:
            return "todo"
        for obj in relevant:
            attr = self.region_attribute(obj)
            if attr:
                if not (obj.get("attrs") or {}).get(attr):
                    return "partial"
            else:
                # Older annotation files stored every attribute in one TOP/BOTTOM box.
                for a in self.focus_attrs(obj.get("cls")):
                    if not (obj.get("attrs") or {}).get(a):
                        return "partial"
        return "done"

    def row_text(self, i):
        f = self.files[i]
        return f"{ICON[self.status[f]]}  {f.relative_to(self.folder)}"

    def fill_list(self):
        self._lock = True
        self.lb.delete(0, "end")
        for i in range(len(self.files)):
            self.lb.insert("end", self.row_text(i))
        self._lock = False
        self.update_progress()

    def refresh_row(self, i):
        self._lock = True
        self.lb.delete(i)
        self.lb.insert(i, self.row_text(i))
        if i == self.cur:
            self.lb.selection_set(i)
        self._lock = False
        self.update_progress()

    def update_progress(self):
        done = sum(1 for s in self.status.values() if s == "done")
        self.progress.set(f"{done} of {len(self.files)} photos done")

    def on_list_select(self, *args):
        if args:
            pass
        if self._lock:
            return
        sel = self.lb.curselection()
        if sel and sel[0] != self.cur:
            self.open_image(sel[0])

    def goto(self, step):
        if self.files and self.cur is not None:
            self.open_image(max(0, min(len(self.files) - 1, self.cur + step)))

    def next_todo(self):
        if not self.files:
            return
        n = len(self.files)
        for k in range(1, n + 1):
            i = ((self.cur or 0) + k) % n
            if self.status[self.files[i]] != "done":
                return self.open_image(i)
        messagebox.showinfo("All done", "Every photo is done ✔")

    def toggle_done(self):
        """Mark the current photo done in the task tracker (or undo that mark).

        The flag is stored as ``"done": true`` in the JSON sidecar, so it
        survives reopening the folder even if some attribute boxes are empty.
        """
        if self.cur is None or not self.files:
            return
        f = self.files[self.cur]
        if f in self.done_forced:
            self.done_forced.discard(f)
            self.meta.pop("done", None)
            self.save()
            self.status[f] = self.status_of(self.objs)
            self.refresh_row(self.cur)
            self.line.set("Marked not done")
        else:
            self.done_forced.add(f)
            self.meta["done"] = True
            self.save()
            self.status[f] = "done"
            self.refresh_row(self.cur)
            self.line.set("✔ Marked done")
            self.next_todo()

    def delete_photo(self):
        if self.cur is None:
            return
        f = self.files[self.cur]
        if not messagebox.askyesno(
            "Move photo",
            f"Move this photo (and its label) to the '{DELETE_DIR}' folder?\n\n{f.name}\n\n"
            "It is not erased; you can drag it back.",
        ):
            return
        i = self.cur
        self.cur, self.objs, self.extra = None, [], []
        move_to_delete(f, self.folder)
        self.files.pop(i)
        self.status.pop(f, None)
        if not self.files:
            self.pil = None
            self.fill_list()
            self.render()
            return
        self.fill_list()
        self.open_image(min(i, len(self.files) - 1))

    # ------------------------------------------------------------- image / data
    def _letterbox_4x3(self, im: Image.Image) -> Image.Image:
        """
        Return a 4:3 portrait version of the image for display only.
        Adds black bars if needed. Never touches the file on disk.
        """
        w, h = im.size
        target_ratio = 4 / 3          # height / width

        if abs(h / w - target_ratio) < 0.02:
            return im                 # already close enough to 4:3

        if h / w < target_ratio:
            # too wide → add bars top & bottom
            new_h = int(round(w * target_ratio))
            new_w = w
        else:
            # too tall → add bars left & right
            new_w = int(round(h / target_ratio))
            new_h = h

        canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        x = (new_w - w) // 2
        y = (new_h - h) // 2
        canvas.paste(im, (x, y))
        return canvas

    def open_image(self, i):
        self.save()
        self.cur, path = i, self.files[i]
        self.jp = path.with_suffix(".json")
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            im = self._letterbox_4x3(im)   # force 4:3 display so labelers are not confused
            im.thumbnail((1600, 1600))
            self.pil = im
        except Exception as e:
            self.pil = None
            messagebox.showwarning("Cannot open photo", f"{path.name}\n{e}")
        self.meta, self.objs, self.extra = {}, [], []
        if self.jp.exists():
            try:
                m = json.loads(self.jp.read_text("utf-8"))
                for o in m.pop("objects", []):
                    if (
                        o.get("cls") in COLORS
                        and isinstance(o.get("box"), list)
                        and len(o["box"]) == 4
                    ):
                        o["attrs"] = o.get("attrs") or {}
                        # Older files used sleeve_left/right as object classes.
                        # Keep their geometry and make them explicit TOP regions.
                        if o.get("cls") in ("sleeve_left", "sleeve_right"):
                            o["region"] = o["cls"]
                            o["cls"] = "top"
                        self.objs.append(o)
                    else:
                        self.extra.append(o)
                self.meta = m
            except Exception:
                bad = self.jp.with_suffix(".json.bad")
                self.jp.rename(bad)
                messagebox.showwarning(
                    "Label file was broken",
                    f"Could not read {self.jp.name}.\nIt was renamed to {bad.name}.",
                )
        self.undo, self.sel = [], None
        if self.force_draw:
            self.set_force_draw(False)
        self.worn.set(self.meta.get("worn", "on_body"))
        try:
            self.warn.set(
                "\n".join(problems(*display_size(path)))
                and "⚠ "
                + " ".join(problems(*display_size(path)))
                + "  (run check_photos.py)"
            )
        except Exception:
            self.warn.set("")
        self._lock = True
        self.lb.selection_clear(0, "end")
        self.lb.selection_set(i)
        self.lb.see(i)
        self._lock = False
        self.build_panel()
        self.render()
        self.refresh_checklist()
        self.update_line()

    def save(self):
        if (
            self.cur is None
            or self.jp is None
            or not self.files
            or self.cur >= len(self.files)
        ):
            return
        if not self.objs and not self.extra and not self.jp.exists():
            return  # just browsing: do not create empty label files
        meta = dict(self.meta)
        current_file = self.files[self.cur]
        meta.update(
            {
                "image": current_file.name,
                "schema_version": self.schema["version"],
                "worn": self.worn.get(),
            }
        )
        if current_file in self.done_forced:
            meta["done"] = True
        else:
            meta.pop("done", None)
        meta["objects"] = [
            {
                "cls": o["cls"],
                **({"region": o["region"]} if o.get("region") else {}),
                "box": [round(v, 6) for v in o["box"]],
                "attrs": {k: v for k, v in o["attrs"].items() if v},
            }
            for o in self.objs
        ] + self.extra
        self.jp.write_text(json.dumps(meta, indent=1), "utf-8")
        f = self.files[self.cur]
        self.status[f] = "done" if f in self.done_forced else self.status_of(self.objs)
        self.refresh_row(self.cur)

    def push_undo(self):
        self.undo.append(copy.deepcopy(self.objs))
        del self.undo[:-50]

    def do_undo(self):
        if not self.undo:
            return
        self.objs = self.undo.pop()
        self.sel = None
        self.changed()

    def changed(self):
        self.save()
        self.build_panel()
        self.render()
        self.refresh_checklist()
        self.update_line()

    def ignore_canvas_scroll(self, event=None):
        """Prevent wheel/touchpad scrolling from moving the annotation canvas."""
        return "break"

    # ------------------------------------------------------------- drawing
    def render(self):
        c = self.canvas
        c.delete("all")
        cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        if self.pil is None:
            c.create_text(
                cw / 2,
                ch / 2,
                fill="#bbb",
                font=self.f_title,
                text="Choose a folder with photos\n(button at the top left)",
                justify="center",
            )
            return
        iw, ih = self.pil.size
        s = min(cw / iw, ch / ih)
        dw, dh = max(1, int(iw * s)), max(1, int(ih * s))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        key = (self.cur, dw, dh)
        if key != self._cache_key:
            self._photo = ImageTk.PhotoImage(self.pil.resize((dw, dh), Image.BILINEAR))
            self._cache_key = key
        c.create_image(ox, oy, image=self._photo, anchor="nw")
        self.geom = (ox, oy, dw, dh)
        for idx, o in enumerate(self.objs):
            self.draw_box(idx, o)

    def to_canvas(self, xn, yn):
        ox, oy, dw, dh = self.geom
        return ox + xn * dw, oy + yn * dh

    def to_norm(self, x, y):
        ox, oy, dw, dh = self.geom
        return min(1, max(0, (x - ox) / dw)), min(1, max(0, (y - oy) / dh))

    def corners(self, o):
        xc, yc, w, h = o["box"]
        x1, y1 = self.to_canvas(xc - w / 2, yc - h / 2)
        x2, y2 = self.to_canvas(xc + w / 2, yc + h / 2)
        return x1, y1, x2, y2

    def label_text(self, o):
        cls = o["cls"]
        region = o.get("region")
        if cls in REGIONS and (cls, region) in REGION_TITLES:
            title = REGION_TITLES[(cls, region)]
            attr = REGION_ATTRS[(cls, region)]
            value = (o.get("attrs") or {}).get(attr)
            return title + (f" · {display_value(value)}" if value else "")
        # Backward-compatible display for old whole-TOP/BOTTOM boxes.
        vals = [v for v in (o.get("attrs") or {}).values() if v]
        title = NICE.get(cls, cls.upper())
        return title + (" · " + " · ".join(v.replace("_", " ") for v in vals) if vals else "")

    def draw_box(self, idx, o):
        c = self.canvas
        region = o.get("region")
        col = COLORS[region] if region in ("sleeve_left", "sleeve_right") else COLORS[o["cls"]]
        x1, y1, x2, y2 = self.corners(o)
        selected = idx == self.sel
        c.create_rectangle(x1, y1, x2, y2, outline=col, width=4 if selected else 2)
        ty, anchor = (y1 - 2, "sw") if y1 > 24 else (y1 + 3, "nw")
        tid = c.create_text(
            x1 + 5,
            ty,
            anchor=anchor,
            text=self.label_text(o),
            fill="white",
            font=self.f_bold,
        )
        bb = c.bbox(tid)
        bg = c.create_rectangle(
            bb[0] - 3, bb[1] - 1, bb[2] + 3, bb[3] + 1, fill=col, outline=col
        )
        c.tag_lower(bg, tid)
        if selected:
            for cx, cy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                c.create_rectangle(
                    cx - 6, cy - 6, cx + 6, cy + 6, fill="white", outline=col, width=2
                )

    def corner_hit(self, x, y):
        if self.sel is None:
            return None
        x1, y1, x2, y2 = self.corners(self.objs[self.sel])
        for name, cx, cy in (
            ("tl", x1, y1),
            ("tr", x2, y1),
            ("bl", x1, y2),
            ("br", x2, y2),
        ):
            if abs(x - cx) <= 10 and abs(y - cy) <= 10:
                return name
        return None

    def is_active_box(self, o):
        """Only boxes of the chosen category can be picked with the mouse.
        Other categories are ignored, so a new box can be drawn on top of them."""
        active = (self.mode.get(), self.region_var.get())
        if o.get("region") is None and o.get("cls") == self.mode.get():
            return True  # an old whole-garment label of the active class
        return (o.get("cls"), o.get("region")) == active

    def box_at(self, x, y):
        best, area = None, None
        for idx, o in enumerate(self.objs):
            if not self.is_active_box(o):
                continue
            x1, y1, x2, y2 = self.corners(o)
            if x1 <= x <= x2 and y1 <= y <= y2:
                a = (x2 - x1) * (y2 - y1)
                if area is None or a < area:
                    best, area = idx, a
        return best

    def set_force_draw(self, on):
        self.force_draw = bool(on)
        if hasattr(self, "new_box_btn"):
            self.new_box_btn.state(["pressed"] if self.force_draw else ["!pressed"])
        self.update_line()

    def toggle_new_box(self):
        self.set_force_draw(not self.force_draw)

    def on_press(self, e):
        if self.pil is None:
            return
        self.canvas.focus_set()
        snap = copy.deepcopy(self.objs)
        if not self.force_draw:
            hit = self.corner_hit(e.x, e.y)
            if hit:
                x1, y1, x2, y2 = self.corners(self.objs[self.sel])
                ax, ay = {"tl": (x2, y2), "tr": (x1, y2), "bl": (x2, y1), "br": (x1, y1)}[
                    hit
                ]
                self.drag = {"kind": "resize", "anchor": self.to_norm(ax, ay), "snap": snap}
                return
            idx = self.box_at(e.x, e.y)
            if idx is not None:
                self.sel = idx
                nx, ny = self.to_norm(e.x, e.y)
                xc, yc = self.objs[idx]["box"][:2]
                self.drag = {"kind": "move", "dx": nx - xc, "dy": ny - yc, "snap": snap}
                self.build_panel()
                self.render()
                self.update_line()
                return
        self.sel = None
        self.drag = {"kind": "draw", "start": (e.x, e.y), "snap": snap}
        self.build_panel()
        self.render()
        self.update_line()

    def on_drag(self, e):
        d = self.drag
        if not d:
            return
        nx, ny = self.to_norm(e.x, e.y)
        if d["kind"] == "draw":
            self.canvas.delete("rubber")
            self.canvas.create_rectangle(
                d["start"][0],
                d["start"][1],
                e.x,
                e.y,
                outline=MODE_COLOR[self.mode.get()],
                width=3,
                dash=(6, 3),
                tags="rubber",
            )
        elif d["kind"] == "move":
            o = self.objs[self.sel]
            w, h = o["box"][2:]
            o["box"][0] = min(1 - w / 2, max(w / 2, nx - d["dx"]))
            o["box"][1] = min(1 - h / 2, max(h / 2, ny - d["dy"]))
            self.render()
        elif d["kind"] == "resize":
            ax, ay = d["anchor"]
            x1, x2, y1, y2 = min(ax, nx), max(ax, nx), min(ay, ny), max(ay, ny)
            if x2 - x1 >= 0.02 and y2 - y1 >= 0.02:
                self.objs[self.sel]["box"] = [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1,
                ]
                self.render()

    def on_release(self, e):
        d, self.drag = self.drag, None
        if not d:
            return
        self.canvas.delete("rubber")
        if d["kind"] == "draw":
            sx, sy = d["start"]
            if abs(e.x - sx) < 12 or abs(e.y - sy) < 12:
                return  # a click, not a drag
            (x1, y1), (x2, y2) = self.to_norm(sx, sy), self.to_norm(e.x, e.y)
            x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            if x2 - x1 < 0.02 or y2 - y1 < 0.02:
                return
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            cls = self.mode.get()
            region = self.region_var.get()
            self.undo.append(d["snap"])
            self.objs.append(
                {
                    "cls": cls,
                    "region": region,
                    "box": [xc, yc, x2 - x1, y2 - y1],
                    "attrs": {},
                }
            )
            self.sel = len(self.objs) - 1
            if self.force_draw:
                self.set_force_draw(False)  # one new box per button press
            self.changed()
        else:
            if self.objs != d["snap"]:
                self.undo.append(d["snap"])
            self.changed()

    def delete_box(self):
        if self.sel is None or self.sel >= len(self.objs):
            return
        self.push_undo()
        del self.objs[self.sel]
        self.sel = None
        self.changed()

    def _build_region_cards(self):
        """One compact button per region; its value is shown inline on the button."""
        for child in self.region_cards.winfo_children():
            child.destroy()
        self.region_card_buttons.clear()
        cls = self.mode.get()
        for index, (region, _title, _attr) in enumerate(REGIONS[cls]):
            btn = ttk.Button(
                self.region_cards,
                text=self._region_button_text(cls, region),
                style="Card.TButton",
                command=lambda r=region: self.choose_region(r),
            )
            btn.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
            self.region_card_buttons[(cls, region)] = btn
        self._highlight_region_card()

    def _region_button_text(self, cls, region):
        attr = REGION_ATTRS[(cls, region)]
        found = [o for o in self.objs if o.get("cls") == cls and o.get("region") == region]
        value = (found[-1].get("attrs") or {}).get(attr) if found else None
        name = SHORT_TITLE.get(region, region)
        if len(found) > 1:
            name = f"{name} x{len(found)}"  # e.g. two pattern areas on one shirt
        return f"{name}: {display_value(value)}" if value else name

    def _highlight_region_card(self):
        for key, button in self.region_card_buttons.items():
            if key == (self.mode.get(), self.region_var.get()):
                button.state(["pressed"])
            else:
                button.state(["!pressed"])

    def choose_region(self, region):
        self.region_var.set(region)
        self.sel = None
        if self.force_draw:
            self.set_force_draw(False)
        self.build_panel()
        self._highlight_region_card()
        self.update_line()

    # ------------------------------------------------------------- right panel
    def build_panel(self):
        for (card_cls, card_region), button in self.region_card_buttons.items():
            button.configure(text=self._region_button_text(card_cls, card_region))
        for widget in self.box_frame.winfo_children():
            widget.destroy()
        if self.sel is None or self.sel >= len(self.objs):
            self.sel = None
            ttk.Label(
                self.box_frame, justify="left", foreground="#666",
                text="Click a category above and draw its box. Select the box to set its value.",
            ).pack(anchor="w")
            return

        obj = self.objs[self.sel]
        cls, region = obj.get("cls"), obj.get("region")
        if cls not in REGIONS or (cls, region) not in REGION_ATTRS:
            # Legacy whole-garment labels remain editable without deleting their data.
            title = NICE.get(cls, str(cls).upper()) + " (older whole-garment label)"
            tk.Label(
                self.box_frame, text=title, bg=COLORS.get(cls, "#555"),
                fg="white", font=self.f_bold, anchor="w", padx=8, pady=3,
            ).pack(fill="x")
            attrs = self.focus_attrs(cls)
        else:
            title = REGION_TITLES[(cls, region)]
            tk.Label(
                self.box_frame, text=f"{cls.upper()}  /  {title}",
                bg=COLORS[cls], fg="white", font=self.f_bold,
                anchor="w", padx=8, pady=3,
            ).pack(fill="x")
            attrs = self.focus_attrs(cls, region)
            if region in ("sleeve_left", "sleeve_right"):
                ttk.Label(
                    self.box_frame, text="Sleeve side (wearer's):", font=self.f_bold,
                ).pack(anchor="w", pady=(6, 1))
                side_var = tk.StringVar(value=region)
                for side, label in (("sleeve_left", "LEFT"), ("sleeve_right", "RIGHT")):
                    ttk.Radiobutton(
                        self.box_frame, text=label, value=side, variable=side_var,
                        command=lambda v=side_var: self.set_sleeve_side(v.get()),
                    ).pack(side="left", padx=(0, 10))

        if not attrs:
            ttk.Label(self.box_frame, text="This attribute is disabled in the label settings.").pack(anchor="w", pady=8)
            return

        self.heads, self.vars = {}, {}
        for attr in attrs:
            frame = ttk.Frame(self.box_frame)
            frame.pack(fill="x", pady=(6, 0))
            head = ttk.Label(frame, font=self.f_bold)
            head.pack(anchor="w")
            var = tk.StringVar(value=(obj.get("attrs") or {}).get(attr, ""))
            self.vars[attr] = var
            self.heads[attr] = head
            values = list(self.schema["attributes"][cls][attr])
            if cls == "top" and region == "length":
                values = list(dict.fromkeys(["cropped", "regular", *values]))
            options = [("", "Not set")] + [
                (value, display_value(value)) for value in values
            ]
            # Grid the radio buttons inside their own frame: a parent cannot use
            # both pack() (head/outer) and grid() (buttons) for its children.
            choices = ttk.Frame(frame)
            choices.pack(fill="x", pady=(2, 0))
            for index, (value, label) in enumerate(options):
                ttk.Radiobutton(
                    choices, text=label, value=value, variable=var,
                    command=lambda a=attr: self.on_attr(a),
                ).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 8))
            self.refresh_head(attr)

    def refresh_head(self, a):
        val = self.vars[a].get()
        title = ATTR_TITLE.get(a, a)
        if val:
            self.heads[a].config(
                text=f"{title}:  {display_value(val)}", foreground="#1b6e2d"
            )
        else:
            self.heads[a].config(text=f"{title}:  not set", foreground="#b00020")

    def on_attr(self, a):
        o = self.objs[self.sel]
        self.push_undo()
        val = self.vars[a].get()
        if val:
            o["attrs"][a] = val
        else:
            o["attrs"].pop(a, None)
        self.refresh_head(a)
        self.save()
        self.render()
        self.refresh_checklist()

    def set_sleeve_side(self, region):
        self.push_undo()
        obj = self.objs[self.sel]
        obj["cls"] = "top"
        obj["region"] = region
        self.changed()

    def set_focus(self, only):
        for key, v in self.attr_on.items():
            a = key[1]
            v.set(True if not only else a in only)
        self.on_focus_change()

    def on_focus_change(self):
        self.recompute_status()
        self.fill_list()
        if self.cur is not None:
            self._lock = True
            self.lb.selection_set(self.cur)
            self._lock = False
        self.build_panel()
        self.refresh_checklist()

    def refresh_checklist(self):
        for widget in self.check_frame.winfo_children():
            widget.destroy()
        items = []
        if self.pil is None:
            items.append(("info", "No photo open."))
        elif not self.objs:
            items.append(("warn", "No region boxes yet. Select TOP/BOTTOM and a box region, then draw."))
        else:
            region_counts = {}
            sleeve_sides = set()
            for obj in self.objs:
                cls, region = obj.get("cls"), obj.get("region")
                if cls in REGIONS and (cls, region) in REGION_ATTRS:
                    region_counts[(cls, region)] = region_counts.get((cls, region), 0) + 1
                    if cls == "top" and region in ("sleeve_left", "sleeve_right"):
                        sleeve_sides.add(region)
                    attr = REGION_ATTRS[(cls, region)]
                    if not (obj.get("attrs") or {}).get(attr):
                        items.append(("warn", f"{REGION_TITLES[(cls, region)]}: choose a value."))
                elif cls in ("top", "bottom"):
                    missing = [
                        ATTR_TITLE.get(attr, attr)
                        for attr in self.focus_attrs(cls)
                        if not (obj.get("attrs") or {}).get(attr)
                    ]
                    if missing:
                        items.append(("warn", f"Older {cls.upper()} box: still empty — {', '.join(missing)}"))
            for (cls, region), count in region_counts.items():
                # A shirt can have several patterns, so multiple pattern boxes are fine.
                if count > 1 and region != "pattern":
                    items.append(("warn", f"More than one {REGION_TITLES[(cls, region)]} box — check for duplicates."))
            if sleeve_sides and sleeve_sides != {"sleeve_left", "sleeve_right"}:
                missing_side = "LEFT" if "sleeve_left" not in sleeve_sides else "RIGHT"
                items.append(("warn", f"Add the other sleeve box for the wearer's {missing_side} arm."))
        if self.objs and not items:
            items.append(("ok", "All drawn regions have labels ✔"))
        colors = {"warn": "#b26a00", "ok": "#1b6e2d", "info": "#555"}
        for kind, message in items:
            tk.Label(
                self.check_frame,
                text=("⚠ " if kind == "warn" else "") + message,
                fg=colors[kind], wraplength=330, justify="left", anchor="w",
            ).pack(fill="x", anchor="w")

    def update_mode(self):
        cls = self.mode.get()
        entries = REGIONS[cls]
        valid = {region for region, _title, _attr in entries}
        if self.region_var.get() not in valid:
            self.region_var.set(entries[0][0])
        self.sel = None                      # never keep a box from the other category selected
        if self.force_draw:
            self.set_force_draw(False)
        if hasattr(self, "region_cards"):
            self._build_region_cards()
        self.update_line()

    def update_line(self):
        if self.force_draw:
            cls, region = self.mode.get(), self.region_var.get()
            title = REGION_TITLES.get((cls, region), region)
            self.line.set(f"NEW {cls.upper()} box for {title}: drag anywhere, even over other boxes (N to stop).")
        elif self.sel is not None:
            self.line.set(
                "Selected box: drag it to move · drag a white corner to resize · press Delete to remove it."
            )
        else:
            cls, region = self.mode.get(), self.region_var.get()
            title = REGION_TITLES.get((cls, region), region)
            self.line.set(f"Draw a {cls.upper()} box for: {title}.")

    def open_guide(self):
        webbrowser.open(GUIDE.as_uri())

    def on_close(self):
        self.save()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="Label tool for raja_fashion")
    ap.add_argument("folder", nargs="?")
    ap.add_argument(
        "--only",
        nargs="*",
        help="only ask for these attributes, e.g. --only neck  or  --only top.neck bottom.fit",
    )
    args = ap.parse_args()
    root = tk.Tk()
    root.title("raja_fashion label tool")
    root.geometry("1400x880")
    root.minsize(1150, 720)
    App(root, args.folder, args.only)
    root.mainloop()


if __name__ == "__main__":
    main()
