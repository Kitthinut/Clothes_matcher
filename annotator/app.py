"""Raja-Fashion desktop annotator (Tkinter, no browser or model download).

Run::

    python -m annotator.app [--images DIR] [--annotations DIR]

The GUI has one left canvas and one compact right panel: a TOP/BOTTOM toggle,
attribute chips, large label chips, read-only automatic-colour cards and the
box list. Every region is a separate YOLO detection box; colour is computed
automatically from a dedicated Colour box.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from annotator.canvas import AnnotationCanvas  # noqa: E402
from annotator.image_check import (  # noqa: E402
    ImageSizeReport, inspect_images, show_image_report,
)
from annotator.resize_tool import parse_size, resize_images  # noqa: E402
from annotator.store import (  # noqa: E402
    AnnotationStore, Box, ImageAnnotation, resolve_annotation_dirs,
)
from annotator.ui_utils import tip  # noqa: E402
from common.colors import ColorInfo, analyze_color  # noqa: E402
from common.config import Config, load_config  # noqa: E402
from common.logging_conf import LOGGER  # noqa: E402
from common.mock_data import ensure_mock_dataset  # noqa: E402
from common.paths import ensure_dir  # noqa: E402
from common.schema import Region, Schema, load_schema  # noqa: E402

LOG = LOGGER.getChild("annotator.desktop")

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")
BG = "#f6f7fb"
PANEL = "#ffffff"
TEXT = "#1f2433"
MUTED = "#6b7280"
ACCENT = "#6d5efc"
ACCENT_DARK = "#5b4ee6"
WARN = "#b91c1c"
OK = "#166534"

# Distinct colour per attribute (so overlapping boxes stay readable).
BOX_COLORS: dict[tuple[str, str], str] = {
    ("top", "length"): "#22c55e", ("top", "type"): "#3b82f6",
    ("top", "pattern"): "#a855f7", ("top", "neck"): "#ec4899",
    ("top", "sleeves"): "#f59e0b", ("top", "color"): "#14b8a6",
    ("bottom", "waist_length"): "#0ea5e9", ("bottom", "bottom_length"): "#8b5cf6",
    ("bottom", "pattern"): "#d946ef", ("bottom", "fit_type"): "#f97316",
    ("bottom", "color"): "#0d9488",
}

HELP = """Raja-Fashion annotator — shortcuts

Mouse
  Left-drag empty area / any box : draw the active attribute's box
  Click a box                    : select it
  Drag the selected box          : move it
  Drag its corner handles        : resize it
  Delete / Backspace             : remove selected box
  Wheel                          : zoom (anchored at pointer)
  Middle-drag / Space+drag       : pan

Keyboard
  T / B              : switch Top / Bottom
  Tab / Shift+Tab    : next / previous attribute chip
  1-9                : label option (when a box is selected), otherwise chip
  Enter              : save and go to next image
  Left / Right       : previous / next image
  Ctrl+Z / Ctrl+Y    : undo / redo
  Ctrl+S             : save
  Ctrl+N             : next unlabelled image
  F1                 : this help

Colour is automatic: draw a Colour box over a representative patch of the
garment. The swatch updates live as you move or resize that box.
"""


class DesktopAnnotator:
    """Tk pop-up window for drawing and labelling attribute-region boxes."""

    def __init__(
        self,
        win: tk.Tk,
        config: Config | None = None,
        schema: Schema | None = None,
        *,
        image_dir: str | Path | None = None,
        annotations_dir: str | Path | None = None,
    ) -> None:
        self.win = win
        self.config = config or load_config()
        self.schema = schema or load_schema()
        images, annotations = resolve_annotation_dirs(self.config, image_dir, annotations_dir)
        ensure_dir(images)
        ensure_dir(annotations)
        self.store = AnnotationStore(images, annotations, self.schema)
        self.paths = self.store.image_paths()
        if not self.paths and image_dir is None and annotations_dir is None:
            ensure_mock_dataset(config=self.config, schema=self.schema)
            self.paths = self.store.image_paths()
        if not self.paths:
            raise RuntimeError("No images found in the selected folder")
        self.image_report: ImageSizeReport = inspect_images(self.paths)

        self.index = max(0, min(int(self.store.load_session().get("last_index", 0)), len(self.paths) - 1))
        self.annotation = ImageAnnotation("", 0, 0)
        self.selected: str | None = None
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.skipped: list[str] = list(self.store.load_session().get("skipped", []))
        self.pil_image: Image.Image | None = None
        self.color_info: dict[str, ColorInfo] = {}
        self.pending_labels: dict[str, str] = {}

        # Attribute chips (whole-item box is internal, not a chip).
        self.choices: dict[str, tuple[str, str, str, str]] = {}
        self.choice_order: list[str] = []
        self.by_garment: dict[str, list[str]] = {g: [] for g in self.schema.garment_names}
        for garment in self.schema.garment_names:
            for region in self.schema.regions(garment):
                if region.name == "garment":
                    continue
                for side in region.sides:
                    title = region.title + (f" {'L' if side == 'left' else 'R'}" if region.paired else "")
                    key = self.choice_key(garment, region.name, side)
                    self.choices[key] = (garment, region.name, side, title)
                    self.choice_order.append(key)
                    self.by_garment[garment].append(key)

        self.active_garment = self.schema.garment_names[0]
        self.active_key = self.by_garment[self.active_garment][0]
        self.absent_vars: dict[str, tk.BooleanVar] = {}
        self.garment_buttons: dict[str, tk.Button] = {}
        self.chip_buttons: dict[str, tk.Button] = {}
        self.label_buttons: list[tk.Button] = []
        self.progress_var = tk.DoubleVar(value=0.0)
        self.counter_var = tk.StringVar()
        self.file_var = tk.StringVar()
        self.label_title_var = tk.StringVar()
        self.label_note_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.selected_var = tk.StringVar(value="None — draw a box first")

        self.build_ui()
        self.load_index(self.index)
        self.win.after(100, self.render_hint)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def choice_key(garment: str, region: str, side: str) -> str:
        return f"{garment}_{region}_{side}" if region != "garment" else f"{garment}_garment"

    def current_path(self) -> Path:
        return self.paths[self.index]

    def boxes(self) -> list[Box]:
        return self.annotation.boxes

    def selected_box(self) -> Box | None:
        return next((b for b in self.boxes() if b.id == self.selected), None)

    def active_choice(self) -> tuple[str, str, str]:
        key = self.active_key if self.active_key in self.choices else self.by_garment[self.active_garment][0]
        garment, region, side, _ = self.choices[key]
        return garment, region, side

    def region_for(self, garment: str, region: str) -> Region | None:
        return next((r for r in self.schema.regions(garment) if r.name == region), None)

    def box_color(self, box: Box) -> str:
        return BOX_COLORS.get((box.garment, box.region), "#22c55e")

    def box_title(self, box: Box) -> str:
        region = self.region_for(box.garment, box.region)
        title = region.title if region else box.region
        if box.region == "color":
            return f"Colour: {box.label}"
        if box.side != "center":
            title += f" {'L' if box.side == 'left' else 'R'}"
        return f"{title}: {box.label}"

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def check_sizes(self) -> None:
        """Show size report and offer resizing when sizes differ."""
        self.offer_resize()

    def offer_resize(self) -> None:
        """At start-up (or on demand), ask whether to normalise image sizes."""
        self.image_report = inspect_images(self.paths)
        if self.image_report.uniform:
            show_image_report(self.win, self.image_report)
            return
        question = (
            self.image_report.message()
            + "\n\nResize ALL images to one size with black padding (no distorting)?\n"
            "Originals are copied to images_original_backup/ first, and existing\n"
            "annotation boxes are rescaled to the new coordinates."
        )
        if not messagebox.askyesno("Images have different sizes", question, parent=self.win):
            show_image_report(self.win, self.image_report)
            return
        raw = simpledialog.askstring(
            "Target size", "Width x Height (4:3 default):",
            initialvalue="384x288", parent=self.win,
        )
        if not raw:
            return
        try:
            target = parse_size(raw)
        except ValueError as exc:
            messagebox.showerror("Invalid size", str(exc), parent=self.win)
            return
        try:
            result = resize_images(
                self.store.images_dir, self.store.annotations_dir, target,
                backup_dir=self.store.images_dir.parent / "images_original_backup",
                schema=self.schema,
            )
        except (OSError, ValueError) as exc:
            LOG.exception("Resize failed")
            messagebox.showerror("Resize failed", str(exc), parent=self.win)
            return
        self.image_report = inspect_images(self.paths)
        self.load_index(self.index)
        messagebox.showinfo("Images resized", result.message(), parent=self.win)
        self.set_status(f"All images resized to {target[0]}x{target[1]}")

    def render_hint(self) -> None:
        """One hint bar derived from the real active state (never out of sync)."""
        garment, region, _side = self.active_choice()
        _, _, _, title = self.choices[self.active_key]
        if garment in self.annotation.absent_garments:
            self.set_status(f"{garment.title()} is marked absent — untick to annotate")
        else:
            self.set_status(f"{garment.title()} > {title}: drag on the image to draw")

    # --------------------------------------------------------------------- UI
    def build_ui(self) -> None:
        w = self.win
        w.title("Raja-Fashion Annotator")
        w.geometry("1420x900")
        w.minsize(1100, 700)
        w.configure(bg=BG)

        style = ttk.Style()
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=FONT)
        style.configure("Header.TLabel", background=BG, foreground=TEXT, font=FONT_TITLE)
        style.configure("TLabelframe", background=PANEL, foreground=TEXT)
        style.configure("TLabelframe.Label", background=PANEL, foreground=TEXT, font=FONT_BOLD)
        style.configure("TButton", font=FONT, padding=5)
        style.configure("Accent.TButton", font=FONT_BOLD, padding=5,
                        foreground="#ffffff", background=ACCENT)
        style.configure("Warn.TButton", font=FONT_BOLD, padding=5, foreground=WARN)

        # ------------------------------------------------------------- top bar
        top = ttk.Frame(w, padding=(12, 9))
        top.pack(side="top", fill="x")
        tip(ttk.Label(top, text="Raja-Fashion", style="Header.TLabel"), "Fashion attribute annotator").pack(side="left")
        ttk.Label(top, textvariable=self.counter_var).pack(side="left", padx=(18, 8))
        ttk.Progressbar(top, variable=self.progress_var, maximum=1.0, length=170, mode="determinate").pack(side="left")
        actions = ttk.Frame(top)
        actions.pack(side="right")
        for text, cmd, help_text in (
            ("Undo", self.undo, "Ctrl+Z"),
            ("Redo", self.redo, "Ctrl+Y"),
        ):
            tip(ttk.Button(actions, text=text, command=cmd), help_text).pack(side="left", padx=2)
        tip(ttk.Button(actions, text="Clear", style="Warn.TButton", command=self.confirm_clear),
            "Remove all boxes on this image").pack(side="left", padx=2)
        tip(ttk.Button(actions, text="Skip", command=lambda: self.navigate(1)),
            "Save and go to the next image").pack(side="left", padx=(10, 2))
        tip(tk.Button(actions, text="Submit", bg=ACCENT, fg="white", activebackground=ACCENT_DARK,
                      activeforeground="white", relief="flat", font=FONT_BOLD,
                      command=lambda: self.navigate(1)), "Save and submit (Enter)").pack(side="left", padx=2)

        # --------------------------------------------------------- main split
        main = ttk.Frame(w)
        main.pack(side="top", fill="both", expand=True)

        canvas_host = ttk.Frame(main)
        canvas_host.pack(side="left", fill="both", expand=True)
        self.canvas = AnnotationCanvas(
            canvas_host,
            on_select=self.select_box,
            on_create=self.on_box_created,
            on_change=self.on_box_changed,
            on_status=self.set_status,
        )
        self.canvas.pack(side="top", fill="both", expand=True)
        tip(ttk.Button(canvas_host, text="Fit image", command=self.canvas.fit),
            "Reset zoom and pan").pack(side="bottom", anchor="w", padx=8, pady=6)

        right = ttk.Frame(main, width=390, padding=(10, 8))
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        file_row = ttk.Frame(right)
        file_row.pack(fill="x", pady=(0, 8))
        tip(ttk.Label(file_row, textvariable=self.file_var, style="Header.TLabel"),
            "Current image file").pack(side="left", fill="x", expand=True)
        tip(ttk.Button(file_row, text="Check sizes", command=self.check_sizes),
            "Show image-size report for this folder").pack(side="right")

        selector = ttk.LabelFrame(right, text="Attribute", padding=8)
        selector.pack(fill="x", pady=(0, 8))
        toggle = ttk.Frame(selector)
        toggle.pack(fill="x")
        for garment in self.schema.garment_names:
            btn = tk.Button(toggle, text=garment.upper(), font=FONT_BOLD, relief="flat",
                            command=lambda g=garment: self.set_garment(g))
            btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
            tip(btn, f"Switch to {garment} (T/B)")
            self.garment_buttons[garment] = btn
        absent_row = ttk.Frame(selector)
        absent_row.pack(fill="x", pady=(6, 2))
        for garment in self.schema.garment_names:
            var = tk.BooleanVar(value=False)
            self.absent_vars[garment] = var
            cb = ttk.Checkbutton(absent_row, text=f"{garment} absent", variable=var,
                                 command=self.on_absent_toggle)
            cb.pack(side="left", padx=(0, 10))
            tip(cb, f"Tick when this image has no {garment}")
        self.chip_frame = ttk.Frame(selector)
        self.chip_frame.pack(fill="x", pady=(6, 0))

        label_section = ttk.LabelFrame(right, text="Label", padding=8)
        label_section.pack(fill="x", pady=(0, 8))
        self.label_title_var.set("No box selected")
        ttk.Label(label_section, textvariable=self.label_title_var, font=FONT_BOLD).pack(fill="x")
        self.label_frame = ttk.Frame(label_section)
        self.label_frame.pack(fill="x", pady=(6, 0))
        ttk.Label(label_section, textvariable=self.label_note_var, foreground=MUTED,
                  wraplength=350, justify="left").pack(fill="x", pady=(4, 0))

        colour_section = ttk.LabelFrame(right, text="Colour (automatic)", padding=8)
        colour_section.pack(fill="x", pady=(0, 8))
        self.color_frame = ttk.Frame(colour_section)
        self.color_frame.pack(fill="x")

        boxes_section = ttk.LabelFrame(right, text="Boxes in this image", padding=6)
        boxes_section.pack(fill="both", expand=True, pady=(0, 8))
        self.region_list = tk.Listbox(boxes_section, height=6, exportselection=False,
                                      font=FONT, selectbackground=ACCENT)
        self.region_list.pack(fill="both", expand=True)
        self.region_list.bind("<<ListboxSelect>>", self.on_region_select)

        nav = ttk.Frame(right)
        nav.pack(fill="x", pady=(4, 0))
        for text, cmd, help_text in (
            ("< Prev", lambda: self.navigate(-1), "Left arrow"),
            ("Save", self.save, "Ctrl+S"),
            ("Next >", lambda: self.navigate(1), "Right arrow / Enter"),
        ):
            tip(ttk.Button(nav, text=text, command=cmd), help_text).pack(
                side="left", expand=True, fill="x", padx=(0, 3))
        tip(ttk.Button(right, text="Next unlabelled (Ctrl+N)", command=self.next_unlabelled),
            "Jump to the next image with no annotations").pack(fill="x", pady=(5, 0))

        hint = ttk.Frame(w, padding=(12, 6))
        hint.pack(side="bottom", fill="x")
        ttk.Label(hint, textvariable=self.status_var, foreground=TEXT).pack(side="left")
        ttk.Label(hint, text="F1 Help · wheel zoom · middle/space-drag pan", foreground=MUTED).pack(side="right")

        for seq, handler in (
            ("<Delete>", lambda e: None if self.typing() else self.delete_box()),
            ("<BackSpace>", lambda e: None if self.typing() else self.delete_box()),
            ("<Escape>", lambda e: self.select_box(None)),
            ("<Return>", lambda e: None if self.typing() else self.navigate(1)),
            ("<Left>", lambda e: None if self.typing() else self.navigate(-1)),
            ("<Right>", lambda e: None if self.typing() else self.navigate(1)),
            ("<Control-z>", lambda e: self.undo()),
            ("<Control-y>", lambda e: self.redo()),
            ("<Control-s>", lambda e: self.save()),
            ("<Control-n>", lambda e: self.next_unlabelled()),
            ("<F1>", lambda e: messagebox.showinfo("Help", HELP)),
            ("<Tab>", lambda e: self._shift_chip(1)),
            ("<Shift-Tab>", lambda e: self._shift_chip(-1)),
            ("<Key-t>", lambda e: None if self.typing() else self.set_garment("top")),
            ("<Key-b>", lambda e: None if self.typing() else self.set_garment("bottom")),
        ):
            w.bind(seq, handler)
        for i in range(1, 10):
            w.bind(f"<Key-{i}>", lambda _e, n=i: self._number_key(n))
        w.protocol("WM_DELETE_WINDOW", self.on_close)

    def typing(self) -> bool:
        try:
            return isinstance(self.win.focus_get(), (tk.Entry, ttk.Combobox, tk.Listbox))
        except tk.TclError:
            return False

    # ------------------------------------------------------------- image/state
    def load_index(self, index: int) -> None:
        index = max(0, min(index, len(self.paths) - 1))
        self.index = index
        path = self.current_path()
        try:
            self.annotation = self.store.load(path)
            with Image.open(path) as src:
                self.pil_image = src.convert("RGB")
        except (OSError, ValueError) as exc:
            LOG.warning("Could not load %s: %s", path, exc)
            messagebox.showwarning("Image problem", f"Could not load {path.name}: {exc}")
            self.annotation = ImageAnnotation(path.name, 0, 0)
            self.pil_image = None
            return
        self.selected = self.boxes()[0].id if self.boxes() else None
        self.color_info = {}
        self.undo_stack, self.redo_stack = [], []
        self.store.save_session(index, self.skipped)
        self.canvas.set_image(self.pil_image)
        self.file_var.set(path.name)
        done, total = self.store.progress()
        self.progress_var.set(done / max(1, total))
        self.counter_var.set(f"Image {index + 1} of {total} · annotated {done}/{total}")
        for garment, var in self.absent_vars.items():
            var.set(garment in self.annotation.absent_garments)
        if self.active_garment in self.annotation.absent_garments:
            self.set_garment(next(g for g in self.schema.garment_names
                                  if g not in self.annotation.absent_garments), render=False)
        for box in self.boxes():
            if box.region == "color":
                self.refresh_color(box)
        self.render_all()

    def render_all(self) -> None:
        self.canvas.set_boxes(self.boxes(), self.selected, self.box_color, self.box_title)
        self.render_selector()
        self.render_labels()
        self.render_color_cards()
        self.render_regions()
        self.render_hint()

    def redraw(self) -> None:
        self.render_all()

    # ------------------------------------------------------------- rendering
    def render_selector(self) -> None:
        for garment, btn in self.garment_buttons.items():
            active = garment == self.active_garment
            btn.configure(bg=ACCENT if active else PANEL,
                          fg="white" if active else TEXT,
                          activebackground=ACCENT_DARK if active else "#eef0f7")
        for child in self.chip_frame.winfo_children():
            child.destroy()
        self.chip_buttons.clear()
        keys = self.by_garment[self.active_garment]
        for i, key in enumerate(keys):
            _g, region, side, title = self.choices[key]
            found = next((b for b in self.boxes()
                          if b.garment == self.active_garment and b.region == region and b.side == side), None)
            text = (f"✓ {title}: {found.label}" if found else f"{i + 1} {title}")
            absent = self.active_garment in self.annotation.absent_garments
            active = key == self.active_key
            bg = ACCENT if active and not absent else ("#e5e7ef" if found else PANEL)
            fg = "white" if active and not absent else (OK if found and not active else TEXT)
            btn = tk.Button(self.chip_frame, text=text, font=FONT, relief="flat", bg=bg, fg=fg,
                            activebackground=ACCENT_DARK, wraplength=155, justify="left",
                            state="disabled" if absent else "normal",
                            command=lambda k=key: self.set_category(k))
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            self.chip_frame.columnconfigure(i % 2, weight=1)
            tip(btn, f"Draw a {title} box for this image")
            self.chip_buttons[key] = btn

    def render_labels(self) -> None:
        for child in self.label_frame.winfo_children():
            child.destroy()
        self.label_buttons.clear()
        box = self.selected_box()
        garment, region, _side = self.active_choice()
        reg = self.region_for(garment, region)
        if box is not None:
            reg = self.region_for(box.garment, box.region)
        if reg is None or reg.name in ("garment", "color"):
            self.label_title_var.set("No label to choose")
            self.label_note_var.set(
                "Colour is detected automatically from the Colour box."
                if box is not None and box.region == "color"
                else "Select a box, or draw one for the active attribute."
            )
            self.selected_var.set("None — draw a box first" if box is None else self.box_title(box))
            return
        current = box.label if box is not None else self.pending_labels.get(reg.key, reg.classes[0])
        self.label_title_var.set(f"{garment.title()} > {reg.title}")
        self.label_note_var.set("Click a label, or press 1-9. The first option is used until you choose.")
        for i, value in enumerate(reg.classes):
            btn = tk.Button(self.label_frame, text=f"{i + 1} {value}", font=FONT, relief="flat",
                            bg=ACCENT if value == current else PANEL,
                            fg="white" if value == current else TEXT,
                            activebackground=ACCENT_DARK,
                            command=lambda v=value: self.choose_label(v))
            btn.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
            self.label_frame.columnconfigure(i % 3, weight=1)
            self.label_buttons.append(btn)
        self.selected_var.set(self.box_title(box) if box is not None else "None — draw a box first")

    def render_color_cards(self) -> None:
        for child in self.color_frame.winfo_children():
            child.destroy()
        for garment in self.schema.garment_names:
            boxes = [b for b in self.boxes() if b.garment == garment and b.region == "color"]
            for box in boxes:
                info = self.color_info.get(box.id)
                if info is None:
                    self.refresh_color(box)
                    info = self.color_info.get(box.id)
                if info is None:
                    continue
                card = tk.Frame(self.color_frame, bg=PANEL, highlightbackground="#d8dbe5",
                                highlightthickness=1)
                card.pack(fill="x", pady=2)
                swatch = tk.Label(card, text="  ", bg=info.hex, width=3, relief="flat")
                swatch.pack(side="left", padx=6, pady=5)
                hsv = ", ".join(f"{v:.2f}" for v in info.hsv)
                tk.Label(card, text=f"{garment.title()}: {info.friendly} {info.hex}\nHSV ({hsv})",
                         bg=PANEL, fg=TEXT, font=FONT, justify="left").pack(side="left", pady=4)
                tip(card, "Detected automatically — move or redraw the Colour box to update")

    def render_regions(self) -> None:
        self.region_list.delete(0, "end")
        for box in self.boxes():
            self.region_list.insert("end", self.box_title(box))
        if self.selected:
            for i, box in enumerate(self.boxes()):
                if box.id == self.selected:
                    self.region_list.selection_clear(0, "end")
                    self.region_list.selection_set(i)
                    self.region_list.see(i)
                    break

    # --------------------------------------------------------------- actions
    def set_garment(self, garment: str, *, render: bool = True) -> None:
        if garment not in self.by_garment or not self.by_garment[garment]:
            return
        self.active_garment = garment
        key = next((k for k in self.by_garment[garment]
                    if k not in self.annotation.absent_garments), self.by_garment[garment][0])
        self.set_category(key, render=render)

    def set_category(self, key: str, *, render: bool = True) -> None:
        if key not in self.choices:
            return
        self.active_key = key
        self.active_garment = self.choices[key][0]
        if self.active_garment in self.annotation.absent_garments:
            self.annotation.absent_garments.remove(self.active_garment)
            self.absent_vars[self.active_garment].set(False)
            self.save(show_status=False)
        self.selected = None
        if render:
            self.render_all()
            self.render_hint()

    def on_absent_toggle(self) -> None:
        self.annotation.absent_garments = [g for g, v in self.absent_vars.items() if v.get()]
        self.save(show_status=False)
        self.render_all()
        self.render_hint()

    def _shift_chip(self, delta: int) -> str:
        keys = self.by_garment[self.active_garment]
        index = keys.index(self.active_key) if self.active_key in keys else 0
        self.set_category(keys[(index + delta) % len(keys)])
        return "break"

    def _number_key(self, n: int) -> str:
        if self.selected_box() is not None and self.label_buttons and n <= len(self.label_buttons):
            self.label_buttons[n - 1].invoke()
            return "break"
        keys = self.by_garment[self.active_garment]
        if n <= len(keys):
            self.set_category(keys[n - 1])
        return "break"

    def choose_label(self, value: str) -> None:
        box = self.selected_box()
        if box is not None and box.region not in ("garment", "color"):
            self.push_undo()
            box.label = value
            self.save(show_status=False)
            self.render_all()
            self.set_status(f"{self.box_title(box)} saved")
        else:
            _g, region, _side = self.active_choice()
            reg = self.region_for(_g, region)
            if reg and value in reg.classes:
                self.pending_labels[reg.key] = value
                self.render_labels()
                self.set_status(f"Next {reg.title} box will be {value}")

    def select_box(self, box_id: str | None) -> None:
        self.selected = box_id
        if box_id:
            box = self.selected_box()
            if box and self.choice_key(box.garment, box.region, box.side) in self.choices:
                self.active_key = self.choice_key(box.garment, box.region, box.side)
                self.active_garment = box.garment
        self.render_all()
        self.render_hint()

    def on_region_select(self, _event: Any) -> None:
        selection = self.region_list.curselection()
        if selection and selection[0] < len(self.boxes()):
            self.select_box(self.boxes()[selection[0]].id)

    def upsert_box(self, garment: str, region: str, side: str, value: str,
                   xyxy: tuple[int, ...]) -> Box | None:
        """Replace or create the single box for a (garment, region, side)."""
        existing = next((b for b in self.boxes()
                         if b.garment == garment and b.region == region and b.side == side), None)
        if existing is not None:
            existing.xyxy = tuple(int(v) for v in xyxy)  # type: ignore[assignment]
            if region != "color":
                existing.label = value
            return existing
        box = Box.create(garment, region, side, value, xyxy)
        self.boxes().append(box)
        return box

    def refresh_color(self, box: Box) -> ColorInfo | None:
        """Compute colour from a Colour box and store label + HSV in the box."""
        if self.pil_image is None:
            return None
        x1, y1, x2, y2 = box.xyxy
        crop = self.pil_image.crop((max(0, x1), max(0, y1),
                                    min(self.pil_image.width, x2), min(self.pil_image.height, y2)))
        info = analyze_color(crop, allowed=set(self.schema.color_classes))
        if info is None:
            return None
        box.label = info.name
        box.hsv = info.hsv
        self.color_info[box.id] = info
        return info

    def on_box_created(self, xyxy: tuple[int, int, int, int]) -> None:
        """Create or replace the box for the active region at *xyxy*."""
        garment, region, side = self.active_choice()
        reg = self.region_for(garment, region)
        value = self.pending_labels.get(reg.key, reg.classes[0]) if reg else garment
        self.push_undo()
        box = self.upsert_box(garment, region, side, value, xyxy)
        if box is None:
            self.render_all()
            return
        self.selected = box.id
        info = self.refresh_color(box) if region == "color" else None
        self.save(show_status=False)
        self.render_all()
        self.set_status(
            f"{garment.title()} > Colour: {info.friendly} {info.hex}" if info
            else f"{garment.title()} > {region}: box saved"
        )

    def on_box_changed(self, box: Box) -> None:
        """Persist a moved/resized box; refresh colour live."""
        if box.region == "color":
            info = self.refresh_color(box)
            if info:
                self.set_status(f"{box.garment.title()} > Colour: {info.friendly} {info.hex}")
        self.save(show_status=False)
        self.render_all()

    # Mouse wrappers kept for compatibility/tests; the canvas owns the logic.
    def on_press(self, event: Any) -> None:
        self.canvas._press_draw(event)

    def on_drag(self, event: Any) -> None:
        self.canvas._motion(event)

    def on_release(self, event: Any) -> None:
        self.canvas._release(event)

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        return self.canvas.to_image(x, y)

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.canvas.to_canvas(x, y)

    # ------------------------------------------------------------ save/history
    def push_undo(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.annotation.to_dict()))
        self.undo_stack = self.undo_stack[-50:]
        self.redo_stack.clear()

    def save(self, show_status: bool = True) -> None:
        try:
            self.annotation.validate(self.schema)
            self.store.save(self.annotation)
            done, total = self.store.progress()
            self.progress_var.set(done / max(1, total))
            self.counter_var.set(f"Image {self.index + 1} of {total} · annotated {done}/{total}")
            if show_status:
                self.set_status("Saved")
        except (ValueError, OSError) as exc:
            LOG.exception("Save failed")
            messagebox.showerror("Save failed", str(exc))

    def _restore(self, data: dict) -> None:
        self.annotation = ImageAnnotation.from_dict(data)
        self.selected = self.boxes()[0].id if self.boxes() else None
        self.color_info = {}
        for box in self.boxes():
            if box.region == "color":
                self.refresh_color(box)
        self.save(show_status=False)
        self.render_all()

    def undo(self) -> None:
        if not self.undo_stack:
            self.set_status("Nothing to undo")
            return
        self.redo_stack.append(copy.deepcopy(self.annotation.to_dict()))
        self._restore(self.undo_stack.pop())
        self.set_status("Undo")

    def redo(self) -> None:
        if not self.redo_stack:
            self.set_status("Nothing to redo")
            return
        self.undo_stack.append(copy.deepcopy(self.annotation.to_dict()))
        self._restore(self.redo_stack.pop())
        self.set_status("Redo")

    def delete_box(self) -> None:
        if not self.selected:
            self.set_status("Select a box first")
            return
        self.push_undo()
        self.boxes()[:] = [b for b in self.boxes() if b.id != self.selected]
        self.selected = None
        self.save(show_status=False)
        self.render_all()
        self.set_status("Box removed")

    def confirm_clear(self) -> None:
        if self.boxes() and messagebox.askyesno("Clear boxes", "Remove all boxes on this image?"):
            self.push_undo()
            self.boxes().clear()
            self.selected = None
            self.color_info.clear()
            self.save(show_status=False)
            self.render_all()
            self.set_status("All boxes cleared")

    def clear_boxes(self) -> None:
        self.confirm_clear()

    def navigate(self, delta: int) -> None:
        self.save(show_status=False)
        new_index = self.index + delta
        if 0 <= new_index < len(self.paths):
            self.load_index(new_index)
        else:
            self.set_status("No more images")

    def next_unlabelled(self) -> None:
        for offset in range(1, len(self.paths) + 1):
            candidate = (self.index + offset) % len(self.paths)
            if not self.store.has_annotation(self.paths[candidate]):
                self.save(show_status=False)
                self.load_index(candidate)
                return
        self.set_status("All images annotated")

    def on_close(self) -> None:
        self.save(show_status=False)
        self.win.destroy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Raja-Fashion desktop annotator")
    parser.add_argument("--images", type=str, default=None)
    parser.add_argument("--annotations", type=str, default=None)
    args = parser.parse_args(argv)
    root = tk.Tk()
    try:
        app = DesktopAnnotator(root, image_dir=args.images, annotations_dir=args.annotations)
    except RuntimeError as exc:
        messagebox.showerror("No images", str(exc))
        root.destroy()
        return
    root.after(300, app.offer_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
