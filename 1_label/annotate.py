"""GUI annotator (1_label/annotate.py). Draw per-category boxes and save annotations."""

import argparse
import json
import os
import shutil
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

# 1_label/ -> project root
ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = Path(__file__).resolve().parent
for p in (ROOT, LABEL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.colors import detect_colors
from shared.data import EXTS, SCHEMA_VERSION
from shared.schema import load_full

NONE = "__none__"


def parse_target_size(size_str):
    """Parse width and height from '288x384', '288,384', or a single integer."""
    if isinstance(size_str, (list, tuple)) and len(size_str) == 2:
        return (int(size_str[0]), int(size_str[1]))
    if isinstance(size_str, str):
        size_str = size_str.lower().replace("x", ",").replace(" ", "")
        if "," in size_str:
            parts = size_str.split(",")
            return (int(parts[0]), int(parts[1]))
    val = int(size_str)
    return (val, val)


class Annotator:
    def __init__(self, win, folder, ann_path, target_size=(288, 384), garment="top"):
        self.win, self.folder, self.ann_path = win, Path(folder), Path(ann_path)
        self.target_size = target_size  # (Width, Height), e.g. (288, 384)
        self.garment = garment
        self.schema = load_full(garment)
        self.files = sorted(
            p.relative_to(self.folder).as_posix()
            for p in self.folder.rglob("*")
            if p.suffix.lower() in EXTS
        )
        if not self.files:
            messagebox.showerror("No images", f"No images found in {self.folder}")
            raise SystemExit
        self.data = (
            json.loads(self.ann_path.read_text(encoding="utf-8"))
            if self.ann_path.exists()
            else {}
        )
        self.source_target_size = tuple(
            self.data.get("target_size", target_size)
        )
        if self.data.get("garment", garment) != garment:
            raise ValueError(
                f"Annotations are for {self.data['garment']}, not {garment}."
            )
        self.data["schema_version"] = SCHEMA_VERSION
        self.data.setdefault("garment", garment)
        self.data.setdefault("images", {})
        self.data["image_root"] = self.image_root_relative()
        self.data["target_size"] = list(self.target_size)
        self.trash = self.folder.resolve().parent / "deleted_pictures"
        self.idx, self.start, self.cur_box, self.box_id = 0, None, None, None
        self.vars = {}
        self.category_frames = {}
        self.upgrade_existing_images()
        self.build_ui()
        win.after(150, self.begin)

    def image_root_relative(self):
        return os.path.relpath(
            self.folder.resolve(), self.ann_path.parent.resolve()
        ).replace("\\", "/")

    def scale_box(self, box, from_size, to_size):
        if not box or len(box) != 4:
            return None
        from_w, from_h = from_size
        to_w, to_h = to_size
        if not from_w or not from_h:
            return None
        return [
            round(box[0] * to_w / from_w),
            round(box[1] * to_h / from_h),
            round(box[2] * to_w / from_w),
            round(box[3] * to_h / from_h),
        ]

    def upgrade_existing_images(self):
        crop_keys = {
            cat["key"] for cat in self.schema["categories"] if cat.get("crop_boxes")
        }
        for info in self.data["images"].values():
            if info.get("regions"):
                self.upgrade_region_list(info, crop_keys)
                continue
            regions = []
            for garment in info.get("garments", []):
                original_size = (
                    info.get("width") or self.target_size[0],
                    info.get("height") or self.target_size[1],
                )
                attrs = garment.get("attributes") or {}
                base_attrs = {key: value for key, value in attrs.items() if key not in crop_keys}
                if base_attrs or garment.get("box"):
                    box = self.scale_box(
                        garment.get("box") or [0, 0, *original_size],
                        original_size,
                        self.target_size,
                    )
                    region = {
                        "category": "garment",
                        "box": box or [0, 0, *self.target_size],
                        "attributes": base_attrs,
                    }
                    if garment.get("colors"):
                        region["shirt_colors"] = garment["colors"]
                    regions.append(region)

                crops = garment.get("attribute_crops") or {}
                neck = crops.get("neck")
                if attrs.get("neck") and neck:
                    scaled = self.scale_box(neck, original_size, self.target_size)
                    if scaled:
                        regions.append({
                            "category": "neck",
                            "box": scaled,
                            "attributes": {"neck": attrs["neck"]},
                        })

                sleeves = crops.get("sleeves")
                if attrs.get("sleeves") and sleeves:
                    if isinstance(sleeves, dict):
                        sleeve_boxes = sleeves.items()
                    else:
                        sleeve_boxes = (("left", sleeves),)
                    for side, box in sleeve_boxes:
                        scaled = self.scale_box(box, original_size, self.target_size)
                        if scaled:
                            regions.append({
                                "category": "sleeves",
                                "side": side,
                                "box": scaled,
                                "attributes": {"sleeves": attrs["sleeves"]},
                            })
            if not regions and info.get("attributes"):
                attrs = info["attributes"]
                base_attrs = {key: value for key, value in attrs.items() if key not in crop_keys}
                if base_attrs:
                    regions.append({
                        "category": "garment",
                        "box": [0, 0, *self.target_size],
                        "attributes": base_attrs,
                    })
            info.pop("garments", None)
            info["regions"] = regions
            info["target_size"] = list(self.target_size)

    def upgrade_region_list(self, info, crop_keys):
        stored_size = tuple(info.get("target_size", self.source_target_size))
        converted, garment_region, sleeve_sides = [], None, set()
        for region in info.get("regions", []):
            attrs = region.get("attributes") or {}
            category = region.get("category")
            if category not in ("garment", "neck", "sleeves"):
                category = next((key for key in crop_keys if key in attrs), "garment")

            if category == "garment":
                if garment_region is None:
                    garment_region = {
                        "category": "garment",
                        "box": region.get("box"),
                        "attributes": {},
                    }
                    converted.append(garment_region)
                garment_region["attributes"].update(attrs)
                for key in ("shirt_colors", "colors"):
                    if region.get(key):
                        garment_region["shirt_colors"] = region[key]
                continue

            new_region = dict(region)
            new_region["category"] = category
            new_region["attributes"] = attrs
            if category == "sleeves":
                side = region.get("side")
                if side not in ("left", "right"):
                    box = region.get("box")
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        side = (
                            "left"
                            if (box[0] + box[2]) / 2 < stored_size[0] / 2
                            else "right"
                        )
                    else:
                        side = next(
                            (candidate for candidate in ("left", "right")
                             if candidate not in sleeve_sides),
                            "right",
                        )
                new_region["side"] = side
                sleeve_sides.add(side)
            converted.append(new_region)

        if stored_size != self.target_size:
            for region in converted:
                region["box"] = self.scale_box(
                    region.get("box"), stored_size, self.target_size
                )
        info.pop("garments", None)
        info["regions"] = converted
        info["target_size"] = list(self.target_size)

    def build_ui(self):
        """Build canvas + right-side category / region / nav controls."""
        w = self.win
        w.title(f"Garment Annotator — Target Res: {self.target_size[0]}x{self.target_size[1]}")
        w.geometry("1380x900")
        self.canvas = tk.Canvas(w, bg="#222", cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        r = tk.Frame(w, width=470, padx=8, pady=6)
        r.pack(side="right", fill="y")
        self.title = tk.Label(
            r, anchor="w", justify="left", wraplength=450, font=("Arial", 10, "bold")
        )
        self.title.pack(fill="x")

        categories = self.schema["categories"]
        self.category_by_key = {cat["key"]: cat for cat in categories}
        self.base_categories = [
            cat for cat in categories if not cat.get("crop_boxes")
        ]
        self.crop_categories = [
            cat for cat in categories if cat.get("crop_boxes")
        ]
        self.category_key_by_title = {"Garment": "garment"}
        self.category_key_by_title.update(
            {cat["title"]: cat["key"] for cat in self.crop_categories}
        )
        self.category_var = tk.StringVar(value="Garment")
        tk.Label(r, text="Category for this region:", anchor="w").pack(
            fill="x", pady=(8, 0)
        )
        self.category_picker = ttk.Combobox(
            r,
            state="readonly",
            textvariable=self.category_var,
            values=["Garment"] + [cat["title"] for cat in self.crop_categories],
        )
        self.category_picker.pack(fill="x")
        self.category_picker.bind("<<ComboboxSelected>>", self.on_category_change)

        self.category_container = tk.Frame(r)
        self.category_container.pack(fill="x")
        garment_frame = tk.LabelFrame(
            self.category_container, text="Garment attributes", padx=4
        )
        self.category_frames["garment"] = garment_frame
        for cat in self.base_categories:
            self.add_category_controls(garment_frame, cat)
        for cat in self.crop_categories:
            frame = tk.LabelFrame(
                self.category_container, text=cat["title"], padx=4
            )
            self.category_frames[cat["key"]] = frame
            self.add_category_controls(frame, cat)

        self.on_category_change()

        row = tk.Frame(r)
        row.pack(fill="x", pady=3)
        tk.Button(row, text="Add region (Enter)", command=self.add_region, bg="#cfe8cf").pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(row, text="Update selected", command=self.update_region).pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(row, text="Delete", command=self.delete_region).pack(
            side="left", expand=True, fill="x"
        )
        self.keep = tk.BooleanVar(value=False)
        tk.Checkbutton(r, text="Keep ticks after adding", variable=self.keep, anchor="w").pack(
            fill="x"
        )

        tk.Label(r, text="Regions in this image:", anchor="w").pack(fill="x")
        self.rl = tk.Listbox(r, height=5, exportselection=False)
        self.rl.pack(fill="x")
        self.rl.bind("<<ListboxSelect>>", self.on_region_select)

        tk.Button(
            r, text="Preview shirt colors (Length crop only)", command=self.analyze_current
        ).pack(fill="x", pady=2)

        self.info = tk.Text(r, height=6, wrap="word", font=("Consolas", 9))
        self.info.pack(fill="both", expand=True)

        row = tk.Frame(r)
        row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text="Go to #").pack(side="left")
        self.goto_entry = tk.Entry(row, width=6)
        self.goto_entry.pack(side="left", padx=2)
        self.goto_entry.bind("<Return>", lambda e: (self.goto_number(), "break")[1])
        tk.Button(row, text="Go", command=self.goto_number).pack(side="left")
        tk.Button(
            row, text="Next unlabelled (Ctrl+N)", command=self.jump_unlabelled, bg="#fff3c4"
        ).pack(side="right", expand=True, fill="x", padx=(6, 0))

        row = tk.Frame(r)
        row.pack(fill="x", pady=4)
        tk.Button(row, text="< Prev", command=lambda: self.goto(self.idx - 1)).pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(row, text="Save", command=self.save).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Next >", command=lambda: self.goto(self.idx + 1)).pack(
            side="left", expand=True, fill="x"
        )
        tk.Button(
            r, text="Delete this picture... (Del)", command=self.delete_picture, fg="#b00020"
        ).pack(fill="x")

        w.bind("<Return>", lambda e: None if self.typing() else self.add_region())
        w.bind("<Delete>", lambda e: None if self.typing() else self.delete_picture())
        w.bind("<Left>", lambda e: None if self.typing() else self.goto(self.idx - 1))
        w.bind("<Right>", lambda e: None if self.typing() else self.goto(self.idx + 1))
        w.bind("<Control-n>", lambda e: self.jump_unlabelled())
        w.bind("<Control-s>", lambda e: self.save())
        w.protocol("WM_DELETE_WINDOW", self.on_close)

    def add_category_controls(self, parent, cat):
        frame = tk.LabelFrame(parent, text=cat["title"], padx=4)
        frame.pack(fill="x", pady=1)
        key, names = cat["key"], [option["name"] for option in cat["options"]]
        if cat["mode"] == "single":
            var = tk.StringVar(value="")
            self.vars[key] = var
            items = [(name, name) for name in names] + [("(skip)", "")]
            for index, (text, value) in enumerate(items):
                tk.Radiobutton(
                    frame, text=text, variable=var, value=value, anchor="w"
                ).grid(row=index // 3, column=index % 3, sticky="w", padx=4)
            return

        variables = {name: tk.BooleanVar() for name in names + [NONE]}
        self.vars[key] = variables
        for index, name in enumerate(names + [NONE]):
            tk.Checkbutton(
                frame,
                text="(none of these)" if name == NONE else name,
                variable=variables[name],
                anchor="w",
            ).grid(row=index // 3, column=index % 3, sticky="w", padx=4)

    def on_category_change(self, _=None):
        title = self.category_var.get()
        key = self.category_key_by_title.get(title)
        if key is None:
            return
        for frame in self.category_frames.values():
            frame.pack_forget()
        self.category_frames[key].pack(fill="x", pady=1)
    def typing(self):
        try:
            return isinstance(self.win.focus_get(), (tk.Entry, tk.Spinbox))
        except KeyError:
            return False

    def labelled(self, rel):
        return bool(self.data["images"].get(rel, {}).get("regions"))

    def first_unlabelled(self, start=0):
        n = len(self.files)
        for k in range(n):
            i = (start + k) % n
            if not self.labelled(self.files[i]):
                return i
        return None

    def begin(self):
        self.idx = self.ask_start()
        self.show_image()

    def ask_start(self):
        n = len(self.files)
        done = [i for i, f in enumerate(self.files) if self.labelled(f)]
        first_un = self.first_unlabelled(0)
        after_last = min(done[-1] + 1, n - 1) if done else 0

        dlg = tk.Toplevel(self.win)
        dlg.title("Where do you want to start?")
        dlg.transient(self.win)
        dlg.resizable(False, False)
        choice = tk.StringVar(
            value="resume" if (done and first_un is not None) else "begin"
        )
        box = tk.Frame(dlg, padx=16, pady=10)
        box.pack()
        tk.Label(
            box,
            text=f"{n} pictures | {len(done)} labelled | {n - len(done)} left\nTarget Size: {self.target_size[0]}x{self.target_size[1]} (W x H)",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")

        def radio(text, val):
            tk.Radiobutton(box, text=text, variable=choice, value=val, anchor="w").pack(
                fill="x"
            )

        if first_un is not None:
            radio(f"First unlabelled (#{first_un + 1})", "resume")
        if done:
            radio(f"After last labelled (#{after_last + 1})", "after")
        radio("Start from beginning", "begin")
        result = {"idx": 0}

        def go(_=None):
            c = choice.get()
            if c == "resume" and first_un is not None:
                i = first_un
            elif c == "after":
                i = after_last
            else:
                i = 0
            result["idx"] = max(0, min(i, n - 1))
            dlg.destroy()

        tk.Button(dlg, text="Start", command=go, bg="#cfe8cf").pack(pady=8)
        dlg.protocol("WM_DELETE_WINDOW", go)
        dlg.grab_set()
        self.win.wait_window(dlg)
        return result["idx"]

    def jump_unlabelled(self):
        j = self.first_unlabelled(self.idx + 1)
        if j is None:
            self.say("All pictures are labelled.")
        else:
            self.goto(j)

    def goto_number(self):
        try:
            i = int(self.goto_entry.get()) - 1
        except ValueError:
            return self.say(f"Type 1..{len(self.files)}")
        self.goto(i)
        self.goto_entry.delete(0, "end")

    def delete_picture(self):
        rel = self.files[self.idx]
        if not messagebox.askyesno("Delete", f"Remove {rel}?"):
            return
        dst = self.trash / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(f"{dst.stem}_{int(time.time())}{dst.suffix}")
        try:
            shutil.move(str(self.folder / rel), str(dst))
        except OSError as ex:
            messagebox.showerror("Error", str(ex))
            return
        self.data["images"].pop(rel, None)
        del self.files[self.idx]
        if not self.files:
            self.save()
            self.win.destroy()
            return
        self.idx = min(self.idx, len(self.files) - 1)
        self.save()
        self.show_image()

    def get_attributes(self):
        region_category = self.category_key_by_title[self.category_var.get()]
        keys = (
            [cat["key"] for cat in self.base_categories]
            if region_category == "garment"
            else [region_category]
        )
        attrs = {}
        for key in keys:
            cat, variable = self.category_by_key[key], self.vars[key]
            if cat["mode"] == "single":
                if variable.get():
                    attrs[key] = variable.get()
                continue
            ticked = [
                name for name, value in variable.items()
                if name != NONE and value.get()
            ]
            if ticked:
                attrs[key] = ticked
            elif variable[NONE].get():
                attrs[key] = []
        return attrs

    def region_category(self, region):
        category = region.get("category")
        if category in self.category_frames:
            return category
        attrs = region.get("attributes", {})
        for cat in self.crop_categories:
            if cat["key"] in attrs:
                return cat["key"]
        return "garment"

    def region_title(self, region):
        category = self.region_category(region)
        if category == "garment":
            return "Garment"
        if category == "sleeves":
            return f"Sleeve {region.get('side', 'left')}"
        return self.category_by_key[category]["title"]

    def set_attributes(self, attrs, region=None):
        selected_crop = next(
            (cat["key"] for cat in self.crop_categories if cat["key"] in attrs),
            None,
        )
        region_category = (
            self.region_category(region)
            if region
            else (
                selected_crop
                or self.category_key_by_title.get(self.category_var.get(), "garment")
            )
        )
        title = (
            self.category_by_key[region_category]["title"]
            if region_category != "garment"
            else "Garment"
        )
        self.category_var.set(title)
        self.on_category_change()

        for cat in self.schema["categories"]:
            key, variable = cat["key"], self.vars[cat["key"]]
            value = attrs.get(key)
            if cat["mode"] == "single":
                variable.set(value if isinstance(value, str) else "")
            else:
                values = value if isinstance(value, list) else []
                for name, checkbox in variable.items():
                    checkbox.set(name in values if name != NONE else value == [])

    def short(self, attrs):
        labels = []
        for key, value in attrs.items():
            if not value:
                continue
            title = self.category_by_key.get(key, {}).get("title", key)
            labels.append(
                f"{title}: {'+'.join(value) if isinstance(value, list) else value}"
            )
        return " | ".join(labels) or "-"

    def entry(self):
        return self.data["images"].setdefault(self.files[self.idx], {"regions": []})

    def sleeve_side_for_box(self, box):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Expected a four-coordinate sleeve box, got {box!r}")
        center_x = (box[0] + box[2]) / 2
        return "left" if center_x < self.img.width / 2 else "right"

    def show_image(self):
        """Loads and resizes image to specified target size first."""
        raw_img = Image.open(self.folder / self.files[self.idx]).convert("RGB")
        self.entry()["width"], self.entry()["height"] = raw_img.size
        # Resize image to target resolution (Width, Height)
        self.img = raw_img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 400)
        self.scale = min(cw / self.img.width, ch / self.img.height)
        self.photo = ImageTk.PhotoImage(
            self.img.resize(
                (int(self.img.width * self.scale), int(self.img.height * self.scale))
            )
        )
        self.cur_box = self.box_id = None
        self.refresh_regions()
        self.redraw()

    def redraw(self, highlight=None):
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=self.photo)
        for i, r in enumerate(self.entry()["regions"]):
            x0, y0, x1, y1 = [v * self.scale for v in r["box"]]
            is_garment = self.region_category(r) == "garment"
            col = (
                ("#ff1744" if i == highlight else "#00e676")
                if is_garment
                else ("#ff9100" if i == highlight else "#00e5ff")
            )
            dash = None if is_garment else (4, 4)
            c.create_rectangle(x0, y0, x1, y1, outline=col, width=2, dash=dash)
            c.create_text(
                x0 + 4, y0 + 4, anchor="nw", fill=col, font=("Arial", 10, "bold"),
                text=f"{self.region_title(r)}: {self.short(r['attributes'])}",
            )
        self.box_id = None

    def refresh_regions(self):
        e = self.entry()
        done = sum(1 for f in self.files if self.labelled(f))
        self.title.config(
            text=(
                f"[{self.idx + 1}/{len(self.files)}] {self.files[self.idx]} "
                f"({self.target_size[0]}x{self.target_size[1]})\n"
                f"({len(e['regions'])} regions) | Labelled: {done}/{len(self.files)}"
            )
        )
        self.rl.delete(0, "end")
        for i, r in enumerate(e["regions"]):
            prefix = f"[{self.region_title(r)}]"
            self.rl.insert("end", f"{i + 1}: {prefix} {self.short(r['attributes'])}")

    def on_press(self, e):
        self.start, self.cur_box = (e.x, e.y), None

    def on_drag(self, e):
        if not self.start:
            return
        if self.box_id:
            self.canvas.delete(self.box_id)
        self.box_id = self.canvas.create_rectangle(
            *self.start, e.x, e.y, outline="yellow", width=2, dash=(4, 2)
        )

    def on_release(self, e):
        if not self.start:
            return
        x0, x1 = sorted((self.start[0], e.x))
        y0, y1 = sorted((self.start[1], e.y))
        self.start = None
        if x1 - x0 < 8 or y1 - y0 < 8:
            self.cur_box = None
            self.redraw()
            return
        W, H, s = self.img.width, self.img.height, self.scale
        cl = lambda v, m: max(0, min(int(v), m))
        self.cur_box = [cl(x0 / s, W), cl(y0 / s, H), cl(x1 / s, W), cl(y1 / s, H)]
        side_note = ""
        if self.category_key_by_title.get(self.category_var.get()) == "sleeves":
            side = self.sleeve_side_for_box(self.cur_box)
            side_note = f"\nDetected sleeve side from image position: {side}."
        self.say(
            f"Region box: {self.cur_box}{side_note}\n"
            "Choose its type and labels, then press Enter."
        )

    def run_analysis(self, box):
        return {"colors": detect_colors(self.img.crop(tuple(box)))}

    def add_region(self):
        attrs = self.get_attributes()
        if not attrs:
            messagebox.showwarning("Nothing picked", "Pick a label first.")
            return
        category = self.category_key_by_title[self.category_var.get()]
        if category != "garment" and self.cur_box is None:
            return self.say("Draw this neck or sleeve box before adding the region.")
        box = self.cur_box or [0, 0, self.img.width, self.img.height]
        side = self.sleeve_side_for_box(box) if category == "sleeves" else None
        existing_regions = self.entry()["regions"]
        for region in existing_regions:
            if self.region_category(region) != category:
                continue
            if category != "sleeves" or region.get("side", "left") == side:
                return self.say(
                    f"A {self.region_title({'category': category, 'side': side})} "
                    "region already exists. Select it to update it."
                )

        shirt_colors = None
        if "length" in attrs:
            try:
                shirt_colors = self.run_analysis(box).get("colors", [])
            except Exception as ex:
                self.say(f"(Color analysis skipped: {ex})")
        region_data = {"category": category, "box": box, "attributes": attrs}
        if side:
            region_data["side"] = side
        if shirt_colors:
            region_data["shirt_colors"] = shirt_colors
        existing_regions.append(region_data)
        self.cur_box = None
        if not self.keep.get():
            self.set_attributes({})
        self.refresh_regions()
        self.redraw()
        self.save()
        if shirt_colors:
            self.show_shirt_colors(shirt_colors)

    def sel_region(self):
        s = self.rl.curselection()
        return s[0] if s else None

    def update_region(self):
        i = self.sel_region()
        if i is None:
            return self.say("Select a region first.")
        attrs = self.get_attributes()
        if not attrs:
            messagebox.showwarning("Nothing picked", "Pick a label first.")
            return
        category = self.category_key_by_title[self.category_var.get()]
        selected_region = self.entry()["regions"][i]
        box = self.cur_box or selected_region["box"]
        side = self.sleeve_side_for_box(box) if category == "sleeves" else None
        for index, candidate in enumerate(self.entry()["regions"]):
            if index == i or self.region_category(candidate) != category:
                continue
            if category != "sleeves" or candidate.get("side", "left") == side:
                return self.say("A region of this type already exists.")

        selected_region.update({"category": category, "attributes": attrs})
        if self.cur_box is not None:
            selected_region["box"] = self.cur_box
            self.cur_box = None
        if side:
            selected_region["side"] = side
        else:
            selected_region.pop("side", None)
        self.refresh_regions()
        self.redraw(i)
        self.save()

    def delete_region(self):
        i = self.sel_region()
        if i is None:
            return self.say("Select a region first.")
        del self.entry()["regions"][i]
        self.refresh_regions()
        self.redraw()
        self.save()

    def on_region_select(self, _=None):
        i = self.sel_region()
        if i is None:
            return
        r = self.entry()["regions"][i]
        self.set_attributes(r["attributes"], region=r)
        self.redraw(i)
        if r.get("shirt_colors"):
            self.show_shirt_colors(r["shirt_colors"])

    def show_shirt_colors(self, colors):
        cols = ", ".join(f"{c['name']} ({c['fraction']:.0%})" for c in colors)
        self.say(f"Shirt Color: {cols}")

    def analyze_current(self):
        if self.category_key_by_title[self.category_var.get()] != "garment":
            return self.say("Color analysis is available for garment regions.")
        box = self.cur_box or [0, 0, self.img.width, self.img.height]
        try:
            self.show_shirt_colors(self.run_analysis(box).get("colors", []))
        except Exception as ex:
            self.say(f"Analysis failed: {ex}")

    def say(self, text):
        self.info.delete("1.0", "end")
        self.info.insert("end", text)

    def goto(self, i):
        self.save()
        self.idx = max(0, min(i, len(self.files) - 1))
        self.show_image()

    def save(self):
        self.data["schema_version"] = SCHEMA_VERSION
        self.data["garment"] = self.garment
        self.data["image_root"] = self.image_root_relative()
        self.data["target_size"] = list(self.target_size)
        self.data["images"] = {
            k: v for k, v in self.data["images"].items() if v["regions"]
        }
        self.ann_path.parent.mkdir(parents=True, exist_ok=True)
        self.ann_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def on_close(self):
        self.save()
        self.win.destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=None)
    ap.add_argument("--garment", choices=("top", "bottom"), default="top")
    ap.add_argument("--ann", default=None, help="annotations JSON path")
    ap.add_argument(
        "--size",
        default="288x384",
        help="Target size format '288x384' (Width x Height)",
    )
    a = ap.parse_args()

    folder = (
        Path(a.folder)
        if a.folder
        else ROOT / "dataset" / a.garment / "images"
    )
    if not folder.exists() and not a.folder:
        folder = ROOT / "dataset" / a.garment
    target_size = parse_target_size(a.size)

    if a.ann:
        ann = Path(a.ann)
    elif (folder / "annotations.json").exists():
        ann = folder / "annotations.json"
    elif folder.name.lower() == "images" and (folder.parent / "annotations.json").exists():
        ann = folder.parent / "annotations.json"
    else:
        ann = ROOT / "dataset" / a.garment / "annotations.json"

    root = tk.Tk()
    Annotator(root, folder, ann, target_size=target_size, garment=a.garment)
    root.mainloop()
