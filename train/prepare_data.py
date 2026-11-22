#!/usr/bin/env python3
"""
Prepare all labeled data before training.

What it does
------------
1. Walks through every folder under --data (your photos + friends' photos).
2. Cleans old / removed labels so the model never sees them:
     - Drops bottom "waist" boxes
     - Clears invalid values (bootcut, bottom cropped, …)
3. Auto-fills absence classes:
     - If a top exists but no "button" region  → button = none
     - If a top/bottom exists but no "pattern" → pattern = plain
4. Optionally writes the cleaned data back into the JSON files (--write).
5. Shows a statistics report (terminal + a dashboard window):
     - How many of each attribute value (crew, plain, none, skinny, ...)
     - How many images / instances
     - Warnings about classes that are too rare (risk of underfitting)

Usage
-----
    # Just see the report (does not change any files)
    python prepare_data.py --data ../data

    # Apply cleaning + auto-fill and save back into the JSONs
    python prepare_data.py --data ../data --write

    # Then build the dataset as usual
    python build_dataset.py neck_crew --data ../data
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT.parent / "annotate" / "labeling_config" / "schema.yaml"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Regions / values that used to exist but must never reach training
REMOVED_BOTTOM_REGIONS = {"waist"}
REMOVED_VALUES = {
    ("bottom", "length"): {"cropped"},
    ("bottom", "fit"): {"bootcut"},
    ("bottom", "waist"): {"low_waist", "high_waist"},
}


def load_schema():
    return yaml.safe_load(SCHEMA_PATH.read_text("utf-8")) or {}


def find_image(jp: Path):
    for e in IMG_EXT:
        for cand in (jp.with_suffix(e), jp.with_suffix(e.upper())):
            if cand.exists():
                return cand
    return None


def clean_objects(objects: list, schema: dict) -> tuple[list, list]:
    """
    1. Drop removed regions (e.g. bottom.waist).
    2. Drop / fix invalid attribute values (bootcut, cropped on bottoms…).
    3. Auto-fill absence classes:
         - top exists + no button region  → button = none
         - top/bottom exists + no pattern → pattern = plain
    Returns (new_objects, list_of_change_strings).
    """
    changes = []
    allowed_attrs = schema.get("attributes", {})
    allowed_regions = {
        "top": {"length", "button", "neck", "pattern", "sleeve_left", "sleeve_right"},
        "bottom": {"length", "pattern", "fit"},
    }

    cleaned = []
    for o in objects:
        cls = o.get("cls")
        region = o.get("region")
        attrs = dict(o.get("attrs") or {})

        # Drop entire region boxes that no longer exist in the schema
        if cls == "bottom" and region in REMOVED_BOTTOM_REGIONS:
            changes.append(f"removed bottom.{region} box")
            continue
        if cls in allowed_regions and region and region not in allowed_regions[cls]:
            changes.append(f"removed unknown {cls}.{region} box")
            continue

        # Clean attribute values
        new_attrs = {}
        for attr, value in attrs.items():
            if not value:
                continue
            allowed = set(allowed_attrs.get(cls, {}).get(attr, []))
            removed = REMOVED_VALUES.get((cls, attr), set())
            if value in removed or (allowed and value not in allowed):
                changes.append(f"cleared invalid {cls}.{attr}={value}")
                continue
            new_attrs[attr] = value

        cleaned.append({
            "cls": cls,
            **({"region": region} if region else {}),
            "box": o.get("box"),
            "attrs": new_attrs,
        })

    # --- auto-fill absence classes on the cleaned list ---
    has_top = any(o.get("cls") == "top" for o in cleaned)
    has_bottom = any(o.get("cls") == "bottom" for o in cleaned)
    top_regions = {o.get("region") for o in cleaned if o.get("cls") == "top"}
    bottom_regions = {o.get("region") for o in cleaned if o.get("cls") == "bottom"}

    def _ref_box(cls, prefer_region="length"):
        ref = next((o for o in cleaned if o.get("cls") == cls and o.get("region") == prefer_region), None)
        if ref is None:
            ref = next((o for o in cleaned if o.get("cls") == cls), None)
        return list(ref["box"]) if ref and ref.get("box") else None

    # button = none
    if has_top and "button" not in top_regions:
        box = _ref_box("top")
        if box:
            cx, cy, w, h = box
            small = [cx, cy - h * 0.15, w * 0.35, h * 0.25]
            cleaned.append({
                "cls": "top",
                "region": "button",
                "box": [round(x, 6) for x in small],
                "attrs": {"button": "none"},
            })
            changes.append("added top.button = none")

    # pattern = plain (tops)
    if has_top and "pattern" not in top_regions:
        box = _ref_box("top")
        if box:
            cleaned.append({
                "cls": "top",
                "region": "pattern",
                "box": [round(x, 6) for x in box],
                "attrs": {"pattern": "plain"},
            })
            changes.append("added top.pattern = plain")

    # pattern = plain (bottoms)
    if has_bottom and "pattern" not in bottom_regions:
        box = _ref_box("bottom")
        if box:
            cleaned.append({
                "cls": "bottom",
                "region": "pattern",
                "box": [round(x, 6) for x in box],
                "attrs": {"pattern": "plain"},
            })
            changes.append("added bottom.pattern = plain")

    return cleaned, changes


def collect_stats(objects: list, stats: dict):
    """Update counters from one image's objects."""
    for o in objects:
        cls = o.get("cls")
        region = o.get("region")
        attrs = o.get("attrs") or {}
        if not cls:
            continue
        stats["images_with"][cls] += 1
        for attr, value in attrs.items():
            if value:
                stats["attr"][cls][attr][value] += 1
        if region:
            stats["regions"][cls][region] += 1


def main():
    ap = argparse.ArgumentParser(description="Clean labels, auto-fill none/plain, report dataset statistics")
    ap.add_argument("--data", default=str(ROOT.parent / "data"),
                    help="Root folder that contains all labeled photos (yours + friends)")
    ap.add_argument("--write", action="store_true",
                    help="Actually write the cleaned values back into the JSON files")
    ap.add_argument("--min-count", type=int, default=30,
                    help="Warn if any attribute value has fewer than this many instances")
    ap.add_argument("--cli", action="store_true",
                    help="Print report to terminal only (no popup window)")
    args = ap.parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        sys.exit(f"Data folder not found: {data_root}")

    schema = load_schema()
    stats = {
        "images": 0,
        "images_with": Counter(),
        "attr": defaultdict(lambda: defaultdict(Counter)),
        "regions": defaultdict(Counter),
        "auto_filled": 0,
        "files_changed": 0,
    }
    changes_log = []

    jsons = sorted(data_root.rglob("*.json"))
    jsons = [j for j in jsons if j.name not in ("contributor.json", "closet.json")]

    for jp in jsons:
        if find_image(jp) is None:
            continue
        try:
            meta = json.loads(jp.read_text("utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict) or "objects" not in meta:
            continue

        stats["images"] += 1
        objects = meta["objects"]
        new_objects, changes = clean_objects(objects, schema)

        if changes:
            stats["auto_filled"] += len(changes)
            stats["files_changed"] += 1
            changes_log.append((jp, changes))
            if args.write:
                meta["objects"] = new_objects
                jp.write_text(json.dumps(meta, indent=1), "utf-8")

        # Always report stats *after* cleaning + auto-fill
        collect_stats(new_objects, stats)

    # ---------- build report text (terminal + "Full text" tab) ----------
    lines = []

    def add(s=""):
        lines.append(s)

    add("=" * 64)
    add("  DATASET PREPARATION REPORT")
    add("=" * 64)
    add(f"  Images scanned              : {stats['images']}")
    add(f"  Files that needed changes   : {stats['files_changed']}")
    add(f"  Clean / auto-fill actions   : {stats['auto_filled']}")
    if args.write:
        add("  Mode                        : WRITE (JSON files were updated)")
    else:
        add("  Mode                        : DRY-RUN (no files changed)")
        add("                                → run again with --write to apply")
    add()

    all_attrs = schema.get("attributes", {})
    warnings = []
    attr_groups = []

    add("─" * 64)
    add("  ATTRIBUTE COUNTS (after cleaning + auto-fill)")
    add("─" * 64)

    for cls in ("top", "bottom"):
        if cls not in all_attrs:
            continue
        add(f"\n  [{cls.upper()}]")
        for attr, allowed in all_attrs[cls].items():
            counts = stats["attr"][cls][attr]
            total = sum(counts.values())
            add(f"    {attr:<10}  total={total}")
            values = []
            for val in allowed:
                n = counts.get(val, 0)
                bar = "█" * min(40, max(0, n // 3)) if n else ""
                flag = ""
                if n < args.min_count:
                    flag = f"  ← low (<{args.min_count})"
                    warnings.append(f"{cls}.{attr}={val} has only {n} instances")
                add(f"      {val:<14} {n:>5}  {bar}{flag}")
                values.append((val, n))
            attr_groups.append({"cls": cls, "attr": attr, "total": total, "values": values})

    add("\n" + "─" * 64)
    add("  REGION COVERAGE (how many boxes of each type)")
    add("─" * 64)
    region_groups = {}
    for cls in ("top", "bottom"):
        add(f"\n  [{cls.upper()}]")
        regions = stats["regions"][cls]
        if not regions:
            add("    (none)")
        for region, n in sorted(regions.items()):
            add(f"    {region:<14} {n:>5}")
        region_groups[cls] = sorted(regions.items())

    add("\n" + "─" * 64)
    if warnings:
        add("  WARNINGS (risk of underfitting these classes)")
        add("─" * 64)
        for w in warnings:
            add(f"  ! {w}")
        add("")
        add("  Suggestion: label more examples of the rare classes before training.")
    else:
        add(f"  No major imbalance warnings (all classes ≥ {args.min_count} instances).")

    if changes_log:
        add("\n" + "─" * 64)
        add("  Examples of clean / auto-fill actions")
        add("─" * 64)
        for jp, ch in changes_log[:12]:
            try:
                rel = jp.relative_to(data_root)
            except ValueError:
                rel = jp
            add(f"  {rel}")
            for c in ch:
                add(f"      → {c}")
        if len(changes_log) > 12:
            add(f"  ... and {len(changes_log) - 12} more files")

    add("\n" + "=" * 64)
    if not args.write and stats["files_changed"]:
        add("  Next step: run again with --write to apply the changes,")
        add("  then run:  python build_dataset.py neck_crew --data ../data")
    elif args.write:
        add("  Cleaning applied. You can now run:")
        add("      python build_dataset.py neck_crew --data ../data")
    add("=" * 64)

    report_text = "\n".join(lines)

    # Always print to terminal as backup
    print(report_text)

    # Structured data for the dashboard
    changes_rows = []
    for jp, ch in changes_log:
        try:
            rel = str(jp.relative_to(data_root))
        except ValueError:
            rel = str(jp)
        changes_rows.append((rel, ", ".join(ch)))

    ui_data = {
        "images": stats["images"],
        "files_changed": stats["files_changed"],
        "auto_filled": stats["auto_filled"],
        "wrote": args.write,
        "min_count": args.min_count,
        "attr_groups": attr_groups,
        "region_groups": region_groups,
        "warnings": warnings,
        "changes": changes_rows,
        "data_root": str(data_root),
    }

    if not args.cli:
        show_popup(ui_data, report_text)


# ════════════════════════════════════════════════════════════════════
#  Dashboard UI
# ════════════════════════════════════════════════════════════════════

C = {
    "bg": "#eef1f5",
    "card": "#ffffff",
    "line": "#d9dfe8",
    "ink": "#1c2433",
    "muted": "#667085",
    "accent": "#2f5d8a",
    "track": "#e6ebf2",
    "low": "#c8452f",
    "low_bg": "#fbe9e5",
    "ok_bg": "#e3f3ea",
    "ok_fg": "#1f6b3f",
    "warn_bg": "#fdf0d5",
    "warn_fg": "#8a5a00",
}


def _fonts(root):
    import tkinter.font as tkfont
    families = set(tkfont.families(root))
    for name in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Noto Sans", "DejaVu Sans"):
        if name in families:
            base = name
            break
    else:
        base = "TkDefaultFont"
    mono = next((m for m in ("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono") if m in families),
                "TkFixedFont")
    return {
        "base": (base, 10),
        "small": (base, 9),
        "bold": (base, 10, "bold"),
        "title": (base, 18, "bold"),
        "big": (base, 24, "bold"),
        "section": (base, 12, "bold"),
        "mono": (mono, 10),
    }


def show_popup(data: dict, report_text: str):
    """Dashboard: summary tiles, bar charts per attribute, regions, changes, raw text."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
    except ImportError:
        print("\n(tkinter not available – report shown in terminal only)")
        return

    root = tk.Tk()
    F = _fonts(root)
    root.title("raja_fashion – Dataset report")
    root.geometry("940x780")
    root.minsize(700, 520)
    root.configure(bg=C["bg"])

    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    st.configure(".", font=F["base"], background=C["bg"], foreground=C["ink"])
    st.configure("TFrame", background=C["bg"])
    st.configure("Card.TFrame", background=C["card"])
    st.configure("TLabel", background=C["bg"], foreground=C["ink"])
    st.configure("TNotebook", background=C["bg"], borderwidth=0, tabmargins=(0, 0, 0, 0))
    st.configure("TNotebook.Tab", padding=(16, 8), font=F["base"], background=C["bg"],
                 foreground=C["muted"], borderwidth=0)
    st.map("TNotebook.Tab",
           background=[("selected", C["card"])],
           foreground=[("selected", C["ink"])])
    st.configure("TButton", padding=(14, 7), background=C["card"], foreground=C["ink"],
                 bordercolor=C["line"], focuscolor=C["accent"])
    st.map("TButton", background=[("active", C["track"])])
    st.configure("Accent.TButton", background=C["accent"], foreground="#ffffff", bordercolor=C["accent"])
    st.map("Accent.TButton", background=[("active", "#264d74")])
    st.configure("TCheckbutton", background=C["bg"], foreground=C["ink"])
    st.configure("Treeview", background=C["card"], fieldbackground=C["card"], foreground=C["ink"],
                 rowheight=28, borderwidth=0, font=F["base"])
    st.configure("Treeview.Heading", background=C["bg"], foreground=C["muted"], font=F["bold"],
                 borderwidth=0, padding=(8, 6))
    st.map("Treeview", background=[("selected", "#dbe6f2")], foreground=[("selected", C["ink"])])
    st.configure("Vertical.TScrollbar", background=C["track"], troughcolor=C["bg"], borderwidth=0,
                 arrowsize=12)

    # ================= header =================
    header = ttk.Frame(root, padding=(24, 20, 24, 8))
    header.pack(fill="x")

    left = ttk.Frame(header)
    left.pack(side="left", fill="x", expand=True)
    ttk.Label(left, text="Dataset report", font=F["title"]).pack(anchor="w")
    ttk.Label(left, text=data["data_root"], font=F["small"], foreground=C["muted"]).pack(anchor="w", pady=(2, 0))

    if data["wrote"]:
        badge_text, badge_bg, badge_fg = "Changes applied", C["ok_bg"], C["ok_fg"]
        hint = "JSON files were updated."
    else:
        badge_text, badge_bg, badge_fg = "Preview only", C["warn_bg"], C["warn_fg"]
        hint = "No files changed. Run with --write to apply."
    right = ttk.Frame(header)
    right.pack(side="right")
    tk.Label(right, text=badge_text, bg=badge_bg, fg=badge_fg, font=F["bold"], padx=12, pady=4).pack(anchor="e")
    ttk.Label(right, text=hint, font=F["small"], foreground=C["muted"]).pack(anchor="e", pady=(4, 0))

    # ================= summary tiles =================
    tiles = ttk.Frame(root, padding=(24, 8, 24, 8))
    tiles.pack(fill="x")
    n_low = len(data["warnings"])
    tile_defs = [
        ("Images scanned", data["images"], None),
        ("Files needing changes", data["files_changed"], None),
        ("Clean / auto-fill actions", data["auto_filled"], None),
        (f"Classes under {data['min_count']}", n_low, C["low"] if n_low else C["ok_fg"]),
    ]
    for i, (label, value, color) in enumerate(tile_defs):
        tiles.columnconfigure(i, weight=1, uniform="tile")
        card = tk.Frame(tiles, bg=C["card"], highlightbackground=C["line"], highlightthickness=1)
        card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
        tk.Label(card, text=f"{value:,}", bg=C["card"], fg=color or C["ink"], font=F["big"],
                 anchor="w").pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(card, text=label, bg=C["card"], fg=C["muted"], font=F["small"],
                 anchor="w").pack(fill="x", padx=16, pady=(0, 12))

    # ================= tabs =================
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=24, pady=(8, 0))

    def card_tab(title):
        outer = ttk.Frame(nb, style="Card.TFrame")
        nb.add(outer, text=title)
        return outer

    class Scrollable(tk.Frame):
        def __init__(self, parent):
            super().__init__(parent, bg=C["card"])
            self.canvas = tk.Canvas(self, bg=C["card"], highlightthickness=0, bd=0)
            sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.inner = tk.Frame(self.canvas, bg=C["card"])
            self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
            self.canvas.configure(yscrollcommand=sb.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            self.inner.bind("<Configure>",
                            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
            self.canvas.bind("<Enter>", self._bind_wheel)
            self.canvas.bind("<Leave>", self._unbind_wheel)

        def _bind_wheel(self, _):
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

        def _unbind_wheel(self, _):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.canvas.unbind_all(ev)

        def _on_wheel(self, e):
            step = -1 * (e.delta // 120) if sys.platform == "win32" else -1 * e.delta
            self.canvas.yview_scroll(step * 2, "units")

        def clear(self):
            for w in self.inner.winfo_children():
                w.destroy()

    class Bar(tk.Canvas):
        def __init__(self, parent, value, scale, low, threshold):
            super().__init__(parent, height=16, bg=C["card"], highlightthickness=0, bd=0)
            self.value, self.scale, self.low, self.threshold = value, scale, low, threshold
            self.bind("<Configure>", lambda e: self.draw())

        def draw(self):
            self.delete("all")
            w = max(self.winfo_width(), 10)
            self.create_rectangle(0, 4, w, 12, fill=C["track"], width=0)
            if self.value:
                bw = max(3, int(w * min(self.value / self.scale, 1.0)))
                self.create_rectangle(0, 4, bw, 12, fill=C["low"] if self.low else C["accent"], width=0)
            if self.threshold:
                tx = int(w * min(self.threshold / self.scale, 1.0))
                self.create_line(tx, 1, tx, 15, fill=C["muted"], dash=(2, 2))

    # ---------- Tab 1: attributes ----------
    t_attr = card_tab("Attributes")
    toolbar = ttk.Frame(t_attr, style="Card.TFrame", padding=(16, 12, 16, 4))
    toolbar.pack(fill="x")
    only_low = tk.BooleanVar(value=False)
    ttk.Label(toolbar, text=f"Dashed line marks the minimum of {data['min_count']}.",
              style="TLabel", background=C["card"], foreground=C["muted"], font=F["small"]).pack(side="left")
    st.configure("Card.TCheckbutton", background=C["card"])
    chk = ttk.Checkbutton(toolbar, text="Show only low-count classes", variable=only_low,
                          command=lambda: render_attrs(), style="Card.TCheckbutton")
    chk.pack(side="right")

    attr_scroll = Scrollable(t_attr)
    attr_scroll.pack(fill="both", expand=True)

    def render_attrs():
        attr_scroll.clear()
        parent = attr_scroll.inner
        shown_any = False
        last_cls = None
        for g in data["attr_groups"]:
            rows = g["values"]
            if only_low.get():
                rows = [(v, n) for v, n in rows if n < data["min_count"]]
                if not rows:
                    continue
            shown_any = True
            if g["cls"] != last_cls:
                last_cls = g["cls"]
                tk.Label(parent, text=g["cls"].capitalize(), bg=C["card"], fg=C["ink"],
                         font=F["section"], anchor="w").pack(fill="x", padx=16, pady=(14, 2))
            head = tk.Frame(parent, bg=C["card"])
            head.pack(fill="x", padx=16, pady=(8, 2))
            tk.Label(head, text=g["attr"], bg=C["card"], fg=C["ink"], font=F["bold"]).pack(side="left")
            tk.Label(head, text=f"{g['total']:,} labeled", bg=C["card"], fg=C["muted"],
                     font=F["small"]).pack(side="left", padx=8)

            scale = max([n for _, n in g["values"]] + [data["min_count"]]) * 1.08 or 1
            for val, n in rows:
                low = n < data["min_count"]
                row = tk.Frame(parent, bg=C["low_bg"] if low else C["card"])
                row.pack(fill="x", padx=16, pady=1)
                row.columnconfigure(1, weight=1)
                tk.Label(row, text=val, bg=row["bg"], fg=C["ink"], font=F["base"], width=16,
                         anchor="w").grid(row=0, column=0, padx=(8, 4), pady=3)
                bar = Bar(row, n, scale, low, data["min_count"])
                bar.configure(bg=row["bg"])
                bar.grid(row=0, column=1, sticky="ew", padx=4)
                tk.Label(row, text=f"{n:,}", bg=row["bg"], fg=C["low"] if low else C["ink"],
                         font=F["bold"] if low else F["base"], width=7, anchor="e").grid(row=0, column=2, padx=(4, 8))
                if low:
                    tk.Label(row, text="needs more", bg=row["bg"], fg=C["low"], font=F["small"],
                             width=10, anchor="w").grid(row=0, column=3, padx=(0, 8))
                else:
                    tk.Label(row, text="", bg=row["bg"], width=10).grid(row=0, column=3, padx=(0, 8))
        if not shown_any:
            tk.Label(parent, text="Every class has enough examples. Nothing to fix here.",
                     bg=C["card"], fg=C["muted"], font=F["base"]).pack(pady=40)
        tk.Frame(parent, bg=C["card"], height=12).pack()

    render_attrs()

    # ---------- Tab 2: regions ----------
    t_reg = card_tab("Regions")
    reg_scroll = Scrollable(t_reg)
    reg_scroll.pack(fill="both", expand=True)
    for cls in ("top", "bottom"):
        rows = data["region_groups"].get(cls, [])
        tk.Label(reg_scroll.inner, text=cls.capitalize(), bg=C["card"], fg=C["ink"], font=F["section"],
                 anchor="w").pack(fill="x", padx=16, pady=(14, 4))
        if not rows:
            tk.Label(reg_scroll.inner, text="No boxes labeled yet.", bg=C["card"], fg=C["muted"],
                     font=F["base"], anchor="w").pack(fill="x", padx=24, pady=2)
            continue
        scale = max(n for _, n in rows) * 1.08 or 1
        for region, n in rows:
            row = tk.Frame(reg_scroll.inner, bg=C["card"])
            row.pack(fill="x", padx=16, pady=1)
            row.columnconfigure(1, weight=1)
            tk.Label(row, text=region, bg=C["card"], font=F["base"], width=16, anchor="w").grid(
                row=0, column=0, padx=(8, 4), pady=3)
            Bar(row, n, scale, False, 0).grid(row=0, column=1, sticky="ew", padx=4)
            tk.Label(row, text=f"{n:,}", bg=C["card"], font=F["base"], width=7, anchor="e").grid(
                row=0, column=2, padx=(4, 8))

    # ---------- Tab 3: changes ----------
    t_chg = card_tab(f"Changes ({len(data['changes'])})")
    if data["changes"]:
        tv_frame = ttk.Frame(t_chg, style="Card.TFrame")
        tv_frame.pack(fill="both", expand=True, padx=8, pady=8)
        tv = ttk.Treeview(tv_frame, columns=("file", "change"), show="headings", selectmode="browse")
        tv.heading("file", text="File", anchor="w")
        tv.heading("change", text="What changed", anchor="w")
        tv.column("file", width=420, anchor="w")
        tv.column("change", width=360, anchor="w")
        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for rel, ch in data["changes"]:
            tv.insert("", "end", values=(rel, ch))
    else:
        tk.Label(t_chg, text="Nothing needed cleaning or auto-filling.",
                 bg=C["card"], fg=C["muted"], font=F["base"], justify="center").pack(pady=48)

    # ---------- Tab 4: full text ----------
    t_txt = card_tab("Full text")
    txt_frame = ttk.Frame(t_txt, style="Card.TFrame")
    txt_frame.pack(fill="both", expand=True, padx=8, pady=8)
    text = tk.Text(txt_frame, wrap="none", font=F["mono"], bg=C["card"], fg=C["ink"], relief="flat",
                   padx=8, pady=8, highlightthickness=0)
    ysb = ttk.Scrollbar(txt_frame, orient="vertical", command=text.yview)
    xsb = ttk.Scrollbar(txt_frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
    text.grid(row=0, column=0, sticky="nsew")
    ysb.grid(row=0, column=1, sticky="ns")
    xsb.grid(row=1, column=0, sticky="ew")
    txt_frame.rowconfigure(0, weight=1)
    txt_frame.columnconfigure(0, weight=1)
    text.insert("1.0", report_text)
    text.tag_config("warn", foreground=C["low"])
    start = "1.0"
    while True:
        pos = text.search("← low", start, stopindex="end")
        if not pos:
            break
        line_end = text.index(f"{pos} lineend")
        text.tag_add("warn", pos, line_end)
        start = line_end
    text.configure(state="disabled")

    # ================= footer =================
    footer = ttk.Frame(root, padding=(24, 12, 24, 16))
    footer.pack(fill="x")

    if data["warnings"]:
        msg = f"{len(data['warnings'])} classes have too few examples. Label more of them before training."
        msg_fg = C["low"]
    else:
        msg = "All classes have enough examples. Ready to build the dataset."
        msg_fg = C["ok_fg"]
    ttk.Label(footer, text=msg, foreground=msg_fg, font=F["bold"]).pack(side="left")

    ttk.Button(footer, text="Close", style="Accent.TButton", command=root.destroy).pack(side="right")

    def copy_all():
        root.clipboard_clear()
        root.clipboard_append(report_text)
        copy_btn.configure(text="Copied")
        root.after(1500, lambda: copy_btn.configure(text="Copy report"))

    def save_txt():
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="dataset_report.txt",
                                            filetypes=[("Text file", "*.txt")])
        if path:
            try:
                Path(path).write_text(report_text, "utf-8")
            except OSError as e:
                messagebox.showerror("Could not save", str(e))

    ttk.Button(footer, text="Save as text", command=save_txt).pack(side="right", padx=(0, 8))
    copy_btn = ttk.Button(footer, text="Copy report", command=copy_all)
    copy_btn.pack(side="right", padx=8)

    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()
