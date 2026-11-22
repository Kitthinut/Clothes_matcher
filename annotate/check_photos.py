#!/usr/bin/env python3
"""
Check photos BEFORE labeling them.

A photo passes only if:
  * width  > 1000 px and height > 1000 px   (not too small)
  * width  < 3000 px and height < 3000 px   (not too big)
  * height : width is 4 : 3                  (portrait, tolerance 2 %)

If the SIZE is fine but the shape is not 4:3, the tool ASKS whether to make it 4:3
by adding black bars (top/bottom or left/right, centered). The photo itself is not
stretched or cropped. Your original is saved in the folder "originals_before_4x3".
If the black bars would push a side to 3000 px or more, the photo cannot be fixed this way.

Photos that are too small, too big, or unfixable can be MOVED (not erased) to a folder
called "delete" inside your photo folder, together with their label .json.

Usage
    python check_photos.py                    # opens a window, choose your photo folder
    python check_photos.py path/to/photos     # opens the window on that folder
    python check_photos.py path/to/photos --cli          # print a report (asks about 4:3 fixes)
    python check_photos.py path/to/photos --cli --pad    # report + make fixable photos 4:3, no questions
    python check_photos.py path/to/photos --cli --pad --move   # ...and move the remaining bad photos

Needs only Pillow. The window uses tkinter (included with most Python installs).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

# ---- the rules (edit here if the team changes them) ---------------------------
MIN_PX = 999          # BOTH sides must be MORE than this
MAX_PX = 3000          # BOTH sides must be LESS than this
RATIO = 4 / 3          # height / width
RATIO_TOL = 0.02       # 2 % relative tolerance, phone photos are rarely exact
DELETE_DIR = "delete"
ORIG_DIR = "originals_before_4x3"
BIG_BLACK_WARN = 0.25  # warn when more than 25 % of the fixed photo would be black
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def display_size(path: Path):
    """Width and height as the photo is DISPLAYED (phone EXIF rotation applied), read cheaply."""
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112) in (5, 6, 7, 8):  # rotated 90 degrees by EXIF
            w, h = h, w
    return w, h


def size_ok(w, h):
    return MIN_PX < w < MAX_PX and MIN_PX < h < MAX_PX


def shape_ok(w, h):
    return abs(h / w - RATIO) / RATIO <= RATIO_TOL


def problems(w: int, h: int):
    out = []
    if w <= MIN_PX or h <= MIN_PX:
        out.append(f"Too small: {w} x {h} px. Both sides must be more than {MIN_PX} px.")
    if w >= MAX_PX or h >= MAX_PX:
        out.append(f"Too big: {w} x {h} px. Both sides must be less than {MAX_PX} px.")
    if not shape_ok(w, h):
        ratio = h / w
        if abs(w / h - RATIO) / RATIO <= RATIO_TOL:
            out.append("Wrong shape: this is a 4:3 LANDSCAPE photo. We need PORTRAIT (height : width = 4 : 3).")
        else:
            out.append(f"Not 4:3: height : width = {ratio:.2f} : 1 (need {RATIO:.2f} : 1).")
    return out


def pad_plan(w, h):
    """
    If the size rules pass but the shape is not 4:3, return the new (width, height) after adding
    black bars so that height : width is exactly 4 : 3. Otherwise return None.
      too wide / too short -> add bars on top and bottom:  W' = 3*ceil(W/3), H' = 4W'/3
      too tall / too narrow -> add bars left and right:    H' = 4*ceil(H/4), W' = 3H'/4
    """
    if not size_ok(w, h) or shape_ok(w, h):
        return None
    if h / w < RATIO:
        nw = -(-w // 3) * 3
        nh = nw // 3 * 4
    else:
        nh = -(-h // 4) * 4
        nw = nh // 4 * 3
    if nw >= MAX_PX or nh >= MAX_PX:
        return None  # the bars would make the photo too big
    return nw, nh


def scan(folder):
    folder = Path(folder)
    skips = [folder / DELETE_DIR, folder / ORIG_DIR]
    rows = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or any(s in p.parents for s in skips):
            continue
        try:
            w, h = display_size(p)
            plan = pad_plan(w, h)
            fix = {"w": plan[0], "h": plan[1], "black": 1 - (w * h) / (plan[0] * plan[1])} if plan else None
            rows.append({"path": p, "w": w, "h": h, "reasons": problems(w, h), "fix": fix})
        except Exception as e:
            rows.append({"path": p, "w": None, "h": None, "reasons": [f"Cannot open this file ({e})."], "fix": None})
    return rows


def fix_text(fix):
    s = f"Can be fixed: add black bars to make it {fix['w']} x {fix['h']} px (4:3), {fix['black']:.0%} of it black."
    if fix["black"] > BIG_BLACK_WARN:
        s += " That is a lot of black; retaking the photo may be better."
    return s


def unique_dest(dest_dir: Path, name: str):
    dest, i = dest_dir / name, 1
    stem, suffix = Path(name).stem, Path(name).suffix
    while dest.exists():
        dest = dest_dir / f"{stem}_{i}{suffix}"
        i += 1
    return dest


def move_to_delete(path: Path, folder: Path):
    """Move photo (+ its sidecar .json) into <folder>/delete. Never overwrites."""
    dest_dir = Path(folder) / DELETE_DIR
    dest_dir.mkdir(exist_ok=True)
    for src in (path, path.with_suffix(".json")):
        if src.exists():
            shutil.move(str(src), str(unique_dest(dest_dir, src.name)))


def to_rgb_on_black(im):
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (0, 0, 0))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def remap_json(jp: Path, w, h, nw, nh, ox, oy):
    """If a label file already exists, keep its boxes correct after padding (boxes are fractions of the photo)."""
    try:
        meta = json.loads(jp.read_text("utf-8"))
        for o in meta.get("objects", []):
            b = o.get("box")
            if isinstance(b, list) and len(b) == 4:
                xc, yc, bw, bh = b
                o["box"] = [round((xc * w + ox) / nw, 6), round((yc * h + oy) / nh, 6),
                            round(bw * w / nw, 6), round(bh * h / nh, 6)]
        jp.write_text(json.dumps(meta, indent=1), "utf-8")
    except Exception as e:
        print(f"  warning: could not update {jp.name} ({e})")


def make_4x3(path: Path, folder: Path):
    """Add centered black bars so the photo becomes 4:3. Original (and its .json) is kept in ORIG_DIR."""
    path, folder = Path(path), Path(folder)
    with Image.open(path) as im:
        base = to_rgb_on_black(ImageOps.exif_transpose(im))  # bakes in phone rotation, drops EXIF
    w, h = base.size
    plan = pad_plan(w, h)
    if plan is None:
        raise ValueError("this photo cannot be made 4:3 within the size rules")
    nw, nh = plan
    ox, oy = (nw - w) // 2, (nh - h) // 2
    canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
    canvas.paste(base, (ox, oy))

    backup = folder / ORIG_DIR
    backup.mkdir(exist_ok=True)
    shutil.copy2(path, unique_dest(backup, path.name))
    jp = path.with_suffix(".json")
    if jp.exists():
        shutil.copy2(jp, unique_dest(backup, jp.name))

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        canvas.save(path, "JPEG", quality=95, subsampling=0)
    elif ext == ".png":
        canvas.save(path, "PNG")
    else:
        canvas.save(path, "WEBP", quality=95)
    if jp.exists():
        remap_json(jp, w, h, nw, nh, ox, oy)


# ---- command line mode --------------------------------------------------------
def run_cli(folder, move, pad):
    folder = Path(folder)
    rows = scan(folder)
    bad = [r for r in rows if r["reasons"]]
    fixable = [r for r in bad if r["fix"]]
    print(f"{len(rows)} photos checked, {len(rows) - len(bad)} OK, {len(bad)} need attention "
          f"({len(fixable)} can be fixed by making them 4:3)\n")
    for r in bad:
        print(f"- {r['path'].relative_to(folder)}")
        for reason in r["reasons"]:
            print(f"    {reason}")
        if r["fix"]:
            print(f"    -> {fix_text(r['fix'])}")

    if fixable:
        do = pad
        if not do and sys.stdin.isatty():
            ans = input(f"\n{len(fixable)} photo(s) have a good size but are not 4:3. "
                        f"Make them 4:3 by adding black bars? [y/N] ")
            do = ans.strip().lower() == "y"
        if do:
            done = 0
            for r in fixable:
                try:
                    make_4x3(r["path"], folder)
                    done += 1
                except Exception as e:
                    print(f"  could not fix {r['path'].name}: {e}")
            print(f"\nMade {done} photo(s) 4:3. Originals are in {folder / ORIG_DIR}")
        else:
            print("\n(Not changed. Use --pad to make them 4:3.)")

    if move:
        remaining = [r for r in scan(folder) if r["reasons"]]
        for r in remaining:
            move_to_delete(r["path"], folder)
        if remaining:
            print(f"\nMoved {len(remaining)} photo(s) to {folder / DELETE_DIR}")


# ---- window mode --------------------------------------------------------------
def run_gui(folder):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import ImageTk

    st = {"folder": Path(folder) if folder else None, "total": 0, "bad": [], "photo": None}

    def show(row):
        if row is None:
            preview.config(image="", text="Select a photo on the left")
            reasons.set("")
            fixinfo.set("")
            st["photo"] = None
            return
        try:
            with Image.open(row["path"]) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((560, 560))
                st["photo"] = ImageTk.PhotoImage(im)
            preview.config(image=st["photo"], text="")
        except Exception:
            preview.config(image="", text="(cannot preview this file)")
        reasons.set("\n".join("• " + x for x in row["reasons"]))
        fixinfo.set("✔ " + fix_text(row["fix"]) if row["fix"] else "")

    def on_select(_event=None):
        sel = lb.curselection()
        show(st["bad"][sel[0]] if sel else None)

    def do_pad(items):
        errors, done = [], 0
        for r in items:
            try:
                make_4x3(r["path"], st["folder"])
                done += 1
            except Exception as e:
                errors.append(f"{r['path'].name}: {e}")
        rescan(prompt=False)
        if errors:
            messagebox.showwarning("Some photos could not be fixed", "\n".join(errors[:10]))
        elif done:
            messagebox.showinfo("Done", f"Made {done} photo(s) 4:3.\nOriginals are saved in '{ORIG_DIR}'.")

    def pad_items(items):
        items = [r for r in items if r["fix"]]
        if not items:
            messagebox.showinfo("Nothing to fix", "Select photos marked [can fix 4:3] in the list first.")
            return
        if messagebox.askyesno(
                "Make 4:3?",
                f"Make {len(items)} photo(s) 4:3 by adding black bars (top/bottom or left/right)?\n\n"
                f"The photo is not stretched or cropped. Originals are saved in '{ORIG_DIR}'."):
            do_pad(items)

    def move(items):
        if not items:
            messagebox.showinfo("Nothing selected", "Select one or more photos in the list first.")
            return
        target = st["folder"] / DELETE_DIR
        if not messagebox.askyesno(
                "Move to delete folder",
                f"Move {len(items)} photo(s) (and their label .json files, if any) to:\n{target}\n\n"
                "They are not erased, you can drag them back."):
            return
        for r in items:
            move_to_delete(r["path"], st["folder"])
        rescan(prompt=False)

    def rescan(prompt=True):
        if not st["folder"]:
            return
        rows = scan(st["folder"])
        st["total"], st["bad"] = len(rows), [r for r in rows if r["reasons"]]
        lb.delete(0, "end")
        for r in st["bad"]:
            size = f"{r['w']} x {r['h']}" if r["w"] else "?"
            tag = "   [can fix 4:3]" if r["fix"] else ""
            lb.insert("end", f"{r['path'].relative_to(st['folder'])}   [{size}]{tag}")
        fixable = [r for r in st["bad"] if r["fix"]]
        ok = st["total"] - len(st["bad"])
        summary.set(f"{st['total']} photos checked   |   {ok} OK   |   {len(st['bad'])} need attention"
                    f"   |   {len(fixable)} can be fixed by making them 4:3"
                    + ("   ✔ all good!" if st["total"] and not st["bad"] else ""))
        show(None)
        if prompt and fixable and messagebox.askyesno(
                "Make them 4:3?",
                f"{len(fixable)} photo(s) have a good size but are not 4:3.\n\n"
                "Make them 4:3 by adding black bars (top/bottom or left/right)?\n"
                f"Nothing is stretched or cropped. Originals are saved in '{ORIG_DIR}'."):
            do_pad(fixable)

    def choose():
        d = filedialog.askdirectory(title="Choose the folder with your photos")
        if d:
            st["folder"] = Path(d)
            folder_var.set(d)
            rescan()

    root = tk.Tk()
    root.title("Photo checker: size and 4:3 ratio")
    root.geometry("1120x740")

    bar = ttk.Frame(root, padding=8)
    bar.pack(fill="x")
    folder_var = tk.StringVar(value=str(st["folder"] or "(no folder chosen)"))
    ttk.Button(bar, text="Choose folder…", command=choose).pack(side="left")
    ttk.Button(bar, text="Re-scan", command=rescan).pack(side="left", padx=6)
    ttk.Label(bar, textvariable=folder_var).pack(side="left", padx=10)

    rules = f"Rules: width and height > {MIN_PX}px and < {MAX_PX}px, height : width = 4 : 3 (portrait)"
    ttk.Label(root, text=rules, padding=(8, 0)).pack(anchor="w")
    summary = tk.StringVar()
    ttk.Label(root, textvariable=summary, padding=8, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

    body = ttk.Frame(root, padding=8)
    body.pack(fill="both", expand=True)
    left = ttk.Frame(body)
    left.pack(side="left", fill="y")
    lb = tk.Listbox(left, selectmode="extended", width=58, activestyle="none")
    sb = ttk.Scrollbar(left, command=lb.yview)
    lb.config(yscrollcommand=sb.set)
    lb.pack(side="left", fill="y", expand=True)
    sb.pack(side="left", fill="y")
    lb.bind("<<ListboxSelect>>", on_select)

    right = ttk.Frame(body, padding=(12, 0))
    right.pack(side="left", fill="both", expand=True)
    preview = ttk.Label(right, anchor="center", text="Select a photo on the left")
    preview.pack(fill="both", expand=True)
    reasons = tk.StringVar()
    fixinfo = tk.StringVar()
    ttk.Label(right, textvariable=reasons, wraplength=560, foreground="#b00020", justify="left").pack(anchor="w", pady=(6, 0))
    ttk.Label(right, textvariable=fixinfo, wraplength=560, foreground="#1b6e2d", justify="left").pack(anchor="w", pady=4)

    row1 = ttk.Frame(root, padding=(8, 4))
    row1.pack(fill="x")
    ttk.Button(row1, text="Make SELECTED 4:3 (add black bars)",
               command=lambda: pad_items([st["bad"][i] for i in lb.curselection()])).pack(side="left")
    ttk.Button(row1, text="Make ALL fixable 4:3 (add black bars)",
               command=lambda: pad_items([r for r in st["bad"] if r["fix"]])).pack(side="left", padx=8)
    row2 = ttk.Frame(root, padding=(8, 4, 8, 10))
    row2.pack(fill="x")
    ttk.Button(row2, text=f"Move SELECTED to '{DELETE_DIR}' folder",
               command=lambda: move([st["bad"][i] for i in lb.curselection()])).pack(side="left")
    ttk.Button(row2, text=f"Move ALL flagged to '{DELETE_DIR}' folder",
               command=lambda: move(list(st["bad"]))).pack(side="left", padx=8)

    if st["folder"]:
        root.after(200, rescan)
    else:
        root.after(200, choose)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Check photo size and 4:3 ratio before labeling.")
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--cli", action="store_true", help="print a report instead of opening a window")
    ap.add_argument("--pad", action="store_true", help="with --cli: make fixable photos 4:3 (black bars) without asking")
    ap.add_argument("--move", action="store_true", help="with --cli: move the remaining bad photos to the delete folder")
    args = ap.parse_args()

    if args.cli or args.move or args.pad:
        if not args.folder:
            sys.exit("Give a folder: python check_photos.py <folder> --cli")
        return run_cli(args.folder, args.move, args.pad)
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.exit("tkinter is not installed. Use:  python check_photos.py <folder> --cli")
    run_gui(args.folder)


if __name__ == "__main__":
    main()