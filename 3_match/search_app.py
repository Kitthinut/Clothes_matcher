"""Closet search GUI (3_match/search_app.py).

Pick garment + backend, choose a reference outfit image, get the ranked
matches from YOUR closet (closet/closet.json). Double-click a result to
open the picture.

Usage:  python 3_match/search_app.py
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match import DEFAULT_W, analyze_reference, search  # noqa: E402  (3_match on path)

CLOSET = ROOT / "closet"


class SearchApp:
    def __init__(self, win):
        self.win = win
        win.title("Outfit Matcher — reference image -> best closet matches")
        win.geometry("860x620")

        top = tk.Frame(win)
        top.pack(fill="x", padx=8, pady=6)
        self.garment = tk.StringVar(value="top")
        self.backend = tk.StringVar(value="pytorch")
        ttk.Combobox(top, textvariable=self.garment, values=("top", "bottom"),
                     width=8, state="readonly").pack(side="left", padx=2)
        ttk.Combobox(top, textvariable=self.backend, values=("pytorch", "tensorflow"),
                     width=11, state="readonly").pack(side="left", padx=2)
        tk.Button(top, text="Pick reference image...", command=self.pick_ref,
                  bg="#cfe8cf").pack(side="left", padx=6)
        tk.Button(top, text="Outfit board (PNG)", command=self.save_board).pack(
            side="left", padx=6)
        tk.Button(top, text="Open closet folder", command=self.open_closet).pack(
            side="left", padx=6)
        self.status = tk.Label(win, anchor="w", justify="left", wraplength=830)
        self.status.pack(fill="x", padx=8)

        cols = ("score", "attr", "color", "emb", "item", "labels")
        heads = ("score", "attr", "color", "emb", "item id", "labels [colors]")
        self.tree = ttk.Treeview(win, columns=cols, show="headings", height=18)
        for c, head, wd in zip(cols, heads, (62, 62, 62, 62, 130, 380)):
            self.tree.heading(c, text=head)
            self.tree.column(c, width=wd, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)
        self.tree.bind("<Double-1>", self.open_item)

        self.ref_path = None
        self.items = []
        self.closet_dir = None

    def say(self, text):
        self.status.config(text=text)

    def pick_ref(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not p:
            return
        self.ref_path = p
        self.say(f"Reference: {p}\nMatching...")
        threading.Thread(target=self.run_match, daemon=True).start()

    def run_match(self):
        garment, backend = self.garment.get(), self.backend.get()
        try:
            ref = analyze_reference(self.ref_path, backend, garment)
            items = search(ref, garment, CLOSET, DEFAULT_W, None, 50)
        except Exception as ex:
            self.win.after(0, lambda: messagebox.showerror("Match failed", str(ex)))
            return
        self.items = items

        def fill():
            self.tree.delete(*self.tree.get_children())
            for it in items:
                labels = ", ".join(f"{k}={v}" for k, v in it["attributes"].items())
                cols = "/".join(c["name"] for c in it["colors"][:2])
                emb = (f"{it['emb_score']:.2f}"
                       if it["emb_score"] is not None else "-")
                self.tree.insert("", "end", values=(
                    f"{it['score']:.2f}", f"{it['attr_score']:.2f}",
                    f"{it['color_score']:.2f}", emb, it["item_id"],
                    f"{labels} [{cols}]"))
            self.say(f"{len(items)} matches — double-click a row to open "
                     f"the picture (same model scores every closet item)")
        self.win.after(0, fill)

    def save_board(self):
        """Render the current results as one outfit_board.png and open it."""
        if not self.items:
            messagebox.showinfo("No matches yet", "Pick a reference image first.")
            return
        try:
            from board import build_board, save_board
            from shared.colors import auto_box
            from PIL import Image
            ref_img = Image.open(self.ref_path).convert("RGB")
            garment, backend = self.garment.get(), self.backend.get()
            from match import analyze_reference
            ref = analyze_reference(self.ref_path, backend, garment)
            board = build_board(ref_img, ref["attributes"], ref["colors"],
                                self.items, CLOSET,
                                title=f"Outfit suggestions — {garment}")
            path = save_board(board, CLOSET)
            os.startfile(path) if sys.platform == "win32" else \
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as ex:
            messagebox.showerror("Board failed", str(ex))

    def open_closet(self):
        CLOSET.mkdir(exist_ok=True)
        if sys.platform == "win32":
            os.startfile(CLOSET)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(CLOSET)])

    def open_item(self, _):
        sel = self.tree.selection()
        if not sel or not self.items:
            return
        item = self.items[self.tree.index(sel[0])]
        path = CLOSET / item["file"]
        if sys.platform == "win32":
            os.startfile(path)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])


if __name__ == "__main__":
    root = tk.Tk()
    SearchApp(root)
    root.mainloop()
