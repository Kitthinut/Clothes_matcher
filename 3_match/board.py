"""Outfit board (3_match/board.py) — the "what should I wear" visual.

Renders one contact sheet: the reference outfit image on the left with its
predicted labels, and the top-N closet matches next to it, each tile showing
its weighted score, attribute labels and dominant colours. Saved as a PNG
(so it can go straight into a report or a chat) and/or opened in a window.

Pure PIL drawing — no extra dependencies.
"""

from pathlib import Path

from PIL import Image, ImageDraw

THUMB_H = 300      # tile height in px
PAD = 10
REF_W = 300        # reference panel width
BG = "#181818"
FG = "#f0f0f0"
ACCENT = "#00e5ff"


def _thumb(img, height):
    scale = height / img.height
    return img.resize((max(1, int(img.width * scale)), height))


def _wrap(text, max_lines, width=46):
    """Split a long comma-joined string into <= max_lines chunks."""
    lines, cur = [], ""
    for part in text.split(", "):
        cand = f"{cur}, {part}" if cur else part
        if len(cand) > width and cur:
            lines.append(cur)
            cur = part
        else:
            cur = cand
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:  # overflow -> ellipsis on the last line
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 3] + "..."
    return lines


def _text_block(draw, x, y, lines, width, color=FG, leading=14):
    for ln in lines:
        # crude wrap: keep under `width` chars is done by callers; clip here
        draw.text((x, y), ln[:46], fill=color)
        y += leading
    return y


def build_board(ref_img, ref_labels, ref_colors, results, closet_dir,
                top_n=9, title="Outfit suggestions"):
    """(PIL reference, labels, colors, match dicts, closet dir) -> board PIL."""
    closet_dir = Path(closet_dir)
    matches = results[:top_n]
    cols = 3
    tile_w = THUMB_H + 2 * PAD  # square-ish tiles
    rows = (len(matches) + cols - 1) // cols
    left_w = REF_W + 2 * PAD
    W = left_w + cols * tile_w + PAD
    head_h = 30
    H = head_h + max(THUMB_H + 90, rows * (THUMB_H + 100)) + PAD

    board = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(board)
    d.text((PAD, 8), title, fill=ACCENT)

    # --- reference panel -------------------------------------------------
    ref = _thumb(ref_img.convert("RGB"), THUMB_H)
    board.paste(ref, (PAD, head_h + PAD))
    y = head_h + PAD + ref.height + 8
    y = _text_block(d, PAD, y, ["REFERENCE:"], left_w - 2 * PAD, ACCENT)
    for k, v in ref_labels.items():
        v = "+".join(v) if isinstance(v, list) else v
        y = _text_block(d, PAD, y, [f"{k}: {v}"], left_w - 2 * PAD)
    cols_txt = ", ".join(c["name"] for c in ref_colors[:3])
    _text_block(d, PAD, y, [f"colors: {cols_txt}"], left_w - 2 * PAD)

    # --- match tiles ------------------------------------------------------
    for i, it in enumerate(matches):
        c, r = i % cols, i // cols
        x0 = left_w + c * tile_w + PAD
        y0 = head_h + PAD + r * (THUMB_H + 100)
        try:
            thumb = _thumb(Image.open(closet_dir / it["file"]).convert("RGB"),
                           THUMB_H)
            board.paste(thumb, (x0, y0))
        except Exception as ex:
            d.text((x0, y0), f"(image failed: {ex})", fill="#ff5555")
        d.rectangle([x0, y0, x0 + THUMB_H - 1, y0 + THUMB_H - 1],
                    outline=ACCENT if i == 0 else "#444444", width=2 if i == 0 else 1)
        ty = y0 + THUMB_H + 4
        lines = [f"#{i + 1}  score {it['score']:.2f}  ({it['item_id']})"]
        attrs = ", ".join(f"{k}={'+'.join(v) if isinstance(v, list) else v}"
                          for k, v in it["attributes"].items())
        lines += _wrap(attrs, 2)  # up to two lines of labels
        cols_txt = ", ".join(c["name"] for c in it["colors"][:2])
        lines.append(f"[{cols_txt}]  attr {it['attr_score']:.2f} "
                     f"col {it['color_score']:.2f}")
        _text_block(d, x0, ty, lines, tile_w - 2 * PAD, FG, leading=13)
    return board


def save_board(board, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "outfit_board.png"
    board.save(path)
    return path


def show_board(board):
    """Open the board in the default image viewer (or the Grad-CAM viewer)."""
    try:
        from shared.visualize import Viewer
        Viewer([("outfit board", board)])
    except Exception:
        import os
        import sys
        path = save_board(board, Path("weights") / "visual_debug")
        if sys.platform == "win32":
            os.startfile(path)
