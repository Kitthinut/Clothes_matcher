"""Image canvas widget: zoom/pan, box drawing, selection, move and resize.

All boxes are stored in **original-image pixel coordinates**. The widget keeps
the displayed scale/offset internally and converts pointer coordinates through
:meth:`to_image` / :meth:`to_canvas`, so drawing stays correct at any zoom or
letterboxing.

Hit-testing uses geometry (not Tk's ``current`` tag), which avoids the old bug
where an overlapping box or its label swallowed the drag and the app silently
moved an existing box instead of drawing the active attribute's box.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any, Callable

from PIL import Image, ImageTk

from annotator.store import Box

HANDLE_HALF = 5       # screen pixels
CLICK_SLOP = 5        # screen pixels: below this a drag is treated as a click
MIN_BOX_PX = 5        # screen pixels: ignore accidental tiny boxes


class AnnotationCanvas(tk.Canvas):
    """Tk canvas specialised for attribute-region box annotation."""

    def __init__(
        self,
        master: Any,
        *,
        on_select: Callable[[str | None], None] | None = None,
        on_create: Callable[[tuple[int, int, int, int]], None] | None = None,
        on_change: Callable[[Box], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master, bg="#202027", cursor="crosshair", highlightthickness=0)
        self.on_select = on_select
        self.on_create = on_create
        self.on_change = on_change
        self.on_status = on_status

        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.boxes: list[Box] = []
        self.selected_id: str | None = None
        self.color_for: Callable[[Box], str] = lambda _b: "#22c55e"
        self.title_for: Callable[[Box], str] = lambda b: f"{b.region}: {b.label}"
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0
        self._drag: dict[str, Any] | None = None
        self._space = False

        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._press_draw)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<ButtonPress-2>", self._press_pan)
        self.bind("<B2-Motion>", self._motion_pan)
        self.bind("<ButtonRelease-2>", self._release_pan)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e, 1.1))
        self.bind("<Button-5>", lambda e: self._zoom_at(e, 1 / 1.1))
        self.bind("<space>", lambda _e: self._set_space(True))
        self.bind("<KeyRelease-space>", lambda _e: self._set_space(False))

    # ---------------------------------------------------------------- state
    def _set_space(self, value: bool) -> None:
        self._space = value
        self.configure(cursor="fleur" if value else "crosshair")

    def set_image(self, image: Image.Image) -> None:
        """Replace the displayed image and reset zoom/pan."""
        self.image = image
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self.redraw()

    def set_boxes(
        self,
        boxes: list[Box],
        selected_id: str | None,
        color_for: Callable[[Box], str],
        title_for: Callable[[Box], str] | None = None,
    ) -> None:
        """Set the box list and callbacks, then redraw."""
        self.boxes, self.selected_id, self.color_for = boxes, selected_id, color_for
        if title_for is not None:
            self.title_for = title_for
        self.redraw()

    # ---------------------------------------------------------- coordinates
    def _layout(self) -> tuple[int, int]:
        """Compute displayed size and origin from canvas size + zoom/pan."""
        cw, ch = self.winfo_width(), self.winfo_height()
        # Before geometry is applied (or in headless tests) use a sane size so
        # image<->canvas coordinate conversion is still correct.
        if cw < 50:
            cw = 900
        if ch < 50:
            ch = 650
        if self.image is None:
            return cw, ch
        iw, ih = self.image.size
        base = min((cw - 32) / max(1, iw), (ch - 32) / max(1, ih))
        self.scale = max(0.02, base * self.zoom)
        dw, dh = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        self.ox = (cw - dw) / 2 + self.pan_x
        self.oy = (ch - dh) / 2 + self.pan_y
        return dw, dh

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        """Canvas coordinate -> original-image pixel coordinate."""
        return (x - self.ox) / self.scale, (y - self.oy) / self.scale

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        """Original-image pixel coordinate -> canvas coordinate."""
        return self.ox + x * self.scale, self.oy + y * self.scale

    def _clamp_image(self, x: float, y: float) -> tuple[float, float]:
        if self.image is None:
            return x, y
        return (max(0.0, min(float(self.image.width), x)),
                max(0.0, min(float(self.image.height), y)))

    # ------------------------------------------------------------- drawing
    def redraw(self) -> None:
        """Redraw image, boxes, labels and handles for the current state."""
        self.delete("all")
        if self.image is None:
            return
        dw, dh = self._layout()
        self.photo = ImageTk.PhotoImage(self.image.resize((dw, dh), Image.Resampling.LANCZOS))
        self.create_image(self.ox, self.oy, image=self.photo, anchor="nw")
        for box in self.boxes:
            x1, y1 = self.to_canvas(*box.xyxy[:2])
            x2, y2 = self.to_canvas(*box.xyxy[2:])
            selected = box.id == self.selected_id
            color = self.color_for(box)
            self.create_rectangle(x1, y1, x2, y2, outline=color,
                                  width=3 if selected else 2,
                                  dash=() if selected else (5, 4),
                                  tags=("box", box.id))
            tag = self.title_for(box)
            tw = max(46, len(tag) * 7 + 12)
            ty = max(self.oy, y1 - 18)
            self.create_rectangle(x1, ty, x1 + tw, ty + 16, fill=color, outline="", tags=("box", box.id))
            self.create_text(x1 + 5, ty + 3, text=tag, fill="#ffffff", anchor="nw",
                             font=("Segoe UI", 8, "bold"), tags=("box", box.id))
            if selected:
                for corner, (hx, hy) in {
                    "nw": (x1, y1), "ne": (x2, y1), "sw": (x1, y2), "se": (x2, y2),
                }.items():
                    self.create_rectangle(hx - HANDLE_HALF, hy - HANDLE_HALF,
                                          hx + HANDLE_HALF, hy + HANDLE_HALF,
                                          fill="#ffffff", outline=color, width=2,
                                          tags=("handle", box.id, corner))

    # ----------------------------------------------------------- hit testing
    def _hit(self, ix: float, iy: float) -> tuple[str, str, str] | None:
        """Return (kind, box_id, corner) for a point in image coordinates."""
        selected = self.selected_id
        # Handles of the selected box win, so resize always works.
        if selected:
            box = next((b for b in self.boxes if b.id == selected), None)
            if box:
                hx = HANDLE_HALF / self.scale
                for corner, (x, y) in {
                    "nw": (box.xyxy[0], box.xyxy[1]), "ne": (box.xyxy[2], box.xyxy[1]),
                    "sw": (box.xyxy[0], box.xyxy[3]), "se": (box.xyxy[2], box.xyxy[3]),
                }.items():
                    if abs(ix - x) <= hx and abs(iy - y) <= hx:
                        return "handle", box.id, corner
        for box in reversed(self.boxes):
            x1, y1, x2, y2 = box.xyxy
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return "box", box.id, ""
        return None

    # ----------------------------------------------------------- mouse flow
    def _press_pan(self, event: Any) -> None:
        self._drag = {"type": "pan", "canvas": (event.x, event.y), "pan": (self.pan_x, self.pan_y)}
        self.configure(cursor="fleur")

    def _motion_pan(self, event: Any) -> None:
        if not self._drag or self._drag["type"] != "pan":
            return
        sx, sy = self._drag["canvas"]
        px, py = self._drag["pan"]
        self.pan_x, self.pan_y = px + event.x - sx, py + event.y - sy
        self.redraw()

    def _release_pan(self, _event: Any) -> None:
        self._drag = None
        self.configure(cursor="fleur" if self._space else "crosshair")

    def _press_draw(self, event: Any) -> None:
        if self._space or getattr(event, "num", 1) == 2:
            self._press_pan(event)
            return
        if self.image is None:
            return
        ix, iy = self._clamp_image(*self.to_image(event.x, event.y))
        hit = self._hit(ix, iy)
        if hit and hit[0] == "handle":
            box = next(b for b in self.boxes if b.id == hit[1])
            self._drag = {"type": "resize", "id": hit[1], "corner": hit[2],
                          "orig": tuple(box.xyxy), "canvas": (event.x, event.y)}
        elif hit and hit[0] == "box" and hit[1] == self.selected_id:
            box = next(b for b in self.boxes if b.id == hit[1])
            self._drag = {"type": "move", "id": hit[1], "start": (ix, iy),
                          "orig": tuple(box.xyxy), "canvas": (event.x, event.y)}
        else:
            # Draw even when the pointer is over another box (nested regions).
            self._drag = {"type": "draw", "start": (ix, iy), "canvas": (event.x, event.y)}

    def _motion(self, event: Any) -> None:
        drag = self._drag
        if not drag or drag["type"] == "pan":
            if drag and drag["type"] == "pan":
                self._motion_pan(event)
            return
        sx, sy = drag["canvas"]
        moved = math.hypot(event.x - sx, event.y - sy) >= CLICK_SLOP
        drag["moved"] = moved
        if not moved:
            return
        if drag["type"] == "draw":
            self.delete("temp")
            x1, y1 = self.to_canvas(*drag["start"])
            self.create_rectangle(x1, y1, event.x, event.y, outline="#ef4444",
                                  width=2, dash=(6, 4), tags="temp")
            return
        box = next((b for b in self.boxes if b.id == drag["id"]), None)
        if box is None:
            return
        ix, iy = self._clamp_image(*self.to_image(event.x, event.y))
        if drag["type"] == "move":
            dx, dy = ix - drag["start"][0], iy - drag["start"][1]
            o = drag["orig"]
            box.xyxy = (round(max(0, o[0] + dx)), round(max(0, o[1] + dy)),
                        round(min(self.image.width, o[2] + dx)),
                        round(min(self.image.height, o[3] + dy)))
        else:
            o, corner = drag["orig"], drag["corner"]
            n = list(o)
            if "w" in corner:
                n[0] = min(o[2] - 1, round(ix))
            if "e" in corner:
                n[2] = max(o[0] + 1, round(ix))
            if "n" in corner:
                n[1] = min(o[3] - 1, round(iy))
            if "s" in corner:
                n[3] = max(o[1] + 1, round(iy))
            box.xyxy = tuple(n)
        self.redraw()
        if self.on_status:
            self.on_status(f"Moving/resizing {box.region}… release to save")

    def _release(self, event: Any) -> None:
        drag = self._drag
        if not drag:
            return
        self._drag = None
        self.delete("temp")
        self.configure(cursor="fleur" if self._space else "crosshair")
        if drag["type"] == "pan":
            return
        if not drag.get("moved"):
            # Click (no drag) selects the box under the pointer, or clears it.
            ix, iy = self._clamp_image(*self.to_image(event.x, event.y))
            hit = self._hit(ix, iy)
            if self.on_select:
                self.on_select(hit[1] if hit and hit[0] == "box" else None)
            return
        if drag["type"] in ("move", "resize"):
            box = next((b for b in self.boxes if b.id == drag["id"]), None)
            if box is not None and self.on_change:
                self.on_change(box)
            self.redraw()
            return
        # New box: convert pointer to image pixels and ignore tiny drags.
        ix, iy = self._clamp_image(*self.to_image(event.x, event.y))
        x1, y1 = drag["start"]
        xyxy = (round(min(x1, ix)), round(min(y1, iy)), round(max(x1, ix)), round(max(y1, iy)))
        if (xyxy[2] - xyxy[0]) * self.scale < MIN_BOX_PX or (xyxy[3] - xyxy[1]) * self.scale < MIN_BOX_PX:
            self.redraw()
            if self.on_status:
                self.on_status("Box too small — drag a little further")
            return
        if self.on_create:
            self.on_create(xyxy)

    # ----------------------------------------------------------------- zoom
    def _zoom_at(self, event: Any, factor: float) -> None:
        if self.image is None:
            return
        ix, iy = self._clamp_image(*self.to_image(event.x, event.y))
        self.zoom = max(0.2, min(8.0, self.zoom * factor))
        self._layout()
        cx, cy = self.to_canvas(ix, iy)
        self.pan_x += event.x - cx
        self.pan_y += event.y - cy
        self.redraw()
        if self.on_status:
            self.on_status(f"Zoom {round(self.zoom * 100)}% · wheel to zoom, middle/space-drag to pan")

    def _wheel(self, event: Any) -> None:
        self._zoom_at(event, 1.1 if event.delta > 0 else 1 / 1.1)

    def fit(self) -> None:
        self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
        self.redraw()
