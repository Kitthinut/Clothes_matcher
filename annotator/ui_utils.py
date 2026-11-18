"""Small Tk helpers for the annotation window (tooltips, styled chips)."""

from __future__ import annotations

import tkinter as tk
from typing import Any


class ToolTip:
    """Show a lightweight hover tooltip for any Tk widget."""

    def __init__(self, widget: Any, text: str, *, delay: int = 350) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.window: tk.Toplevel | None = None
        self._after: str | None = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event: Any = None) -> None:
        self._after = self.widget.after(self.delay, self._show)

    def _show(self) -> None:
        if self.window or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(self.window, text=self.text, background="#1f2433", foreground="#ffffff",
                 font=("Segoe UI", 9), padx=8, pady=4).pack()

    def _hide(self, _event: Any = None) -> None:
        if self._after:
            self.widget.after_cancel(self._after)
            self._after = None
        if self.window:
            self.window.destroy()
            self.window = None


def tip(widget: Any, text: str) -> Any:
    """Attach a :class:`ToolTip` and return the widget (chainable)."""
    ToolTip(widget, text)
    return widget
