"""Check that a folder of training images is consistent before annotating.

YOLO can letterbox different sizes, but a uniform, reasonably large set is
much easier to train and usually what collaborators expect. The desktop app
shows this report in a pop-up when it opens.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image

MIN_SIDE = 224  # smallest side we consider usable for YOLO-style training


@dataclass
class ImageSizeReport:
    """Summary of image dimensions in a folder."""

    total: int = 0
    sizes: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)
    min_side: int = MIN_SIDE

    @property
    def uniform(self) -> bool:
        return len(self.sizes) == 1

    @property
    def too_small(self) -> list[str]:
        out: list[str] = []
        for size, names in self.sizes.items():
            if min(size) < self.min_side:
                out.extend(names)
        return sorted(out)

    @property
    def common_size(self) -> tuple[int, int] | None:
        if not self.sizes:
            return None
        return max(self.sizes, key=lambda s: len(self.sizes[s]))

    def message(self) -> str:
        """Human-readable message for a pop-up dialog."""
        if self.total == 0:
            return "No images were found in the selected folder."
        lines = [f"Checked {self.total} image(s)."]
        if self.unreadable:
            lines.append(f"Unreadable: {len(self.unreadable)} — " + ", ".join(self.unreadable[:5]))
        if self.uniform:
            w, h = self.common_size or (0, 0)
            lines.append(f"All images are the same size: {w}x{h} px.")
        else:
            lines.append("Images are NOT all the same size:")
            for size, names in sorted(self.sizes.items(), key=lambda kv: -len(kv[1])):
                lines.append(f"  {size[0]}x{size[1]} px — {len(names)} image(s)")
            lines.append("You can still annotate, but resizing them to one size is recommended before training.")
        small = self.too_small
        if small:
            lines.append(
                f"{len(small)} image(s) have a side smaller than {self.min_side} px "
                f"(e.g. {', '.join(small[:5])}) — consider replacing them."
            )
        if self.uniform and not small and not self.unreadable:
            lines.append("This set is uniform and large enough for training.")
        return "\n".join(lines)

    @property
    def ok(self) -> bool:
        return self.uniform and not self.too_small and not self.unreadable


def inspect_images(paths: Iterable[Path], *, min_side: int = MIN_SIDE) -> ImageSizeReport:
    """Read image headers and group paths by pixel size."""
    report = ImageSizeReport(min_side=min_side)
    for path in paths:
        path = Path(path)
        report.total += 1
        try:
            with Image.open(path) as image:
                size = (int(image.width), int(image.height))
        except (OSError, ValueError) as exc:
            report.unreadable.append(f"{path.name} ({exc.__class__.__name__})")
            continue
        report.sizes.setdefault(size, []).append(path.name)
    return report


def show_image_report(parent: object, report: ImageSizeReport) -> None:
    """Show the size report as an info (all good) or warning (problems) pop-up."""
    from tkinter import messagebox

    title = "Image size check"
    if report.ok:
        messagebox.showinfo(title, report.message(), parent=parent)
    else:
        messagebox.showwarning(title, report.message(), parent=parent)
