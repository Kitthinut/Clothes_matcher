"""Local Roboflow/Label Studio-style annotation GUI built with Gradio.

Run with::

    python -m annotator.app_web        # from the project root
    python annotator/app_web.py        # also works (adds project root to sys.path)

The UI is a custom HTML/CSS/JS canvas hosted inside a single Gradio ``gr.HTML``
block. Python remains the source of truth: annotations are saved through
``AnnotationStore`` as attribute JSON + YOLO detection labels, and every edit
is mirrored from the browser through hidden Gradio components.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from annotator.store import (  # noqa: E402
    AnnotationStore, Box, ImageAnnotation, resolve_annotation_dirs,
)
from common.config import Config, load_config  # noqa: E402
from common.logging_conf import LOGGER  # noqa: E402
from common.paths import ensure_dir  # noqa: E402
from common.mock_data import ensure_mock_dataset  # noqa: E402
from common.schema import Schema, load_schema  # noqa: E402

LOG = LOGGER.getChild("annotator.app")
UI_DIR = Path(__file__).resolve().parent / "ui"


def _assets() -> str:
    """Load the UI assets. CSS is injected via JS because gr.HTML escapes <style>."""
    css = (UI_DIR / "editor.css").read_text(encoding="utf-8")
    js = (UI_DIR / "editor_v2.js").read_text(encoding="utf-8")
    inject = "(function(){var s=document.createElement('style');s.textContent=" + json.dumps(css) + ";document.head.appendChild(s);})();\n"
    return inject + js + "\nwindow.rfInitEditor && window.rfInitEditor();"


_SHELL = """
<div class="rf-app">
  <div class="rf-header">
    <div class="rf-brand"><span class="rf-brand-mark"></span> Raja-Fashion</div>
    <span class="rf-crumb">/ <b>Annotation</b></span>
    <span id="rf-header-image" class="rf-crumb"></span>
    <div class="rf-spacer"></div>
    <span id="rf-header-progress-text" class="rf-crumb"></span>
    <div class="rf-progress"><div id="rf-progress-bar"></div></div>
    <button class="rf-btn" data-action="undo" title="Ctrl+Z">↶</button>
    <button class="rf-btn" data-action="redo" title="Ctrl+Shift+Z">⟲</button>
    <button class="rf-btn" data-action="reset" title="Clear all boxes">✕</button>
    <button class="rf-btn" data-action="skip">Skip</button>
    <button class="rf-btn primary" data-action="submit">Submit</button>
  </div>
  <div class="rf-body">
    <div class="rf-sidebar">
      <div class="rf-tabs">
        <div class="rf-tab active">Regions</div>
        <div class="rf-tab">Boxes</div>
      </div>
      <div class="rf-panel-section" id="rf-panel-classes"></div>
      <div class="rf-panel-section" id="rf-panel-regions"></div>
    </div>
    <div class="rf-stage">
      <div class="rf-scroll">
        <div class="rf-canvas-wrap" id="rf-canvas-wrap">
          <img class="rf-image" id="rf-image" alt="annotation canvas">
          <svg class="rf-svg" id="rf-svg" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
      </div>
      <div id="rf-editor-host"></div>
    </div>
    <div class="rf-tools">
      <button class="rf-icon-btn" data-action="tool" data-tool="select" title="Select / move (V)">↖</button>
      <button class="rf-icon-btn active" data-action="tool" data-tool="box" title="Draw box (B)">▣</button>
      <button class="rf-icon-btn" data-action="tool" data-tool="pan" title="Pan (H / Space-drag)">✋</button>
      <div class="rf-tools-sep"></div>
      <button class="rf-icon-btn" data-action="zoom-in" title="Zoom in (+)">＋</button>
      <button class="rf-icon-btn" data-action="zoom-out" title="Zoom out (-)">－</button>
      <button class="rf-icon-btn" data-action="fit" title="Fit image (0)">⤢</button>
      <div class="rf-tools-sep"></div>
      <button class="rf-icon-btn" data-action="undo" title="Undo (Ctrl+Z)">↶</button>
      <button class="rf-icon-btn" data-action="redo" title="Redo (Ctrl+Shift+Z)">⟲</button>
    </div>
  </div>
  <div class="rf-footer">
    <button class="rf-btn" data-action="prev">← Prev</button>
    <button class="rf-btn" data-action="next">Next →</button>
    <div class="rf-spacer"></div>
    <button class="rf-icon-btn" data-action="zoom-out">－</button>
    <span id="rf-zoom-label" class="rf-crumb">100%</span>
    <button class="rf-icon-btn" data-action="zoom-in">＋</button>
    <button class="rf-btn" data-action="fit">Fit</button>
    <div class="rf-spacer"></div>
    <span class="rf-crumb">1/2 class · drag to box · V/B/H tools · Del remove · Ctrl+Z undo · Enter submit</span>
  </div>
</div>
"""


def _data_url(path: Path) -> str:
    """Return a base64 data URL so the JS canvas can draw local images."""
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def make_app(
    config: Config | None = None,
    schema: Schema | None = None,
    *,
    image_dir: str | Path | None = None,
    annotations_dir: str | Path | None = None,
) -> gr.Blocks:
    """Build the annotator Blocks app.

    Args:
        config: Config override (used by tests).
        schema: Schema override (used by tests).
        image_dir: Optional folder of images to label.
        annotations_dir: Optional folder for saved labels.
    """
    cfg = config or load_config()
    sc = schema or load_schema()
    images, annotations = resolve_annotation_dirs(cfg, image_dir, annotations_dir)
    ensure_dir(images)
    ensure_dir(annotations)
    store = AnnotationStore(images, annotations, sc)
    paths = store.image_paths()
    if not paths and image_dir is None and annotations_dir is None:
        LOG.warning("No images found; generating the mock dataset")
        ensure_mock_dataset(config=cfg, schema=sc)
        paths = store.image_paths()
    if not paths:
        raise RuntimeError("No images available for annotation")
    total = len(paths)
    js = _assets()

    def dataset_counts() -> dict[str, int]:
        counts = {name: 0 for name in sc.detection_classes}
        for ann in store.all_annotations().values():
            for box in ann.boxes:
                name = box.detection_name(sc)
                if name in counts:
                    counts[name] += 1
        return counts

    def build_payload(state: dict) -> str:
        """Serialise everything the browser needs for the current image."""
        index = max(0, min(int(state.get("index", 0)), total - 1))
        ann = store.load(paths[index])
        done, count = store.progress()
        payload = {
            "image_name": paths[index].name,
            "image_src": _data_url(paths[index]),
            "width": ann.width,
            "height": ann.height,
            "boxes": [b.to_dict() for b in ann.boxes],
            "schema": {
                "classes": list(sc.detection_classes),
                "color_classes": list(sc.color_classes),
                "regions": [
                    {
                        "key": f"{g}_{r.name}_{side}" if r.paired else f"{g}_{r.name}",
                        "garment": g, "region": r.name, "side": side,
                        "title": f"{g} · {r.title}" + (f" ({side})" if r.paired else ""),
                        "classes": list(r.classes), "paired": r.paired, "source": r.source,
                    }
                    for g in sc.garment_names for r in sc.regions(g) for side in r.sides
                ],
            },
            "counts": dataset_counts(),
            "index": index,
            "total": count,
            "annotated": done,
            "fit_on_load": True,
        }
        return json.dumps(payload, ensure_ascii=False)

    def current_annotation(state: dict) -> ImageAnnotation:
        return ImageAnnotation.from_dict(state["annotation"])

    def push_undo(state: dict) -> None:
        state.setdefault("undo", []).append(copy.deepcopy(state["annotation"]))
        state["undo"] = state["undo"][-50:]
        state["redo"] = []

    def load_index(index: int, state: dict) -> dict:
        index = max(0, min(index, total - 1))
        ann = store.load(paths[index])
        state.update({"index": index, "annotation": ann.to_dict(), "undo": [], "redo": []})
        store.save_session(index, state.get("skipped", []))
        return state

    with gr.Blocks(title="Raja-Fashion Annotator", analytics_enabled=False) as demo:
        gr.HTML(value=_SHELL, elem_id="rf-ui-host")

        # Hidden DOM-resident channels (visible=False would remove them).
        payload = gr.Textbox(visible="hidden", elem_id="rf-payload")
        boxes_state = gr.Textbox(visible="hidden", elem_id="rf-boxes-state")
        action_state = gr.Textbox(visible="hidden", elem_id="rf-action-state")
        color_result = gr.Textbox(visible="hidden", elem_id="rf-color-result")
        state = gr.State({})
        sync_btn = gr.Button(visible="hidden", elem_id="rf-sync")
        action_btn = gr.Button(visible="hidden", elem_id="rf-action")

        demo.load(
            fn=lambda: (build_payload({"index": store.load_session().get("last_index", 0)}),
                        {"index": store.load_session().get("last_index", 0), "undo": [], "redo": [], "skipped": []}),
            outputs=[payload, state],
        )
        demo.load(fn=None, js=js)
        payload.change(
            fn=None, inputs=[payload],
            js="(s) => { window.rfLoadPayload ? window.rfLoadPayload(s) : (window.__rfPending = s); }",
        )
        color_result.change(fn=None, inputs=[color_result], js="(s) => window.rfLoadColor(s)")
        sync_btn.click(fn=None, outputs=[boxes_state], js="() => JSON.stringify(window.__rfBoxes || [])")
        action_btn.click(fn=None, outputs=[action_state], js="() => JSON.stringify(window.__rfAction || {})")

        def on_sync(raw_json: str, current: dict) -> dict:
            """Mirror browser boxes into Python state and persist them."""
            try:
                raw = json.loads(raw_json or "[]")
            except ValueError:
                LOG.warning("Ignoring malformed box payload")
                return current
            ann = current_annotation(current)
            new_boxes: list[Box] = []
            for item in raw:
                box = Box.from_dict(item)
                try:
                    sc.validate_box(box.garment, box.region, box.side, box.label)
                except ValueError:
                    LOG.warning("Ignoring invalid region box %s", item)
                    continue
                new_boxes.append(box)
            if [b.to_dict() for b in new_boxes] == [b.to_dict() for b in ann.boxes]:
                return current
            push_undo(current)
            ann.boxes = new_boxes
            current["annotation"] = ann.to_dict()
            store.save(ann)
            return current

        def on_action(raw_json: str, current: dict) -> tuple[str, str, dict]:
            """Handle navigation, undo/redo, reset and automatic colour detection."""
            try:
                data = json.loads(raw_json or "{}")
            except ValueError:
                data = {}
            name = data.get("name", "")
            output_color = ""

            if name == "color":
                garment = str(data.get("garment", "top"))
                xyxy = [int(v) for v in data.get("xyxy", [])]
                if garment in sc.garment_names and len(xyxy) == 4:
                    from common.colors import dominant_color_name
                    index = current.get("index", 0)
                    with Image.open(paths[index]) as src:
                        crop = src.convert("RGB").crop((max(0, xyxy[0]), max(0, xyxy[1]), xyxy[2], xyxy[3]))
                    detected = dominant_color_name(crop, set(sc.color_classes))
                    output_color = json.dumps({"color": detected, "garment": garment, "xyxy": xyxy})
            elif name in ("undo", "redo"):
                stack = current.get("undo" if name == "undo" else "redo", [])
                if stack:
                    other = "redo" if name == "undo" else "undo"
                    current.setdefault(other, []).append(copy.deepcopy(current["annotation"]))
                    current["annotation"] = stack.pop()
                    store.save(current_annotation(current))
                else:
                    LOG.info("Nothing to %s", name)
            elif name == "reset":
                ann = current_annotation(current)
                if ann.boxes:
                    push_undo(current)
                    ann.boxes = []
                    current["annotation"] = ann.to_dict()
                    store.save(ann)
            elif name in ("prev", "next", "skip", "submit"):
                store.save(current_annotation(current))
                delta = -1 if name == "prev" else 1
                load_index(current.get("index", 0) + delta, current)
            elif name:
                LOG.debug("Unhandled annotator action: %s", name)

            return build_payload(current), output_color, current

        boxes_state.change(fn=on_sync, inputs=[boxes_state, state], outputs=[state])
        action_state.change(
            fn=on_action, inputs=[action_state, state],
            outputs=[payload, color_result, state],
        )

    return demo


def main(argv: list[str] | None = None) -> None:
    """Launch the browser annotator on localhost (optionally with custom folders)."""
    parser = argparse.ArgumentParser(description="Raja-Fashion browser annotator")
    parser.add_argument("--images", type=str, default=None,
                        help="Folder of images to label (default: config paths.images_raw)")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Folder for saved labels (default: config paths.annotations)")
    args = parser.parse_args(argv)
    demo = make_app(image_dir=args.images, annotations_dir=args.annotations)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
