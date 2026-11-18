"""Gradio UI for attribute-aware similarity search.

Run with::

    python -m search.app
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from common.config import Config, load_config  # noqa: E402
from common.schema import Schema, load_schema  # noqa: E402
from search.analyze import Analyzer  # noqa: E402
from search.db import connect, get_candidates, init_db  # noqa: E402
from search.scoring import search_catalog  # noqa: E402

MODES = ["ignore", "soft", "lock"]


def make_app(config: Config | None = None, schema: Schema | None = None, analyzer: Analyzer | None = None) -> gr.Blocks:
    """Build the Gradio search app; controls are generated from the schema."""
    cfg = config or load_config()
    sc = schema or load_schema()
    engine = analyzer or Analyzer(config=cfg, schema=sc)
    db_path = cfg.path("paths.database")
    # One filter per region; paired regions (sleeves) get left/right filters.
    filter_keys: list[str] = []
    for task in sc.classification_tasks():
        if task.paired:
            filter_keys.extend(f"{task.name}_{side}" for side in task.sides)
        else:
            filter_keys.append(task.name)

    def do_search(image: str | None, garment_type: str, *values: Any) -> tuple[list, dict, str]:
        """Run analyze → score → top-K and format gallery + explanation."""
        if image is None:
            return [], {}, "Upload a reference image first."
        filters: dict[str, dict[str, Any]] = {}
        offset = 0
        for key in filter_keys:
            mode, weight = values[offset], values[offset + 1]
            offset += 2
            filters[key] = {"mode": mode, "weight": float(weight)}
        embedding_weight = float(values[offset])
        top_k = int(values[offset + 1])

        reference = engine.analyze(image)
        init_db(db_path)
        with connect(db_path) as conn:
            candidates = (
                get_candidates(conn, garment_type=garment_type)
                if garment_type in ("top", "bottom") else get_candidates(conn)
            )
        results = search_catalog(
            candidates, reference,
            filters=filters,
            embedding_weight=embedding_weight,
            garment_type=garment_type,
            top_k=top_k,
        )

        gallery = [
            (r["product"]["image_path"], f"#{i + 1} {r['score']:.3f} — {r['product']['title']}")
            for i, r in enumerate(results)
        ]
        explanation = {str(i + 1): r for i, r in enumerate(results)}
        summary = (
            f"Found **{len(results)}** matches for "
            f"**{len(reference.garments)}** reference garment(s). "
            f"Seek type: `{garment_type}`."
        )
        return gallery, explanation, summary

    with gr.Blocks(title="Raja-Fashion Search", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Raja-Fashion similarity search\n"
            "Upload a reference outfit, lock or weight the attributes you care about, "
            "and rank the catalog by attribute match + visual similarity."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Reference image")
                garment_type = gr.Radio(["both", "top", "bottom"], value="both", label="Garment type")
            with gr.Column(scale=2):
                gr.Markdown("### Attribute filters")
                filter_controls: list[Any] = []
                for key in filter_keys:
                    with gr.Row():
                        mode = gr.Dropdown(MODES, value="soft", label=f"{key} — mode", scale=1)
                        weight = gr.Slider(0, 5, 1, step=0.5, label=f"{key} — weight", scale=2)
                    filter_controls.extend([mode, weight])
                with gr.Row():
                    embedding_weight = gr.Slider(0, 5, 1, step=0.25, label="Visual similarity weight")
                    top_k = gr.Slider(1, 20, 10, step=1, label="Top-K")
                search_btn = gr.Button("Search catalog", variant="primary")

        summary_md = gr.Markdown()
        with gr.Row():
            gallery = gr.Gallery(label="Matches", columns=4, height=420, object_fit="contain")
        explanation = gr.JSON(label="Per-attribute match explanation")

        inputs = [image_input, garment_type, *filter_controls, embedding_weight, top_k]
        search_btn.click(fn=do_search, inputs=inputs, outputs=[gallery, explanation, summary_md])

    return demo


def main() -> None:
    """Launch the search UI."""
    make_app().launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
