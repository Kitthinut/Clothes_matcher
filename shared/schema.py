"""Schema loading (shared/schema.py).

Turns attributes_<garment>.json into the compact spec dict that both backends
use to build model heads, and exposes the full schema (mode + aliases) to the
labelling / validation tools.

Theory: "Multi-task learning" - one shared model with one small classifier
head per attribute, instead of one big model per attribute.
"""

import json
from pathlib import Path

SHARED = Path(__file__).resolve().parent
GARMENTS = ("top", "bottom")


def load_full(garment):
    """Full raw schema for a garment, e.g.
    {schema_version, garment, categories: [{key, title, mode, options: [{name, aliases}]}]}."""
    if garment not in GARMENTS:
        raise ValueError(f"garment must be one of {GARMENTS}, got '{garment}'")
    return json.loads((SHARED / f"attributes_{garment}.json").read_text(encoding="utf-8"))


def load_spec(garment):
    """garment 'top'/'bottom' ->
    {key: {'title': str, 'options': [str, ...], 'mode': 'single'|'multi',
           'crop_boxes': int (optional)}}.

    'options' stays a plain list of names so model-head code and the search
    GUI can keep using it unchanged.
    """
    schema = load_full(garment)
    return {
        cat["key"]: {
            "title": cat["title"],
            "options": [o["name"] for o in cat["options"]],
            "mode": cat.get("mode", "single"),
            "crop_boxes": cat.get("crop_boxes", 0),
        }
        for cat in schema["categories"]
    }


def aliases(garment):
    """Option name -> list of search aliases (also includes the name itself)."""
    out = {}
    for cat in load_full(garment)["categories"]:
        for o in cat["options"]:
            out[o["name"]] = [o["name"]] + list(o.get("aliases", []))
    return out


def weights(garment):
    """Search weight per attribute key (default 1.0; optional 'weight' field
    in attributes_<garment>.json, e.g. neck matters more than sleeves)."""
    return {cat["key"]: float(cat.get("weight", 1.0))
            for cat in load_full(garment)["categories"]}


def valid_values(garment, key):
    """All valid option names for one attribute key (None if key unknown)."""
    for cat in load_full(garment)["categories"]:
        if cat["key"] == key:
            return [o["name"] for o in cat["options"]]
    return None
