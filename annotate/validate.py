#!/usr/bin/env python3
"""
Check your labels before sending them back.

    python validate.py path/to/your_folder

Looks at every <name>.json that sits next to <name>.jpg/png/webp and reports
anything that does not follow labeling_config/schema.yaml.
Exit code 0 = all good, 1 = problems found.
"""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "labeling_config" / "schema.yaml"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def load_schema():
    return yaml.safe_load(SCHEMA.read_text("utf-8"))


def find_image(jp: Path):
    for e in IMG_EXT:
        for cand in (jp.with_suffix(e), jp.with_suffix(e.upper())):
            if cand.exists():
                return cand
    return None


def object_problems(o, schema):
    p = []
    cls = o.get("cls")
    if cls not in schema["classes"]:
        return [f"unknown class '{cls}' (allowed: {schema['classes']})"]
    allowed = schema["attributes"].get(cls, {})
    for k, v in (o.get("attrs") or {}).items():
        if v is None:
            continue
        if k not in allowed:
            p.append(f"'{cls}' has no attribute '{k}'")
        elif v not in allowed[k]:
            p.append(f"{cls}.{k} = '{v}' is not one of {allowed[k]}")
    b = o.get("box")
    if not (isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in b)
            and b[2] > 0 and b[3] > 0):
        p.append(f"box must be [x_center, y_center, width, height] between 0 and 1, got {b}")
    return p


def check_folder(folder: Path, schema=None):
    """Returns (list of ok json paths, list of error strings)."""
    schema = schema or load_schema()
    ok, errors = [], []
    jsons = [j for j in sorted(folder.rglob("*.json")) if j.name != "contributor.json"]
    for jp in jsons:
        try:
            meta = json.loads(jp.read_text("utf-8"))
        except Exception as e:
            errors.append(f"{jp.name}: not valid JSON ({e})")
            continue
        if not isinstance(meta, dict) or not isinstance(meta.get("objects"), list):
            errors.append(f"{jp.name}: missing 'objects' list")
            continue
        if find_image(jp) is None:
            errors.append(f"{jp.name}: no image with the same name next to it")
            continue
        errs = [f"{jp.name}: object {i + 1}: {p}" for i, o in enumerate(meta["objects"]) for p in object_problems(o, schema)]
        if errs:
            errors.extend(errs)
        else:
            ok.append(jp)
    for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT):
        if not img.with_suffix(".json").exists():
            errors.append(f"{img.name}: photo has no label file (.json)")
    return ok, errors


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python validate.py <folder>")
    folder = Path(sys.argv[1])
    ok, errors = check_folder(folder)
    print(f"{len(ok)} label files OK, {len(errors)} problem(s)")
    for e in errors[:50]:
        print("  -", e)
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
