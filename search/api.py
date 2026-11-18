"""FastAPI endpoint for attribute-aware similarity search.

Run with::

    uvicorn search.api:app --reload --port 8000
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from common.config import load_config
from common.logging_conf import LOGGER
from search.analyze import Analyzer
from search.db import connect, get_candidates, get_stats, init_db
from search.scoring import parse_filters, search_catalog

LOG = LOGGER.getChild("search.api")
app = FastAPI(title="Raja-Fashion Search", version="1.0")


@lru_cache(maxsize=1)
def get_analyzer() -> Analyzer:
    """Lazily build a single shared analyzer (avoids loading models at import)."""
    return Analyzer()


@app.get("/health")
def health() -> dict[str, Any]:
    """Simple readiness check."""
    return {"status": "ok", "ultralytics_model_family": load_config().model_family}


@app.get("/stats")
def stats() -> dict[str, Any]:
    """Catalog row counts."""
    config = load_config()
    init_db(config.path("paths.database"))
    with connect(config.path("paths.database")) as conn:
        return get_stats(conn)


@app.post("/search", response_model=None)
async def search(
    image: UploadFile = File(...),
    garment_type: str = Form("both"),
    filters: str = Form("{}"),
    embedding_weight: float = Form(0.0),
    top_k: int = Form(10),
) -> Any:
    """Analyse an uploaded image and return top-K matching catalog garments.

    ``filters`` is a JSON object mapping attribute names to
    ``{"mode": "lock"|"soft"|"ignore", "weight": 0-5, "value": optional}``.
    """
    if garment_type not in ("both", "top", "bottom"):
        return JSONResponse({"error": "garment_type must be both/top/bottom"}, status_code=422)
    try:
        parsed_filters = parse_filters(filters)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        temp_path = Path(tmp.name)
    try:
        reference = get_analyzer().analyze(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    config = load_config()
    db_path = config.path("paths.database")
    init_db(db_path)
    with connect(db_path) as conn:
        candidates = get_candidates(conn, garment_type=garment_type) if garment_type in ("top", "bottom") else get_candidates(conn)

    results = search_catalog(
        candidates, reference,
        filters=parsed_filters,
        embedding_weight=embedding_weight,
        garment_type=garment_type,
        top_k=top_k,
    )
    return {
        "query": {
            "garments": [g.to_dict() for g in reference.garments],
            "garment_type": garment_type,
            "filters": {k: v.__dict__ for k, v in parsed_filters.items()},
        },
        "count": len(results),
        "results": results,
    }
