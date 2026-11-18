"""End-to-end smoke tests for the search stack in mock mode."""

from __future__ import annotations

import json

from common.config import Config
from common.mock_data import ensure_mock_dataset
from common.schema import load_schema
from search.analyze import Analyzer
from search.db import connect, get_candidates, get_stats, init_db
from search.ingest import ingest_catalog
from search.scoring import search_catalog


def _setup(config: Config) -> Analyzer:
    ensure_mock_dataset(n_images=6, n_products=4, config=config, schema=load_schema(), force=True)
    return Analyzer(config=config, mode="mock")


def test_analyze_returns_schema_driven_predictions(temp_config: Config) -> None:
    from PIL import Image

    analyzer = _setup(temp_config)
    image = temp_config.path("paths.images_raw") / "mock_0000.jpg"
    with Image.open(image) as src:
        width, height = src.size
    result = analyzer.analyze(image).to_dict()

    assert set(result) == {"garments"}
    assert result["garments"]
    for garment in result["garments"]:
        assert garment["type"] in ("top", "bottom")
        assert all(0 <= v <= width for v in (garment["box"][0], garment["box"][2]))
        sc = load_schema()
        expected_attrs: set[str] = set()
        for region in sc.regions(garment["type"]):
            if region.paired:
                expected_attrs.update(f"{region.name}_{side}" for side in region.sides)
            else:
                expected_attrs.add(region.name)
        assert expected_attrs.issubset(set(garment["attributes"]))
        assert garment["embedding"]
        for prediction in garment["attributes"].values():
            assert abs(sum(prediction["probs"].values()) - 1.0) < 1e-6


def test_ingest_is_idempotent_and_resumable(temp_config: Config) -> None:
    analyzer = _setup(temp_config)
    first = ingest_catalog(temp_config, analyzer=analyzer)
    assert first["ingested"] == 4
    assert first["garments"] > 0
    stats = first["database_stats"]
    assert stats["products"] == 4
    assert stats["garments"] == first["garments"]
    assert stats["embeddings"] == first["garments"]

    second = ingest_catalog(temp_config, analyzer=analyzer)
    assert second["skipped"] == 4
    assert second["ingested"] == 0
    with connect(temp_config.path("paths.database")) as conn:
        assert get_stats(conn) == stats  # no duplicates after re-run

    forced = ingest_catalog(temp_config, analyzer=analyzer, force=True)
    assert forced["ingested"] == 4
    with connect(temp_config.path("paths.database")) as conn:
        assert get_stats(conn)["garments"] == stats["garments"]


def test_search_over_ingested_catalog(temp_config: Config) -> None:
    analyzer = _setup(temp_config)
    ingest_catalog(temp_config, analyzer=analyzer)
    reference_image = temp_config.path("paths.images_raw") / "mock_0000.jpg"
    reference = analyzer.analyze(reference_image)
    with connect(temp_config.path("paths.database")) as conn:
        candidates = get_candidates(conn)

    results = search_catalog(candidates, reference, top_k=3, embedding_weight=1.0)
    assert len(results) <= 3
    assert all(0.0 <= r["score"] <= 1.0 for r in results)
    assert all(r["explanation"] for r in results)
    assert all(scores == sorted(scores, reverse=True) for scores in [[r["score"] for r in results]])


def test_api_and_gradio_app(temp_config: Config, monkeypatch) -> None:
    import search.api as api
    from fastapi.testclient import TestClient

    analyzer = _setup(temp_config)
    ingest_catalog(temp_config, analyzer=analyzer)
    monkeypatch.setattr(api, "load_config", lambda: temp_config)
    monkeypatch.setattr(api, "get_analyzer", lambda: analyzer)

    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/stats").json()["products"] == 4
        image_bytes = (temp_config.path("paths.images_raw") / "mock_0000.jpg").read_bytes()
        response = client.post(
            "/search",
            files={"image": ("ref.jpg", image_bytes, "image/jpeg")},
            data={"garment_type": "top", "filters": json.dumps({"length": {"mode": "lock", "weight": 3}}),
                  "embedding_weight": "1", "top_k": "3"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] <= 3
        assert len(body["query"]["garments"]) >= 0

    from search.app import make_app

    demo = make_app(config=temp_config, analyzer=analyzer)
    assert demo is not None


def test_init_db_creates_all_tables(temp_config: Config, tmp_path) -> None:
    import sqlite3

    db = tmp_path / "fresh.sqlite3"
    init_db(db)
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"products", "garments", "garment_attributes", "garment_embeddings"}.issubset(tables)
