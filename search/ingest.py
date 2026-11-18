"""Idempotent, resumable catalog ingestion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import Config, load_config
from common.logging_conf import LOGGER
from search.analyze import Analyzer
from search.db import clear_garments_for_product, connect, get_stats, init_db, insert_garment, product_exists, upsert_product

LOG = LOGGER.getChild("search.ingest")


def load_products(catalog_dir: Path) -> list[dict[str, Any]]:
    """Read the catalog manifest."""
    manifest = Path(catalog_dir) / "products.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Catalog manifest not found: {manifest}")
    products = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(products, list):
        raise ValueError("products.json must contain a list")
    return products


def ingest_catalog(
    config: Config | None = None,
    *,
    analyzer: Analyzer | None = None,
    force: bool = False,
    db_path: str | Path | None = None,
    catalog_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyse and store every catalog product, skipping already-ingested ones.

    Each product is committed separately so an interrupted run can simply be
    re-executed to resume where it stopped.
    """
    cfg = config or load_config()
    db = Path(db_path) if db_path is not None else cfg.path("paths.database")
    catalog = Path(catalog_dir) if catalog_dir is not None else cfg.path("paths.catalog")
    init_db(db)
    engine = analyzer or Analyzer(config=cfg)

    products = load_products(catalog)
    stats = {"total": len(products), "ingested": 0, "skipped": 0, "garments": 0, "failed": []}

    with connect(db) as conn:
        for product in products:
            product_id = str(product["id"])
            if product_exists(conn, product_id) and not force:
                LOG.debug("Already ingested %s; skipping", product_id)
                stats["skipped"] += 1
                continue
            image_path = Path(str(product["image_path"]))
            if not image_path.is_absolute():
                image_path = catalog / image_path
            if not image_path.exists():
                LOG.warning("Missing image for %s: %s", product_id, image_path)
                stats["failed"].append(str(product_id))
                continue
            try:
                result = engine.analyze(image_path)
            except Exception as exc:  # keep one bad product from aborting the run
                LOG.exception("Failed to analyse %s: %s", product_id, exc)
                stats["failed"].append(str(product_id))
                continue

            upsert_product(conn, product, datetime.now(timezone.utc).isoformat())
            clear_garments_for_product(conn, product_id)
            for garment in result.garments:
                insert_garment(conn, product_id, garment)
                stats["garments"] += 1
            conn.commit()
            stats["ingested"] += 1
            LOG.info("Ingested %s (%d garments)", product_id, len(result.garments))

        stats["database_stats"] = get_stats(conn)
    return stats


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest the Raja-Fashion catalog")
    parser.add_argument("--force", action="store_true", help="Re-ingest products already in the DB")
    args = parser.parse_args(argv)
    print(ingest_catalog(force=args.force))


if __name__ == "__main__":
    main()
