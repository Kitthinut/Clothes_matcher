"""SQLite catalog storage (plain ``sqlite3``, no ORM)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common.logging_conf import LOGGER
from search.types import AttributePrediction, GarmentResult

LOG = LOGGER.getChild("search.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS products (
    id          TEXT PRIMARY KEY,
    image_path  TEXT NOT NULL,
    title       TEXT,
    price       REAL,
    url         TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS garments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    box         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_garments_product ON garments(product_id);
CREATE TABLE IF NOT EXISTS garment_attributes (
    garment_id  INTEGER NOT NULL REFERENCES garments(id) ON DELETE CASCADE,
    attribute   TEXT NOT NULL,
    label       TEXT NOT NULL,
    probs_json  TEXT NOT NULL,
    PRIMARY KEY (garment_id, attribute)
);
CREATE TABLE IF NOT EXISTS garment_embeddings (
    garment_id  INTEGER PRIMARY KEY REFERENCES garments(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL
);
"""


@dataclass
class CatalogGarment:
    """A catalog garment with its source product metadata."""

    garment: GarmentResult
    product: dict[str, Any]


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with foreign keys enabled and row access by name."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    """Create all tables if they do not exist."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def product_exists(conn: sqlite3.Connection, product_id: str) -> bool:
    """Return whether a product has already been ingested."""
    return conn.execute("SELECT 1 FROM products WHERE id = ?", (product_id,)).fetchone() is not None


def upsert_product(conn: sqlite3.Connection, product: Mapping[str, Any], created_at: str) -> None:
    """Insert or update one product row."""
    conn.execute(
        "INSERT INTO products (id, image_path, title, price, url, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET image_path=excluded.image_path, title=excluded.title, "
        "price=excluded.price, url=excluded.url",
        (
            str(product["id"]), str(product["image_path"]), product.get("title"),
            float(product.get("price") or 0.0), product.get("url"), created_at,
        ),
    )


def clear_garments_for_product(conn: sqlite3.Connection, product_id: str) -> None:
    """Remove old garments (and cascading rows) so re-ingest is idempotent."""
    conn.execute("DELETE FROM garments WHERE product_id = ?", (product_id,))


def insert_garment(conn: sqlite3.Connection, product_id: str, garment: GarmentResult) -> int:
    """Insert one garment, its attributes and its embedding; return its id."""
    cursor = conn.execute(
        "INSERT INTO garments (product_id, type, box) VALUES (?, ?, ?)",
        (product_id, garment.type, json.dumps(garment.box)),
    )
    garment_id = int(cursor.lastrowid)
    for name, prediction in garment.attributes.items():
        conn.execute(
            "INSERT INTO garment_attributes (garment_id, attribute, label, probs_json) VALUES (?, ?, ?, ?)",
            (garment_id, name, prediction.label, json.dumps(prediction.probs)),
        )
    if garment.embedding:
        vector = np.asarray(garment.embedding, dtype=np.float32)
        conn.execute(
            "INSERT INTO garment_embeddings (garment_id, vector, dim) VALUES (?, ?, ?)",
            (garment_id, vector.tobytes(), int(vector.size)),
        )
    return garment_id


def get_candidates(conn: sqlite3.Connection, garment_type: str = "both") -> list[CatalogGarment]:
    """Load all catalog garments (optionally filtered by type)."""
    query = (
        "SELECT g.id AS garment_id, g.type AS garment_type, g.box, "
        "p.id AS product_id, p.image_path, p.title, p.price, p.url, p.created_at "
        "FROM garments g JOIN products p ON p.id = g.product_id"
    )
    params: tuple[Any, ...] = ()
    if garment_type in ("top", "bottom"):
        query += " WHERE g.type = ?"
        params = (garment_type,)
    rows = conn.execute(query, params).fetchall()

    results: list[CatalogGarment] = []
    for row in rows:
        attrs: dict[str, AttributePrediction] = {}
        for a in conn.execute(
            "SELECT attribute, label, probs_json FROM garment_attributes WHERE garment_id = ?",
            (row["garment_id"],),
        ):
            attrs[a["attribute"]] = AttributePrediction(
                label=a["label"], probs=json.loads(a["probs_json"])
            )
        embedding: list[float] = []
        emb_row = conn.execute(
            "SELECT vector, dim FROM garment_embeddings WHERE garment_id = ?", (row["garment_id"],)
        ).fetchone()
        if emb_row is not None:
            vector = np.frombuffer(emb_row["vector"], dtype=np.float32)
            embedding = vector[: emb_row["dim"]].tolist()

        garment = GarmentResult(
            type=row["garment_type"],
            box=json.loads(row["box"]),
            attributes=attrs,
            embedding=embedding,
        )
        product = {
            "id": row["product_id"], "image_path": row["image_path"],
            "title": row["title"], "price": row["price"], "url": row["url"],
            "created_at": row["created_at"],
        }
        results.append(CatalogGarment(garment=garment, product=product))
    return results


def get_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Return simple row counts for diagnostics."""
    def count(table: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    return {
        "products": count("products"),
        "garments": count("garments"),
        "attributes": count("garment_attributes"),
        "embeddings": count("garment_embeddings"),
    }
