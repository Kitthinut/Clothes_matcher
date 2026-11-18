"""Attribute-driven scoring with lock / weighted / ignore filters.

Score = ``sum(weight_a * match_a) / sum(weight_a)`` over active attributes,
optionally blended with an embedding-similarity term::

    final = (base + embedding_weight * similarity) / (1 + embedding_weight)

``match_a`` is the reference classifier's probability for the candidate's
label (soft match). A locked attribute is a hard filter: a mismatching
candidate is excluded entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from search.db import CatalogGarment
from search.embeddings import cosine_similarity
from search.types import AnalysisResult, GarmentResult

FilterMode = Literal["lock", "soft", "ignore"]


@dataclass
class AttributeFilter:
    """User filter for one attribute category."""

    mode: FilterMode = "soft"
    weight: float = 1.0
    value: str | None = None  # optional explicit target label (mainly for lock)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "AttributeFilter") -> "AttributeFilter":
        if isinstance(data, AttributeFilter):
            return data
        mode = str(data.get("mode", "soft")).lower()
        if mode not in ("lock", "soft", "ignore"):
            raise ValueError(f"Invalid filter mode: {mode}")
        return cls(mode=mode, weight=float(data.get("weight", 1.0)), value=data.get("value"))


@dataclass
class AttributeMatch:
    """Per-attribute explanation of a score."""

    name: str
    mode: str
    weight: float
    reference_label: str | None
    candidate_label: str | None
    match: float = 0.0
    contribution: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.name, "mode": self.mode, "weight": self.weight,
            "reference_label": self.reference_label, "candidate_label": self.candidate_label,
            "match": round(self.match, 4), "contribution": round(self.contribution, 4),
            "note": self.note,
        }


@dataclass
class ScoreResult:
    """A scored catalog garment with its full explanation."""

    product: dict[str, Any]
    garment: GarmentResult
    score: float
    base_score: float
    embedding_similarity: float
    explanation: list[AttributeMatch] = field(default_factory=list)
    excluded: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "garment": {
                "type": self.garment.type, "box": self.garment.box,
                "attributes": {k: v.label for k, v in self.garment.attributes.items()},
            },
            "score": round(self.score, 4),
            "base_score": round(self.base_score, 4),
            "embedding_similarity": round(self.embedding_similarity, 4),
            "explanation": [e.to_dict() for e in self.explanation],
            "excluded": self.excluded,
            "reason": self.reason,
        }


def _coerce_filters(filters: Mapping[str, Any] | None) -> dict[str, AttributeFilter]:
    return {str(k): AttributeFilter.from_dict(v) for k, v in (filters or {}).items()}


def score_garment(
    reference: GarmentResult,
    candidate: GarmentResult,
    filters: Mapping[str, Any] | None = None,
    *,
    embedding_weight: float = 0.0,
    garment_type: str = "both",
    embedding_similarity: float | None = None,
) -> ScoreResult:
    """Score one candidate garment against one reference garment.

    Args:
        reference: Garment from the uploaded reference image.
        candidate: Catalog garment to score.
        filters: Per-attribute ``AttributeFilter`` (or dict) overrides.
        embedding_weight: Relative weight of the visual similarity term.
        garment_type: ``top``, ``bottom`` or ``both``.
        embedding_similarity: Precomputed cosine similarity; computed if omitted.
    """
    coerce = _coerce_filters(filters)
    product: dict[str, Any] = {}

    if garment_type in ("top", "bottom") and candidate.type != garment_type:
        return ScoreResult(product, candidate, 0.0, 0.0, 0.0, excluded=True, reason=f"type != {garment_type}")

    weighted_sum = 0.0
    weight_total = 0.0
    explanation: list[AttributeMatch] = []

    for name, ref_pred in reference.attributes.items():
        filt = coerce.get(name, AttributeFilter())
        cand_pred = candidate.attributes.get(name)
        if cand_pred is None:
            continue
        if filt.mode == "ignore" or filt.weight <= 0:
            explanation.append(
                AttributeMatch(name, filt.mode, max(0.0, filt.weight), ref_pred.label,
                               cand_pred.label, note="ignored")
            )
            continue

        if filt.mode == "lock":
            target = filt.value or ref_pred.label
            if cand_pred.label != target:
                return ScoreResult(
                    product, candidate, 0.0, 0.0, 0.0,
                    explanation=explanation, excluded=True,
                    reason=f"locked {name}: {cand_pred.label!r} != {target!r}",
                )
            match = 1.0
            note = "locked match"
        else:
            target = filt.value or cand_pred.label
            match = float(ref_pred.probs.get(target, 0.0))
            note = f"soft match via reference P({target})"

        contribution = filt.weight * match
        weighted_sum += contribution
        weight_total += filt.weight
        explanation.append(
            AttributeMatch(name, filt.mode, filt.weight, ref_pred.label,
                           cand_pred.label, match, contribution, note)
        )

    base = min(1.0, max(0.0, weighted_sum / weight_total)) if weight_total > 0 else 0.0
    sim = (
        float(embedding_similarity)
        if embedding_similarity is not None
        else cosine_similarity(reference.embedding, candidate.embedding)
    )
    weight = max(0.0, float(embedding_weight))
    score = (base + weight * sim) / (1.0 + weight) if weight > 0 else base
    score = min(1.0, max(0.0, score))
    return ScoreResult(product, candidate, score, base, sim, explanation)


def search_catalog(
    candidates: list[CatalogGarment],
    reference: AnalysisResult,
    *,
    filters: Mapping[str, Any] | None = None,
    embedding_weight: float = 0.0,
    garment_type: str = "both",
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Score every catalog garment against every reference garment; return top-K."""
    results: list[ScoreResult] = []
    for candidate in candidates:
        for ref_garment in reference.garments:
            if garment_type in ("top", "bottom") and ref_garment.type != garment_type:
                continue
            if ref_garment.type != candidate.garment.type:
                continue
            scored = score_garment(
                ref_garment, candidate.garment, filters,
                embedding_weight=embedding_weight,
                garment_type=garment_type,
            )
            if not scored.excluded:
                scored.product = candidate.product
            results.append(scored)

    # Deduplicate by catalog garment id, keeping the best reference match.
    best: dict[int, ScoreResult] = {}
    for result in results:
        key = id(result.garment)
        if key not in best or result.score > best[key].score:
            best[key] = result
    ranked = sorted(
        (r for r in best.values() if not r.excluded),
        key=lambda r: r.score, reverse=True,
    )[:max(0, top_k)]
    return [r.to_dict() for r in ranked]


def parse_filters(raw: str | Mapping[str, Any] | None) -> dict[str, AttributeFilter]:
    """Parse a JSON string (from the API/UI) into attribute filters."""
    if raw is None or raw == "":
        return {}
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        raise ValueError("filters must be a JSON object")
    return _coerce_filters(data)
