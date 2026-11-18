"""Unit tests for the attribute scoring function and top-K search."""

from __future__ import annotations

from search.db import CatalogGarment
from search.scoring import AttributeFilter, parse_filters, score_garment, search_catalog
from search.types import AnalysisResult, AttributePrediction, GarmentResult


def _garment(garment_type: str = "top", *, attrs: dict, embedding: list[float] | None = None) -> GarmentResult:
    return GarmentResult(
        type=garment_type,
        box=[0, 0, 10, 10],
        attributes={
            name: AttributePrediction(label=label, probs=probs)
            for name, (label, probs) in attrs.items()
        },
        embedding=embedding or [1.0, 0.0],
    )


REFERENCE = _garment(
    attrs={
        "length": ("regular", {"cropped": 0.2, "regular": 0.8}),
        "pattern": ("plain", {"plain": 0.7, "stripe": 0.3}),
    }
)


def test_soft_match_uses_reference_probability_for_candidate_label() -> None:
    candidate = _garment(attrs={"length": ("cropped", {"cropped": 1.0})})
    result = score_garment(REFERENCE, candidate)
    assert not result.excluded
    assert result.explanation[0].match == 0.2  # P_ref("cropped")
    assert abs(result.score - 0.2) < 1e-6
    assert result.explanation[0].note.startswith("soft match")


def test_weighted_average() -> None:
    candidate = _garment(attrs={
        "length": ("cropped", {"cropped": 1.0}),   # match 0.2
        "pattern": ("stripe", {"stripe": 1.0}),    # match 0.3
    })
    filters = {"length": AttributeFilter(weight=3.0), "pattern": AttributeFilter(weight=1.0)}
    result = score_garment(REFERENCE, candidate, filters)
    expected = (3 * 0.2 + 1 * 0.3) / 4
    assert abs(result.score - expected) < 1e-6


def test_ignore_and_zero_weight_are_skipped() -> None:
    candidate = _garment(attrs={
        "length": ("cropped", {}),
        "pattern": ("stripe", {}),
    })
    filters = {
        "length": AttributeFilter(mode="ignore", weight=5.0),
        "pattern": AttributeFilter(weight=0.0),
    }
    result = score_garment(REFERENCE, candidate, filters)
    assert result.base_score == 0.0
    assert all(m.contribution == 0.0 for m in result.explanation)


def test_lock_is_a_hard_filter() -> None:
    candidate = _garment(attrs={"length": ("cropped", {"cropped": 1.0})})
    filters = {"length": AttributeFilter(mode="lock")}
    excluded = score_garment(REFERENCE, candidate, filters)
    assert excluded.excluded and excluded.score == 0.0
    assert "locked length" in excluded.reason

    matching = _garment(attrs={"length": ("regular", {"regular": 1.0})})
    included = score_garment(REFERENCE, matching, filters)
    assert not included.excluded
    assert included.explanation[0].match == 1.0

    explicit = score_garment(REFERENCE, candidate, {"length": AttributeFilter(mode="lock", value="cropped")})
    assert not explicit.excluded


def test_embedding_term_blends_scores() -> None:
    candidate = _garment(
        attrs={"length": ("unknown_label", {})},  # label absent from reference probs -> base = 0
        embedding=[1.0, 0.0],
    )
    result = score_garment(REFERENCE, candidate, embedding_weight=1.0)
    assert abs(result.embedding_similarity - 1.0) < 1e-6
    assert abs(result.score - 0.5) < 1e-6  # (0 + 1*1)/(1+1)


def test_garment_type_filter() -> None:
    bottom = _garment("bottom", attrs={"length": ("regular", {"regular": 1.0})})
    result = score_garment(REFERENCE, bottom, garment_type="top")
    assert result.excluded and result.reason == "type != top"


def test_search_catalog_ranks_and_limits_top_k() -> None:
    reference = AnalysisResult("q.jpg", [REFERENCE])
    candidates = [
        CatalogGarment(_garment(attrs={"length": ("cropped", {})}), {"id": "a"}),
        CatalogGarment(_garment(attrs={"length": ("regular", {})}), {"id": "b"}),
        CatalogGarment(_garment("bottom", attrs={"waist_length": ("high", {})}), {"id": "c"}),
    ]
    results = search_catalog(candidates, reference, top_k=2)
    assert len(results) == 2
    assert results[0]["product"]["id"] == "b"
    assert results[0]["score"] >= results[1]["score"]
    assert all(r["product"]["id"] != "c" for r in results)


def test_parse_filters_json() -> None:
    parsed = parse_filters('{"neck": {"mode": "lock", "weight": 4, "value": "crew"}}')
    filt = parsed["neck"]
    assert filt.mode == "lock" and filt.weight == 4 and filt.value == "crew"
