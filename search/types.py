"""Serializable result types shared by analyze, ingest, scoring and the APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class AttributePrediction:
    """One attribute prediction: winning label + full class-probability map."""

    label: str
    probs: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttributePrediction":
        return cls(label=str(data["label"]), probs={str(k): float(v) for k, v in (data.get("probs") or {}).items()})

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "probs": self.probs}


@dataclass
class GarmentResult:
    """One detected garment with all attribute predictions and an embedding."""

    type: str
    box: list[int]  # xyxy pixel coordinates
    attributes: dict[str, AttributePrediction] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GarmentResult":
        return cls(
            type=str(data["type"]),
            box=[int(v) for v in data["box"]],
            attributes={str(k): AttributePrediction.from_dict(v) for k, v in (data.get("attributes") or {}).items()},
            embedding=[float(v) for v in (data.get("embedding") or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "box": self.box,
            "attributes": {k: v.to_dict() for k, v in self.attributes.items()},
            "embedding": self.embedding,
        }


@dataclass
class AnalysisResult:
    """Result of :meth:`search.analyze.Analyzer.analyze`."""

    image_path: str
    garments: list[GarmentResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisResult":
        return cls(
            image_path=str(data.get("image_path", "")),
            garments=[GarmentResult.from_dict(g) for g in data.get("garments", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        # The public shape matches the requested analyze() contract.
        return {"garments": [g.to_dict() for g in self.garments]}

    def to_stored_dict(self) -> dict[str, Any]:
        """Include the image path for persistence/metadata."""
        return {"image_path": self.image_path, **self.to_dict()}
