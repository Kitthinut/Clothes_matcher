"""Two-stage inference for schema v2 region detections.

Stage 1 predicts region boxes with their class names (e.g.
``top_neck_crew``). Stage 2 optionally refines each region's probabilities
with a crop classifier. Boxes are grouped into garments; an image may contain
only a top, only a bottom, or both.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from common.colors import dominant_color_name
from common.config import Config, load_config
from common.logging_conf import LOGGER
from common.schema import Region, Schema, load_schema
from search.embeddings import EmbeddingBackend, get_embedding_backend
from search.types import AnalysisResult, AttributePrediction, GarmentResult

LOG = LOGGER.getChild("search.analyze")


@dataclass
class _Detection:
    """Internal region detection with hinted attribute value."""

    garment: str
    region: str
    side: str
    value: str
    xyxy: list[int]
    confidence: float = 1.0


class MockDetector:
    """Reads ground-truth region boxes from annotation JSON when present."""

    def __init__(self, annotations_dir: Path, schema: Schema) -> None:
        self.annotations_dir = Path(annotations_dir)
        self.schema = schema

    def detect(self, image_path: Path, image: Image.Image) -> list[_Detection]:
        from annotator.store import AnnotationStore

        store = AnnotationStore(image_path.parent, self.annotations_dir, self.schema)
        if store.has_annotation(image_path):
            annotation = store.load(image_path)
            detections = []
            for box in annotation.boxes:
                if box.garment not in self.schema.garment_names or box.region == "garment":
                    continue
                detections.append(_Detection(
                    box.garment, box.region, box.side, box.label, list(box.xyxy)
                ))
            return detections
        # Deterministic fallback: a whole-garment box for the 'top'.
        w, h = image.size
        return [_Detection("top", "garment", "center", "top",
                           [int(w * 0.25), int(h * 0.2), int(w * 0.75), int(h * 0.7)])]


class YOLODetector:
    """Stage-1 region detector backed by a trained YOLO checkpoint."""

    def __init__(self, weights: str | Path, config: Config, schema: Schema) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.config = config
        self.schema = schema

    def detect(self, image_path: Path, image: Image.Image) -> list[_Detection]:
        results = self.model(
            image,
            conf=float(self.config.get("model.conf", 0.15)),
            iou=float(self.config.get("model.iou", 0.5)),
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        detections: list[_Detection] = []
        if not results:
            return detections
        names = results[0].names or {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            class_name = str(names.get(cls_id, ""))
            parsed = self.schema.parse_detection_name(class_name)
            if parsed is None:
                continue
            garment, region, side, value = parsed
            xyxy = [int(round(float(v))) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 1.0
            detections.append(_Detection(garment, region, side, value, xyxy, conf))
        return detections


class MockClassifier:
    """Attribute classifier using the detection value or a stable image hash."""

    def predict(self, crop: Image.Image, region: Region, hint: str | None) -> AttributePrediction:
        classes = list(region.classes)
        value = hint if hint in classes else classes[hashlib.md5(crop.tobytes()).digest()[0] % len(classes)]
        others = [c for c in classes if c != value]
        probs = {c: (1.0 - 0.85) / max(1, len(others)) for c in others}
        probs[value] = 0.85
        return AttributePrediction(label=value, probs=probs)


class YOLOAttributeClassifier:
    """Stage-2 classifier for one region, backed by YOLO cls weights."""

    def __init__(self, weights: str | Path, region: Region) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.region = region

    def predict(self, crop: Image.Image, region: Region, hint: str | None = None) -> AttributePrediction:
        results = self.model(crop, verbose=False)
        if not results or results[0].probs is None:
            raise RuntimeError(f"No probabilities for {region.key}")
        tensor = results[0].probs.data
        values = [float(v) for v in tensor] if hasattr(tensor, "__iter__") else [float(tensor)]
        names = self.model.names or {}
        by_name = {str(label): values[idx] for idx, label in names.items() if idx < len(values)}
        probs = {label: float(by_name.get(label, 0.0)) for label in region.classes}
        if not any(probs.values()):
            ordered = sorted(region.classes)
            probs = {label: float(values[ordered.index(label)]) for label in region.classes}
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        return AttributePrediction(label=max(probs, key=probs.get), probs=probs)


class Analyzer:
    """Analyse an image into garment results with region attributes."""

    def __init__(
        self,
        config: Config | None = None,
        schema: Schema | None = None,
        *,
        mode: str = "auto",
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.config = config or load_config()
        self.schema = schema or load_schema()
        self.mode = mode
        models_dir = self.config.path("paths.models")
        detect_weights = models_dir / "detect" / "best.pt"
        registry = models_dir / "labels.json"
        cls_weights: dict[str, Path] = {}
        if registry.exists():
            try:
                data = json.loads(registry.read_text(encoding="utf-8"))
                for key, entry in (data.get("classifiers") or {}).items():
                    candidate = models_dir / entry["weights"]
                    if candidate.exists():
                        cls_weights[key] = candidate
            except (ValueError, OSError, KeyError) as exc:
                LOG.warning("Could not parse model registry: %s", exc)

        use_real = mode == "real" or (mode == "auto" and detect_weights.exists())
        if mode == "real" and not detect_weights.exists():
            LOG.warning("Real mode requested but %s missing; using mock detector", detect_weights)
            use_real = False
        self.detector: Any = (
            YOLODetector(detect_weights, self.config, self.schema) if use_real
            else MockDetector(self.config.path("paths.annotations"), self.schema)
        )
        self.classifiers: dict[str, Any] = {}
        for task in self.schema.classification_tasks():
            weights = cls_weights.get(task.key)
            self.classifiers[task.key] = (
                YOLOAttributeClassifier(weights, task) if use_real and weights else MockClassifier()
            )
        self.embeddings = embedding_backend or get_embedding_backend(self.config)

    def _as_path(self, image: str | Path | Image.Image | np.ndarray) -> tuple[Path, Image.Image]:
        if isinstance(image, Image.Image):
            tmp = Path(tempfile.gettempdir()) / "raja_fashion_upload.png"
            image.convert("RGB").save(tmp)
            return tmp, image.convert("RGB")
        if isinstance(image, np.ndarray):
            return self._as_path(Image.fromarray(image.astype(np.uint8)).convert("RGB"))
        path = Path(image)
        with Image.open(path) as src:
            return path, src.convert("RGB")

    def _probabilities(self, detection: _Detection, crop: Image.Image) -> AttributePrediction:
        if detection.region == "garment":
            return AttributePrediction(label=detection.value, probs={detection.value: detection.confidence})
        region = self.schema.region(detection.garment, detection.region)
        classifier = self.classifiers.get(region.key)
        if classifier is not None and not isinstance(classifier, MockClassifier):
            try:
                return classifier.predict(crop, region, detection.value)
            except Exception as exc:  # pragma: no cover - model dependent
                LOG.warning("Classifier %s failed: %s", region.key, exc)
        if classifier is not None:
            return classifier.predict(crop, region, detection.value)
        # Fallback: soft probabilities from detector confidence.
        others = [c for c in region.classes if c != detection.value]
        probs = {c: (1.0 - detection.confidence) / max(1, len(others)) for c in others}
        probs[detection.value] = detection.confidence
        return AttributePrediction(label=detection.value, probs=probs)

    def analyze(self, image: str | Path | Image.Image | np.ndarray) -> AnalysisResult:
        path, pil = self._as_path(image)
        detections = self.detector.detect(path, pil)
        by_garment: dict[str, list[_Detection]] = defaultdict(list)
        for detection in detections:
            if detection.garment in self.schema.garment_names:
                by_garment[detection.garment].append(detection)

        garments: list[GarmentResult] = []
        for garment, dets in by_garment.items():
            garment_boxes = [d for d in dets if d.region == "garment"]
            if garment_boxes:
                gxyxy = garment_boxes[0].xyxy
            elif dets:
                gxyxy = [
                    min(d.xyxy[0] for d in dets), min(d.xyxy[1] for d in dets),
                    max(d.xyxy[2] for d in dets), max(d.xyxy[3] for d in dets),
                ]
            else:
                continue
            x1, y1, x2, y2 = (max(0, gxyxy[0]), max(0, gxyxy[1]),
                              min(pil.width, gxyxy[2]), min(pil.height, gxyxy[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            garment_crop = pil.crop((x1, y1, x2, y2))
            attributes: dict[str, AttributePrediction] = {}

            for detection in dets:
                if detection.region == "garment":
                    continue
                dx1, dy1 = max(0, detection.xyxy[0]), max(0, detection.xyxy[1])
                dx2, dy2 = min(pil.width, detection.xyxy[2]), min(pil.height, detection.xyxy[3])
                if dx2 <= dx1 or dy2 <= dy1:
                    continue
                crop = pil.crop((dx1, dy1, dx2, dy2))
                key = detection.region
                region = self.schema.region(garment, detection.region)
                if region.paired:
                    key = f"{detection.region}_{detection.side}"
                attributes[key] = self._probabilities(detection, crop)

            # HSV colour from the whole-garment crop if no colour detection exists.
            if not any(k.startswith("color") for k in attributes):
                name = dominant_color_name(garment_crop, set(self.schema.color_classes))
                if name:
                    values = self.schema.color_classes
                    others = [c for c in values if c != name]
                    attributes["color"] = AttributePrediction(
                        label=name, probs={**{c: 0.05 / max(1, len(others)) for c in others}, name: 0.95},
                    )

            garments.append(GarmentResult(
                type=garment, box=[x1, y1, x2, y2], attributes=attributes,
                embedding=self.embeddings.extract(garment_crop),
            ))

        return AnalysisResult(image_path=str(path), garments=garments)


def analyze(image: str | Path | Image.Image | np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Convenience function returning the public ``{"garments": [...]}`` dict."""
    return Analyzer(**kwargs).analyze(image).to_dict()
