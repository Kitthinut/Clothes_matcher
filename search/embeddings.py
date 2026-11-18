"""Swappable visual embedding backends.

Default: the penultimate (pooled) layer of a trained YOLO classifier — no extra
dependencies beyond Ultralytics. A CLIP-style backend and a deterministic mock
backend are also available; every backend exposes the same tiny interface.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common.config import Config, load_config
from common.logging_conf import LOGGER

LOG = LOGGER.getChild("search.embeddings")


def _normalise(vector: np.ndarray) -> list[float]:
    arr = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(arr))
    if norm == 0 or not math.isfinite(norm):
        return arr.tolist()
    return (arr / norm).tolist()


class EmbeddingBackend(ABC):
    """Interface every embedding backend must implement."""

    name: str = "base"

    @abstractmethod
    def extract(self, crop: Image.Image) -> list[float]:
        """Return a normalised embedding vector for a garment crop."""

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimensionality."""


class MockEmbedding(EmbeddingBackend):
    """Deterministic, dependency-free embeddings (average colour + thumbnail).

    Similar-looking synthetic crops produce similar vectors, which makes the
    whole search pipeline smoke-testable without any downloaded weights.
    """

    name = "mock"

    def __init__(self, size: int = 8) -> None:
        self.size = size

    def extract(self, crop: Image.Image) -> list[float]:
        image = crop.convert("RGB")
        thumb = np.asarray(image.resize((self.size, self.size)), dtype=np.float32) / 255.0
        mean_rgb = thumb.mean(axis=(0, 1))
        grey = thumb.mean(axis=2).flatten()
        return _normalise(np.concatenate([mean_rgb, grey]))

    def dimension(self) -> int:
        return 3 + self.size * self.size


class YOLOClassifierEmbedding(EmbeddingBackend):
    """Embedding from the pooled penultimate layer of a YOLO classifier."""

    name = "yolo"

    def __init__(self, weights_path: str | Path) -> None:
        import torch  # noqa: F401  (required by Ultralytics internals)

        from ultralytics import YOLO

        self._model = YOLO(str(weights_path))
        self._dim: int | None = None
        # The classification head ends with Classify(AdaptiveAvgPool2d, Linear).
        # Hooking the pool yields a compact, transferable feature vector.
        self._layer = None
        for module in self._model.model.model.modules():
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):
                self._layer = module
        if self._layer is None:
            raise RuntimeError(f"No AdaptiveAvgPool2d layer found in {weights_path}")

    def extract(self, crop: Image.Image) -> list[float]:
        import torch

        captured: list[np.ndarray] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            captured.append(tensor.detach().cpu().numpy())

        handle = self._layer.register_forward_hook(hook)
        try:
            self._model(crop, verbose=False)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Embedding hook did not fire during inference")
        flat = captured[-1].reshape(captured[-1].shape[0], -1)
        self._dim = int(flat.shape[1])
        return _normalise(flat[0])

    def dimension(self) -> int:
        return self._dim or 0


class CLIPEmbedding(EmbeddingBackend):
    """Optional CLIP-style backend via ``open-clip-torch`` (not installed by default)."""

    name = "clip"

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        import open_clip  # type: ignore[import-not-found]
        import torch

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self._model.eval()
        self._torch = torch

    def extract(self, crop: Image.Image) -> list[float]:
        tensor = self._preprocess(crop.convert("RGB")).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
        return _normalise(features.numpy())

    def dimension(self) -> int:
        return int(self._model.visual.output_dim)


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Return cosine similarity in ``[-1, 1]`` (0 for empty vectors)."""
    va, vb = np.asarray(a, dtype=np.float32).flatten(), np.asarray(b, dtype=np.float32).flatten()
    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _first_classifier_weights(models_dir: Path) -> Path | None:
    """Find the first trained classifier checkpoint in the model registry."""
    registry = models_dir / "labels.json"
    if not registry.exists():
        return None
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
        classifiers = data.get("classifiers") or {}
        for entry in classifiers.values():
            weights = models_dir / entry["weights"]
            if weights.exists():
                return weights
    except (KeyError, ValueError, OSError) as exc:
        LOG.warning("Could not read model registry: %s", exc)
    return None


def get_embedding_backend(
    config: Config | None = None,
    *,
    name: str | None = None,
    models_dir: Path | None = None,
) -> EmbeddingBackend:
    """Create the requested backend, falling back to mock when unavailable.

    Args:
        config: Config override.
        name: ``yolo``, ``clip`` or ``mock``; defaults to config setting.
        models_dir: Override for the models directory (used by tests).
    """
    cfg = config or load_config()
    backend_name = (name or cfg.get("search.embedding_backend", "yolo")).lower()
    directory = Path(models_dir) if models_dir is not None else cfg.path("paths.models")

    if backend_name == "mock":
        return MockEmbedding()
    if backend_name == "yolo":
        weights = _first_classifier_weights(directory)
        if weights is None:
            LOG.warning("No trained classifier checkpoint found; using mock embeddings")
            return MockEmbedding()
        try:
            return YOLOClassifierEmbedding(weights)
        except Exception as exc:  # pragma: no cover - depends on installed weights
            LOG.warning("Could not load YOLO embedding backend (%s); using mock", exc)
            return MockEmbedding()
    if backend_name == "clip":
        try:
            return CLIPEmbedding()
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.warning("Could not load CLIP backend (%s); using mock", exc)
            return MockEmbedding()
    raise ValueError(f"Unknown embedding backend: {backend_name}")
