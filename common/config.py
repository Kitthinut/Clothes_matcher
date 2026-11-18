"""Configuration loading with dot-path access and derived weights names."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from common.paths import project_path


@dataclass(frozen=True)
class Config:
    """Validated view of ``config.yaml``."""

    raw: Mapping[str, Any]
    root: Path

    @property
    def project(self) -> str:
        return str(self.get("project", default="Raja-Fashion"))

    @property
    def model_family(self) -> str:
        """The single chosen weights name, e.g. ``yolo26n``."""
        return str(self.get("model.family"))

    @property
    def detect_weights(self) -> str:
        """Detection weights name derived from :attr:`model_family`."""
        return f"{self.model_family}.pt"

    @property
    def cls_weights(self) -> str:
        """Classification weights name derived from :attr:`model_family`."""
        return f"{self.model_family}-cls.pt"

    @property
    def imgsz(self) -> int:
        return int(self.get("model.imgsz", default=640))

    @property
    def device(self) -> str:
        return str(self.get("model.device", default="auto"))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a value using dot notation, e.g. ``get("train.batch")``."""
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted_key: str) -> Path:
        """Resolve a configured project-relative path to an absolute path."""
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Unknown path key: {dotted_key}")
        p = Path(str(value))
        return p if p.is_absolute() else self.root / p

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable deep copy of the raw config."""
        import copy

        return copy.deepcopy(dict(self.raw))


@lru_cache(maxsize=4)
def _load_cached(raw_path: str) -> Config:
    path = Path(raw_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    if "model" not in data or "family" not in data["model"]:
        raise ValueError("config.yaml must define model.family")
    return Config(data, project_path())


def load_config(path: str | Path | None = None, *, reload: bool = False) -> Config:
    """Load and cache ``config.yaml``.

    Args:
        path: Optional alternate config file. Defaults to the project-root one.
        reload: Clear the cache first.
    """
    resolved = Path(path) if path is not None else project_path("config.yaml")
    if reload:
        _load_cached.cache_clear()
    return _load_cached(str(resolved.resolve()))
