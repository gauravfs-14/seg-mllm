from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentationInstance:
    """One predicted instance from open-vocabulary segmentation."""

    xy: dict[str, float]
    hw: dict[str, float]
    mask: np.ndarray  # H x W bool


@dataclass(frozen=True)
class SegmentationResult:
    instances: tuple[SegmentationInstance, ...]
    query: str


@runtime_checkable
class PerceptionClient(Protocol):
    """Abstraction for dense perception / segmentation tools."""

    def segment(self, image: Image.Image, query: str) -> SegmentationResult:
        ...
