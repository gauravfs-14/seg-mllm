from __future__ import annotations

import numpy as np
from PIL import Image

from seg_mllm.contracts.perception import SegmentationInstance


def render_instance_overlay(
    base: Image.Image,
    instances: tuple[SegmentationInstance, ...],
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend semi-transparent colors over each instance mask for VLM follow-up."""
    arr = np.asarray(base.convert("RGB"), dtype=np.float32)
    out = arr.copy()
    rng = np.random.default_rng(42)
    for inst in instances:
        color = rng.integers(40, 255, size=3, dtype=np.int64).astype(np.float32)
        m = inst.mask
        if not np.any(m):
            continue
        for c in range(3):
            channel = out[:, :, c]
            channel[m] = channel[m] * (1.0 - alpha) + color[c] * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
