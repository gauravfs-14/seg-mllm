from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from seg_mllm.config import Settings, normalize_falcon_output_mode
from seg_mllm.contracts.perception import PerceptionClient, SegmentationInstance, SegmentationResult


def _use_mlx_backend() -> bool:
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def _pair_bbox_entries(raw: list[dict]) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...] (upstream demo helper)."""
    bboxes: list[dict] = []
    current: dict = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # Some backends / variants may already return full bbox dicts.
        if all(k in entry for k in ("x", "y", "h", "w")):
            bboxes.append({"x": float(entry["x"]), "y": float(entry["y"]), "h": float(entry["h"]), "w": float(entry["w"])})
            current = {}
            continue
        current.update(entry)
        if all(k in current for k in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def _decode_rle_to_bool(rle: dict) -> np.ndarray | None:
    try:
        return mask_utils.decode(rle).astype(bool)
    except Exception:
        return None


def _mask_at_image_size(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    w, h = image.size
    if mask.shape == (h, w):
        return mask
    return np.array(
        Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST),
    ).astype(bool)


def _mask_from_normalized_box(bb: dict[str, float], *, width: int, height: int) -> np.ndarray:
    """Axis-aligned box mask from Falcon-style normalized center + size (same convention as upstream demos)."""
    cx = float(bb["x"]) * width
    cy = float(bb["y"]) * height
    bw = float(bb["w"]) * width
    bh = float(bb["h"]) * height
    x0 = int(np.clip(np.floor(cx - bw / 2), 0, width - 1))
    x1 = int(np.clip(np.ceil(cx + bw / 2), 0, width))
    y0 = int(np.clip(np.floor(cy - bh / 2), 0, height - 1))
    y1 = int(np.clip(np.ceil(cy + bh / 2), 0, height))
    mask = np.zeros((height, width), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


class FalconPerceptionService:
    """Falcon-Perception via the official package (MLX on Apple Silicon, PyTorch elsewhere)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mlx: bool = _use_mlx_backend()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._model_args: Any | None = None
        self._engine: Any | None = None
        self._process_batch_and_generate: Any | None = None

    def _resolve_load_kwargs(self) -> tuple[str | None, str | None]:
        raw = (self._settings.falcon_model_path or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if not p.is_dir():
                msg = f"Falcon local model path is not a directory: {raw}"
                raise FileNotFoundError(msg)
            return None, str(p.resolve())
        return self._settings.falcon_model_id, None

    def _ensure_model(self) -> None:
        if self._engine is not None:
            return

        from falcon_perception import load_and_prepare_model

        hf_model_id, hf_local_dir = self._resolve_load_kwargs()

        if self._mlx:
            from falcon_perception.mlx.batch_inference import (
                BatchInferenceEngine,
                process_batch_and_generate,
            )

            model, tokenizer, model_args = load_and_prepare_model(
                hf_model_id=hf_model_id,
                hf_local_dir=hf_local_dir,
                backend="mlx",
                dtype="float16",
                compile=False,
            )
            self._process_batch_and_generate = process_batch_and_generate
            self._engine = BatchInferenceEngine(model, tokenizer)
        else:
            import torch  # type: ignore[import-not-found]
            from falcon_perception.batch_inference import (
                BatchInferenceEngine,
                process_batch_and_generate,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer, model_args = load_and_prepare_model(
                hf_model_id=hf_model_id,
                hf_local_dir=hf_local_dir,
                device=device,
                dtype="float32",
                compile=self._settings.falcon_compile,
                backend="torch",
            )
            self._process_batch_and_generate = process_batch_and_generate
            self._engine = BatchInferenceEngine(model, tokenizer)

        self._model = self._engine.model
        self._tokenizer = tokenizer
        self._model_args = model_args

    def segment(self, image: Image.Image, query: str) -> SegmentationResult:
        from falcon_perception import build_prompt_for_task

        self._ensure_model()
        assert self._tokenizer is not None and self._model_args is not None and self._engine is not None
        assert self._process_batch_and_generate is not None

        rgb = image.convert("RGB")
        w_i, h_i = rgb.size

        mode = normalize_falcon_output_mode(self._settings.falcon_output_mode)
        model_do_seg = bool(getattr(self._model_args, "do_segmentation", True))

        if mode == "auto":
            falcon_task = "segmentation" if model_do_seg else "detection"
        elif mode == "segmentation":
            falcon_task = "segmentation"
        else:
            falcon_task = "detection"

        prompt = build_prompt_for_task(query, falcon_task)
        max_dim = max(256, min(1024, self._settings.falcon_max_dimension))
        min_dim = max(128, min(max_dim, self._settings.falcon_min_dimension))
        max_tok = max(32, min(512, self._settings.falcon_max_new_tokens))

        batch = self._process_batch_and_generate(
            self._tokenizer,
            [(rgb, prompt)],
            max_length=self._model_args.max_seq_len,
            min_dimension=min_dim,
            max_dimension=max_dim,
        )

        if self._mlx:
            _, aux_outputs = self._engine.generate(
                tokens=batch["tokens"],
                pos_t=batch["pos_t"],
                pos_hw=batch["pos_hw"],
                pixel_values=batch["pixel_values"],
                pixel_mask=batch["pixel_mask"],
                max_new_tokens=max_tok,
                temperature=0.0,
                task=falcon_task,
            )
            if self._settings.falcon_mlx_clear_cache:
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:
                    pass
        else:
            dev = next(self._model.parameters()).device
            dt = self._model.dtype
            batch = {
                "tokens": batch["tokens"].to(dev),
                "pos_t": batch["pos_t"].to(dev),
                "pos_hw": batch["pos_hw"].to(dev),
                "pixel_values": batch["pixel_values"].to(device=dev, dtype=dt),
                "pixel_mask": batch["pixel_mask"].to(dev),
            }
            _, aux_outputs = self._engine.generate(
                tokens=batch["tokens"],
                pos_t=batch["pos_t"],
                pos_hw=batch["pos_hw"],
                pixel_values=batch["pixel_values"],
                pixel_mask=batch["pixel_mask"],
                max_new_tokens=max_tok,
                temperature=0.0,
                task=falcon_task,
            )

        aux = aux_outputs[0]
        bboxes = _pair_bbox_entries(aux.bboxes_raw)
        instances: list[SegmentationInstance] = []

        def _instances_from_masks() -> None:
            for i, rle in enumerate(aux.masks_rle):
                mask_arr = _decode_rle_to_bool(rle)
                if mask_arr is None:
                    continue
                mask_arr = _mask_at_image_size(mask_arr, rgb)
                bb = bboxes[i] if i < len(bboxes) else {"x": 0.5, "y": 0.5, "h": 1.0, "w": 1.0}
                instances.append(
                    SegmentationInstance(
                        xy={"x": float(bb["x"]), "y": float(bb["y"])},
                        hw={"h": float(bb["h"]), "w": float(bb["w"])},
                        mask=mask_arr,
                    )
                )

        def _instances_from_boxes() -> None:
            for bb in bboxes:
                if not all(k in bb for k in ("x", "y", "h", "w")):
                    continue
                mask_arr = _mask_from_normalized_box(bb, width=w_i, height=h_i)
                instances.append(
                    SegmentationInstance(
                        xy={"x": float(bb["x"]), "y": float(bb["y"])},
                        hw={"h": float(bb["h"]), "w": float(bb["w"])},
                        mask=mask_arr,
                    )
                )

        if mode == "boxes":
            _instances_from_boxes()
        elif aux.masks_rle:
            _instances_from_masks()
        elif mode in ("auto", "segmentation") and bboxes:
            _instances_from_boxes()

        return SegmentationResult(instances=tuple(instances), query=query)
