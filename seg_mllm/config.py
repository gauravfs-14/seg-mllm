from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def _env_seed() -> int | None:
    raw = (os.environ.get("OLLAMA_SEED") or "").strip()
    if not raw:
        return None
    return int(raw, 10)


def normalize_falcon_output_mode(raw: str | None) -> str:
    """``auto`` | ``segmentation`` | ``boxes`` — used by Falcon service and chat settings."""
    if raw is None:
        return "auto"
    s = str(raw).strip().lower()
    if s in ("segmentation", "seg", "masks"):
        return "segmentation"
    if s in ("boxes", "box", "detection", "detect"):
        return "boxes"
    return "auto"


def _env_falcon_output_mode() -> str:
    return normalize_falcon_output_mode(os.environ.get("FALCON_OUTPUT_MODE"))


@dataclass(frozen=True)
class Settings:
    """Application configuration (immutable for predictable behavior in Streamlit reruns)."""

    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "gemma4:latest"))
    falcon_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "FALCON_MODEL_ID",
            "tiiuae/falcon-perception",
        ),
    )
    # If set to an existing directory with config.json + model.safetensors, loads locally (no Hub download).
    falcon_model_path: str = field(
        default_factory=lambda: (os.environ.get("FALCON_MODEL_PATH") or "").strip(),
    )
    falcon_compile: bool = field(
        default_factory=lambda: os.environ.get("FALCON_COMPILE", "1").lower() in ("1", "true", "yes"),
    )
    # ``auto``: follow checkpoint (``do_segmentation``). ``segmentation``: pixel masks when emitted; box fallback if none.
    # ``boxes``: detection decode + rectangular overlays only (ignores RLE masks).
    falcon_output_mode: str = field(default_factory=_env_falcon_output_mode)
    # Vision preprocess longest side for Falcon (lower = faster, less Metal/GPU contention on Mac).
    falcon_max_dimension: int = field(default_factory=lambda: _env_int("FALCON_MAX_DIMENSION", "768"))
    falcon_min_dimension: int = field(default_factory=lambda: _env_int("FALCON_MIN_DIMENSION", "256"))
    # Autoregressive decode cap during segmentation (lower = quicker tool return).
    falcon_max_new_tokens: int = field(default_factory=lambda: _env_int("FALCON_MAX_NEW_TOKENS", "96"))
    # After MLX segmentation, free Metal cache (helps UI recover sooner; small extra latency).
    falcon_mlx_clear_cache: bool = field(
        default_factory=lambda: os.environ.get("FALCON_MLX_CLEAR_CACHE", "1").lower() in ("1", "true", "yes"),
    )
    # Longest side when encoding overlay for Chainlit (full-res overlay is costly to PNG/JPEG encode).
    falcon_overlay_display_max_side: int = field(
        default_factory=lambda: _env_int("FALCON_OVERLAY_DISPLAY_MAX", "1280"),
    )
    # Passed to Ollama as ``options`` (see ``build_ollama_options``).
    ollama_temperature: float = field(default_factory=lambda: _env_float("OLLAMA_TEMPERATURE", "0.8"))
    ollama_top_p: float = field(default_factory=lambda: _env_float("OLLAMA_TOP_P", "0.9"))
    ollama_top_k: int = field(default_factory=lambda: _env_int("OLLAMA_TOP_K", "40"))
    ollama_num_ctx: int = field(default_factory=lambda: _env_int("OLLAMA_NUM_CTX", "8192"))
    # 0 = omit (use model/server default max generation length).
    ollama_num_predict: int = field(default_factory=lambda: _env_int("OLLAMA_NUM_PREDICT", "0"))
    ollama_repeat_penalty: float = field(default_factory=lambda: _env_float("OLLAMA_REPEAT_PENALTY", "1.1"))
    ollama_seed: int | None = field(default_factory=_env_seed)
    # How many past messages (excluding system) to keep and resend to Ollama each turn.
    chat_history_max_messages: int = field(default_factory=lambda: _env_int("CHAT_HISTORY_MAX_MESSAGES", "24"))
    # Video sampling sent to the VLM (too low misses quick events like crashes).
    video_sample_fps: float = field(default_factory=lambda: _env_float("VIDEO_SAMPLE_FPS", "4"))
    video_max_frames: int = field(default_factory=lambda: _env_int("VIDEO_MAX_FRAMES", "120"))
