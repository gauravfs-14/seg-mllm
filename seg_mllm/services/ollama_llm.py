from __future__ import annotations

import base64
import io
from typing import Any

import ollama
from PIL import Image

from seg_mllm.config import Settings
from seg_mllm.contracts.llm import ThinkOption


def build_ollama_options(settings: Settings) -> dict[str, Any]:
    """Map ``Settings`` fields to Ollama ``options`` for chat/generate APIs."""
    opts: dict[str, Any] = {
        "temperature": settings.ollama_temperature,
        "top_p": settings.ollama_top_p,
        "top_k": settings.ollama_top_k,
        "num_ctx": settings.ollama_num_ctx,
        "repeat_penalty": settings.ollama_repeat_penalty,
    }
    if settings.ollama_num_predict > 0:
        opts["num_predict"] = settings.ollama_num_predict
    if settings.ollama_seed is not None:
        opts["seed"] = settings.ollama_seed
    return opts


class OllamaLLMClient:
    """Ollama-backed chat completion with optional vision (dependency on host service)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = ollama.Client(host=settings.ollama_host)

    def complete(
        self,
        system_prompt: str,
        user_text: str,
        images: list[Image.Image] | None = None,
        *,
        json_mode: bool = False,
        think: ThinkOption = None,
    ) -> str:
        messages: list[dict] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        user_msg: dict = {"role": "user", "content": user_text}
        if images:
            user_msg["images"] = [_pil_to_b64(im) for im in images]
        messages.append(user_msg)
        kwargs: dict = {}
        if json_mode:
            kwargs["format"] = "json"
        t = _normalize_think(think)
        if t is not None:
            kwargs["think"] = t
        response = self._client.chat(
            model=self._settings.ollama_model,
            messages=messages,
            options=build_ollama_options(self._settings),
            **kwargs,
        )
        content = response.message.content
        if content is None:
            return ""
        return content


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def normalize_think(think: ThinkOption):
    """Map app reasoning settings to Ollama's ``think`` argument."""
    if think is None or think == "" or think == "default":
        return None
    if think is True:
        return True
    if think is False:
        return False
    if isinstance(think, str):
        s = think.strip().lower()
        if s in ("default", "off", "none"):
            return None
        if s in ("full", "on", "true", "yes"):
            return True
        if s in ("low", "medium", "high"):
            return s
    return think


def _normalize_think(think: ThinkOption):
    return normalize_think(think)
