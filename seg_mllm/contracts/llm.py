from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

ThinkOption = bool | str | None


@runtime_checkable
class LLMClient(Protocol):
    """Abstraction for text (+ optional vision) chat completion."""

    def complete(
        self,
        system_prompt: str,
        user_text: str,
        images: list[Image.Image] | None = None,
        *,
        json_mode: bool = False,
        think: ThinkOption = None,
    ) -> str:
        """Return assistant text. When ``json_mode`` is True, text should be valid JSON."""
        ...
