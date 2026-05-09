from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import ollama
from ollama import ShowResponse


def discover_vision_and_tool_models(host: str) -> list[str]:
    """
    Installed models that report **both** ``vision`` and ``tools`` on ``/api/show`` (capabilities).

    Ollama lists these when the model supports multimodal input and function/tool calling.
    """
    client = ollama.Client(host=host)
    try:
        listed = client.list()
    except Exception:
        return []

    names: list[str] = []
    for entry in listed.models:
        name = entry.model
        if not name:
            continue
        try:
            info = client.show(model=name)
        except Exception:
            continue
        if _show_indicates_vision_and_tools(info):
            names.append(name)

    return sorted(set(names))


def _show_indicates_vision_and_tools(info: ShowResponse) -> bool:
    caps = info.capabilities
    if caps is None:
        return False
    cset = {str(c).lower() for c in caps}
    return "vision" in cset and "tools" in cset


def discover_vision_capable_models(host: str) -> list[str]:
    """
    List installed Ollama models that report vision support.

    Uses ``/api/tags`` for names, then ``/api/show`` per model (tags do not include capabilities).
    """
    client = ollama.Client(host=host)
    try:
        listed = client.list()
    except Exception:
        return []

    names: list[str] = []
    for entry in listed.models:
        name = entry.model
        if not name:
            continue
        try:
            info = client.show(model=name)
        except Exception:
            continue
        if _show_indicates_vision(info):
            names.append(name)

    return sorted(set(names))


def _show_indicates_vision(info: ShowResponse) -> bool:
    caps = info.capabilities
    if caps is not None and "vision" in caps:
        return True
    if caps is not None and len(caps) > 0:
        return False
    return _modelinfo_suggests_vision(info.modelinfo)


def _modelinfo_suggests_vision(modelinfo: Mapping[str, Any] | None) -> bool:
    if not modelinfo:
        return False
    hints = ("vision", "mmproj", "image_encoder", "clip", "siglip", "multimodal", "image.projector")
    for key in modelinfo:
        kl = str(key).lower()
        if any(h in kl for h in hints):
            return True
    return False
