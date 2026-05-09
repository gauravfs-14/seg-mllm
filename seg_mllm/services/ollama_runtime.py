from __future__ import annotations

import shutil

import ollama

from seg_mllm.config import Settings


def ollama_cli_installed() -> bool:
    """True if the ``ollama`` executable is on ``PATH`` (install hint for users)."""
    return shutil.which("ollama") is not None


def ollama_daemon_reachable(settings: Settings) -> bool:
    """True if a server responds at ``settings.ollama_host``."""
    try:
        client = ollama.Client(host=settings.ollama_host)
        client.list()
        return True
    except Exception:
        return False
