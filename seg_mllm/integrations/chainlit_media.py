from __future__ import annotations

import base64
import os
from typing import Any

from PIL import Image

from seg_mllm.media import (
    MediaLoadError,
    load_image_from_upload,
    sniff_media_kind,
    video_first_frame_from_upload,
    video_sample_frames_from_upload,
)


def _read_element_bytes(el: Any) -> bytes | None:
    path = getattr(el, "path", None)
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    content = getattr(el, "content", None)
    if content is None:
        return None
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(content, str):
        s = content.strip()
        # Chainlit/web may send data URLs: "data:video/mp4;base64,AAAA..."
        if "base64," in s[:64]:
            s = s.split("base64,", 1)[-1].strip()
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return None
    return None


def frame_from_message_elements(elements: list[Any] | None) -> tuple[Image.Image | None, str | None]:
    """
    Build a single RGB frame from the first usable image or video attachment.

    Returns ``(frame, error)``. ``error`` is set when bytes were present but decoding failed.
    """
    if not elements:
        return None, None

    for el in elements:
        name = getattr(el, "name", None) or ""
        mime = (getattr(el, "mime", None) or "").lower()
        data = _read_element_bytes(el)
        if not data:
            continue
        kind = sniff_media_kind(str(name) if name else None, data)
        try:
            if kind == "video" or "video/" in mime:
                suffix = ""
                if name and "." in name:
                    suffix = "." + str(name).rsplit(".", 1)[-1].lower()
                elif "video/" in mime:
                    suffix = "." + mime.split("/", 1)[-1].split(";", 1)[0].strip().lower()
                return video_first_frame_from_upload(data, suffix=suffix or ".mp4"), None
            if kind == "image" or "image/" in mime:
                return load_image_from_upload(data), None
            return None, "Unsupported attachment type (only image/* and video/* are supported)."
        except MediaLoadError as exc:
            return None, str(exc)

    return None, None


def frames_from_message_elements(
    elements: list[Any] | None,
    *,
    video_sample_fps: float = 1.0,
    video_max_frames: int = 60,
) -> tuple[list[Image.Image] | None, str | None]:
    """
    Build a list of RGB frames from the first usable image or video attachment.

    - image: returns [image]
    - video: returns sampled frames in time order
    """
    if not elements:
        return None, None

    for el in elements:
        name = getattr(el, "name", None) or ""
        mime = (getattr(el, "mime", None) or "").lower()
        data = _read_element_bytes(el)
        if not data:
            if "video/" in mime or str(name).lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")):
                return (
                    None,
                    "No video bytes were attached. This is usually an unsupported video format/codec in the browser UI. "
                    "Try an H.264/AAC `.mp4` or `.webm` and re-upload.",
                )
            continue
        kind = sniff_media_kind(str(name) if name else None, data)
        try:
            if kind == "video" or "video/" in mime:
                suffix = ""
                if name and "." in name:
                    suffix = "." + str(name).rsplit(".", 1)[-1].lower()
                elif "video/" in mime:
                    suffix = "." + mime.split("/", 1)[-1].split(";", 1)[0].strip().lower()
                frames = video_sample_frames_from_upload(
                    data,
                    suffix=suffix or ".mp4",
                    sample_fps=video_sample_fps,
                    max_frames=video_max_frames,
                )
                return frames, None
            if kind == "image" or "image/" in mime:
                return [load_image_from_upload(data)], None
            return None, "Unsupported attachment type (only image/* and video/* are supported)."
        except MediaLoadError as exc:
            return None, str(exc)

    return None, None

