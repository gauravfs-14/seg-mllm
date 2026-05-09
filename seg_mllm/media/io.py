from __future__ import annotations

import io
import os
import tempfile

import cv2
from PIL import Image


class MediaLoadError(RuntimeError):
    pass


def load_image_from_upload(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive for arbitrary uploads
        raise MediaLoadError(f"Could not decode image: {exc}") from exc


def video_first_frame_from_upload(data: bytes, suffix: str = ".mp4") -> Image.Image:
    """Extract the first decodable frame as RGB for VLM / segmentation pipelines."""
    if not data:
        raise MediaLoadError("Empty video upload.")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise MediaLoadError(
                "OpenCV could not open the uploaded video. "
                "If this is an .mp4/.mov from a phone, it may use a codec your OpenCV build can't decode. "
                "Try re-encoding to H.264/AAC (mp4) or upload a different format."
            )
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise MediaLoadError("Could not read a frame from the video.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.unlink(tmp_path)


def video_sample_frames_from_upload(
    data: bytes,
    *,
    suffix: str = ".mp4",
    sample_fps: float = 1.0,
    max_frames: int = 60,
) -> list[Image.Image]:
    """Sample frames at ~sample_fps (default 1fps) across the video (RGB PIL images)."""
    if not data:
        raise MediaLoadError("Empty video upload.")
    fps = float(sample_fps)
    if fps <= 0:
        fps = 1.0
    cap_max = int(max_frames)
    if cap_max <= 0:
        cap_max = 60
    cap_max = min(cap_max, 600)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise MediaLoadError(
                "OpenCV could not open the uploaded video. "
                "Try re-encoding to H.264/AAC (mp4) or upload a different format."
            )

        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        duration_s = (frame_count / native_fps) if (frame_count > 0 and native_fps > 0) else 0.0

        # Sample one frame per second (or per 1/fps) by seeking via milliseconds.
        step_ms = max(int(round(1000.0 / fps)), 1)
        timestamps_ms: list[int] = []
        if duration_s > 0:
            total_ms = int(duration_s * 1000.0)
            t = 0
            while t <= total_ms and len(timestamps_ms) < cap_max:
                timestamps_ms.append(t)
                t += step_ms
            if not timestamps_ms:
                timestamps_ms = [0]
        else:
            # Unknown duration: fall back to reading sequential frames.
            timestamps_ms = []

        frames: list[Image.Image] = []
        if timestamps_ms:
            for t_ms in timestamps_ms:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t_ms))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        else:
            while len(frames) < cap_max:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

        cap.release()
        if not frames:
            raise MediaLoadError("Could not read any frames from the video.")
        return frames
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.unlink(tmp_path)


def sniff_media_kind(filename: str | None, data: bytes) -> str:
    """
    STRICT mode: do not guess from bytes.

    We only classify based on file extension. If we can't classify, return "unknown".
    """
    del data
    name = (filename or "").lower()
    if any(name.endswith(ext) for ext in (".mp4", ".webm", ".mov", ".mkv", ".avi")):
        return "video"
    if any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
        return "image"
    return "unknown"
