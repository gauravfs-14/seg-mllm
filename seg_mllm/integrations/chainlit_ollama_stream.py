"""Chainlit + Ollama streaming helpers (official ``Message.stream_token`` pattern)."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any, Mapping

import chainlit as cl
import ollama
from ollama import ChatResponse
from PIL import Image

from seg_mllm.services.ollama_llm import normalize_think

if TYPE_CHECKING:
    from seg_mllm.contracts.llm import ThinkOption


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _delta_from_stream_piece(seen: str, piece: str) -> tuple[str, str]:
    """Support both incremental and cumulative Ollama stream ``message.content`` chunks."""
    if not piece:
        return seen, ""
    if piece.startswith(seen):
        d = piece[len(seen) :]
        return piece, d
    return seen + piece, piece


def _stream_chunks(text: str, *, chunk_size: int = 48) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def ollama_performance_parts(
    resp: ChatResponse | None,
    *,
    model: str,
    wall_seconds: float | None = None,
) -> dict[str, str | float | int | None]:
    """Structured timing / throughput for ``Message.metadata`` (and optional compact UI text)."""
    out: dict[str, str | float | int | None] = {"model": model}
    if wall_seconds is not None:
        out["client_wall_s"] = round(wall_seconds, 3)
    if resp is None:
        return out
    if resp.prompt_eval_count is not None:
        out["prompt_tokens"] = resp.prompt_eval_count
    if resp.eval_count is not None:
        out["output_tokens"] = resp.eval_count
    if resp.total_duration is not None:
        out["server_total_s"] = round(resp.total_duration / 1e9, 3)
    if resp.eval_duration is not None and resp.eval_duration > 0 and resp.eval_count:
        gen_s = resp.eval_duration / 1e9
        out["gen_s"] = round(gen_s, 3)
        out["tok_per_s"] = round(resp.eval_count / gen_s, 2)
    if resp.done_reason:
        out["done"] = resp.done_reason
    return out


def format_ollama_performance_caption(parts: dict[str, str | float | int | None]) -> str:
    """Single-line caption (e.g. italic footnote) for generation rate and latency."""
    bits: list[str] = []
    if parts.get("tok_per_s") is not None:
        bits.append(f"{parts['tok_per_s']} tok/s")
    if parts.get("server_total_s") is not None:
        bits.append(f"{parts['server_total_s']}s server")
    elif parts.get("client_wall_s") is not None:
        bits.append(f"{parts['client_wall_s']}s wall")
    if parts.get("output_tokens") is not None:
        bits.append(f"{int(parts['output_tokens'])} tok out")
    if parts.get("note") == "direct_agent":
        bits.append("direct")
    if parts.get("note") == "agentic" and parts.get("agentic_rounds") is not None:
        bits.append(f"{int(parts['agentic_rounds'])} agent rounds")
    return " · ".join(bits) if bits else ""


async def stream_ollama_chat_to_message(
    *,
    host: str,
    model: str,
    system_prompt: str,
    user_text: str,
    images: list[Image.Image] | None,
    think: ThinkOption,
    options: Mapping[str, Any] | None = None,
) -> tuple[ChatResponse | None, cl.Message, None]:
    """
    Stream an Ollama completion using Chainlit **Step** lifecycle.

    When ``think`` is enabled, only a **Model reasoning** step (``type="llm"``,
    ``auto_collapse=True``) wraps extended thinking — entered via ``__aenter__`` so
    ``parent_id`` attaches to the active handler run. It is closed before answer tokens.

    The assistant **Message** uses the default parent from ``local_steps`` after reasoning
    exits, so it stays a sibling under the same run (no extra wrapper step).

    See Chainlit streaming: https://docs.chainlit.io/advanced-features/streaming
    """
    client = ollama.AsyncClient(host=host)
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    user_msg: dict = {"role": "user", "content": user_text}
    if images:
        user_msg["images"] = [_pil_to_b64(im) for im in images]
    messages.append(user_msg)

    kwargs: dict = {}
    t_norm = normalize_think(think)
    if t_norm is not None:
        kwargs["think"] = t_norm
    if options:
        kwargs["options"] = dict(options)

    want_reasoning_ui = t_norm is not None

    stream = await client.chat(model=model, messages=messages, stream=True, **kwargs)

    if want_reasoning_ui:
        seen_content = ""
        seen_thinking = ""
        last: ChatResponse | None = None
        msg: cl.Message | None = None
        thinking_seen = False
        early_answer_buf = ""
        reasoning_ctx: cl.Step | None = None

        async def close_reasoning() -> None:
            nonlocal reasoning_ctx
            if reasoning_ctx is not None:
                await reasoning_ctx.__aexit__(None, None, None)
                reasoning_ctx = None

        async for chunk in stream:
            last = chunk
            m = chunk.message
            if not m:
                continue

            t_piece = m.thinking or ""
            seen_thinking, t_delta = _delta_from_stream_piece(seen_thinking, t_piece)

            c_piece = m.content or ""
            seen_content, c_delta = _delta_from_stream_piece(seen_content, c_piece)

            if t_delta:
                thinking_seen = True
                if reasoning_ctx is None:
                    reasoning_ctx = cl.Step(
                        name="Model reasoning",
                        type="llm",
                        show_input=False,
                        default_open=False,
                        auto_collapse=True,
                    )
                    await reasoning_ctx.__aenter__()
                await reasoning_ctx.stream_token(t_delta)

            if c_delta:
                if not thinking_seen:
                    early_answer_buf += c_delta
                    continue

                await close_reasoning()

                if msg is None:
                    msg = await cl.Message(content="").send()
                if early_answer_buf:
                    for piece in _stream_chunks(early_answer_buf):
                        await msg.stream_token(piece)
                    early_answer_buf = ""
                await msg.stream_token(c_delta)

        await close_reasoning()

        if early_answer_buf and not thinking_seen:
            if msg is None:
                msg = await cl.Message(content="").send()
            for piece in _stream_chunks(early_answer_buf):
                await msg.stream_token(piece)

        if msg is None:
            msg = await cl.Message(content="").send()

        return last, msg, None

    seen_content = ""
    seen_thinking = ""
    last: ChatResponse | None = None
    msg: cl.Message | None = None

    async for chunk in stream:
        last = chunk
        m = chunk.message
        if not m:
            continue

        t_piece = m.thinking or ""
        seen_thinking, _ = _delta_from_stream_piece(seen_thinking, t_piece)

        c_piece = m.content or ""
        seen_content, c_delta = _delta_from_stream_piece(seen_content, c_piece)

        if c_delta:
            if msg is None:
                msg = await cl.Message(content="").send()
            await msg.stream_token(c_delta)

    if msg is None:
        msg = await cl.Message(content="").send()

    return last, msg, None
