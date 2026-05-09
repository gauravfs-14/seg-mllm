"""Ollama tool-calling loop for vision + Falcon segmentation (ReAct-style, server-side)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import ollama
from ollama import ChatResponse
from PIL import Image

from seg_mllm.config import Settings
from seg_mllm.contracts.llm import ThinkOption
from seg_mllm.contracts.perception import PerceptionClient
from seg_mllm.media.overlay import render_instance_overlay
from seg_mllm.prompts.agent_prompts import AGENTIC_VISION_SYSTEM
from seg_mllm.services.ollama_llm import build_ollama_options, normalize_think
from seg_mllm.services.task_agent import AgentStep, _format_segmentation_summary


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _last_message_is_tool(messages: list[Any]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, dict):
        return last.get("role") == "tool"
    return getattr(last, "role", None) == "tool"


def _coerce_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        return dict(json.loads(s))
    return dict(raw)


@dataclass
class _SegmentToolState:
    last_overlay: Image.Image | None = None
    last_frame_index: int | None = None


def _make_segment_tool(
    perception: PerceptionClient,
    frame: Image.Image | None,
    frames: tuple[Image.Image, ...] | None,
    state: _SegmentToolState,
) -> Callable[..., str]:
    def segment_open_vocab(query: str, frame_index: int | None = None) -> str:
        """
        Run open-vocabulary instance segmentation on the current user image using Falcon-Perception.

        Args:
            query: Short phrase for what to segment (e.g. "all pedestrians", "the red mug").
            frame_index: (Video only) which frame to segment (0-based). If omitted, uses frame 0.

        Returns:
            Text summary: instance count, normalized centers/sizes, and masked pixel counts.
        """
        q = (query or "").strip() or "objects"
        selected: Image.Image | None = frame
        if frames:
            idx = 0 if frame_index is None else int(frame_index)
            if idx < 0 or idx >= len(frames):
                return f"Error: frame_index must be in [0, {len(frames)-1}] (got {frame_index})."
            selected = frames[idx]
            state.last_frame_index = idx
        if selected is None:
            return (
                "Error: no image is available for this turn. Tell the user you cannot segment "
                "without an image, and answer any non-visual parts of their question if possible."
            )
        if len(q) > 500:
            q = q[:500] + "…"
        seg = perception.segment(selected, q)
        if seg.instances:
            state.last_overlay = render_instance_overlay(selected, seg.instances)
        else:
            state.last_overlay = None
        return _format_segmentation_summary(seg)

    return segment_open_vocab


@dataclass(frozen=True)
class AgenticVisionResult:
    answer: str
    steps: tuple[AgentStep, ...]
    overlay: Image.Image | None
    last_response: ChatResponse | None
    rounds: int
    messages: tuple[dict[str, Any], ...]


@runtime_checkable
class AgenticVisionHooks(Protocol):
    """Optional UI hooks (e.g. Chainlit steps) interleaved with the agent loop."""

    async def on_thinking(self, round: int, text: str) -> None:
        ...

    async def on_segmentation(
        self,
        round: int,
        query: str,
        frame_index: int | None,
        summary: str,
        overlay: Image.Image | None,
    ) -> None:
        ...


async def _emit_thinking(hooks: AgenticVisionHooks | None, round: int, msg: Any) -> None:
    if hooks is None:
        return
    text = (getattr(msg, "thinking", None) or "").strip()
    if text:
        await hooks.on_thinking(round, text)


async def run_agentic_vision_chat(
    *,
    settings: Settings,
    model: str,
    user_text: str,
    frame: Image.Image | None,
    vision_frames: tuple[Image.Image, ...] | None = None,
    perception: PerceptionClient,
    think: ThinkOption = None,
    system_prompt: str = AGENTIC_VISION_SYSTEM,
    max_rounds: int = 8,
    hooks: AgenticVisionHooks | None = None,
    prior_messages: list[dict[str, Any]] | None = None,
) -> AgenticVisionResult:
    """
    Multi-turn Ollama chat with ``segment_open_vocab`` tool until the model stops calling tools
    or ``max_rounds`` API calls are reached.

    When ``hooks`` is set, model thinking is emitted per round and each segmentation runs
    through ``on_segmentation`` as the tool executes (for intermediate UI steps).
    """
    seg_state = _SegmentToolState()
    segment_tool = _make_segment_tool(perception, frame, vision_frames, seg_state)
    tools: list[Any] = [segment_tool]
    client = ollama.Client(host=settings.ollama_host)

    def _msg_to_dict(m: Any) -> dict[str, Any]:
        if isinstance(m, dict):
            return dict(m)
        dump = getattr(m, "model_dump", None)
        if callable(dump):
            return dict(dump(exclude_none=True))
        # Fallback: best-effort
        out: dict[str, Any] = {}
        for k in ("role", "content", "thinking", "tool_name", "tool_calls"):
            v = getattr(m, k, None)
            if v is not None:
                out[k] = v
        return out

    # Full chat: start from prior messages (already includes system), else start fresh.
    messages: list[dict[str, Any]] = [dict(m) for m in (prior_messages or [{"role": "system", "content": system_prompt.strip()}])]
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})

    user_msg: dict[str, Any] = {"role": "user", "content": user_text.strip()}
    # For video, pass multiple frames (in time order) so the model can reason temporally.
    # For images, this is a single frame.
    imgs = vision_frames if vision_frames else ((frame,) if frame is not None else ())
    if imgs:
        user_msg["content"] = (
            "You are given frames from a video (in time order). "
            "Reason about changes across frames.\n\n" + user_msg["content"]
            if vision_frames and len(vision_frames) > 1
            else user_msg["content"]
        )
        user_msg["images"] = [_pil_to_b64(im) for im in imgs if im is not None]
    messages.append(user_msg)

    kwargs: dict[str, Any] = {
        "model": model,
        "options": build_ollama_options(settings),
        "tools": tools,
        "stream": False,
    }
    t_norm = normalize_think(think)
    if t_norm is not None:
        kwargs["think"] = t_norm

    trace: list[AgentStep] = []
    last_resp: ChatResponse | None = None
    rounds = 0

    def _chat() -> ChatResponse:
        return client.chat(messages=messages, **kwargs)

    while rounds < max_rounds:
        rounds += 1
        resp = await asyncio.to_thread(_chat)
        last_resp = resp
        msg = resp.message
        msg_d = _msg_to_dict(msg)
        messages.append(msg_d)

        await _emit_thinking(hooks, rounds, msg)

        calls = msg.tool_calls
        if not calls:
            content = msg.content or ""
            trace.append(
                AgentStep(
                    f"Agentic round {rounds}",
                    "Model finished (no tool calls).\n" + (content[:2000] if content else "(empty)"),
                )
            )
            return AgenticVisionResult(
                answer=content,
                steps=tuple(trace),
                overlay=seg_state.last_overlay,
                last_response=last_resp,
                rounds=rounds,
                messages=tuple(messages),
            )

        names = ", ".join(c.function.name for c in calls)
        trace.append(
            AgentStep(
                f"Agentic round {rounds}",
                f"Model requested tool call(s): {names}",
            )
        )

        for call in calls:
            name = call.function.name
            args = _coerce_tool_args(call.function.arguments)
            if name != "segment_open_vocab":
                err = f"Unknown tool {name!r}"
                messages.append({"role": "tool", "tool_name": name, "content": err})
                trace.append(AgentStep(f"Tool error ({name})", err))
                continue
            query = str(args.get("query", ""))
            frame_index = args.get("frame_index", None)
            try:
                result = await asyncio.to_thread(segment_tool, query, frame_index)
            except Exception as exc:  # pragma: no cover
                result = f"Tool execution failed: {exc}"
            messages.append({"role": "tool", "tool_name": name, "content": result})
            preview = result if len(result) <= 1200 else result[:1200] + "\n…"
            trace.append(AgentStep("segment_open_vocab", preview))
            if hooks is not None:
                await hooks.on_segmentation(rounds, query, seg_state.last_frame_index, result, seg_state.last_overlay)

    # If we hit the round cap right after emitting tool results, run one more model pass
    # so the assistant can answer instead of ending on a dangling ``tool`` message.
    if _last_message_is_tool(messages):
        try:
            resp = await asyncio.to_thread(_chat)
            last_resp = resp
            messages.append(_msg_to_dict(resp.message))
            rounds += 1
            await _emit_thinking(hooks, rounds, resp.message)
            content = resp.message.content or ""
            if resp.message.tool_calls:
                trace.append(
                    AgentStep(
                        "Agentic (post-cap)",
                        "Model still requested tools after the round cap; returning any text below.",
                    )
                )
            else:
                trace.append(
                    AgentStep(
                        "Agentic (post-cap)",
                        "Final model pass after round cap (tool results consumed).",
                    )
                )
            return AgenticVisionResult(
                answer=content
                or "The model did not return a final answer after the tool loop hit the round cap.",
                steps=tuple(trace),
                overlay=seg_state.last_overlay,
                last_response=last_resp,
                rounds=rounds,
                messages=tuple(messages),
            )
        except Exception as exc:  # pragma: no cover
            trace.append(AgentStep("Agentic (post-cap error)", str(exc)))

    final = ""
    if last_resp and last_resp.message.content:
        final = last_resp.message.content
    trace.append(
        AgentStep(
            "Agentic limit",
            f"Stopped after {max_rounds} model rounds (cap). Last assistant text may be incomplete.",
        )
    )
    return AgenticVisionResult(
        answer=final
        or "The agent reached the maximum number of tool rounds without a final answer. Try a simpler question.",
        steps=tuple(trace),
        overlay=seg_state.last_overlay,
        last_response=last_resp,
        rounds=rounds,
        messages=tuple(messages),
    )
