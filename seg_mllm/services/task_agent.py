from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from seg_mllm.contracts.llm import LLMClient, ThinkOption
from seg_mllm.contracts.perception import PerceptionClient, SegmentationResult
from seg_mllm.media.overlay import render_instance_overlay
from seg_mllm.prompts.agent_prompts import FINAL_SYSTEM, UNIFIED_VISION_SYSTEM

_SEG_HINT = re.compile(
    r"\b(segment|segmentation|seg\b|mask(s)?|contour(s)?|outline(s)?|silhouette|polygon|"
    r"boundary|boundaries|pixel[- ]accurate|pixel[- ]level|-instance\b|instances\b|"
    r"falcon|masked pixels|which pixels|trace (the )?outline|draw (the )?outline|"
    r"\bcount\b.{0,48}\b(object|instance|person|people|thing)s?\b|"
    r"how many\b.{0,40}\b(separate|distinct|individual)\b)",
    re.I | re.DOTALL,
)


def _needs_segmentation(user_text: str, *, has_image: bool) -> bool:
    """Heuristic: Falcon only when there is an image and the query asks for masks or fine regions."""
    if not has_image:
        return False
    t = user_text.strip()
    if not t:
        return False
    return bool(_SEG_HINT.search(t))


def _segmentation_query(user_text: str) -> str:
    """Short phrase for Falcon; defaults to the user's wording."""
    q = user_text.strip()
    if len(q) > 500:
        q = q[:500] + "…"
    return q or "objects"


@dataclass(frozen=True)
class AgentStep:
    title: str
    detail: str


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    steps: tuple[AgentStep, ...]
    overlay: Image.Image | None


@dataclass(frozen=True)
class AgentPrepared:
    """Plan from one unified agent path; streaming messages use ``stream_*`` when set."""

    steps: tuple[AgentStep, ...]
    overlay: Image.Image | None
    instant_answer: str | None
    stream_system: str | None
    stream_user: str | None
    stream_images: tuple[Image.Image, ...] | None

    def show_tool_trace(self) -> bool:
        """Collapsible trace only when Falcon / segmentation actually ran."""
        return any(s.title == "Falcon-Perception" for s in self.steps)


class TaskAgent:
    """Single vision agent: answer directly when possible; delegate to Falcon only when needed."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        perception: PerceptionClient,
    ) -> None:
        self._llm = llm
        self._perception = perception

    @property
    def perception(self) -> PerceptionClient:
        return self._perception

    def prepare(
        self,
        user_text: str,
        frame: Image.Image | None,
        *,
        think: ThinkOption = None,
    ) -> AgentPrepared:
        steps: list[AgentStep] = []
        has_image = frame is not None

        if _needs_segmentation(user_text, has_image=has_image):
            assert frame is not None
            query = _segmentation_query(user_text)
            steps.append(
                AgentStep(
                    "Segmentation",
                    f"Requested ({query[:120]}{'…' if len(query) > 120 else ''})",
                )
            )
            seg = self._perception.segment(frame, query)
            steps.append(
                AgentStep(
                    "Falcon-Perception",
                    _format_segmentation_summary(seg),
                )
            )
            overlay = render_instance_overlay(frame, seg.instances) if seg.instances else None
            tool_text = _format_segmentation_summary(seg)
            final_user = (
                f"Original user request:\n{user_text}\n\n"
                f"Segmentation query used:\n{query}\n\n"
                f"Tool output:\n{tool_text}"
            )
            images: list[Image.Image] = [frame]
            if overlay is not None:
                images.append(overlay)
            steps.append(AgentStep("Synthesis", "Vision model with frame + overlay + tool summary."))
            return AgentPrepared(
                steps=tuple(steps),
                overlay=overlay,
                instant_answer=None,
                stream_system=FINAL_SYSTEM,
                stream_user=final_user,
                stream_images=tuple(images),
            )

        # Single-pass vision/text answer — no separate router model.
        steps.append(
            AgentStep(
                "Agent",
                "Direct answer (no Falcon segmentation).",
            )
        )
        return AgentPrepared(
            steps=tuple(steps),
            overlay=None,
            instant_answer=None,
            stream_system=UNIFIED_VISION_SYSTEM,
            stream_user=user_text.strip(),
            stream_images=(frame,) if frame else None,
        )

    def run(
        self,
        user_text: str,
        frame: Image.Image | None,
        *,
        think: ThinkOption = None,
    ) -> AgentRunResult:
        prepared = self.prepare(user_text, frame, think=think)
        if prepared.instant_answer is not None:
            return AgentRunResult(
                answer=prepared.instant_answer,
                steps=prepared.steps,
                overlay=prepared.overlay,
            )
        assert prepared.stream_system and prepared.stream_user is not None
        answer = self._llm.complete(
            prepared.stream_system,
            prepared.stream_user,
            list(prepared.stream_images) if prepared.stream_images else None,
            think=think,
        )
        return AgentRunResult(answer=answer, steps=prepared.steps, overlay=prepared.overlay)


def _format_segmentation_summary(result: SegmentationResult) -> str:
    lines = [f"query: {result.query}", f"instance_count: {len(result.instances)}"]
    for i, inst in enumerate(result.instances, 1):
        px = int(inst.mask.sum())
        x = float(inst.xy.get("x", 0.0))
        y = float(inst.xy.get("y", 0.0))
        h = float(inst.hw.get("h", 0.0))
        w = float(inst.hw.get("w", 0.0))
        lines.append(
            f"  - instance {i}: center=({x:.4f},{y:.4f}) size=({h:.4f},{w:.4f}) masked_pixels={px}",
        )
    return "\n".join(lines)
