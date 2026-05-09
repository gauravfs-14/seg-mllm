from __future__ import annotations

import asyncio
import io
import time
from functools import lru_cache

import chainlit as cl
from chainlit.chat_settings import ChatSettings
from chainlit.input_widget import Slider, Switch, Tab, TextInput
from PIL import Image

from seg_mllm.config import Settings, normalize_falcon_output_mode
from seg_mllm.integrations.chainlit_media import frame_from_message_elements, frames_from_message_elements
from seg_mllm.integrations.chainlit_ollama_stream import (
    format_ollama_performance_caption,
    ollama_performance_parts,
)
from seg_mllm.services import (
    FalconPerceptionService,
    OllamaLLMClient,
    TaskAgent,
    discover_vision_and_tool_models,
    ollama_cli_installed,
    ollama_daemon_reachable,
)
from seg_mllm.prompts.agent_prompts import AGENTIC_VISION_SYSTEM
from seg_mllm.services.agentic_ollama import run_agentic_vision_chat


def _normalize_llm_markdown(text: str) -> str:
    """Best-effort cleanup so headings/tables render in Chainlit."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n")
    # Models sometimes emit "*** ###" inline; make separators and headings start new lines.
    s = s.replace("***###", "\n\n###").replace("*** ###", "\n\n###")
    s = s.replace(" *** ", "\n\n---\n\n").replace("\n***\n", "\n\n---\n\n")
    # Ensure headings begin on a fresh line.
    s = s.replace("###", "\n\n###")
    # Tables: ensure a blank line before any table block.
    lines = s.split("\n")
    out_lines: list[str] = []
    for i, line in enumerate(lines):
        is_tableish = line.lstrip().startswith("|") and line.count("|") >= 2
        if is_tableish and out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append(line)
    s = "\n".join(out_lines)
    # Collapse accidental extra blank lines.
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s.strip() + "\n"


async def _stream_markdown(msg: cl.Message, text: str, *, chunk_size: int = 200) -> None:
    """Stream while preserving whitespace/newlines (word-splitting breaks Markdown)."""
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        await msg.stream_token(text[i : i + chunk_size])


def _trim_history(messages: list[dict], *, max_messages: int) -> list[dict]:
    """Keep system + last N messages; drop any embedded images from history."""
    if not messages:
        return []
    sys_msg = messages[0] if messages[0].get("role") == "system" else None
    rest = messages[1:] if sys_msg else messages
    trimmed = rest[-max(0, int(max_messages)) :]
    out: list[dict] = []
    if sys_msg:
        out.append(dict(sys_msg))
    for m in trimmed:
        d = dict(m)
        d.pop("images", None)  # avoid re-sending huge base64 blobs
        out.append(d)
    return out


def _video_element_from_uploads(elements: list[object] | None) -> cl.Video | None:
    """Best-effort: re-display the uploaded video as a page element."""
    if not elements:
        return None
    for el in elements:
        mime = (getattr(el, "mime", None) or "").lower()
        name = getattr(el, "name", None) or "uploaded_video.mp4"
        path = getattr(el, "path", None)
        if "video/" not in mime and not str(name).lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")):
            continue
        if path:
            return cl.Video(name=str(name), path=str(path), display="page", size="large", mime=mime or "video/mp4")
        content = getattr(el, "content", None)
        if isinstance(content, (bytes, bytearray)):
            return cl.Video(
                name=str(name),
                content=bytes(content),
                display="page",
                size="large",
                mime=mime or "video/mp4",
            )
    return None


def _overlay_for_display(
    overlay: Image.Image,
    *,
    max_side: int,
) -> tuple[bytes, str, str]:
    """Resize and encode overlay so the browser/UI does less work (full 4K PNGs are expensive)."""
    im = overlay.convert("RGB")
    if max_side > 0:
        w, h = im.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / long_side
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue(), "segmentation_overlay.jpg", "image/jpeg"


class _ChainlitVisionHooks:
    """Emits Chain of Thought steps as the agent runs (not bundled at the end)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def on_thinking(self, round: int, text: str) -> None:
        async with cl.Step(
            name=f"Reasoning (round {round})",
            type="llm",
            default_open=False,
            auto_collapse=True,
        ) as step:
            step.output = text

    async def on_segmentation(
        self,
        round: int,
        query: str,
        frame_index: int | None,
        summary: str,
        overlay: Image.Image | None,
    ) -> None:
        async with cl.Step(
            name=f"Segmentation (round {round})",
            type="tool",
            default_open=True,
            show_input="json",
        ) as step:
            step.input = {"query": query, "frame_index": frame_index}
            # Some UIs won't visibly render a step that has only elements but no text.
            # Keep the parent step lightweight and always show the overlay here.
            step.output = "Segmentation overlay"
            if overlay is not None:
                data, fname, mime = _overlay_for_display(
                    overlay,
                    max_side=self._settings.falcon_overlay_display_max_side,
                )
                step.elements = [
                    cl.Image(
                        content=data,
                        name=fname,
                        display="inline",
                        size="large",
                        mime=mime,
                    )
                ]

            # Keep the overlay uncollapsed in the parent step, but move the very long
            # per-instance text summary into a collapsed child step.
            async with cl.Step(
                name="Segmentation details",
                type="tool",
                default_open=False,
                auto_collapse=True,
                show_input=False,
            ) as details:
                details.output = summary


@lru_cache(maxsize=64)
def _cached_agent(settings: Settings) -> TaskAgent:
    return TaskAgent(
        llm=OllamaLLMClient(settings),
        perception=FalconPerceptionService(settings),
    )


def _composer_modes(host: str, base: Settings) -> list[cl.Mode]:
    strict = discover_vision_and_tool_models(host)
    models = strict if strict else [base.ollama_model]
    default_id = base.ollama_model if base.ollama_model in models else models[0]
    return [
        cl.Mode(
            id="model",
            name="Model",
            options=[
                cl.ModeOption(
                    id=m,
                    name=m,
                    description=f"Capabilities include vision + tools — try: ollama show {m}",
                    icon="sparkles",
                    default=(m == default_id),
                )
                for m in models
            ],
        ),
        cl.Mode(
            id="reasoning",
            name="Reasoning",
            options=[
                cl.ModeOption(id="low", name="Low", description="Fast and efficient", icon="rocket", default=True),
                cl.ModeOption(id="medium", name="Medium", description="Balanced", icon="rocket"),
                cl.ModeOption(id="high", name="High", description="Think harder", icon="rocket"),
                cl.ModeOption(id="full", name="Full", description="Maximum effort (if supported)", icon="rocket"),
            ],
        ),
        cl.Mode(
            id="falcon_output",
            name="Falcon output",
            options=[
                cl.ModeOption(
                    id="auto",
                    name="Auto",
                    description="Follow checkpoint (masks if available; else boxes).",
                    icon="scan",
                    default=(normalize_falcon_output_mode(base.falcon_output_mode) == "auto"),
                ),
                cl.ModeOption(
                    id="segmentation",
                    name="Segmentation",
                    description="Prefer pixel masks (falls back to boxes if the model emits none).",
                    icon="scan",
                    default=(normalize_falcon_output_mode(base.falcon_output_mode) == "segmentation"),
                ),
                cl.ModeOption(
                    id="boxes",
                    name="Boxes",
                    description="Bounding boxes only (rectangular overlays).",
                    icon="scan",
                    default=(normalize_falcon_output_mode(base.falcon_output_mode) == "boxes"),
                ),
            ],
        ),
        cl.Mode(
            id="falcon_model",
            name="Falcon model",
            options=[
                cl.ModeOption(
                    id="full",
                    name="Full",
                    description="Falcon-Perception (segmentation-capable; heavier).",
                    icon="layers",
                    default=("300m" not in (base.falcon_model_id or "").lower()),
                ),
                cl.ModeOption(
                    id="300m",
                    name="300M",
                    description="Falcon-Perception-300M (boxes/detection; faster).",
                    icon="layers",
                    default=("300m" in (base.falcon_model_id or "").lower()),
                ),
            ],
        ),
    ]


def _as_float(v: object, default: float) -> float:
    if v is None:
        return default
    return float(v)  # type: ignore[arg-type]


def _as_int(v: object, default: int) -> int:
    if v is None:
        return default
    return int(round(float(v)))


def _ollama_seed_from_session(d: dict, base: Settings) -> int | None:
    if "ollama_seed" not in d:
        return base.ollama_seed
    raw = d.get("ollama_seed")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return base.ollama_seed


def _seed_text_for_form(d: dict, base: Settings) -> str:
    if "ollama_seed" in d:
        raw = d.get("ollama_seed")
        if raw is None or str(raw).strip() == "":
            return ""
        return str(raw).strip()
    if base.ollama_seed is not None:
        return str(base.ollama_seed)
    return ""


def _settings_from_session(sess: dict | None, *, model: str) -> Settings:
    base = Settings()
    d = sess or {}
    host = (d.get("ollama_host") or "").strip() or base.ollama_host
    falcon_path = (d.get("falcon_model_path") or "").strip() or base.falcon_model_path
    falcon_c = d.get("falcon_compile")
    if falcon_c is None:
        falcon_c = base.falcon_compile
    return Settings(
        ollama_host=host,
        ollama_model=model,
        falcon_model_path=falcon_path,
        falcon_compile=bool(falcon_c),
        # falcon_output_mode is intentionally NOT persisted in chat settings; use composer mode per-message.
        ollama_temperature=_as_float(d.get("ollama_temperature"), base.ollama_temperature),
        ollama_top_p=_as_float(d.get("ollama_top_p"), base.ollama_top_p),
        ollama_top_k=_as_int(d.get("ollama_top_k"), base.ollama_top_k),
        ollama_num_ctx=_as_int(d.get("ollama_num_ctx"), base.ollama_num_ctx),
        ollama_num_predict=_as_int(d.get("ollama_num_predict"), base.ollama_num_predict),
        ollama_repeat_penalty=_as_float(d.get("ollama_repeat_penalty"), base.ollama_repeat_penalty),
        ollama_seed=_ollama_seed_from_session(d, base),
    )


def _build_chat_settings(settings_dict: dict | None) -> ChatSettings:
    base = Settings()
    d = settings_dict or {}
    host = (d.get("ollama_host") or "").strip() or base.ollama_host
    falcon_path = d.get("falcon_model_path") or base.falcon_model_path
    falcon_c = d.get("falcon_compile")
    if falcon_c is None:
        falcon_c = base.falcon_compile
    t = _as_float(d.get("ollama_temperature"), base.ollama_temperature)
    tp = _as_float(d.get("ollama_top_p"), base.ollama_top_p)
    tk = _as_int(d.get("ollama_top_k"), base.ollama_top_k)
    nctx = _as_int(d.get("ollama_num_ctx"), base.ollama_num_ctx)
    npred = _as_int(d.get("ollama_num_predict"), base.ollama_num_predict)
    rp = _as_float(d.get("ollama_repeat_penalty"), base.ollama_repeat_penalty)
    seed_txt = _seed_text_for_form(d, base)
    # ChatSettings takes ``inputs`` (flat widgets or Tab rows). A ``tabs=`` kwarg is ignored
    # and would yield an empty form in the UI.
    return ChatSettings(
        [
            Tab(
                id="server",
                label="Server",
                inputs=[
                    TextInput(
                        id="ollama_host",
                        label="Ollama host",
                        initial=host,
                        placeholder="http://127.0.0.1:11434",
                    ),
                ],
            ),
            Tab(
                id="ollama",
                label="Ollama generation",
                inputs=[
                    Slider(
                        id="ollama_temperature",
                        label="Temperature",
                        initial=t,
                        min=0.0,
                        max=2.0,
                        step=0.05,
                    ),
                    Slider(
                        id="ollama_top_p",
                        label="Top P",
                        initial=tp,
                        min=0.0,
                        max=1.0,
                        step=0.01,
                    ),
                    Slider(
                        id="ollama_top_k",
                        label="Top K",
                        initial=float(tk),
                        min=0.0,
                        max=100.0,
                        step=1.0,
                        description="0 disables; typical 20–40.",
                    ),
                    Slider(
                        id="ollama_num_ctx",
                        label="Context length (num_ctx)",
                        initial=float(nctx),
                        min=512.0,
                        max=131072.0,
                        step=512.0,
                    ),
                    Slider(
                        id="ollama_num_predict",
                        label="Max output tokens (num_predict)",
                        initial=float(npred),
                        min=0.0,
                        max=32768.0,
                        step=256.0,
                        description="0 = model default (unlimited).",
                    ),
                    Slider(
                        id="ollama_repeat_penalty",
                        label="Repeat penalty",
                        initial=rp,
                        min=0.5,
                        max=2.0,
                        step=0.05,
                    ),
                    TextInput(
                        id="ollama_seed",
                        label="Seed (empty = random)",
                        initial=seed_txt,
                        placeholder="e.g. 42",
                    ),
                ],
            ),
            Tab(
                id="falcon",
                label="Falcon",
                inputs=[
                    TextInput(
                        id="falcon_model_path",
                        label="Falcon local directory",
                        initial=falcon_path,
                        placeholder="/path/to/export with config.json + model.safetensors",
                    ),
                    Switch(
                        id="falcon_compile",
                        label="Falcon torch.compile (CUDA/CPU only; first call slower)",
                        initial=bool(falcon_c),
                    ),
                ],
            ),
        ]
    )


async def _maybe_refresh_settings_after_host_change(
    previous: dict | None,
    current: dict,
) -> None:
    if not previous:
        return
    p_host = (previous.get("ollama_host") or "").strip()
    n_host = (current.get("ollama_host") or "").strip()
    if p_host == n_host:
        return
    await _build_chat_settings(current).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("vision_frame", None)
    cl.user_session.set("vision_frames", None)
    # Full chat history for Ollama (system + turns).
    cl.user_session.set(
        "ollama_messages",
        [{"role": "system", "content": AGENTIC_VISION_SYSTEM.strip()}],
    )
    base = Settings()
    initial = {
        "ollama_host": base.ollama_host,
        "falcon_model_path": base.falcon_model_path,
        "falcon_compile": base.falcon_compile,
        "ollama_temperature": base.ollama_temperature,
        "ollama_top_p": base.ollama_top_p,
        "ollama_top_k": base.ollama_top_k,
        "ollama_num_ctx": base.ollama_num_ctx,
        "ollama_num_predict": base.ollama_num_predict,
        "ollama_repeat_penalty": base.ollama_repeat_penalty,
        "ollama_seed": "" if base.ollama_seed is None else str(base.ollama_seed),
    }
    s = await _build_chat_settings(initial).send()
    cl.user_session.set("settings", s)

    host = (s.get("ollama_host") or "").strip() or base.ollama_host
    await cl.context.emitter.set_modes(_composer_modes(host, base))

    lines = [
        "Welcome to the seg-mllm! This is a Chainlit app that runs an Ollama vision+tools model with a Falcon-Perception tool for open-vocabulary detection/segmentation.",
    ]
    if not ollama_cli_installed():
        lines.append(
            "\n⚠️ The `ollama` CLI is not on your PATH. Install from https://ollama.com "
            "so the server can be managed from the terminal."
        )
    elif not ollama_daemon_reachable(base):
        lines.append(
            "\n⚠️ No Ollama server responded at the configured host. Start it (e.g. `ollama serve`) or fix the host in **Server** settings."
        )
    elif not discover_vision_and_tool_models(host):
        lines.append(
            "\n⚠️ No installed models report **vision** and **tools** together. "
            "The picker falls back to `OLLAMA_MODEL` until you pull a multimodal tool model (check `ollama show <name>` → Capabilities)."
        )

    await cl.Message(content="\n".join(lines)).send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    prev = cl.user_session.get("settings")
    cl.user_session.set("settings", settings)
    await _maybe_refresh_settings_after_host_change(
        previous=prev if isinstance(prev, dict) else None,
        current=settings,
    )
    # Refresh model mode when host changes (models are host-dependent)
    base = Settings()
    host = (settings.get("ollama_host") or "").strip() or base.ollama_host
    await cl.context.emitter.set_modes(_composer_modes(host, base))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    raw = (message.content or "").strip()
    if not raw:
        await cl.Message(content="Please enter a text message.").send()
        return

    settings = cl.user_session.get("settings")
    if not isinstance(settings, dict):
        await cl.Message(
            content="Open **Chat settings** in the **left sidebar**, adjust options, and confirm so configuration is saved."
        ).send()
        return

    base = Settings()
    model = (message.modes or {}).get("model") or base.ollama_model
    app_settings = _settings_from_session(settings, model=model)
    # Per-message Falcon checkpoint selection from composer.
    falcon_model_sel = ((message.modes or {}).get("falcon_model") or "full").strip().lower()
    if not (app_settings.falcon_model_path or "").strip():
        falcon_model_id = "tiiuae/falcon-perception-300m" if falcon_model_sel == "300m" else "tiiuae/falcon-perception"
        app_settings = Settings(**{**app_settings.__dict__, "falcon_model_id": falcon_model_id})
    falcon_mode = normalize_falcon_output_mode((message.modes or {}).get("falcon_output"))
    if falcon_mode != "auto":
        # Per-message override from the composer (next to Reasoning).
        app_settings = Settings(**{**app_settings.__dict__, "falcon_output_mode": falcon_mode})
    reasoning = (message.modes or {}).get("reasoning") or "low"
    think = "default"
    if reasoning in ("low", "medium", "high"):
        think = reasoning
    elif reasoning == "full":
        think = True

    new_frames, attach_err = frames_from_message_elements(
        message.elements,
        video_sample_fps=app_settings.video_sample_fps,
        video_max_frames=app_settings.video_max_frames,
    )
    if attach_err:
        await cl.Message(content=f"Could not use the attachment: {attach_err}").send()
        return

    vid = _video_element_from_uploads(message.elements)
    if vid is not None:
        await cl.Message(content="Video", elements=[vid]).send()

    if new_frames is not None:
        cl.user_session.set("vision_frames", tuple(new_frames))
        cl.user_session.set("vision_frame", new_frames[0] if new_frames else None)
    frames = cl.user_session.get("vision_frames")
    frame = cl.user_session.get("vision_frame")

    agent = _cached_agent(app_settings)

    try:
        prior = cl.user_session.get("ollama_messages")
        prior_msgs = prior if isinstance(prior, list) else [{"role": "system", "content": AGENTIC_VISION_SYSTEM.strip()}]
        ares = await run_agentic_vision_chat(
            settings=app_settings,
            model=model,
            user_text=raw,
            frame=frame,
            vision_frames=frames if isinstance(frames, tuple) else None,
            perception=agent.perception,
            think=think,
            hooks=_ChainlitVisionHooks(app_settings),
            prior_messages=prior_msgs,
        )
    except Exception as exc:  # pragma: no cover
        await cl.Message(content=f"Agent pipeline error: {exc}").send()
        return

    # Persist updated history for next turns (cap length).
    cl.user_session.set(
        "ollama_messages",
        _trim_history(list(ares.messages), max_messages=app_settings.chat_history_max_messages),
    )

    msg = await cl.Message(content="").send()
    t0 = time.perf_counter()
    answer = _normalize_llm_markdown(ares.answer)
    await _stream_markdown(msg, answer)
    wall = time.perf_counter() - t0
    perf = ollama_performance_parts(ares.last_response, model=model, wall_seconds=wall)
    perf["note"] = "agentic"
    perf["agentic_rounds"] = ares.rounds

    caption = format_ollama_performance_caption(perf)
    msg.metadata = {**(msg.metadata or {}), "seg_mllm_perf": perf}
    if caption:
        # Use Chainlit Actions (supported) so it shows beside the copy button row.
        msg.actions = [
            cl.Action(
                name="seg_mllm_perf",
                payload={},
                label=caption,
                tooltip="Generation performance",
            )
        ]
    await msg.update()


@cl.action_callback("seg_mllm_perf")
async def _noop_perf_action(action: cl.Action) -> None:
    # Intentionally no-op: the action is a UI label.
    return None
