# seg-mllm

Chainlit app that runs an **Ollama vision+tools model** with a **Falcon-Perception** tool for open-vocabulary detection/segmentation.

## Demo

Screen recording on Apple Silicon: **Falcon-Perception 300M**, **bounding-box output**, **Gemma 4** in Ollama (fast interactive run).

[![Watch the video](https://raw.githubusercontent.com/gauravfs-14/seg-mllm/main/public/segment-300m-thumb.jpg)](https://raw.githubusercontent.com/gauravfs-14/seg-mllm/main/public/segment-300m.mp4)

## Run

```bash
uv sync
uv run main.py
```

## Requirements

- Ollama configured with a vision+tools model
- Falcon-Perception installed and configured
- ffmpeg installed and configured in the path for video processing

> [!NOTE]
> This repository currently only support MLX on Apple Silicon. Why? because I use a Mac, and I don't want to deal with CUDA.
> If you want to use this thing, idk, maybe you want to, then you can make it work on your CUDA, and open a PR. I'll be more than happy to merge it.

## How it works

- **Chat**: full chat-style history is kept per session and sent back to Ollama each turn (capped).
- **Images**: sent as a single frame to Ollama.
- **Videos**:
  - frames are sampled at `VIDEO_SAMPLE_FPS` (default **4 fps**) up to `VIDEO_MAX_FRAMES` (default **120**)
  - the VLM sees all frames (time order) for temporal context
  - Falcon can be called on specific frames via `frame_index` (0-based), and should be used on up to 3 “most important” frames.

## UI controls (composer bar)

- **Model**: picks an Ollama model that supports **vision + tools**.
- **Reasoning**: Low / Medium / High / Full.
- **Falcon output**: Auto / Segmentation / Boxes.
- **Falcon model**: Full / 300M.

## Configuration (env vars)

### Ollama

- **`OLLAMA_HOST`**: Ollama server URL (default `http://127.0.0.1:11434`)
- **`OLLAMA_MODEL`**: default fallback model id
- **`OLLAMA_SEED`**: set for determinism (optional)

### Falcon

- **`FALCON_MODEL_PATH`**: local Falcon export dir (contains `config.json` + `model.safetensors`); when set, Hub download is skipped
- **`FALCON_MODEL_ID`**: Hub id to use when `FALCON_MODEL_PATH` is empty (default `tiiuae/falcon-perception`)
- **`FALCON_OUTPUT_MODE`**: `auto|segmentation|boxes` (default `auto`)
- **`FALCON_MAX_DIMENSION`** / **`FALCON_MIN_DIMENSION`**: image resize bounds for Falcon preprocessing
- **`FALCON_MAX_NEW_TOKENS`**: decode cap for Falcon tool runs
- **`FALCON_MLX_CLEAR_CACHE`**: clear MLX Metal cache after tool runs (default on)
- **`FALCON_OVERLAY_DISPLAY_MAX`**: downscale overlay before sending to UI

### Video sampling

- **`VIDEO_SAMPLE_FPS`**: frames per second to sample for the VLM (default **4**)
- **`VIDEO_MAX_FRAMES`**: hard cap on sampled frames (default **120**)

### Chat history

- **`CHAT_HISTORY_MAX_MESSAGES`**: number of past messages (excluding system) kept per session (default **24**)
