## seg-mllm

Chainlit app that runs an **Ollama VLM** with a **Falcon-Perception** tool for detection/segmentation.

### Run

```bash
uv sync
python main.py
```

### Notes

- **Falcon weights**: set `FALCON_MODEL_PATH` to a local export dir (with `config.json` + `model.safetensors`) to avoid Hub downloads.
- **Falcon output mode** (UI + env): `FALCON_OUTPUT_MODE=auto|segmentation|boxes`.
