# seg-mllm

Attach an **image** or **video** and ask a question.

## Tips

- For **videos**, the model receives multiple frames (time order). For best results, ask for **events over time** (“what changed?”, “did a crash happen?”, “list the sequence of events”).
- Use **Falcon output** / **Falcon model** in the composer bar to switch between **segmentation** vs **boxes**, and **Full** vs **300M**.
- If you want reproducible results, set a **Seed** in chat settings.
