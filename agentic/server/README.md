# DiffusionGemma OpenAI-compatible server

`serve_openai.py` serves `google/diffusiongemma-26B-A4B-it` (dInfer, expert parallel over the listed GPUs) as
`/v1/chat/completions` with tool calling and per-turn prefix reuse. Results: `../results/PHASE1_RESULTS.md`.

The Polestar clone must be on a branch with prefix reuse (`feature/dgemma-prefix-reuse`), since
`/scratch/mi72/envs/polestar` uses editable installs of it.

Start the server (loads in ~1 min; uses both TITANs):
```bash
cd /scratch/mi72/agentic-dgemma/server
PYTHONPATH= HF_HOME=/scratch/mi72/hf TMPDIR=/scratch/mi72/tmp /scratch/mi72/envs/polestar/bin/python serve_openai.py --gpus 0,1 --port 8000 --log-dir ../results/server-logs
```
Useful flags:
- `--polestar` (canvas cache);
- `--thinking`;
- `--entropy-bound`, `--max-denoising-steps`;
- `--cache-slots`;
- `--default-max-tokens`.

Stop it with Ctrl-C, or with `pkill -f "^/scratch/mi72/envs/polestar/bin/python serve_openai"`.

Use from any OpenAI / LiteLLM client (one request at a time; the model name is ignored):
```python
litellm.completion(model="openai/diffusiongemma", api_base="http://127.0.0.1:8000/v1", api_key="x",
                   messages=..., tools=...)
```

Files:
| File | Purpose |
|---|---|
| `gemma_format.py` | OpenAI ↔ Gemma translation (rendering, tool-call parsing) |
| `test_gemma_format.py` | CPU tests with the real tokenizer: `PYTHONPATH= HF_HOME=/scratch/mi72/hf /scratch/mi72/envs/polestar/bin/python -m pytest -q` |
| `phase1_check.py` | Probes, multi-turn reuse timing and API checks against a running server |
| `litellm_check.py` | A two-turn tool exchange through LiteLLM (run with `/scratch/mi72/envs/agents/bin/python`) |
