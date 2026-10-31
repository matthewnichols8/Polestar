# Polestar in dInfer

This optional package runs Polestar-Cache and Polestar-Commit in a dInfer LLaDA
worker. The accelerated dInfer path uses the same LLaDA checkpoint as
standalone Polestar. Within dInfer's prefix decoder, a penultimate-layer
proxy updates the final-layer KV cache for the next forward pass, while
final-layer attention drift guides token commitment.
`parallel_decoding="polestar", cache="polestar"` selects both Cache and Commit.
dInfer's threshold decoder with a prefix cache is available as a comparison
baseline; it is not a Polestar path.

The bundled dInfer code retains its [Apache-2.0 license](LICENSE).
[Upstream dInfer](https://github.com/inclusionAI/dInfer) documents its other
models and serving options.

## Install

From the release root, with a CUDA-capable PyTorch build:

```bash
python -m pip install -e '.[eval]'
python -m pip install -e integrations/dinfer
```

## Serve Polestar

```python
from dinfer.decoding.serving import DiffusionLLMServing, SamplingParams
from transformers import AutoTokenizer

model_id = "GSAI-ML/LLaDA-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 12 times 8?"}],
    add_generation_prompt=True,
    tokenize=False,
)
prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
params = SamplingParams(
    parallel_decoding="polestar",
    cache="polestar",
    use_bd=False,
    mask_id=126336,
    eos_id=126081,
    polestar_seed=42,
)
server = DiffusionLLMServing(
    model_id,
    model_type="llada",
    sample_params=params,
    num_gpus=1,
    dp_size=1,
    tpep_size=1,
    backend="torch",
)
try:
    output_ids = server.generate(prompt_ids, gen_length=256, block_length=32)
    print(tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True))
finally:
    server.stop_serving()
```

The adapter uses one GPU per worker and compiles the dInfer model on first use.
The first request includes compilation; a new prompt length may compile an
additional graph.
`polestar_seed` defaults to 42 and seeds cache clustering for each request.
The dInfer path requires greedy decoding (`temperature=0`) and currently uses
two upper layers for selective Cache refresh every eight decode forwards. Its
prefix KV patches take effect on the next model forward. The standalone
Polestar interface retains the paper's full-layer cache path; see the
[standalone benchmark](../../README.md#performance-benchmarking).

## Check GSM8K

Run two questions first:

```bash
python integrations/dinfer/benchmarks/evaluate_gsm8k.py \
  --backend polestar --config configs/llada8b.json --task gsm8k \
  --limit 2 --seed 42 --output-path results/dinfer-polestar-smoke
```

For a larger comparison, run `--limit 16` with a new output directory. The
benchmark uses the same 5-shot prompts, generation length 256, and scoring
as `evaluation/run.py`. It writes accuracy to `results.json`, responses to
`samples_gsm8k.jsonl`, and tokens per second and request times to
`serving_metrics.json`. To measure dInfer alone on the same questions:

```bash
python integrations/dinfer/benchmarks/evaluate_gsm8k.py \
  --backend dinfer --config configs/llada8b.json --task gsm8k \
  --limit 2 --seed 42 --output-path results/dinfer-baseline-smoke
```

Both decoders compile their model on the first request. Add
`--prewarm-request` to both commands to run each prompt once before timing
it. The timing file also records whether the repeated answers match.

## Measured comparison

On one GB300, the two decoders ran sequentially with the same LLaDA-8B
checkpoint, 16 GSM8K questions, 5-shot prompts, seed 42, generation length
256, block length 32, confidence threshold 0.9, and one untimed pass for each
question. Both repeated all 16 answers exactly.

| Decoder | Accuracy | Tokens per second | Measured request time |
| --- | ---: | ---: | ---: |
| dInfer threshold + prefix cache | 10/16 | 330.32 | 11.39 s |
| dInfer with Polestar Cache + Commit | 12/16 | 383.42 | 10.06 s |

The Polestar path was 16.1% faster by tokens per second on this comparison.
Those rates exclude the first measured request. Request time is the sum
across all 16 measured requests, excluding each prompt's untimed pass.
[Recorded metrics](../../assets/data/dinfer-gsm8k16-gb300.json) include the
settings and exact values. This small check is separate from the paper's full
GSM8K evaluation.
