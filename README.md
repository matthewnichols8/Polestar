<div align="center">

# Polestar

### Drift-Aware Cache Calibration and Token Commitment for Efficient Inference of Diffusion LLMs

**Official implementation · NeurIPS 2026**

[![Website](https://img.shields.io/badge/Website-Polestar-76B900)](https://polestar-dllm.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.14107-b31b1b)](https://arxiv.org/abs/2607.14107)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-5b4ab0)](https://arxiv.org/pdf/2607.14107)
[![Checks](https://github.com/gt-synergy-lab/Polestar/actions/workflows/checks.yml/badge.svg)](https://github.com/gt-synergy-lab/Polestar/actions/workflows/checks.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-28736c)](LICENSE)

Mingyu Lee\*, Akshat Ramachandran\*, Souvik Kundu, and Tushar Krishna<br>
Georgia Institute of Technology · Intel AI Group<br>
\* Equal contribution

**[Website](https://polestar-dllm.github.io/) · [Paper](https://arxiv.org/pdf/2607.14107) · [Video](#video) · [Results](#performance) · [Quickstart](#quickstart) · [Models](#models-and-examples) · [Evaluation](#evaluation) · [Method](#method-overview)**

</div>

## Video

https://github.com/user-attachments/assets/5fc2c4c0-0985-4036-b8fd-8819079f94bd

## News

- **October 2026:** [Project website](https://polestar-dllm.github.io/) launched with interactive demos and benchmark results.
- **October 2026:** Algorithm code released for LLaDA, Dream, and LLaDA-V.
- **September 2026:** Polestar accepted at **NeurIPS 2026**.

## Method Overview

**Polestar uses representation drift to decide which cached states to refresh
and when to commit tokens during diffusion LLM inference. It needs no
additional training.**

- **Refresh where it matters.** Polestar-Cache locates high-drift regions and selectively updates cached representations.
- **Commit when tokens are ready.** Polestar-Commit combines drift events with confidence to increase decoding parallelism.
- **Use existing checkpoints.** Run LLaDA-8B-Instruct, LLaDA-1.5, Dream-7B-Instruct, and LLaDA-V without additional training.

<p align="center">
  <img src="assets/figures/polestar-overview.png" alt="Polestar-Cache and Polestar-Commit methodology" width="1000">
</p>

<p align="center"><em>Polestar-Cache and Polestar-Commit.</em> <a href="https://arxiv.org/pdf/2607.14107">Paper and algorithm</a></p>

## Performance

Across the paper's mathematics and coding benchmarks, Polestar achieves **up to 3.7× higher throughput** and **up to 10.73% accuracy improvement** over existing baselines, with decoding parallelism reaching **3.67 tokens per forward pass**.

<p align="center">
  <a href="assets/figures/accuracy-throughput.png"><img src="assets/figures/accuracy-throughput.png" alt="Polestar accuracy-throughput trade-off on GSM8K" width="480"></a>
</p>

<p align="center"><em>Accuracy–throughput trade-off.</em></p>

<p align="center">
  <a href="assets/figures/table1-256.svg"><img src="assets/figures/table1-256.png" alt="Accuracy bars and throughput curves across six methods on mathematics and code benchmarks at generation length 256" width="1000"></a>
</p>

<p align="center"><em>Accuracy and throughput · 256 tokens.</em> <a href="assets/figures/table1-256.svg">Figure</a> · <a href="assets/data/table1.json">Data</a></p>

**Results across all four models:**

| Model | Benchmark | Accuracy (%) ↑ | TPF ↑ | TPS ↑ |
|---|---|---:|---:|---:|
| LLaDA-8B-Instruct | GSM8K · 5-shot | 78.33 | 3.67 | 87.57 |
| LLaDA-1.5 | GSM8K · 5-shot | 81.06 | 3.40 | 79.65 |
| Dream-7B-Instruct | GSM8K · 5-shot | 72.40 | 2.39 | 52.80 |
| LLaDA-V | MathVista | 63.58 | 2.17 | 37.91 |
| LLaDA-V | MathVerse | 36.12 | 2.85 | 42.95 |

<p align="center"><em>Paper results · NVIDIA A100 80GB · batch size 1 · text generation length 256.</em> <a href="assets/data/table1.json">Full table data</a></p>

TPF is non-EOS response tokens per model forward; TPS is non-EOS response
tokens per second. See the [paper](https://arxiv.org/pdf/2607.14107) for the
full evaluation protocol.

## Quickstart

Use Python 3.10 or newer and a CUDA-capable NVIDIA GPU. Install Polestar from source:

```bash
git clone https://github.com/gt-synergy-lab/Polestar.git
cd Polestar
python -m pip install -e .
```

Generate a response with LLaDA-8B-Instruct:

```bash
python examples/generate.py --config configs/llada8b.json \
  --prompt "Natalia sold 48 clips in April and half as many in May. How many did she sell altogether?"
```

The example prints the generated response followed by `Model evaluations: ...`.
The checkpoint and tokenizer are loaded from Hugging Face and cached locally.
Set `HF_HOME` to a directory with room for the checkpoint, or set `model_id`
in the preset to a local checkpoint directory. Install a CUDA-enabled PyTorch
build compatible with your GPU before installing Polestar.

## Models and examples

| Model | Input | Preset | Generate | Evaluate |
|---|---|---|---|---|
| LLaDA-8B-Instruct | Text | [llada8b.json](configs/llada8b.json) | [Text example](examples/generate.py) | [Text benchmarks](#evaluation) |
| LLaDA-1.5 | Text | [llada15.json](configs/llada15.json) | [Text example](examples/generate.py) | [Text benchmarks](#evaluation) |
| Dream-7B-Instruct | Text | [dream7b.json](configs/dream7b.json) | [Text example](examples/generate.py) | [Text benchmarks](#evaluation) |
| LLaDA-V | Image + text | [llada-v.json](configs/llada-v.json) | [Image example](examples/image_generation.py) | [Image benchmarks](#evaluation) |

Switch text models by replacing the `--config` path in the quickstart. For an image-conditioned response with LLaDA-V:

```bash
python -m pip install -e '.[vision]'
python examples/image_generation.py --config configs/llada-v.json \
  --image path/to/image.jpg --prompt "Describe this image."
```

Each preset sets `model_type`, a Hugging Face or local `model_id`, and
generation length, steps, block length, and confidence thresholds. For a
512-token response, set both `max_new_tokens` and `steps` to 512 in the
preset. Dream uses the same standalone API.

<details>
<summary><b>Python API: load a preset and generate</b></summary>

```python
from polestar import PolestarConfig, generate, load_model

config = PolestarConfig.from_json("configs/dream7b.json")
model, tokenizer = load_model(config)
result = generate(model, tokenizer, "Explain bidirectional attention.", config)
print(result.text)
```

</details>

## LLaDA system presets

| Preset | Cache path |
|---|---|
| [llada8b-cuda-graphs.json](configs/llada8b-cuda-graphs.json) | Fixed-shape CUDA graphs for decode projections |
| [llada8b-stream-overlap.json](configs/llada8b-stream-overlap.json) | CUDA stream overlap |
| [llada8b-nvfp4.json](configs/llada8b-nvfp4.json) | Centroid-residual NVFP4 simulation |
| [llada8b-nvfp4-overlap.json](configs/llada8b-nvfp4-overlap.json) | Quantization, pinned-CPU offload, and overlap |

With the CUDA-graph preset, standalone LLaDA scored 13/16 on a 5-shot GSM8K
check on one GB300, with 3.332 non-EOS tokens per forward and 57.90 non-EOS
tokens/s. This small check uses different hardware from the paper's full
A100 evaluation. The offload path retains a GPU hidden-state copy for sparse
patching, so it does not provide the paper's GPU-memory reduction.

## dInfer serving

The optional [dInfer package](integrations/dinfer/README.md) runs
Polestar-Cache and Polestar-Commit in a dInfer worker. Set
`parallel_decoding="polestar", cache="polestar"`; the guide includes a complete
serving example and a two-question GSM8K check. In a matched 16-question
GB300 run, this path scored 12/16 and reached 383.42 warmed non-EOS tokens/s,
versus 10/16 and 330.32 tokens/s for dInfer's threshold and prefix baseline.
See the [measurement protocol](integrations/dinfer/README.md#measured-comparison).

## Evaluation

Install the text evaluation dependencies and run a benchmark:

```bash
python -m pip install -e '.[eval]'
python evaluation/run.py --config configs/llada8b.json --task gsm8k \
  --output-path results/llada8b-gsm8k-256
```

Add `--limit 16` for a short check, or `--limit 0.1` for 10% of the task.
The text evaluation presets use 5-shot GSM8K, 4-shot MATH
(`minerva_math`), 0-shot HumanEval, and 3-shot MBPP. Use
`--allow-code-execution` when scoring generated programs with the benchmark's
tests. The runner writes `results.json` and `samples_<task>.jsonl` to the
output directory.

For LLaDA-V evaluation, install `'.[vision-eval]'` in a separate environment
and run `evaluation/run.py` with `mathvista_testmini` or
`mathverse_testmini_vision`. Those task scorers require `OPENAI_API_KEY`;
`--predict-only` saves generations without scoring.

Questions and contributions are welcome through [issues](https://github.com/gt-synergy-lab/Polestar/issues) and [pull requests](https://github.com/gt-synergy-lab/Polestar/pulls).

## Citation

```bibtex
@inproceedings{lee2026polestar,
  title={Polestar: Drift-Aware Cache Calibration and Token Commitment for Efficient Inference of Diffusion {LLM}s},
  author={Mingyu Lee and Akshat Ramachandran and Souvik Kundu and Tushar Krishna},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2607.14107}
}
```

## License

Polestar's original code is provided under the [MIT License](LICENSE). Bundled model implementations retain their original copyright notices and licenses; see [Third-party notices](THIRD_PARTY_NOTICES.md).
