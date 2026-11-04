# Third-party notices

Polestar's original package interfaces, presets, and documentation use the root MIT license. Imported and adapted code retains its original license and copyright notices.

| Code | Origin | License |
|---|---|---|
| LLaDA/Dream model and generation implementations; code sanitization/evaluation utilities | [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), with source references to [LLaDA](https://github.com/ML-GSAI/LLaDA) and [Dream](https://github.com/HKUNLP/Dream) in the files | Apache-2.0 for files marked by NVIDIA; preserve each file's header |
| LLaDA-V language/vision model, multimodal preparation, and generation integration | [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) / source LLaVA implementation | Apache-2.0 |
| LLaDA cache offload and Triton dequantization components; optional dInfer integration | Supplied `dllmKV` and `dInfer` system sources | Apache-2.0; the dInfer license is bundled in `integrations/dinfer/LICENSE` |
| DiffusionGemma model, entropy-bound decoder, per-layer prefix cache, and canvas diffusion loop in the dInfer integration | Re-implemented from the [Hugging Face Transformers](https://github.com/huggingface/transformers) DiffusionGemma text model and sampler, inside the bundled dInfer package | Apache-2.0 |

The bundled [dInfer Apache-2.0 license](integrations/dinfer/LICENSE) contains the full Apache-2.0 text applicable to the listed Apache-licensed components. File-level notices govern bundled third-party code. Model checkpoints, datasets, and separately installed evaluation libraries retain their own licenses and are not distributed with this repository.
