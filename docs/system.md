# LLaDA system paths

The standalone LLaDA runner uses the model, decoder, and cache implementation
in `src/polestar/models/llada`. The dInfer package lives under
`integrations/dinfer` and applies a dInfer prefix-cache variant of
Polestar-Cache and Polestar-Commit. Dream uses the release's Dream decoder
and cluster cache.

## Standalone LLaDA

Install the package with evaluation dependencies and run GSM8K:

```bash
python -m pip install -e '.[eval]'
python evaluation/run.py --config configs/llada8b-cuda-graphs.json --task gsm8k \
  --limit 16 --seed 42 --output-path results/llada8b-cuda-graphs-gsm8k-16
```

The standard preset uses the paper's LLaDA-8B-Instruct checkpoint, 5-shot GSM8K, 256 response positions, 32-token blocks, batch size one, confidence threshold 0.9, an 8-centroid local cache, and a two-block prefix plus one-block suffix cache window. Use `configs/llada8b-stream-overlap.json` for overlapping cache work with the transformer forward pass. Use `configs/llada8b-nvfp4.json` for centroid-residual NVFP4 simulation, or `configs/llada8b-nvfp4-overlap.json` to combine quantization, pinned-CPU offload, and overlap.

The NVFP4 path uses E2M1 values and per-16-channel FP8 E4M3 scales. For offloaded cache rows, it packs two values per byte and reconstructs selected rows using their block-entry centroid. Sparse updates refresh the offloaded rows. The implementation currently retains a GPU copy of the hidden-state cache for sparse patching, so offload does not yet provide the paper's GPU-memory reduction. The offload and stream paths may be slower on some GPUs; measure them on the target hardware.

The evaluator reports accuracy, non-EOS generated tokens per model forward, and non-EOS generated tokens per wall-clock second. It completes the full denoising trajectory even if EOS appears early. Accuracy uses the benchmark's stop-sequence and answer-extraction rules.

## Measured LLaDA validation

On one NVIDIA GB300 with 16 GSM8K test questions, 5-shot prompts, generation length 256, block length 32, and seed 42, the CUDA-graph preset scored 13/16 with the harness's `flexible-extract` metric, with 3.332 non-EOS tokens per forward and 57.90 non-EOS tokens per second. These are small-sample measurements on different hardware from the paper's A100 full-test result, so neither the accuracy nor throughput is directly comparable to its final table. The measured throughput is 66.1% of the paper's 87.57 tokens per second. A profile of one 14-forward block shows substantial host-side work in the variable-size proxy and cache update path.

`configs/llada8b-cuda-graphs.json` is the measured faster preset on GB300. Graph preparation runs after model loading and before timed generation. `configs/llada8b-stream-overlap.json` enables stream overlap for GPUs where it helps. CUDA graphs require a CUDA device and a batch-one sequential worker.

## dInfer

Install the adapter from the release root:

```bash
python -m pip install -e integrations/dinfer
```

Set `parallel_decoding="polestar"` and `cache="polestar"` to run
Polestar-Cache and Polestar-Commit through dInfer. This path uses dInfer's
compiled LLaDA prefix decoder. It selects prefix rows with clustered
attention drift, updates the last layer's cache from the preceding layer,
and applies the update on the next forward pass. Final-layer attention
drift guides token commitment. This two-layer prefix implementation
differs from the standalone path's same-forward, multi-layer cache updates.
The adapter supports one LLaDA request per worker on one GPU. See the
[dInfer guide](../integrations/dinfer/README.md).
