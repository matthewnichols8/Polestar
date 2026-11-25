"""Trace Polestar-Cache inside dInfer DiffusionGemma: prove the cache is called and that it cuts work.

Instruments (by wrapping, no library edits):
  - CanvasPolestarCache._run_full / _run_partial  : which path every layer of every denoising step took
  - Layer._ffn                                     : how many canvas tokens actually went through the FFN/MoE
  - Experts._weights / _weights_subset             : full vs sparse expert dequantization (experts materialized)
  - DiffusionGemmaModelLM.denoise                  : wall time of each step, split by full/partial
Then runs the same prompts with and without the cache (same decoder seed) and compares outputs.

    PYTHONPATH= HF_HOME=/scratch/mi72/hf TMPDIR=/scratch/mi72/tmp \
    /scratch/mi72/envs/polestar/bin/python trace_polestar_cache.py --gpus 0,1 --out trace.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import torch
import torch.distributed as dist

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
    "friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
]


class Trace:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = None                # "full" / "partial" for the step in progress
        self.layer_paths = defaultdict(int)
        self.ffn_tokens = defaultdict(list)       # mode -> tokens per layer call
        self.experts_full = defaultdict(int)       # mode -> full dequant calls
        self.experts_sparse = defaultdict(list)    # mode -> experts materialized per sparse call
        self.step_seconds = defaultdict(list)


def instrument(trace, device):
    from dinfer.model import modeling_diffusion_gemma as mdg
    from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

    def wrap(cls, name, before=None, after=None):
        original = getattr(cls, name)

        def wrapper(self, *args, **kwargs):
            if before:
                before(self, *args, **kwargs)
            result = original(self, *args, **kwargs)
            if after:
                after(self, result, *args, **kwargs)
            return result

        setattr(cls, name, wrapper)

    def mark(mode):
        def before(self, *a, **k):
            trace.mode = mode
            trace.layer_paths[mode] += 1
        return before

    wrap(CanvasPolestarCache, "_run_full", before=mark("full"))
    wrap(CanvasPolestarCache, "_run_partial", before=mark("partial"))
    wrap(mdg.Layer, "_ffn", before=lambda self, h: trace.ffn_tokens[trace.mode or "no_cache"].append(h.shape[1]))
    wrap(mdg.Experts, "_weights", before=lambda self: trace.experts_full.__setitem__(
        trace.mode or "no_cache", trace.experts_full[trace.mode or "no_cache"] + 1))
    wrap(mdg.Experts, "_weights_subset", before=lambda self, used: trace.experts_sparse[trace.mode].append(
        int(used.numel())))

    original_denoise = mdg.DiffusionGemmaModelLM.denoise

    def timed_denoise(self, canvas, cache, sc=None, layer_runner=None):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        trace.mode = None
        out = original_denoise(self, canvas, cache, sc, layer_runner=layer_runner)
        torch.cuda.synchronize(device)
        mode = "no_cache" if layer_runner is None else ("full" if layer_runner.full else "partial")
        trace.step_seconds[mode].append(time.perf_counter() - start)
        return out

    mdg.DiffusionGemmaModelLM.denoise = timed_denoise


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def worker(rank, gpus, args):
    device = torch.device("cuda", gpus[rank])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{args.port}", rank=rank,
                            world_size=len(gpus), device_id=device)
    from huggingface_hub import snapshot_download
    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, EntropyBoundParallelDecoder, KVCacheFactory
    from dinfer.model import DiffusionGemmaModelLM, load_diffusion_gemma_tokenizer
    from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

    trace = Trace()
    instrument(trace, device)
    path = snapshot_download(MODEL_ID)
    model = DiffusionGemmaModelLM.from_pretrained(path, device=device, expert_dtype="int8")
    tokenizer = load_diffusion_gemma_tokenizer(path)
    n_local_experts = model.layers[0].experts.n_loc
    report = {"n_local_experts_per_rank": n_local_experts, "num_layers": len(model.layers),
              "canvas_length": model.config.canvas_length, "runs": []}

    for prompt_text in PROMPTS:
        ids = torch.tensor([tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}],
                                                          add_generation_prompt=True, tokenize=True)], device=device)
        outputs = {}
        for backend in ("no_cache", "polestar"):
            decoder = EntropyBoundParallelDecoder(model.config.vocab_size, seed=42, device=device)
            cache = CanvasPolestarCache(len(model.layers), topk=2, refresh_every=8, polestar_seed=42) \
                if backend == "polestar" else None
            llm = CanvasDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
                                     canvas_cache=cache)
            trace.reset()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = llm.generate(ids, gen_length=512)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            outputs[backend] = text
            stats = llm.last_stats
            entry = {
                "prompt": prompt_text[:60], "backend": backend, "seconds": round(seconds, 2),
                "steps": stats["num_forwards"], "steps_per_canvas": stats["steps_per_canvas"],
                "new_tokens": stats["new_tokens"],
                "layer_calls": dict(trace.layer_paths),
                "ffn_tokens_mean": {m: mean(v) for m, v in trace.ffn_tokens.items()},
                "ffn_tokens_total": {m: sum(v) for m, v in trace.ffn_tokens.items()},
                "full_expert_dequants": dict(trace.experts_full),
                "sparse_expert_dequants": {m: len(v) for m, v in trace.experts_sparse.items()},
                "experts_materialized_mean_sparse": {m: mean(v) for m, v in trace.experts_sparse.items()},
                "ms_per_step": {m: round(1000 * mean(v), 1) for m, v in trace.step_seconds.items()},
                "steps_by_mode": {m: len(v) for m, v in trace.step_seconds.items()},
                "cache_stats": stats.get("canvas_cache"),
                "text": text,
            }
            report["runs"].append(entry)
            if rank == 0:
                print(json.dumps({k: v for k, v in entry.items() if k not in ("text", "cache_stats")}), flush=True)
        if rank == 0:
            report["runs"][-1]["same_text_as_no_cache"] = outputs["polestar"] == outputs["no_cache"]
            print("same output:", outputs["polestar"] == outputs["no_cache"], flush=True)
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--out", default="trace.json")
    p.add_argument("--port", type=int, default=29661)
    args = p.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    torch.multiprocessing.spawn(worker, args=(gpus, args), nprocs=len(gpus))
