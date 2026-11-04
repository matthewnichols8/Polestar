"""Compare dInfer DiffusionGemma with and without Polestar-Cache on the release text evaluator.

Both backends run google/diffusiongemma-26B-A4B-it through dInfer's CanvasDiffusionLLM with the same prompts
and lm-evaluation-harness scoring as evaluation/run.py (tasks and few-shot counts from
configs/evaluation/text.json; the few-shot context is sent as one chat turn and the answer is cut at the
task's stop sequences). `--backend polestar` adds CanvasPolestarCache.

The model's experts are split across the listed GPUs (expert parallel): rank 0 runs lm-eval and broadcasts
each prompt; the other ranks decode in lockstep.

    python integrations/dinfer/benchmarks/evaluate_gsm8k_diffusion_gemma.py \
        --backend polestar --task gsm8k --limit 16 --seed 42 --gpus 0,1 \
        --output-path results/dgemma-polestar-gsm8k16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

RELEASE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "diffusiongemma26b.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate DiffusionGemma in dInfer, with or without Polestar-Cache")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backend", choices=["dinfer", "polestar"], default="polestar")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, help="number of examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids; experts are split across them")
    parser.add_argument("--expert-dtype", choices=["bf16", "int8"], help="override the preset")
    parser.add_argument("--max-new-tokens", type=int, help="override the preset")
    parser.add_argument("--polestar-topk", type=int, help="override the preset")
    parser.add_argument("--polestar-refresh-every", type=int, help="override the preset")
    parser.add_argument("--prewarm-request", action="store_true", help="run each prompt once before timing it")
    parser.add_argument("--allow-code-execution", action="store_true")
    parser.add_argument("--master-port", type=int, default=29613)
    return parser.parse_args(argv)


def load_settings(args):
    preset = json.loads(Path(args.config).read_text())
    if preset.get("model_type") != "diffusion_gemma":
        raise ValueError("the preset must set model_type to diffusion_gemma")
    generation = dict(preset.get("generation", {}))
    polestar = dict(preset.get("polestar", {}))
    for key, value in [("expert_dtype", args.expert_dtype), ("max_new_tokens", args.max_new_tokens)]:
        if value is not None:
            generation[key] = value
    for key, value in [("topk", args.polestar_topk), ("refresh_every", args.polestar_refresh_every)]:
        if value is not None:
            polestar[key] = value
    return preset["model_id"], generation, polestar


def resolve_checkpoint(model_id):
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id)


class _Generator:
    """One rank's model, decoder and (optionally) Polestar-Cache."""

    def __init__(self, path, generation, polestar, backend, seed, device):
        from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, EntropyBoundParallelDecoder, KVCacheFactory
        from dinfer.model import DiffusionGemmaModelLM

        self.model = DiffusionGemmaModelLM.from_pretrained(
            path, device=device, expert_dtype=generation.get("expert_dtype", "bf16")
        )
        decoder = EntropyBoundParallelDecoder(
            self.model.config.vocab_size,
            max_denoising_steps=generation.get("max_denoising_steps", 48),
            entropy_bound=generation.get("entropy_bound", 0.1),
            t_min=generation.get("t_min", 0.4),
            t_max=generation.get("t_max", 0.8),
            seed=seed,
            device=device,
        )
        self.cache = None
        if backend == "polestar":
            from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

            self.cache = CanvasPolestarCache(len(self.model.layers), polestar_seed=seed, **polestar)
        self.llm = CanvasDiffusionLLM(
            self.model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
            canvas_cache=self.cache,
        )
        self.gen_length = generation.get("max_new_tokens", 512)

    def generate(self, ids):
        return self.llm.generate(ids, gen_length=self.gen_length)


def _broadcast_ids(ids, device):
    """Rank 0 sends a prompt (or None to stop); returns the prompt on every rank."""
    length = torch.tensor([-1 if ids is None else ids.shape[1]], device=device)
    dist.broadcast(length, src=0)
    if int(length) < 0:
        return None
    if ids is None:
        ids = torch.empty(1, int(length), dtype=torch.long, device=device)
    ids = ids.to(device)
    dist.broadcast(ids, src=0)
    return ids


def _harness(generator, tokenizer, device, prewarm, timings):
    from lm_eval.api.model import LM

    class DiffusionGemmaHarness(LM):
        def generate_until(self, requests):
            outputs = []
            for request in requests:
                context, options = request.args
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": context}], add_generation_prompt=True, tokenize=True
                )
                ids = torch.tensor([prompt], device=device)
                if prewarm:
                    _broadcast_ids(ids, device)
                    generator.generate(ids)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                _broadcast_ids(ids, device)
                out = generator.generate(ids)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                stats = generator.llm.last_stats
                text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                for stop in options.get("until", []):
                    text = text.split(stop)[0]
                fractions = [f for canvas in stats.get("canvas_cache", []) for f in canvas["recompute_frac"]]
                timings.append({
                    "prompt_tokens": ids.shape[1], "new_tokens": stats["new_tokens"], "seconds": elapsed,
                    "steps": stats["num_forwards"], "steps_per_canvas": stats["steps_per_canvas"],
                    "recompute_frac": fractions,
                })
                outputs.append(text)
            return outputs

        def loglikelihood(self, requests):
            raise NotImplementedError("DiffusionGemma evaluation supports generation tasks only")

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("DiffusionGemma evaluation supports generation tasks only")

    return DiffusionGemmaHarness()


def _summary(timings):
    tokens = sum(t["new_tokens"] for t in timings)
    seconds = sum(t["seconds"] for t in timings)
    steps = sum(t["steps"] for t in timings)
    fractions = [f for t in timings for f in t["recompute_frac"]]
    return {
        "requests": len(timings),
        "new_tokens": tokens,
        "seconds": seconds,
        "steps": steps,
        "tokens_per_second": tokens / seconds if seconds else None,
        "seconds_per_step": seconds / steps if steps else None,
        "tokens_per_step": tokens / steps if steps else None,
        "mean_recompute_frac_partial_steps": sum(fractions) / len(fractions) if fractions else None,
    }


def worker(rank, gpus, args):
    device = torch.device("cuda", gpus[rank])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{args.master_port}",
                            rank=rank, world_size=len(gpus), device_id=device)
    try:
        model_id, generation, polestar = load_settings(args)
        path = resolve_checkpoint(model_id)
        generator = _Generator(path, generation, polestar, args.backend, args.seed, device)
        if rank != 0:
            while (ids := _broadcast_ids(None, device)) is not None:
                generator.generate(ids)
            return

        from dinfer.model import load_diffusion_gemma_tokenizer
        import lm_eval

        suite = json.loads((RELEASE_ROOT / "configs" / "evaluation" / "text.json").read_text())
        task = suite["tasks"][args.task]
        if task.get("code_execution"):
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print(generator.model.placement_summary(), flush=True)
        timings = []
        harness = _harness(generator, load_diffusion_gemma_tokenizer(path), device, args.prewarm_request, timings)
        try:
            results = lm_eval.simple_evaluate(
                model=harness, tasks=[args.task], num_fewshot=task["num_fewshot"], batch_size=1,
                limit=args.limit, log_samples=True, confirm_run_unsafe_code=args.allow_code_execution,
                random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
                fewshot_random_seed=args.seed,
            )
        finally:
            _broadcast_ids(None, device)
        samples = results.pop("samples", {})
        results["polestar"] = {
            "model_preset": args.config, "model_id": model_id, "backend": args.backend,
            "generation": generation, "polestar_cache": polestar if args.backend == "polestar" else None,
            "seed": args.seed, "limit": args.limit, "gpus": gpus,
        }
        output = Path(args.output_path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
        for task_name, rows in samples.items():
            with (output / f"samples_{task_name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str) + "\n")
        metrics = {"summary": _summary(timings), "requests": timings}
        (output / "serving_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(results.get("results", {}), indent=2, default=str))
        print(json.dumps(metrics["summary"], indent=2))
        print(f"Saved results, samples and serving metrics to {output}")
    finally:
        dist.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the DiffusionGemma benchmark requires CUDA")
    if (Path(args.output_path) / "results.json").exists():
        raise SystemExit("results.json already exists; choose a new output directory")
    suite = json.loads((RELEASE_ROOT / "configs" / "evaluation" / "text.json").read_text())
    if args.task not in suite["tasks"]:
        raise SystemExit("select a task from configs/evaluation/text.json")
    if suite["tasks"][args.task].get("code_execution") and not args.allow_code_execution:
        raise SystemExit("this task executes generated code; pass --allow-code-execution")
    load_settings(args)  # validate the preset before starting workers
    gpus = [int(g) for g in args.gpus.split(",")]
    torch.multiprocessing.spawn(worker, args=(gpus, args), nprocs=len(gpus))


if __name__ == "__main__":
    main()
