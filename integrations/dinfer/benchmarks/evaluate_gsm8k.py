"""Compare Polestar and dInfer on the release GSM8K evaluator.

Both backends use dInfer's serving worker, the same LLaDA model and prompts,
and the same lm-evaluation-harness scoring as evaluation/run.py.
"""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


RELEASE_ROOT = Path(__file__).resolve().parents[3]

from dinfer.decoding.serving import DiffusionLLMServing, SamplingParams


def install_dinfer_decoder(*, backend: str = "polestar", prewarm_request: bool = False):
    """Route the existing evaluator's model call through dInfer."""
    if backend not in {"dinfer", "polestar"}:
        raise ValueError(f"unknown dInfer benchmark backend: {backend}")
    service = [None]
    previous_forwards = [0]
    request_timings = []
    warmup_matches = []
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 42

    def generate_via_dinfer(
        model,
        prompt,
        *,
        steps,
        gen_length,
        block_length,
        temperature,
        remasking,
        mask_id,
        threshold,
        threshold_early,
        use_cluster,
        enable_quantization,
        quantization_precision,
        enable_overlap,
        factor=None,
    ):
        if not use_cluster or factor is not None:
            raise ValueError("dInfer's Polestar adapter requires the clustered release decoder")
        if enable_quantization or enable_overlap:
            raise ValueError("The dInfer benchmark does not support quantization or overlap")
        params = SamplingParams(
            parallel_decoding="threshold" if backend == "dinfer" else "polestar",
            cache="polestar" if backend == "polestar" else "prefix",
            use_bd=False,
            enable_torch_compile=True,
            cont_weight=0.0,
            mask_id=mask_id,
            temperature=temperature,
            threshold=threshold,
            current_commit_lower_bound=threshold_early,
            polestar_steps=steps,
            polestar_seed=seed,
            polestar_profile_timing=True,
            eos_id=126081,
        )
        start = time.perf_counter()
        if service[0] is None:
            service[0] = DiffusionLLMServing(
                model.config.name_or_path,
                model_type="llada",
                sample_params=params,
                num_gpus=1,
                dp_size=1,
                tpep_size=1,
                backend="torch",
                timeout=600,
            )
        if prewarm_request:
            warmup_output = service[0].generate(
                prompt, gen_length=gen_length, block_length=block_length
            )
            previous_forwards[0] = service[0].num_forwards
            start = time.perf_counter()
        output = service[0].generate(prompt, gen_length=gen_length, block_length=block_length)
        if prewarm_request:
            warmup_matches.append(torch.equal(warmup_output, output))
        elapsed = time.perf_counter() - start
        response = output[:, prompt.shape[1]:]
        request_timings.append(
            (
                int((response != params.eos_id).sum().item()),
                gen_length,
                elapsed,
                service[0].last_decode_seconds,
            )
        )
        nfe = service[0].num_forwards - previous_forwards[0]
        previous_forwards[0] = service[0].num_forwards
        return output, nfe, [], prompt.shape[1]

    return generate_via_dinfer, service, request_timings, warmup_matches


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("The dInfer GSM8K benchmark requires CUDA")
    if "--serving" in sys.argv:
        sys.argv.remove("--serving")
    if "--cuda-graphs" in sys.argv:
        raise ValueError("--cuda-graphs is not a dInfer benchmark option")
    prewarm_request = "--prewarm-request" in sys.argv
    if prewarm_request:
        sys.argv.remove("--prewarm-request")
    backend = "polestar"
    if "--backend" in sys.argv:
        index = sys.argv.index("--backend")
        backend = sys.argv[index + 1]
        del sys.argv[index:index + 2]
    output_dir = Path(sys.argv[sys.argv.index("--output-path") + 1])
    generate_fn, service, request_timings, warmup_matches = install_dinfer_decoder(
        backend=backend, prewarm_request=prewarm_request,
    )
    print(f"dInfer LLaDA GPU: {torch.cuda.get_device_name(0)}", flush=True)
    # The worker owns the model. The evaluator only needs the model
    # path to start that worker, so avoid loading a duplicate 8B model here.
    model_factory = lambda path, _device: SimpleNamespace(
        config=SimpleNamespace(name_or_path=path)
    )
    try:
        runpy.run_path(str(RELEASE_ROOT / "evaluation" / "run.py"))["main"](
            llada_model_factory=model_factory,
            llada_generate_fn=generate_fn,
            execution_metadata={
                "integration": "dinfer",
                "backend": backend,
                "serving": True,
                "model_compile": True,
                "torch_compile_mode": "reduce-overhead",
                "prewarm_each_request": prewarm_request,
            },
        )
    finally:
        if service[0] is not None:
            service[0].stop_serving()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    if request_timings:
        total_tokens = sum(tokens for tokens, _, _, _ in request_timings)
        total_positions = sum(positions for _, positions, _, _ in request_timings)
        total_seconds = sum(seconds for _, _, seconds, _ in request_timings)
        decode_seconds = sum(seconds for _, _, _, seconds in request_timings)
        metrics = {
            "backend": backend,
            "prewarm_each_request": prewarm_request,
            "request_tps": total_tokens / total_seconds,
            "worker_decode_tps": total_tokens / decode_seconds,
            "worker_capacity_positions_per_second": total_positions / decode_seconds,
            "requests": [
                {
                    "generated_tokens": tokens,
                    "response_capacity_positions": positions,
                    "request_seconds": seconds,
                    "worker_decode_seconds": decode,
                }
                for tokens, positions, seconds, decode in request_timings
            ],
        }
        if prewarm_request:
            metrics["warmup_matches"] = warmup_matches
            metrics["all_warmups_match"] = all(warmup_matches)
        print(f"dInfer serving end-to-end request TPS: {metrics['request_tps']:.2f}")
        print(f"dInfer serving decode-only TPS: {metrics['worker_decode_tps']:.2f}")
        if len(request_timings) > 1:
            steady_tokens = sum(tokens for tokens, _, _, _ in request_timings[1:])
            steady_positions = sum(positions for _, positions, _, _ in request_timings[1:])
            steady_seconds = sum(seconds for _, _, seconds, _ in request_timings[1:])
            metrics["warm_worker_tps"] = steady_tokens / steady_seconds
            metrics["warm_worker_capacity_positions_per_second"] = steady_positions / steady_seconds
            print(f"dInfer serving warm-worker TPS: {metrics['warm_worker_tps']:.2f}")
            print(
                "dInfer serving warm-worker response capacity positions/s: "
                f"{metrics['warm_worker_capacity_positions_per_second']:.2f}"
            )
        (output_dir / "serving_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
