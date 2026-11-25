"""OpenAI-compatible chat-completions server for DiffusionGemma in dInfer (Phase 1).

    PYTHONPATH= HF_HOME=/scratch/mi72/hf TMPDIR=/scratch/mi72/tmp \
    /scratch/mi72/envs/polestar/bin/python serve_openai.py --gpus 0,1 --port 8000

Any OpenAI/LiteLLM client can then use it (model name is ignored):
    client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
    litellm.completion(model="openai/diffusiongemma", api_base="http://localhost:8000/v1", api_key="x", ...)

Design
- One process per GPU (expert parallel). Rank 0 runs the HTTP server in a thread and the generation loop in its
  main thread; for each request it broadcasts (prompt ids, budget, reuse length, seed) and every rank decodes in
  lockstep, exactly like the GSM8K benchmark. Requests are served one at a time (batch size 1).
- Prompts are rendered with the checkpoint's chat template (see gemma_format.py); the model's output is parsed
  back into OpenAI `tool_calls`, dropping any tool response the model writes for itself.
- Prefix reuse: the server keeps the prompt KV caches of the last `--cache-slots` requests. If a new prompt
  starts with one of them (an agent conversation that grew by appending), only the new tokens are encoded.
- `max_tokens` is rounded up to whole 256-token canvases. `temperature`/`top_p` are ignored: the decoder has its
  own noise schedule. `seed` re-seeds the decoder for that request. `n` must be 1. The non-standard
  `cache_reuse: false` forces a full prompt encode (for measuring the benefit of reuse).
- Per-request diagnostics are returned in a non-standard `dgemma` field and appended to --log-dir/requests.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import datetime
import faulthandler
import json
import queue
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from fastapi import Request  # module level: FastAPI must resolve the (postponed) annotation below

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gemma_format import (RequestError, apply_stop, gemma_special_tokens, openai_message,  # noqa: E402
                          parse_output, render)

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
SERVED_NAME = "diffusiongemma"


# ---------------------------------------------------------------------------------------------- engine (all ranks)
class Engine:
    """The model, decoder, optional Polestar-Cache and the prompt-cache slots. Identical on every rank."""

    def __init__(self, args, device):
        from huggingface_hub import snapshot_download
        from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, EntropyBoundParallelDecoder, KVCacheFactory
        from dinfer.model import DiffusionGemmaModelLM, load_diffusion_gemma_tokenizer

        self.device = device
        path = snapshot_download(MODEL_ID)
        self.model = DiffusionGemmaModelLM.from_pretrained(path, device=device, expert_dtype=args.expert_dtype)
        self.tokenizer = load_diffusion_gemma_tokenizer(path)
        self.canvas_length = self.model.config.canvas_length
        decoder_kwargs = dict(entropy_bound=args.entropy_bound, max_denoising_steps=args.max_denoising_steps,
                              seed=args.seed, device=device)
        if args.polestar_commit:
            # Full Polestar-Commit, as in integrations/dinfer/configs/diffusiongemma26b.json: drift-gated acceptance,
            # permanent freeze at high confidence, and (with --polestar) Cache refreshes driven by commits.
            from polestar.integrations.dinfer_canvas_commit import PolestarCommitEntropyDecoder

            self.decoder = PolestarCommitEntropyDecoder(
                self.model.config.vocab_size, commit_mode="accept", commit_threshold=args.commit_threshold,
                commit_lower_bound=args.commit_lower_bound, commit_window=args.commit_window,
                commit_gate_scale=args.commit_gate_scale, commit_drift_cap=args.commit_drift_cap,
                commit_confidence="raw", commit_freeze_threshold=args.commit_freeze_threshold, **decoder_kwargs)
        else:
            self.decoder = EntropyBoundParallelDecoder(self.model.config.vocab_size, **decoder_kwargs)
        canvas_cache = None
        if args.polestar:
            from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

            canvas_cache = CanvasPolestarCache(len(self.model.layers), topk=args.polestar_topk,
                                               refresh_every=args.polestar_refresh_every, polestar_seed=args.seed,
                                               refresh_after_commits=(args.polestar_refresh_after_commits
                                                                      if args.polestar_commit else None))
        self.llm = CanvasDiffusionLLM(self.model, self.decoder, BlockIteratorFactory(),
                                      KVCacheFactory("prefix", per_layer=True), canvas_cache=canvas_cache)
        self.cache_slots = args.cache_slots
        self.slots = []  # [(ids list, PerLayerKVCache snapshot)], most recently used last

    def find_prefix(self, ids):
        """(slot index, reuse length) of the longest cached prompt that `ids` starts with; (-1, 0) if none."""
        best, best_len = -1, 0
        for i, (cached, _) in enumerate(self.slots):
            if best_len < len(cached) <= len(ids) and ids[:len(cached)] == cached:
                best, best_len = i, len(cached)
        return best, best_len

    def longest_common_prefix(self, ids):
        """Diagnostics only: how far the best slot matches before diverging."""
        best = 0
        for cached, _ in self.slots:
            n = 0
            for a, b in zip(cached, ids):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        return best

    def generate(self, ids, gen_length, slot, seed):
        prefix_cache = None
        if slot >= 0:
            prefix_cache = self.slots[slot][1]
            self.slots.append(self.slots.pop(slot))
        if seed >= 0:
            self.decoder.generator.manual_seed(seed)
        prompt = torch.tensor([ids], device=self.device)
        out = self.llm.generate(prompt, gen_length=gen_length, prefix_cache=prefix_cache)
        self.slots.append((list(ids), self.llm.last_prompt_cache))
        del self.slots[:max(0, len(self.slots) - self.cache_slots)]  # --cache-slots 0 disables reuse
        return out[0, len(ids):]


def _broadcast_request(header, ids, device):
    """Rank 0 sends header [n_ids, gen_length, slot, seed] (n_ids < 0 = stop) and the ids; all ranks get both."""
    h = torch.tensor(header if header is not None else [0, 0, 0, 0], dtype=torch.long, device=device)
    dist.broadcast(h, src=0)
    n_ids, gen_length, slot, seed = (int(v) for v in h.tolist())
    if n_ids < 0:
        return None
    t = torch.tensor([ids], dtype=torch.long, device=device) if ids is not None else \
        torch.empty(1, n_ids, dtype=torch.long, device=device)
    dist.broadcast(t, src=0)
    return t[0].tolist(), gen_length, slot, seed


# ---------------------------------------------------------------------------------------------- HTTP (rank 0)
class Job:
    def __init__(self, ids, gen_length, seed, reuse=True):
        self.ids, self.gen_length, self.seed, self.reuse = ids, gen_length, seed, reuse
        self.future = concurrent.futures.Future()


def build_app(engine, jobs, args):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    specials = gemma_special_tokens(engine.tokenizer)
    log_file = None
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        log_file = (Path(args.log_dir) / "requests.jsonl").open("a")

    def error(status, message, code=None):
        return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error",
                                                                   "code": code}})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("n", 1) != 1:
            return error(400, "n must be 1")
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", args.thinking))
        try:
            ids = render(engine.tokenizer, body.get("messages") or [], tools, thinking)
        except RequestError as exc:
            return error(400, str(exc))
        if len(ids) > args.max_prompt_tokens:
            # OpenAI's wording and code, so clients (LiteLLM) recognise it as a context-window error and stop
            return error(400, f"This model's maximum context length is {args.max_prompt_tokens} tokens. However, your "
                              f"messages resulted in {len(ids)} tokens.", code="context_length_exceeded")
        budget = body.get("max_completion_tokens") or body.get("max_tokens") or args.default_max_tokens
        budget = max(1, min(int(budget), args.max_new_tokens))
        gen_length = -(-budget // engine.canvas_length) * engine.canvas_length
        seed = body.get("seed")
        job = Job(ids, gen_length, -1 if seed is None else int(seed) % (2 ** 62), body.get("cache_reuse", True))
        jobs.put(job)
        try:
            new_ids, stats = await asyncio.wrap_future(job.future)
        except Exception as exc:  # generation failed on rank 0
            return error(500, f"generation failed: {exc}")

        text = engine.tokenizer.decode(new_ids, skip_special_tokens=False)
        parsed = parse_output(text, specials)
        message = openai_message(parsed)
        hit_stop = False
        if message["content"] and not message.get("tool_calls"):
            message["content"], hit_stop = apply_stop(message["content"], body.get("stop"))
        if message.get("tool_calls"):
            finish = "tool_calls"
        elif stats["finished"] or hit_stop:
            finish = "stop"
        else:
            finish = "length"
        completion_tokens = len(new_ids)
        diag = {
            "prompt_tokens": len(ids), "reused_prompt_tokens": stats["reused"],
            "common_prefix_tokens": stats["common_prefix"], "new_tokens": completion_tokens,
            "steps": stats["steps"], "steps_per_canvas": stats["steps_per_canvas"],
            "seconds": round(stats["seconds"], 3), "queue_seconds": round(stats["queue_seconds"], 3),
            "self_tool_response": parsed["self_tool_response"], "malformed_calls": len(parsed["call_errors"]),
            "thinking": thinking, "polestar_cache": stats["canvas_cache"] is not None,
        }
        if stats["canvas_cache"] is not None:
            diag["canvas_cache"] = stats["canvas_cache"]
        diag["polestar_commit"] = stats["commit"] is not None
        if stats["commit"] is not None:
            diag["commit"] = json.loads(json.dumps(stats["commit"], default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": SERVED_NAME,
            "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": completion_tokens,
                      "total_tokens": len(ids) + completion_tokens},
            "dgemma": diag,
        }
        if log_file:
            log_file.write(json.dumps({"time": time.time(), **diag, "finish_reason": finish,
                                       "n_tool_calls": len(message.get("tool_calls") or []), "raw": text}) + "\n")
            log_file.flush()
        print(f"[serve] prompt={len(ids):6d} reused={stats['reused']:6d} new={completion_tokens:4d} "
              f"steps={stats['steps']:3d} {stats['seconds']:6.2f}s finish={finish}", flush=True)
        if body.get("stream"):
            return StreamingResponse(_stream(response), media_type="text/event-stream")
        return response

    return app


def _stream(response):
    """Streaming fallback: the whole completion arrives in one delta (diffusion has no token-by-token order)."""
    choice = response["choices"][0]
    message = choice["message"]
    base = {k: response[k] for k in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    delta = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        delta["tool_calls"] = [dict(c, index=i) for i, c in enumerate(message["tool_calls"])]
    if message.get("reasoning_content"):
        delta["reasoning_content"] = message["reasoning_content"]
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
             "usage": response["usage"]}
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _serve_http(app, args):
    import uvicorn

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    uvicorn.Server(config).run()  # in a non-main thread uvicorn skips installing signal handlers


# ---------------------------------------------------------------------------------------------- workers
def _exit_with_parent():
    """Followers otherwise outlive a killed launcher, blocked in a broadcast and holding their GPU."""
    import os

    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(2)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def worker(rank, gpus, args):
    _exit_with_parent()
    # `kill -USR1 <pid>` prints every thread's Python stack to the server log (py-spy needs ptrace, which is blocked
    # here) - run_all.sh does this before restarting a server that has stopped making progress.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    device = torch.device("cuda", gpus[rank])
    torch.cuda.set_device(device)
    # The followers block in a broadcast while the server is idle; don't let NCCL's watchdog kill them.
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{args.dist_port}", rank=rank,
                            world_size=len(gpus), device_id=device, timeout=datetime.timedelta(days=30))
    try:
        engine = Engine(args, device)
        if rank != 0:
            while (request := _broadcast_request(None, None, device)) is not None:
                ids, gen_length, slot, seed = request
                engine.generate(ids, gen_length, slot, seed)
            return

        print(engine.model.placement_summary(), flush=True)
        jobs = queue.Queue()
        threading.Thread(target=_serve_http, args=(build_app(engine, jobs, args), args), daemon=True).start()
        print(f"[serve] listening on http://{args.host}:{args.port}/v1 (model name: {SERVED_NAME})", flush=True)
        try:
            while True:
                job = jobs.get()
                queued = time.perf_counter()
                try:
                    slot, reused = engine.find_prefix(job.ids) if job.reuse else (-1, 0)
                    common = reused if slot >= 0 else engine.longest_common_prefix(job.ids)
                    _broadcast_request([len(job.ids), job.gen_length, slot, job.seed], job.ids, device)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    new = engine.generate(job.ids, job.gen_length, slot, job.seed)
                    torch.cuda.synchronize(device)
                    s = engine.llm.last_stats
                    job.future.set_result((new.tolist(), {
                        "reused": s["reused_prompt_tokens"], "common_prefix": common, "finished": s["finished"],
                        "steps": s["num_forwards"], "steps_per_canvas": s["steps_per_canvas"],
                        "seconds": time.perf_counter() - start, "queue_seconds": start - queued,
                        "canvas_cache": s.get("canvas_cache"), "commit": s.get("commit"),
                    }))
                except Exception as exc:  # noqa: BLE001 - report to the client; ranks may now be out of step
                    job.future.set_exception(exc)
                    # Exit at once: the other ranks are still inside this request's collectives, so the normal
                    # teardown (stop broadcast, destroy_process_group) would block forever and look like a hang.
                    # Exiting makes it a visible crash (traceback in the log) that the launcher/run_all.sh restarts.
                    import os
                    import traceback

                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    time.sleep(1)   # let the HTTP thread send the 500 to the client
                    os._exit(1)
        finally:
            _broadcast_request([-1, 0, 0, 0], None, device)
    finally:
        dist.destroy_process_group()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gpus", default="0,1", help="comma-separated GPU ids; experts are split across them")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dist-port", type=int, default=29651)
    p.add_argument("--expert-dtype", choices=["bf16", "int8"], default="int8")
    p.add_argument("--entropy-bound", type=float, default=0.1)
    p.add_argument("--max-denoising-steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--polestar", action="store_true", help="enable Polestar-Cache on the canvas")
    p.add_argument("--polestar-topk", type=int, default=2)
    p.add_argument("--polestar-refresh-every", type=int, default=8)
    p.add_argument("--polestar-commit", action="store_true",
                   help="enable full Polestar-Commit (drift-gated acceptance + freeze + commit-driven Cache refresh)")
    p.add_argument("--commit-threshold", type=float, default=0.9)
    p.add_argument("--commit-lower-bound", type=float, default=0.7)
    p.add_argument("--commit-window", type=int, default=3)
    p.add_argument("--commit-gate-scale", type=float, default=10.0)
    p.add_argument("--commit-drift-cap", type=float, default=2.0)
    p.add_argument("--commit-freeze-threshold", type=float, default=0.999)
    p.add_argument("--polestar-refresh-after-commits", type=int, default=24)
    p.add_argument("--thinking", action="store_true", help="default for chat_template_kwargs.enable_thinking")
    p.add_argument("--default-max-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-prompt-tokens", type=int, default=32768)
    p.add_argument("--cache-slots", type=int, default=4, help="prompt caches kept for prefix reuse")
    p.add_argument("--log-dir", help="append per-request diagnostics to LOG_DIR/requests.jsonl")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the server requires CUDA")
    gpus = [int(g) for g in args.gpus.split(",")]
    torch.multiprocessing.spawn(worker, args=(gpus, args), nprocs=len(gpus))


if __name__ == "__main__":
    main()
