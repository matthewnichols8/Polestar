"""Polestar-Commit for DiffusionGemma's uniform-noise canvas in dInfer.

DiffusionGemma has no mask token: every canvas position is re-predicted on every denoising step, and the
entropy-bound sampler only *accepts* (keeps the sampled token of) the lowest-entropy positions for the next step,
re-noising the rest. So LLaDA's Polestar-Commit ("commit a masked token early when its final-layer attention drift
jumps", see `dinfer_commit.py`) is adapted as **drift-gated acceptance**: on each step, a position whose confidence
lies in [lower_bound, threshold) and whose drift jump clears the confidence-dependent gate is also accepted.

Modes of `PolestarCommitEntropyDecoder` (the drift rule):
  * ``off``    - no drift rule;
  * ``record`` - the drift rule is measured (and traced, with a recorder) but changes no decision;
  * ``accept`` - drift-gated acceptance on top of the entropy bound (never removes an entropy-bound acceptance).

Mapping of LLaDA's full Polestar-Commit (`dinfer_commit.py`) onto the canvas:
  1. Threshold commit -> **freeze** (`commit_freeze_threshold`, off by default). A position whose raw confidence
     reaches the threshold is committed permanently to its argmax token: it is accepted with that token on every
     later step of the canvas and is the token the canvas commits. LLaDA uses 0.9; on DiffusionGemma's GSM8K traces
     a token's first crossing of 0.9 matches the final canvas only 92-94% of the time (0.999: 99.7-99.8%), so a
     much higher threshold is needed. LLaDA's forced progress (commit at least one token per step) is not ported:
     the entropy bound already accepts at least one position per step and the step budget ends every canvas.
  2. Drift commit -> the modes above (per-step acceptance, not a freeze; drift-flagged tokens are final only
     56-72% of the time on GSM8K).
  3. Suffix commit -> no analogue: a denoising pass sees only the prefix and the current canvas, never the next.
  4. Commits drive Cache refreshes -> newly frozen positions are reported to `commit_callback` (CanvasDiffusionLLM
     connects it to `CanvasPolestarCache.record_commits`, see its `refresh_after_commits`).

The drift signal is the head-averaged KL between consecutive steps' final-layer attention distributions of each
canvas query (`compute_token_drift_kl`, as in LLaDA); a drift *jump* is the drift minus its mean over the previous
`window` steps of the same canvas. The suffix-block part of LLaDA's Commit has no analogue (one canvas at a time).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dinfer.decoding.parallel_strategy import EntropyBoundParallelDecoder, broadcast_if_needed
from polestar.models.llada.system_cache import compute_token_drift_kl

COMMIT_MODES = ("off", "record", "accept")
CONFIDENCE_KINDS = ("raw", "tempered")


class CanvasDriftTracker:
    """Per-token final-layer attention drift across the denoising steps of one canvas."""

    def __init__(self, window: int = 3):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.reset()

    def reset(self):
        self.previous = None
        self.history = []

    def update(self, attention: torch.Tensor):
        """attention: [B, heads, C, S]. Returns (drift, delta), each [B, C], or (None, None) on the first step
        (no previous distribution to compare with)."""
        if self.previous is None or self.previous.shape != attention.shape:
            self.reset()
            self.previous = attention.detach()
            return None, None
        drift = compute_token_drift_kl(attention, self.previous)
        delta = drift - torch.stack(self.history).mean(dim=0) if self.history else drift
        self.history.append(drift)
        if len(self.history) > self.window:
            self.history.pop(0)
        self.previous = attention.detach()
        return drift, delta


def select_canvas_drift_commits(confidence, delta, *, threshold, lower_bound, gate_scale=10.0, drift_cap=2.0):
    """LLaDA's Polestar-Commit rule without a mask: positions with lower_bound < conf < threshold whose drift jump
    clears `gate_scale * (threshold - conf)^2` (tighter as confidence falls) and stays below `drift_cap`."""
    gate = gate_scale * torch.clamp(threshold - confidence, min=0.0).square()
    return ((confidence > lower_bound) & (confidence < threshold)
            & (delta >= gate) & (delta < drift_cap))


def max_probability(logits):
    """Probability of the argmax token, [B, C] fp32."""
    return torch.softmax(logits.float(), dim=-1).amax(dim=-1)


class CommitTraceRecorder:
    """Writes per-canvas, per-step signals to `<out_dir>/<request>_c<canvas>.npz` for offline analysis.

    Arrays (T steps x C positions unless noted): cur_step [T], conf_raw, conf_tempered, entropy, drift (NaN on the
    first step), base_accepted, accepted, frozen (after the step), argmax, sampled (the model's, before freezing
    overrides them); final [C] (the committed canvas) and valid_len (positions
    before the first EOS, or C). Batch size 1.
    """

    def __init__(self, out_dir, eos_ids=(1, 106, 50)):
        self.out_dir = out_dir
        self.eos_ids = tuple(eos_ids)
        os.makedirs(out_dir, exist_ok=True)
        self.request = "request"
        self.canvas = 0
        self.steps = []

    def begin_request(self, name):
        self.request, self.canvas, self.steps = str(name), 0, []

    def record_step(self, **arrays):
        self.steps.append({k: (v[0].detach().float().cpu().numpy() if torch.is_tensor(v) else v)
                           for k, v in arrays.items()})

    def end_canvas(self, committed):
        final = committed[0].detach().cpu().numpy()
        eos = np.isin(final, self.eos_ids)
        valid_len = int(eos.argmax()) if eos.any() else final.shape[0]
        out = {"final": final, "valid_len": np.int64(valid_len),
               "cur_step": np.array([s["cur_step"] for s in self.steps], dtype=np.int64)}
        for key in ("conf_raw", "conf_tempered", "entropy", "drift"):
            out[key] = np.stack([s[key] for s in self.steps]).astype(np.float32)
        for key in ("base_accepted", "accepted", "frozen"):
            out[key] = np.stack([s[key] for s in self.steps]).astype(bool)
        for key in ("argmax", "sampled"):
            out[key] = np.stack([s[key] for s in self.steps]).astype(np.int64)
        path = os.path.join(self.out_dir, f"{self.request}_c{self.canvas}.npz")
        np.savez_compressed(path, **out)
        self.canvas += 1
        self.steps = []
        return path


class PolestarCommitEntropyDecoder(EntropyBoundParallelDecoder):
    """Entropy-bound decoder with Polestar-Commit's drift-gated acceptance (see the module docstring).

    `CanvasDiffusionLLM` passes the final-layer attention of every step to `observe_attention` (because
    `wants_final_attention` is True unless the mode is ``off``) and calls `end_canvas(committed)` when a canvas is
    done; the latter returns the canvas's commit statistics. Rank 0's decisions reach other ranks through the base
    decoder's broadcast of `x` and `argmax`; the newly frozen mask is broadcast too, so that every rank reports the
    same commit counts to the cache (whose full/partial step choice must agree across ranks).
    """

    def __init__(self, vocab_size, *, commit_mode="record", commit_threshold=0.9, commit_lower_bound=0.7,
                 commit_window=3, commit_gate_scale=10.0, commit_drift_cap=2.0, commit_confidence="raw",
                 commit_freeze_threshold=None, recorder=None, **kwargs):
        super().__init__(vocab_size, **kwargs)
        if commit_mode not in COMMIT_MODES:
            raise ValueError(f"commit_mode must be one of {COMMIT_MODES}")
        if commit_confidence not in CONFIDENCE_KINDS:
            raise ValueError(f"commit_confidence must be one of {CONFIDENCE_KINDS}")
        if not 0.0 <= commit_lower_bound < commit_threshold <= 1.0:
            raise ValueError("need 0 <= commit_lower_bound < commit_threshold <= 1")
        if commit_freeze_threshold is not None and not 0.0 < commit_freeze_threshold <= 1.0:
            raise ValueError("commit_freeze_threshold must be in (0, 1]")
        self.commit_mode = commit_mode
        self.commit_threshold = commit_threshold
        self.commit_lower_bound = commit_lower_bound
        self.commit_gate_scale = commit_gate_scale
        self.commit_drift_cap = commit_drift_cap
        self.commit_confidence = commit_confidence
        self.commit_freeze_threshold = commit_freeze_threshold
        self.commit_callback = None   # called with the number of newly frozen positions after every step
        self.tracker = CanvasDriftTracker(commit_window)
        self.recorder = recorder
        self._attention = None
        self._canvas_steps = []
        self.frozen = self.frozen_tokens = None

    @property
    def wants_final_attention(self):
        return self.commit_mode != "off"

    def observe_attention(self, probs):
        self._attention = probs

    def block_init(self, block_x, block_id):
        super().block_init(block_x, block_id)
        self.tracker.reset()
        self._attention = None
        self._canvas_steps = []
        self.frozen = self.frozen_tokens = None

    def _freeze(self, proposal, accepted, conf_raw):
        """Freeze newly confident positions, then force every frozen position to its frozen token."""
        argmax = proposal["argmax"]
        if self.frozen is None:
            self.frozen = torch.zeros_like(accepted)
            self.frozen_tokens = argmax.clone()
        new = ((conf_raw >= self.commit_freeze_threshold) & ~self.frozen).to(torch.uint8)
        broadcast_if_needed(new)
        new = new.bool()
        self.frozen_tokens = torch.where(new, argmax, self.frozen_tokens)
        self.frozen = self.frozen | new
        proposal["sampled"] = torch.where(self.frozen, self.frozen_tokens, proposal["sampled"])
        proposal["argmax"] = torch.where(self.frozen, self.frozen_tokens, argmax)
        return accepted | self.frozen, int(new.sum())

    def decode(self, logits, block_start, block_end, x, cur_step):
        proposal = self._propose(logits, cur_step)
        model_argmax, model_sampled = proposal["argmax"], proposal["sampled"]
        accepted = proposal["accepted"]
        conf_raw = max_probability(logits) if self.commit_mode != "off" or self.commit_freeze_threshold else None
        newly_frozen = 0
        commits = None
        if self.commit_mode != "off":
            drift = delta = None
            if self._attention is not None:
                drift, delta = self.tracker.update(self._attention)
                self._attention = None
            conf_tempered = max_probability(proposal["proc"])
            confidence = conf_raw if self.commit_confidence == "raw" else conf_tempered
            if delta is None:
                commits = torch.zeros_like(accepted)
            else:
                commits = select_canvas_drift_commits(
                    confidence, delta, threshold=self.commit_threshold, lower_bound=self.commit_lower_bound,
                    gate_scale=self.commit_gate_scale, drift_cap=self.commit_drift_cap)
            if self.commit_mode == "accept":
                accepted = accepted | commits
        if self.commit_freeze_threshold:
            accepted, newly_frozen = self._freeze(proposal, accepted, conf_raw)
        if self.commit_mode != "off" or self.commit_freeze_threshold:
            self._canvas_steps.append({
                "base": int(proposal["accepted"].sum()),
                "drift_commits": int(commits.sum()) if commits is not None else 0,
                "extra": int((commits & ~proposal["accepted"]).sum()) if commits is not None else 0,
                "frozen": newly_frozen,
            })
        if self.recorder is not None and self.commit_mode != "off":
            self.recorder.record_step(
                cur_step=cur_step, conf_raw=conf_raw, conf_tempered=conf_tempered,
                entropy=proposal["entropy"],
                drift=drift if drift is not None else torch.full_like(conf_raw, float("nan")),
                base_accepted=proposal["accepted"], accepted=accepted,
                frozen=self.frozen if self.frozen is not None else torch.zeros_like(accepted),
                argmax=model_argmax, sampled=model_sampled)
        done = self._finish(proposal, accepted, block_start, block_end, x)
        if self.commit_callback is not None:
            self.commit_callback(newly_frozen)
        return done

    def end_canvas(self, committed):
        steps = self._canvas_steps
        stats = {
            "mode": self.commit_mode,
            "steps": len(steps),
            "base_accepted": sum(s["base"] for s in steps),
            "drift_commits": sum(s["drift_commits"] for s in steps),
            "extra_accepted": sum(s["extra"] for s in steps),
            "frozen": sum(s.get("frozen", 0) for s in steps),
        }
        if self.recorder is not None and steps:
            stats["trace"] = self.recorder.end_canvas(committed)
        self._canvas_steps = []
        return stats
