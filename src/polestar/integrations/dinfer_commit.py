"""Attention-drift commitment and selective prefix caching for dInfer LLaDA."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dinfer.decoding.generate_uniform import BlockWiseDiffusionLLM
from dinfer.decoding.parallel_strategy import ParallelDecoder, broadcast_if_needed
from dinfer.decoding.utils import BlockIteratorFactory, KVCacheFactory
from polestar.models.llada.system_cache import compute_token_drift_kl

from .dinfer_cache import PolestarPrefixCache


@torch.compile(dynamic=True)
def _select_commits(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    drift_delta: torch.Tensor,
    *,
    mask_id: int,
    threshold: float,
    lower_bound: float,
    allow_drift: bool,
):
    prediction = logits.argmax(dim=-1)
    confidence = F.softmax(logits.float(), dim=-1).gather(
        -1, prediction.unsqueeze(-1)
    ).squeeze(-1)
    masked = tokens == mask_id
    masked_confidence = torch.where(
        masked, confidence, torch.full_like(confidence, -torch.inf)
    )
    best_position = masked_confidence.argmax(dim=1, keepdim=True)
    forced_progress = torch.zeros_like(masked).scatter_(1, best_position, True)
    high_confidence = masked & (
        (masked_confidence >= threshold) | forced_progress
    )
    if allow_drift:
        gate = 10 * torch.clamp(threshold - confidence, min=0.0).square()
        drift_commit = (
            masked
            & (confidence > lower_bound)
            & (confidence < threshold)
            & (drift_delta >= gate)
            & (drift_delta < 2.0)
        )
        high_confidence = high_confidence | drift_commit
    return torch.where(masked, prediction, tokens), high_confidence


class PolestarDriftDecoder(ParallelDecoder):
    """Use final-layer attention KL to commit eligible masked positions."""

    def __init__(
        self, *, threshold: float, lower_bound: float, mask_id: int, eos_id: int
    ):
        super().__init__(temperature=0.0, remasking="low_confidence", mask_id=mask_id)
        self.threshold = threshold
        self.lower_bound = lower_bound
        self.eos_id = eos_id
        self.current_attention = None
        self.previous_attention = None
        self.drift_window = []
        self.block_length = None
        self.nominal_block_length = None
        self.first_block_start = None
        self.block_start = None

    def block_init(self, block_x, block_id):
        self.block_length = block_x.shape[1]
        self.block_start = self.first_block_start + block_id * self.nominal_block_length
        self.current_attention = None
        self.previous_attention = None
        self.drift_window.clear()

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        attention = self.current_attention
        drift_delta = torch.zeros_like(logits[..., 0])
        allow_drift = attention is not None and self.previous_attention is not None
        if allow_drift:
            common_queries = min(attention.size(-2), self.previous_attention.size(-2))
            common_keys = min(attention.size(-1), self.previous_attention.size(-1))
            drift = compute_token_drift_kl(
                attention[..., -common_queries:, -common_keys:],
                self.previous_attention[..., -common_queries:, -common_keys:],
            )
            if self.drift_window:
                drift_delta = drift - torch.stack(self.drift_window).mean(dim=0)
            else:
                drift_delta = drift
            self.drift_window.append(drift.detach())
            if len(self.drift_window) > 5:
                self.drift_window.pop(0)
        if attention is not None:
            # The bridge creates attention outside the compiled model on each call.
            self.previous_attention = attention.detach()
        tokens = x[:, block_start:block_end]
        predicted, transfer = _select_commits(
            logits,
            tokens,
            drift_delta,
            mask_id=self.mask_id,
            threshold=self.threshold if iter_threshold is None else iter_threshold,
            lower_bound=self.lower_bound,
            allow_drift=allow_drift,
        )
        x[:, block_start:block_end] = torch.where(transfer, predicted, tokens)
        broadcast_if_needed(x.data)


class _AttentionBridge:
    """Reconstruct current-block attention from explicit compiled-model outputs."""

    def __init__(self, model, decoder: PolestarDriftDecoder, cache: PolestarPrefixCache):
        self.model = model
        self.decoder = decoder
        self.cache = cache

    @property
    def device(self):
        return self.model.device

    def __call__(self, *args, **kwargs):
        is_prefill = kwargs.get("past_key_values") is None
        kwargs["output_polestar_states"] = True
        torch.compiler.cudagraph_mark_step_begin()
        output = self.model(*args, **kwargs)
        start = self.decoder.block_start if is_prefill else 0
        end = start + self.decoder.block_length
        final_block = self.model.model.transformer.blocks[-1]
        final_input = output.hidden_states[-1][:, start:end, :]
        normalized = final_block.attn_norm(final_input)
        batch, length, _ = normalized.shape
        query = final_block.q_proj(normalized)
        key = output.past_key_values[-2]
        if final_block.q_norm is not None:
            query = final_block.q_norm(query).to(dtype=key.dtype)
        query = query.view(
            batch, length, final_block.config.n_heads // final_block.tp_size, -1
        ).transpose(1, 2)
        query, key = final_block.rotary_emb(
            query, key, block_end_index=self.decoder.block_start + self.decoder.block_length
        )
        if key.size(1) != query.size(1):
            key = key.repeat_interleave(query.size(1) // key.size(1), dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (
            query.size(-1) ** -0.5
        )
        self.decoder.current_attention = torch.softmax(scores, dim=-1)
        if is_prefill:
            self.cache.begin_block(
                output,
                (self.decoder.block_start, self.decoder.block_start + self.decoder.block_length),
                hidden_by_layer={
                    self.cache.source_layer: output.hidden_states[-2],
                    self.cache.target_layer: output.hidden_states[-1],
                },
            )
        else:
            self.cache.after_dense_step(output, kwargs["past_key_values"])
        return output


class PolestarPrefixDiffusionLLM(BlockWiseDiffusionLLM):
    request_seed: int = 42
    max_steps: int | None = None

    def generate(self, prompt, gen_length=128, block_length=128):
        if self.max_steps is not None and self.max_steps != gen_length:
            raise ValueError("Polestar dInfer decoding requires steps=gen_length")
        torch.manual_seed(self.request_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.request_seed)
        first_unmasked = (prompt != self.decoder.mask_id).sum(dim=-1).min().item()
        self.decoder.first_block_start = first_unmasked // block_length * block_length
        self.decoder.nominal_block_length = block_length
        self.model.cache.refresh_count = 0
        self.model.cache.effective_patched_rows = 0
        return super().generate(prompt, gen_length=gen_length, block_length=block_length)


def make_polestar_drift_generator(
    model,
    *,
    threshold: float,
    lower_bound: float,
    mask_id: int,
    eos_id: int,
    cache_interval: int = 8,
    request_seed: int = 42,
    steps: int | None = None,
):
    """Build dInfer decoding with Polestar Cache and attention-KL Commit."""
    decoder = PolestarDriftDecoder(
        threshold=threshold, lower_bound=lower_bound, mask_id=mask_id, eos_id=eos_id
    )
    cache = PolestarPrefixCache(model, refresh_interval=cache_interval)
    generator = PolestarPrefixDiffusionLLM(
        _AttentionBridge(model, decoder, cache),
        decoder,
        BlockIteratorFactory(start_block_align=True),
        cache_factory=KVCacheFactory("prefix"),
        early_stop=True,
    )
    generator.request_seed = request_seed
    generator.max_steps = steps
    return generator
