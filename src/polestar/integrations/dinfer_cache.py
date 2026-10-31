"""Selective two-layer Polestar Cache for dInfer's packed prefix KV storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polestar.models.llada.system_cache import (
    apply_state_patching,
    build_cluster_update_packet,
    initialize_clustered_hidden_state,
)


@dataclass
class _LayerState:
    hidden: torch.Tensor
    clusters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class _RotaryQueries:
    def __init__(self, rotary):
        self.rotary = rotary

    def apply_rotary_q_positions(self, q, positions, key_len):
        source = q.float() if self.rotary.config.rope_full_precision else q
        sin, cos = self.rotary.get_rotary_embedding(key_len, q.device)
        sin = sin.to(source.dtype)[:, :, positions, :]
        cos = cos.to(source.dtype)[:, :, positions, :]
        return self.rotary.apply_rotary_pos_emb(sin, cos, source).to(q.dtype)


def _proxy_attention(block, q, k, v, **kwargs):
    return block._scaled_dot_product_attention(q, k, v), None


class PolestarPrefixCache:
    """Patch the final layer's packed KV using penultimate-layer proxy updates."""

    def __init__(
        self, model, *, refresh_interval: int = 8,
    ):
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be positive")
        self.model = model
        self.refresh_interval = refresh_interval
        blocks = model.model.transformer.blocks
        if len(blocks) < 2:
            raise ValueError("selective Cache requires at least two layers")
        self.source_layer = len(blocks) - 2
        self.target_layer = len(blocks) - 1
        self.source = blocks[self.source_layer]
        self.target = blocks[self.target_layer]
        if self.source.q_norm is not None or self.target.k_norm is not None:
            raise ValueError(
                "Polestar's dInfer cache requires unnormalized source queries "
                "and target keys"
            )
        self.source_state = None
        self.target_state = None
        self.previous_scores = None
        self.steps_in_block = 0
        self.refresh_count = 0
        self.effective_patched_rows = 0

    def _proxy(self, raw_kv, active, *, use_previous):
        key, value = raw_kv
        query = key.new_zeros(
            (key.size(0), self.source.config.n_heads, 1, key.size(-1))
        )
        _, rope_key = self.source.rotary_emb(query, key)
        return build_cluster_update_packet(
            clustered_hidden_state=self.source_state.clusters,
            cache=(key, value),
            hidden_state_past=self.source_state.hidden if use_previous else None,
            replace_position=active,
            topk=4,
            rope_k=rope_key,
            attn_norm=self.source.attn_norm,
            q_proj=self.source.q_proj,
            ff_norm=self.source.ff_norm,
            ff_proj=self.source.ff_proj,
            up_proj=self.source.up_proj,
            act=self.source.act,
            ff_out=self.source.ff_out,
            attn_out=self.source.attn_out,
            dropout=self.source.dropout,
            rotary_emb=_RotaryQueries(self.source.rotary_emb),
            scaled_dot_product_attention_fn=lambda q, k, v, **kw: _proxy_attention(
                self.source, q, k, v, **kw
            ),
            d_model=self.source.config.d_model,
            n_heads=self.source.config.n_heads,
            prev_attn_scores=self.previous_scores if use_previous else None,
            candidate_mask=self.prefix_mask if use_previous else None,
        )

    @torch.inference_mode()
    def begin_block(self, output, block_range, *, hidden_by_layer):
        start, end = block_range
        self.block_start = start
        flat_key_values = output.past_key_values
        full_length = flat_key_values[2 * self.source_layer].size(-2)
        active = torch.zeros(
            (flat_key_values[0].size(0), full_length),
            dtype=torch.bool,
            device=flat_key_values[0].device,
        )
        active[:, start:end] = True
        self.active = active
        self.prefix_mask = torch.zeros_like(active)
        self.prefix_mask[:, :start] = True
        self.steps_in_block = 0
        states = []
        for index, block in (
            (self.source_layer, self.source),
            (self.target_layer, self.target),
        ):
            hidden_input = hidden_by_layer[index]
            hidden = hidden_input.detach().clone()
            clusters = initialize_clustered_hidden_state(
                x=hidden,
                current_hidden_state=hidden,
                replace_position=active,
                window_size=end - start,
                target_total=8,
                block_range=block_range,
            )
            states.append(_LayerState(hidden, clusters))
        self.source_state, self.target_state = states
        self.previous_scores = None
        _, self.previous_scores = self._proxy(
            (
                flat_key_values[2 * self.source_layer],
                flat_key_values[2 * self.source_layer + 1],
            ),
            active,
            use_previous=False,
        )

    @torch.inference_mode()
    def after_dense_step(self, output, packed_cache):
        self.steps_in_block += 1
        if (self.steps_in_block - 1) % self.refresh_interval:
            return
        self.refresh_count += 1
        flat_key_values = output.past_key_values
        packet, self.previous_scores = self._proxy(
            (
                flat_key_values[2 * self.source_layer],
                flat_key_values[2 * self.source_layer + 1],
            ),
            self.active,
            use_previous=True,
        )
        if packet is None:
            return
        self.effective_patched_rows += packet[0].numel()
        next_hidden, next_clusters, next_kv = apply_state_patching(
            update_packet_prev=packet,
            hidden_state_past=self.target_state.hidden,
            clustered_hidden_state=self.target_state.clusters,
            layer_past=(
                packed_cache.get_keys(self.target_layer),
                packed_cache.get_values(self.target_layer),
            ),
            attn_norm=self.target.attn_norm,
            k_proj=self.target.k_proj,
            v_proj=self.target.v_proj,
            d_model=self.target.config.d_model,
            n_heads=self.target.config.n_heads,
            effective_n_kv_heads=self.target.config.effective_n_kv_heads,
        )
        self.target_state = _LayerState(next_hidden, next_clusters)
        packed_cache.get_keys(self.target_layer).copy_(next_kv[0])
        packed_cache.get_values(self.target_layer).copy_(next_kv[1])
