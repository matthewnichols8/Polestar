from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clustering import cluster_past_hidden_states
from .system_clustering import (
    pack_codes,
    quantize_residual_codes,
    quantize_residual_given_clusters,
)
from .triton_quant import decode_packed_residuals, encode_packed_residuals, triton as quant_triton
from ...cuda_graph import CapturedCall

@dataclass
class QuantizedOffloadState:
    q_cpu: torch.Tensor
    scales_cpu: torch.Tensor
    centroids: torch.Tensor
    labels: torch.Tensor
    start: int
    precision: str
    group_size: int
    ready_event: Optional[torch.cuda.Event] = None


def _wait_for_offload(state: QuantizedOffloadState) -> None:
    if state.ready_event is not None:
        state.ready_event.synchronize()
        state.ready_event = None


def quantize_hidden_for_offload(
    hidden: torch.Tensor,
    *,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    start: int,
    precision: str = "fp4",
    group_size: int = 16,
    use_triton: bool = True,
) -> QuantizedOffloadState:
    """Offload packed centroid residuals and FP8 scales to pinned host memory."""
    if labels.shape[1] < start + hidden.shape[1]:
        raise ValueError("cluster labels do not cover the hidden-state window")
    if (
        precision == "fp4"
        and group_size == 16
        and hidden.is_cuda
        and quant_triton is not None
        and use_triton
    ):
        q, scales = encode_packed_residuals(hidden, centroids, labels, start=start)
    else:
        local_labels = labels[:, start:start + hidden.shape[1]]
        reference = centroids.gather(
            1, local_labels.clamp_min(0).unsqueeze(-1).expand_as(hidden)
        )
        codes, scales = quantize_residual_codes(
            hidden.float() - reference.float(), precision=precision, group_size=group_size
        )
        q = pack_codes(codes)
    q_cpu = torch.empty_like(q, device="cpu", pin_memory=hidden.is_cuda)
    # CPU advanced indexing does not support float8; keep its exact bit pattern.
    scale_payload = scales.view(torch.uint8) if precision == "fp4" else scales
    s_cpu = torch.empty_like(scale_payload, device="cpu", pin_memory=hidden.is_cuda)
    q_cpu.copy_(q, non_blocking=hidden.is_cuda)
    s_cpu.copy_(scale_payload, non_blocking=hidden.is_cuda)
    ready_event = None
    if hidden.is_cuda:
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(hidden.device))

    return QuantizedOffloadState(
        q_cpu=q_cpu,
        scales_cpu=s_cpu,
        centroids=centroids.detach().clone(),
        labels=labels,
        start=start,
        precision=precision,
        group_size=group_size,
        ready_event=ready_event,
    )


def load_selected_hidden_from_offload(
    state: QuantizedOffloadState,
    idx_global: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_triton: bool = True,
) -> torch.Tensor:
    """Retrieve selected residuals and add their block-entry centroids."""
    _wait_for_offload(state)
    idx_local = idx_global - state.start
    idx_cpu = idx_local.detach().to(device="cpu", dtype=torch.long)
    q_sel_cpu = torch.empty(
        (state.q_cpu.size(0), idx_cpu.numel(), state.q_cpu.size(-1)),
        dtype=state.q_cpu.dtype, device="cpu", pin_memory=state.q_cpu.is_pinned(),
    )
    s_sel_cpu = torch.empty(
        (state.scales_cpu.size(0), idx_cpu.numel(), state.scales_cpu.size(-1)),
        dtype=state.scales_cpu.dtype, device="cpu", pin_memory=state.scales_cpu.is_pinned(),
    )
    torch.index_select(state.q_cpu, 1, idx_cpu, out=q_sel_cpu)
    torch.index_select(state.scales_cpu, 1, idx_cpu, out=s_sel_cpu)

    q_sel = q_sel_cpu.to(device=device, non_blocking=True)
    s_sel = s_sel_cpu.to(device=device, non_blocking=True)
    if state.precision == "fp4":
        s_sel = s_sel.view(torch.float8_e4m3fn)

    labels = state.labels[:, idx_global]
    reference = state.centroids.gather(
        1, labels.unsqueeze(-1).expand(-1, -1, state.centroids.shape[-1])
    ).to(dtype)
    return decode_packed_residuals(
        q_sel, s_sel, reference,
        precision=state.precision,
        group_size=state.group_size,
        use_triton=use_triton,
    )


def load_hidden_window_from_offload(
    state: QuantizedOffloadState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stage all layers' local hidden rows with one host-to-device transfer."""
    _wait_for_offload(state)
    packed = state.q_cpu.to(device=device, non_blocking=True)
    scales = state.scales_cpu.to(device=device, non_blocking=True)
    if state.precision == "fp4":
        scales = scales.view(torch.float8_e4m3fn)
    width = packed.size(1)
    labels = state.labels[:, state.start:state.start + width].clamp_min(0)
    reference = state.centroids.gather(
        1, labels.unsqueeze(-1).expand(-1, -1, state.centroids.size(-1))
    ).to(dtype)
    return decode_packed_residuals(
        packed, scales, reference,
        precision=state.precision,
        group_size=state.group_size,
    )


def store_hidden_window_to_offload(
    state: QuantizedOffloadState, hidden: torch.Tensor
) -> None:
    """Encode a temporary GPU window and replace its pinned CPU contents."""
    if (
        state.precision == "fp4"
        and state.group_size == 16
        and hidden.is_cuda
        and quant_triton is not None
    ):
        packed, scales = encode_packed_residuals(
            hidden, state.centroids, state.labels, start=state.start
        )
    else:
        labels = state.labels[:, state.start:state.start + hidden.size(1)].clamp_min(0)
        reference = state.centroids.gather(
            1, labels.unsqueeze(-1).expand_as(hidden)
        )
        codes, scales = quantize_residual_codes(
            hidden.float() - reference.float(),
            precision=state.precision,
            group_size=state.group_size,
        )
        packed = pack_codes(codes)
    state.q_cpu.copy_(packed, non_blocking=hidden.is_cuda)
    state.scales_cpu.copy_(
        scales.view(torch.uint8) if state.precision == "fp4" else scales,
        non_blocking=hidden.is_cuda,
    )
    if hidden.is_cuda:
        state.ready_event = torch.cuda.Event()
        state.ready_event.record(torch.cuda.current_stream(hidden.device))


def update_quantized_offload_state(
    state: QuantizedOffloadState, idx_global: torch.Tensor, hidden: torch.Tensor
) -> None:
    """Keep offloaded residuals coherent after a sparse hidden-state update."""
    _wait_for_offload(state)
    labels = state.labels[:, idx_global]
    reference = state.centroids.gather(
        1, labels.unsqueeze(-1).expand(-1, -1, state.centroids.shape[-1])
    )
    codes, scales = quantize_residual_codes(
        hidden.float() - reference.float(), precision=state.precision, group_size=state.group_size
    )
    idx_cpu = (idx_global - state.start).to("cpu", dtype=torch.long)
    state.q_cpu[:, idx_cpu, :] = pack_codes(codes).to("cpu")
    state.scales_cpu[:, idx_cpu, :] = (
        scales.view(torch.uint8) if state.precision == "fp4" else scales
    ).to("cpu")


def apply_state_patching(
    *,
    update_packet_prev: Tuple[torch.Tensor, torch.Tensor],
    hidden_state_past: Optional[torch.Tensor],
    clustered_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
    attn_norm: nn.Module,
    k_proj: nn.Module,
    v_proj: nn.Module,
    d_model: int,
    n_heads: int,
    effective_n_kv_heads: int,
    prefetched_h_old_raw: Optional[torch.Tensor] = None,
    quant_offload_state: Optional[QuantizedOffloadState] = None,
    cached_hidden_window: Optional[torch.Tensor] = None,
    window_start: int = 0,
    inplace_kv: bool = False,
    inplace_centroids: bool = False,
    mirror_layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    Optional[Tuple[torch.Tensor, torch.Tensor]],
]:
    """Patch hidden-state/KV caches and update centroids for selected indices."""
    idx_raw, h_new_raw = update_packet_prev[:2]  # idx: (S,), h_new: (B, S, D)
    selection = update_packet_prev[2] if len(update_packet_prev) == 3 else None

    if selection is not None:
        idx = idx_raw
        h_new = h_new_raw
    elif clustered_hidden_state is not None:
        _, current_labels, _ = clustered_hidden_state
        active_mask = current_labels[0, idx_raw] != -1
        idx = idx_raw[active_mask]
        h_new = h_new_raw[:, active_mask, :]
    else:
        idx = idx_raw
        h_new = h_new_raw

    if hidden_state_past is None and quant_offload_state is None and cached_hidden_window is None:
        raise ValueError("state patching requires cached or offloaded hidden states")
    idx = idx.to(h_new.device)

    # 1) Patch hidden-state cache.
    if hidden_state_past is not None:
        hidden_state_past = hidden_state_past.clone()
    if prefetched_h_old_raw is not None:
        if clustered_hidden_state is not None and selection is None:
            h_old = prefetched_h_old_raw[:, active_mask, :].clone()
        else:
            h_old = prefetched_h_old_raw.clone()
    elif hidden_state_past is not None:
        h_old = hidden_state_past[:, idx, :].clone()
    elif cached_hidden_window is not None:
        h_old = cached_hidden_window[:, idx - window_start, :].clone()
    else:
        h_old = load_selected_hidden_from_offload(
            quant_offload_state, idx, device=h_new.device, dtype=h_new.dtype
        )
    if selection is not None:
        h_new = torch.where(selection.view(1, -1, 1), h_new, h_old)
    if hidden_state_past is not None:
        hidden_state_past.index_copy_(1, idx, h_new)
    if cached_hidden_window is not None:
        cached_hidden_window.index_copy_(1, idx - window_start, h_new)
    elif quant_offload_state is not None and idx.numel():
        update_quantized_offload_state(quant_offload_state, idx, h_new)

    # 2) Patch KV cache at updated positions.
    if layer_past is not None:
        past_key, past_value = layer_past
        head_dim = d_model // n_heads

        h_new_norm = attn_norm(h_new if selection is not None else h_new_raw)
        k_new = k_proj(h_new_norm)
        v_new = v_proj(h_new_norm)

        bsz, ssz, _ = k_new.shape
        k_new = k_new.view(bsz, ssz, effective_n_kv_heads, head_dim).transpose(1, 2)
        v_new = v_new.view(bsz, ssz, effective_n_kv_heads, head_dim).transpose(1, 2)

        if not inplace_kv:
            past_key = past_key.clone()
            past_value = past_value.clone()
        if selection is not None:
            selected = selection.view(1, 1, -1, 1)
            k_new = torch.where(selected, k_new, past_key[:, :, idx_raw, :])
            v_new = torch.where(selected, v_new, past_value[:, :, idx_raw, :])
        past_key.index_copy_(2, idx_raw, k_new)
        past_value.index_copy_(2, idx_raw, v_new)
        if mirror_layer_past is not None:
            mirror_layer_past[0].index_copy_(2, idx_raw, k_new)
            mirror_layer_past[1].index_copy_(2, idx_raw, v_new)
        layer_past = (past_key, past_value)

    # 3) Incrementally update centroids using old/new selected token states.
    if clustered_hidden_state is not None:
        centroids, labels, counts = clustered_hidden_state
        updated_centroids = incremental_centroids(
            centroids, labels[:, idx], counts, h_old, h_new
        )
        if inplace_centroids:
            centroids.copy_(updated_centroids)
        else:
            centroids = updated_centroids
        clustered_hidden_state = (centroids, labels, counts)

    return hidden_state_past, clustered_hidden_state, layer_past


def incremental_centroids(
    centroids: torch.Tensor,
    selected_labels: torch.Tensor,
    counts: torch.Tensor,
    old_hidden: torch.Tensor,
    new_hidden: torch.Tensor,
) -> torch.Tensor:
    """Update each centroid from the selected tokens' representation changes."""
    batch, n_clusters, _ = centroids.shape
    if batch != old_hidden.shape[0]:
        raise ValueError("batch size mismatch between centroids and hidden states")
    valid = (selected_labels >= 0) & (selected_labels < n_clusters)
    labels = selected_labels.clamp(min=0)
    onehot = F.one_hot(labels, num_classes=n_clusters).to(centroids.dtype)
    onehot = onehot * valid.unsqueeze(-1)
    old_sum = torch.einsum("bsk,bsd->bkd", onehot, old_hidden.to(centroids.dtype))
    new_sum = torch.einsum("bsk,bsd->bkd", onehot, new_hidden.to(centroids.dtype))
    updated = centroids + (new_sum - old_sum) / (counts + 1e-6)
    return torch.where(counts > 0, updated, centroids)


def initialize_clustered_hidden_state(
    *,
    x: torch.Tensor,
    current_hidden_state: torch.Tensor,
    replace_position: torch.Tensor,
    window_size: int = 32,
    target_total: int = 8,
    enable_quantization: bool = False,
    quantization_precision: str = "fp4",
    block_range: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build initial local-window clusters and labels for cache step 0."""
    block_mask = replace_position[0]
    l_total = block_mask.size(0)
    if block_range is None:
        block_pos = block_mask.nonzero(as_tuple=True)[0]
        curr_start = block_pos[0].item()
        curr_end = block_pos[-1].item() + 1
    else:
        curr_start, curr_end = block_range

    w_start = max(0, curr_start - 2 * window_size)
    w_end = min(l_total, curr_end + window_size)

    if current_hidden_state.size(1) == l_total:
        pre_window = current_hidden_state[:, w_start:curr_start, :]
        post_window = current_hidden_state[:, curr_end:w_end, :]
        local_start = 0
    elif current_hidden_state.size(1) == w_end - w_start:
        pre_window = current_hidden_state[:, :curr_start - w_start, :]
        post_window = current_hidden_state[:, curr_end - w_start:, :]
        local_start = w_start
    else:
        raise ValueError("cached hidden states must cover the full sequence or local block window")
    total_window_len = (curr_start - w_start) + (w_end - curr_end)

    num_pre_clusters = max(1, target_total * (curr_start - w_start) // total_window_len)
    num_pre_clusters = min(target_total - 1, num_pre_clusters)
    if pre_window.size(1) == 0:
        num_pre_clusters = 0
    num_post_clusters = target_total - num_pre_clusters

    all_centroids = []
    full_labels = torch.full((x.size(0), l_total), -1, dtype=torch.long, device=x.device)

    if pre_window.size(1) > 0:
        pre_centroids, pre_labels = cluster_past_hidden_states(
            [pre_window.detach()],
            num_clusters=num_pre_clusters,
        )[0]
        if enable_quantization:
            pre_window = quantize_residual_given_clusters(
                hidden_window=pre_window,
                centroids=pre_centroids,
                labels=pre_labels,
                precision=quantization_precision,
            )
            current_hidden_state[:, w_start - local_start:curr_start - local_start, :] = pre_window
        all_centroids.append(pre_centroids)
        full_labels[:, w_start:curr_start] = pre_labels
    else:
        all_centroids.append(torch.empty((x.size(0), 0, pre_window.size(-1)), device=x.device))

    if post_window.size(1) > 0:
        post_centroids, post_labels = cluster_past_hidden_states(
            [post_window.detach()],
            num_clusters=num_post_clusters,
        )[0]
        if enable_quantization:
            post_window = quantize_residual_given_clusters(
                hidden_window=post_window,
                centroids=post_centroids,
                labels=post_labels,
                precision=quantization_precision,
            )
            current_hidden_state[:, curr_end - local_start:w_end - local_start, :] = post_window
        all_centroids.append(post_centroids)
        full_labels[:, curr_end:w_end] = post_labels + num_pre_clusters
    else:
        all_centroids.append(torch.empty((x.size(0), 0, post_window.size(-1)), device=x.device))

    centroids = torch.cat(all_centroids, dim=1).to(x.dtype)
    b_c, k_clusters, _ = centroids.shape
    valid = (full_labels >= 0) & (full_labels < k_clusters)
    labels_safe = full_labels.clamp(min=0)
    counts = torch.zeros((b_c, k_clusters), device=x.device, dtype=torch.float32)
    counts.scatter_add_(1, labels_safe, valid.to(counts.dtype))
    counts = counts.unsqueeze(-1)

    return centroids, full_labels, counts.to(x.dtype)


def initialize_clustered_hidden_state_with_offload(
    *,
    x: torch.Tensor,
    current_hidden_state: torch.Tensor,
    replace_position: torch.Tensor,
    window_size: int = 32,
    target_total: int = 8,
    quantization_precision: str = "fp4",
    group_size: int = 16,
    block_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], QuantizedOffloadState]:
    """
    Step-0 helper:
    1) build clustered state (GPU centroids/labels/counts)
    2) offload quantized hidden-state cache to CPU
    """
    clustered_state = initialize_clustered_hidden_state(
        x=x,
        current_hidden_state=current_hidden_state,
        replace_position=replace_position,
        window_size=window_size,
        target_total=target_total,
        enable_quantization=False,
        quantization_precision=quantization_precision,
        block_range=block_range,
    )
    if block_range is None:
        block_positions = replace_position[0].nonzero(as_tuple=True)[0]
        block_start = block_positions[0].item()
        block_end = block_positions[-1].item() + 1
    else:
        block_start, block_end = block_range
    window_start = max(0, block_start - 2 * window_size)
    window_end = min(replace_position.shape[1], block_end + window_size)
    if current_hidden_state.shape[1] == replace_position.shape[1]:
        hidden_window = current_hidden_state[:, window_start:window_end, :]
    else:
        hidden_window = current_hidden_state
        if hidden_window.shape[1] != window_end - window_start:
            raise ValueError("hidden-state cache does not match the local window")
    quant_state = quantize_hidden_for_offload(
        hidden_window,
        centroids=clustered_state[0],
        labels=clustered_state[1],
        start=window_start,
        precision=quantization_precision,
        group_size=group_size,
    )
    return clustered_state, quant_state


def compute_token_drift_kl(
    curr_attn: torch.Tensor,
    prev_attn: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute token-wise KL drift, averaged over attention heads."""
    curr = curr_attn.detach().to(torch.float32).clamp_min(eps)
    prev = prev_attn.detach().to(torch.float32).clamp_min(eps)
    kl_bhq = (curr * (torch.log(curr) - torch.log(prev))).sum(dim=-1)
    return kl_bhq.clamp_min(0.0).mean(dim=1)


def compute_cluster_drift_kl(
    curr_attn: torch.Tensor,
    prev_attn: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """KL drift per centroid, averaged after reducing each attention head."""
    curr = curr_attn.detach().to(torch.float32)
    prev = prev_attn.detach().to(torch.float32)
    per_head = (curr * (torch.log(curr + eps) - torch.log(prev + eps))).sum(dim=-1)
    return per_head.clamp_min(0.0).mean(dim=1)


def build_cluster_update_packet(
    *,
    clustered_hidden_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cache: Tuple[torch.Tensor, torch.Tensor],
    hidden_state_past: Optional[torch.Tensor],
    replace_position: Optional[torch.Tensor],
    topk: Optional[int],
    rope_k: torch.Tensor,
    attn_norm: nn.Module,
    q_proj: nn.Module,
    ff_norm: nn.Module,
    ff_proj: nn.Module,
    up_proj: nn.Module,
    act: nn.Module,
    ff_out: nn.Module,
    attn_out: nn.Module,
    dropout: nn.Module,
    rotary_emb: object,
    scaled_dot_product_attention_fn: Callable,
    d_model: int,
    n_heads: int,
    prev_attn_scores: Optional[torch.Tensor],
    quant_offload_state: Optional[QuantizedOffloadState] = None,
    cached_hidden_window: Optional[torch.Tensor] = None,
    window_start: int = 0,
    use_triton_dequant: bool = True,
    centroid_q_graph: Optional[CapturedCall] = None,
    candidate_mask: Optional[torch.Tensor] = None,
    candidate_range: Optional[Tuple[int, int]] = None,
    candidate_indices: Optional[torch.Tensor] = None,
) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Build per-layer update packet from clustered token importance scores."""
    centroids, labels, _ = clustered_hidden_state
    assert centroids.dim() == 3, "centroids should be (B, K, D)"
    assert labels.dim() == 2, "labels should be (B, L_total)"
    if candidate_mask is not None and (
        candidate_mask.shape != labels.shape or candidate_mask.dtype != torch.bool
    ):
        raise ValueError("candidate_mask must be a boolean mask over cache positions")

    head_dim = d_model // n_heads
    bsz, k_clusters, d_hidden = centroids.shape
    _, l_total = labels.shape

    key, value = cache
    _, _, l_kv, _ = key.shape
    assert l_kv == l_total, "Seq size mismatch in KV"

    if centroid_q_graph is not None and centroid_q_graph.accepts(centroids):
        cluster_q = centroid_q_graph(centroids)
    else:
        centers_norm = attn_norm(centroids)
        cluster_q = q_proj(centers_norm)
    cluster_q = cluster_q.view(bsz, k_clusters, n_heads, head_dim).transpose(1, 2)

    n_kv_heads = key.size(1)
    if n_kv_heads != n_heads:
        if n_heads % n_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        key_for_proxy = key.repeat_interleave(n_heads // n_kv_heads, dim=1)
    else:
        key_for_proxy = key
    attn_scores = torch.matmul(cluster_q, key_for_proxy.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_scores = F.softmax(attn_scores, dim=-1)

    scores0 = None
    if prev_attn_scores is not None and (
        hidden_state_past is not None
        or quant_offload_state is not None
        or cached_hidden_window is not None
    ):
        scores0 = compute_cluster_drift_kl(attn_scores, prev_attn_scores)[0]

    update_packet = None
    if scores0 is not None and replace_position is not None:
        block_mask = replace_position[0]
        non_block_mask = ~block_mask

        labels_b0 = labels[0]
        valid = labels_b0 >= 0
        labels_safe = labels_b0.clamp(min=0)
        eligible = valid & non_block_mask
        if candidate_mask is not None:
            eligible = eligible & candidate_mask[0]
            eligible_clusters = torch.zeros(
                k_clusters, device=labels_b0.device, dtype=torch.int32
            )
            eligible_clusters.scatter_add_(0, labels_safe, eligible.to(torch.int32))
            scores0 = scores0.masked_fill(eligible_clusters == 0, -torch.inf)

        m_req = topk if topk is not None else 0
        m_top = min(m_req, k_clusters)
        top_clusters = torch.topk(scores0, k=m_top, dim=-1).indices

        top_mask = torch.zeros(k_clusters, device=labels_b0.device, dtype=torch.bool)
        top_mask.scatter_(0, top_clusters, True)

        sel_mask = eligible & top_mask[labels_safe]
        if candidate_range is None:
            idx_global = sel_mask.nonzero(as_tuple=True)[0]
        else:
            idx_global = (
                candidate_indices
                if candidate_indices is not None
                else torch.arange(*candidate_range, device=labels.device)
            )

        if idx_global.numel() > 0:
            if cached_hidden_window is not None:
                hidden_cache = cached_hidden_window[:, idx_global - window_start, :]
            elif quant_offload_state is not None:
                hidden_cache = load_selected_hidden_from_offload(
                    quant_offload_state,
                    idx_global,
                    device=centroids.device,
                    dtype=centroids.dtype,
                    use_triton=use_triton_dequant,
                )
            elif hidden_state_past is not None:
                hidden_cache = hidden_state_past[:, idx_global, :]
            else:
                return None, attn_scores.detach()

            h_cache_normed = attn_norm(hidden_cache)
            q_cache_sub = q_proj(h_cache_normed)
            q_cache_sub = q_cache_sub.view(bsz, -1, n_heads, head_dim).transpose(1, 2)
            q_sel = rotary_emb.apply_rotary_q_positions(
                q_cache_sub,
                positions=idx_global,
                key_len=l_total,
            )
            h_sel = hidden_cache

            attn, _ = scaled_dot_product_attention_fn(
                q_sel,
                rope_k,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                output_attentions=False,
            )
            att = attn.transpose(1, 2).contiguous().view(bsz, -1, d_hidden)
            att = attn_out(att)
            h_sel = h_sel + dropout(att)

            og_h_sel = h_sel
            h_sel = ff_norm(h_sel)
            h_sel, h_sel_up = ff_proj(h_sel), up_proj(h_sel)
            h_sel = act(h_sel)
            h_sel = h_sel * h_sel_up
            h_sel = ff_out(h_sel)
            h_sel = dropout(h_sel)
            h_sel = og_h_sel + h_sel

            update_packet = (
                (idx_global, h_sel, sel_mask[idx_global])
                if candidate_range is not None else (idx_global, h_sel)
            )

    return update_packet, attn_scores.detach()
