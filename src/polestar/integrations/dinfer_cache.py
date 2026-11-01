"""Layer-wise Polestar Cache updates for dInfer's packed prefix KV storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from polestar.models.llada.clustering import k_means
from polestar.models.llada.system_cache import (
    QuantizedOffloadState,
    apply_state_patching,
    build_cluster_update_packet,
    compute_cluster_drift_kl,
    load_hidden_window_from_offload,
    quantize_hidden_for_offload,
    store_hidden_window_to_offload,
)


_MAX_GRAPH_SHAPES = 4


def _refresh_source_layers(layer_count: int, selected_count: int) -> tuple[int, ...]:
    """Choose source layers evenly, including both ends when possible."""
    source_count = layer_count - 1
    if selected_count == 1:
        return (source_count - 1,)
    return tuple(
        round(index * (source_count - 1) / (selected_count - 1))
        for index in range(selected_count)
    )


@dataclass
class _LayerState:
    clusters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    offload: QuantizedOffloadState


class _PackedCacheView:
    def __init__(self, data):
        self._data = data

    def get_keys(self, layer_idx):
        return self._data[layer_idx, 0]

    def get_values(self, layer_idx):
        return self._data[layer_idx, 1]


@dataclass
class _RefreshWorkspace:
    graph: torch.cuda.CUDAGraph
    key_values: list[torch.Tensor]
    staged: torch.Tensor
    packed: _PackedCacheView
    states: list[_LayerState]
    scores: list[torch.Tensor | None]
    active: torch.Tensor
    prefix_mask: torch.Tensor
    candidate_indices: torch.Tensor
    window_start: torch.Tensor


def _cluster_layers(hidden, *, block_start, block_end, window_start, window_end, full_length):
    """Cluster all layers in two batched spherical K-means calls."""
    layers = hidden.size(1)
    prefix_length = block_start - window_start
    suffix_start = block_end - window_start
    suffix_length = window_end - block_end
    window_length = prefix_length + suffix_length
    if window_length == 0:
        raise ValueError("Polestar Cache requires a prefix or suffix window")
    total_clusters = min(8, window_length)
    if prefix_length and suffix_length:
        prefix_clusters = max(
            1,
            min(
                prefix_length,
                total_clusters - 1,
                total_clusters * prefix_length // window_length,
            ),
        )
        suffix_clusters = min(suffix_length, total_clusters - prefix_clusters)
        prefix_clusters = total_clusters - suffix_clusters
    else:
        prefix_clusters = total_clusters if prefix_length else 0
        suffix_clusters = total_clusters - prefix_clusters
    centroids = []
    labels = torch.full((1, layers, full_length), -1, dtype=torch.long, device=hidden.device)
    if prefix_length:
        prefix_centroids, prefix_labels = k_means(
            hidden[:, :, :prefix_length, :], prefix_clusters
        )
        centroids.append(prefix_centroids)
        labels[:, :, window_start:block_start] = prefix_labels
    if suffix_length:
        suffix_centroids, suffix_labels = k_means(
            hidden[:, :, suffix_start:, :], suffix_clusters
        )
        centroids.append(suffix_centroids)
        labels[:, :, block_end:window_end] = suffix_labels + prefix_clusters
    centers = torch.cat(centroids, dim=2)
    valid = labels >= 0
    counts = torch.zeros(
        (1, layers, centers.size(2)), dtype=torch.float32, device=hidden.device
    )
    counts.scatter_add_(2, labels.clamp_min(0), valid.to(torch.float32))
    return centers, labels, counts.unsqueeze(-1).to(hidden.dtype)


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


class PolestarCache:
    """Pass sparse update packets through the layers of dInfer's prefix cache."""

    def __init__(
        self,
        model,
        *,
        refresh_after_tokens: int = 3,
        use_cuda_graph: bool = True,
        refresh_layer_count: int | None = None,
    ):
        if refresh_after_tokens < 1:
            raise ValueError("refresh_after_tokens must be positive")
        self.refresh_after_tokens = refresh_after_tokens
        self.use_cuda_graph = use_cuda_graph
        self.blocks = model.model.transformer.blocks
        self._compiled_packet = torch.compile(
            build_cluster_update_packet, backend="inductor", mode="default"
        )
        blocks = self.blocks
        if len(blocks) < 2:
            raise ValueError("selective Cache requires at least two layers")
        if refresh_layer_count is not None and not (
            isinstance(refresh_layer_count, int)
            and 1 <= refresh_layer_count < len(blocks)
        ):
            raise ValueError(
                f"refresh_layer_count must be between 1 and {len(blocks) - 1}"
            )
        self.parallel_refresh_layers = (
            _refresh_source_layers(len(blocks), refresh_layer_count)
            if refresh_layer_count is not None
            else None
        )
        if any(block.q_norm is not None or block.k_norm is not None for block in blocks):
            raise ValueError(
                "Polestar's dInfer cache requires unnormalized queries and keys"
            )
        self.states = []
        self.previous_scores = []
        self.decoded_since_refresh = 0
        self._offload_stream = None
        self._refresh_streams = None
        self._graph_pool = {}
        self._shared_key_values = None
        self._shared_packed = None
        self.clear_graph()
        self.refresh_count = 0
        self.effective_patched_rows = 0
        self.suffix_eligible = None

    def clear_graph(self):
        self._active_workspace = None

    def finish_request(self):
        self.clear_graph()
        # Graphs remain available for later requests with the same shape.
        # Their offload buffers belong to the completed request.
        for workspace in self._graph_pool.values():
            for state in workspace.states:
                state.offload = None

    def _stream_for(self, device):
        if self._offload_stream is None:
            self._offload_stream = torch.cuda.Stream(device=device)
        return self._offload_stream

    def _proxy(self, layer_idx, raw_kv, *, use_previous, staged_hidden=None):
        block = self.blocks[layer_idx]
        key, value = raw_kv
        if use_previous:
            query = key.new_zeros(
                (key.size(0), block.config.n_heads, 1, key.size(-1))
            )
            _, rope_key = block.rotary_emb(query, key)
        else:
            # Initial proxy scoring does not run a sparse token forward.
            rope_key = key
        packet_fn = (
            self._compiled_packet
            if (
                use_previous
                and key.is_cuda
                and self.use_cuda_graph
                and self.parallel_refresh_layers is None
            )
            else build_cluster_update_packet
        )
        return packet_fn(
            clustered_hidden_state=self.states[layer_idx].clusters,
            cache=(key, value),
            hidden_state_past=None,
            replace_position=self.active,
            topk=4,
            rope_k=rope_key,
            attn_norm=block.attn_norm,
            q_proj=block.q_proj,
            ff_norm=block.ff_norm,
            ff_proj=block.ff_proj,
            up_proj=block.up_proj,
            act=block.act,
            ff_out=block.ff_out,
            attn_out=block.attn_out,
            dropout=block.dropout,
            rotary_emb=_RotaryQueries(block.rotary_emb),
            scaled_dot_product_attention_fn=lambda q, k, v, **kw: _proxy_attention(
                block, q, k, v, **kw
            ),
            d_model=block.config.d_model,
            n_heads=block.config.n_heads,
            prev_attn_scores=self.previous_scores[layer_idx] if use_previous else None,
            quant_offload_state=self.states[layer_idx].offload if use_previous else None,
            cached_hidden_window=staged_hidden,
            window_start=(
                self._active_workspace.window_start
                if self._active_workspace is not None
                else self.window_start
            ),
            candidate_mask=self.prefix_mask if use_previous else None,
            candidate_range=(self.window_start, self.block_start) if use_previous else None,
            candidate_indices=(
                self._active_workspace.candidate_indices
                if use_previous and self._active_workspace is not None
                else None
            ),
        )

    @torch.inference_mode()
    def begin_block(self, output, block_range):
        if self.states and self.flat_offload.ready_event is not None:
            self.flat_offload.ready_event.synchronize()
        self.clear_graph()
        start, end = block_range
        self.block_start = start
        self.block_end = end
        self.suffix_eligible = None
        flat_key_values = output.past_key_values
        full_length = flat_key_values[0].size(-2)
        active = torch.zeros(
            (flat_key_values[0].size(0), full_length),
            dtype=torch.bool,
            device=flat_key_values[0].device,
        )
        active[:, start:end] = True
        self.active = active
        self.prefix_mask = torch.zeros_like(active)
        self.prefix_mask[:, :start] = True
        self.decoded_since_refresh = 0
        if len(output.hidden_states) != len(self.blocks):
            raise ValueError("Polestar Cache needs the input hidden state of every layer")
        if flat_key_values[0].size(0) != 1:
            raise ValueError("Polestar Cache currently requires batch size one")
        window_start = max(0, start - 2 * (end - start))
        window_end = min(full_length, end + (end - start))
        self.window_start = window_start
        self.window_end = window_end
        hidden_window = torch.stack(
            [state[:, window_start:window_end, :] for state in output.hidden_states],
            dim=1,
        )
        centers, labels, counts = _cluster_layers(
            hidden_window,
            block_start=start,
            block_end=end,
            window_start=window_start,
            window_end=window_end,
            full_length=full_length,
        )
        def encode_window():
            return quantize_hidden_for_offload(
                hidden_window.flatten(0, 1),
                centroids=centers.flatten(0, 1),
                labels=labels.flatten(0, 1),
                start=window_start,
            )

        if hidden_window.is_cuda:
            offload_stream = self._stream_for(hidden_window.device)
            offload_stream.wait_stream(torch.cuda.current_stream(hidden_window.device))
            with torch.cuda.stream(offload_stream):
                flat_offload = encode_window()
            hidden_window.record_stream(offload_stream)
        else:
            flat_offload = encode_window()
        self.flat_offload = flat_offload
        self.states = []
        for layer_idx in range(len(self.blocks)):
            clusters = (
                centers[:, layer_idx],
                labels[:, layer_idx],
                counts[:, layer_idx],
            )
            offload = QuantizedOffloadState(
                q_cpu=flat_offload.q_cpu[layer_idx:layer_idx + 1],
                scales_cpu=flat_offload.scales_cpu[layer_idx:layer_idx + 1],
                centroids=flat_offload.centroids[layer_idx:layer_idx + 1],
                labels=clusters[1],
                start=window_start,
                precision=flat_offload.precision,
                group_size=flat_offload.group_size,
                ready_event=flat_offload.ready_event,
            )
            self.states.append(_LayerState(clusters, offload))
        self.previous_scores = []
        for layer_idx in range(len(self.blocks)):
            _, scores = self._proxy(
                layer_idx,
                (flat_key_values[2 * layer_idx], flat_key_values[2 * layer_idx + 1]),
                use_previous=False,
            )
            self.previous_scores.append(scores)

    def record_commits(self, count: int) -> None:
        self.decoded_since_refresh += count

    def _patch_packet(self, layer_idx, packet, flat_key_values, staged, packed_cache):
        if packet is None:
            return None
        target_idx = layer_idx + 1
        target = self.blocks[target_idx]
        next_hidden, _, _ = apply_state_patching(
            update_packet_prev=packet,
            hidden_state_past=None,
            clustered_hidden_state=self.states[target_idx].clusters,
            layer_past=(
                packed_cache.get_keys(target_idx),
                packed_cache.get_values(target_idx),
            ),
            attn_norm=target.attn_norm,
            k_proj=target.k_proj,
            v_proj=target.v_proj,
            d_model=target.config.d_model,
            n_heads=target.config.n_heads,
            effective_n_kv_heads=target.config.effective_n_kv_heads,
            cached_hidden_window=staged[target_idx:target_idx + 1],
            window_start=(
                self._active_workspace.window_start
                if self._active_workspace is not None
                else self.window_start
            ),
            inplace_kv=True,
            inplace_centroids=True,
            mirror_layer_past=(
                flat_key_values[2 * target_idx],
                flat_key_values[2 * target_idx + 1],
            ),
        )
        if next_hidden is not None:
            raise RuntimeError("offloaded cache unexpectedly retained GPU hidden states")
        return packet[2].sum()

    def _refresh_layers_sequential(self, flat_key_values, staged, packed_cache):
        selected_counts = []
        for layer_idx in range(len(self.blocks) - 1):
            raw_kv = (
                flat_key_values[2 * layer_idx],
                flat_key_values[2 * layer_idx + 1],
            )
            packet, scores = self._proxy(
                layer_idx, raw_kv, use_previous=True,
                staged_hidden=staged[layer_idx:layer_idx + 1],
            )
            self.previous_scores[layer_idx].copy_(scores)
            count = self._patch_packet(
                layer_idx, packet, flat_key_values, staged, packed_cache
            )
            if count is not None:
                selected_counts.append(count)
        if selected_counts:
            self.effective_patched_rows.add_(torch.stack(selected_counts).sum())

    def _refresh_layers_parallel(self, flat_key_values, staged, packed_cache):
        """Score selected source layers, then patch their targets concurrently."""
        def make_packet(layer_idx):
            packet, scores = self._proxy(
                layer_idx,
                (
                    flat_key_values[2 * layer_idx],
                    flat_key_values[2 * layer_idx + 1],
                ),
                use_previous=True,
                staged_hidden=staged[layer_idx:layer_idx + 1],
            )
            self.previous_scores[layer_idx].copy_(scores)
            return packet

        selected = self.parallel_refresh_layers
        main = torch.cuda.current_stream(staged.device) if staged.is_cuda else None
        packets = []
        for stream_idx, layer_idx in enumerate(selected):
            if main is None:
                packets.append(make_packet(layer_idx))
            else:
                stream = self._refresh_streams[stream_idx]
                stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    packets.append(make_packet(layer_idx))
        if main is not None:
            for stream in self._refresh_streams:
                main.wait_stream(stream)
        selected_counts = []
        for stream_idx, (layer_idx, packet) in enumerate(zip(selected, packets)):
            if main is None:
                selected_counts.append(self._patch_packet(
                    layer_idx, packet, flat_key_values, staged, packed_cache
                ))
            else:
                stream = self._refresh_streams[stream_idx]
                stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    selected_counts.append(self._patch_packet(
                        layer_idx, packet, flat_key_values, staged, packed_cache
                    ))
        if main is not None:
            for stream in self._refresh_streams:
                main.wait_stream(stream)
        selected_counts = [count for count in selected_counts if count is not None]
        if selected_counts:
            self.effective_patched_rows.add_(torch.stack(selected_counts).sum())

    def _refresh_layers(self, flat_key_values, staged, packed_cache):
        if self.parallel_refresh_layers is None:
            self._refresh_layers_sequential(flat_key_values, staged, packed_cache)
        else:
            self._refresh_layers_parallel(flat_key_values, staged, packed_cache)

    def _select_suffix(self, flat_key_values):
        if self.window_end <= self.block_end:
            return
        final_layer = len(self.blocks) - 1
        _, current_scores = self._proxy(
            final_layer,
            (
                flat_key_values[2 * final_layer],
                flat_key_values[2 * final_layer + 1],
            ),
            use_previous=False,
        )
        drift = compute_cluster_drift_kl(
            current_scores, self.previous_scores[final_layer]
        )[0]
        top_clusters = drift.topk(min(4, drift.numel())).indices
        suffix_labels = self.states[final_layer].clusters[1][
            0, self.block_end:self.window_end
        ]
        self.suffix_eligible = (
            suffix_labels[:, None] == top_clusters[None, :]
        ).any(dim=1)
        self.previous_scores[final_layer].copy_(current_scores)

    def _graph_key(self, flat_key_values, staged, packed_cache):
        return (
            tuple(flat_key_values[0].shape),
            flat_key_values[0].dtype,
            staged.shape,
            self.block_start - self.window_start,
            tuple(packed_cache._data.shape),
        )

    def _activate_workspace(self, workspace):
        for old, new in zip(self.states, workspace.states):
            for source, target in zip(old.clusters, new.clusters):
                target.copy_(source)
            new.offload = old.offload
        for old, new in zip(self.previous_scores, workspace.scores):
            if old is not None:
                new.copy_(old)
        workspace.active.copy_(self.active)
        workspace.prefix_mask.copy_(self.prefix_mask)
        workspace.candidate_indices.copy_(
            torch.arange(
                self.window_start, self.block_start,
                device=workspace.candidate_indices.device,
            )
        )
        workspace.window_start.fill_(self.window_start)
        self.states = workspace.states
        self.previous_scores = workspace.scores
        self.active = workspace.active
        self.prefix_mask = workspace.prefix_mask
        self._active_workspace = workspace

    def _shared_workspace_storage(self, flat_key_values, packed_cache):
        """Share KV input storage across graph shapes that replay sequentially."""
        if not all(
            kv.is_contiguous()
            and kv.dtype == flat_key_values[0].dtype
            and kv.device == flat_key_values[0].device
            for kv in flat_key_values
        ):
            return None
        packed_data = packed_cache._data
        if not packed_data.is_contiguous():
            return None
        slot_size = max(kv.numel() for kv in flat_key_values)
        packed_size = packed_data.numel()
        raw = self._shared_key_values
        packed = self._shared_packed
        needs_storage = (
            raw is None
            or raw.size(0) != len(flat_key_values)
            or raw.size(1) < slot_size
            or raw.device != flat_key_values[0].device
            or raw.dtype != flat_key_values[0].dtype
            or packed is None
            or packed.numel() < packed_size
            or packed.device != packed_data.device
            or packed.dtype != packed_data.dtype
        )
        if needs_storage:
            # A larger sequence needs new backing addresses, so graphs that
            # captured the previous addresses must be rebuilt.
            self._graph_pool.clear()
            self._shared_key_values = torch.empty(
                (len(flat_key_values), slot_size),
                dtype=flat_key_values[0].dtype,
                device=flat_key_values[0].device,
            )
            self._shared_packed = torch.empty(
                packed_size, dtype=packed_data.dtype, device=packed_data.device
            )
        keys = [
            self._shared_key_values[i, :kv.numel()].view_as(kv)
            for i, kv in enumerate(flat_key_values)
        ]
        packed_view = self._shared_packed[:packed_size].view_as(packed_data)
        return keys, packed_view

    def _make_workspace(self, flat_key_values, staged, packed_cache):
        """Capture once for a shape with persistent tensor inputs and outputs."""
        graph = torch.cuda.CUDAGraph()
        shared = self._shared_workspace_storage(flat_key_values, packed_cache)
        if shared is None:
            key_values = [torch.empty_like(kv) for kv in flat_key_values]
            packed_data = torch.empty_like(packed_cache._data)
        else:
            key_values, packed_data = shared
        workspace = _RefreshWorkspace(
            graph=graph,
            key_values=key_values,
            staged=torch.empty_like(staged),
            packed=_PackedCacheView(packed_data),
            states=[
                _LayerState(tuple(t.clone() for t in state.clusters), state.offload)
                for state in self.states
            ],
            scores=[
                score.clone() if score is not None else None
                for score in self.previous_scores
            ],
            active=self.active.clone(),
            prefix_mask=self.prefix_mask.clone(),
            candidate_indices=torch.arange(
                self.window_start, self.block_start, device=staged.device
            ),
            window_start=torch.tensor(self.window_start, device=staged.device),
        )
        self._activate_workspace(workspace)
        torch._foreach_copy_(workspace.key_values, flat_key_values)
        workspace.staged.copy_(staged)
        workspace.packed._data.copy_(packed_cache._data)
        # dInfer's SDPA is torch.compiled. Compile this shape before stream
        # capture; Dynamo cannot compile a new specialization inside capture.
        warm_layers = (
            range(len(self.blocks) - 1)
            if self.parallel_refresh_layers is None
            else self.parallel_refresh_layers[:1]
        )
        for layer_idx in warm_layers:
            self._proxy(
                layer_idx,
                (
                    workspace.key_values[2 * layer_idx],
                    workspace.key_values[2 * layer_idx + 1],
                ),
                use_previous=True,
                staged_hidden=workspace.staged[layer_idx:layer_idx + 1],
            )
        torch.cuda.current_stream(staged.device).synchronize()
        with torch.cuda.graph(graph):
            self._refresh_layers(workspace.key_values, workspace.staged, workspace.packed)
        return workspace

    @torch.inference_mode()
    def after_dense_step(self, output, packed_cache):
        self.suffix_eligible = None
        if self.decoded_since_refresh <= self.refresh_after_tokens:
            if self.parallel_refresh_layers is not None:
                self._select_suffix(output.past_key_values)
            return
        if self.window_start == self.block_start:
            self.decoded_since_refresh = 0
            self._select_suffix(output.past_key_values)
            return
        self.refresh_count += 1
        self.decoded_since_refresh = 0
        flat_key_values = output.past_key_values
        staged = load_hidden_window_from_offload(
            self.flat_offload,
            device=flat_key_values[0].device,
            dtype=flat_key_values[0].dtype,
        )
        if isinstance(self.effective_patched_rows, int):
            self.effective_patched_rows = torch.zeros(
                (), dtype=torch.int64, device=staged.device
            )
        if (
            staged.is_cuda
            and self.parallel_refresh_layers is not None
            and self._refresh_streams is None
        ):
            self._refresh_streams = [
                torch.cuda.Stream(device=staged.device)
                for _ in self.parallel_refresh_layers
            ]
        if staged.is_cuda and self.use_cuda_graph:
            key = self._graph_key(flat_key_values, staged, packed_cache)
            if self._active_workspace is None:
                workspace = self._graph_pool.pop(key, None)
                if workspace is None:
                    workspace = self._make_workspace(
                        flat_key_values, staged, packed_cache
                    )
                else:
                    self._activate_workspace(workspace)
                self._graph_pool[key] = workspace
                if len(self._graph_pool) > _MAX_GRAPH_SHAPES:
                    del self._graph_pool[next(iter(self._graph_pool))]
            workspace = self._active_workspace
            torch._foreach_copy_(workspace.key_values, flat_key_values)
            workspace.staged.copy_(staged)
            workspace.packed._data.copy_(packed_cache._data)
            workspace.graph.replay()
            packed_cache._data.copy_(workspace.packed._data)
            staged = workspace.staged
        else:
            self._refresh_layers(flat_key_values, staged, packed_cache)
        refreshed_kv = (
            self._active_workspace.key_values
            if self._active_workspace is not None else flat_key_values
        )
        self._select_suffix(refreshed_kv)
        if staged.is_cuda:
            offload_stream = self._stream_for(staged.device)
            offload_stream.wait_stream(torch.cuda.current_stream(staged.device))
            with torch.cuda.stream(offload_stream):
                store_hidden_window_to_offload(self.flat_offload, staged)
            staged.record_stream(offload_stream)
        else:
            store_hidden_window_to_offload(self.flat_offload, staged)
        for state in self.states:
            state.offload.ready_event = self.flat_offload.ready_event
