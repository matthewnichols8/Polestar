"""Polestar-Cache for DiffusionGemma in dInfer: reuse canvas states across denoising steps.

DiffusionGemma writes its prefix KV cache with a causal encode, so cached prefix states never drift and there is
nothing to refresh there. The expensive part is the canvas: every denoising step recomputes all canvas tokens in
every layer. This cache applies Polestar's drift-guided selective update across the denoising steps of a canvas.

On a partial step, each layer recomputes only

    dirty tokens (input token id changed since the previous step)
    U tokens in the `topk` clusters whose centroid proxy attention drifted most (KL vs the previous step).

Every other token reuses its cached layer update (reuse="delta": out = s * h_now + (out_old - s * h_old), with s
the layer scalar, so changes from earlier layers still propagate) or its cached output (reuse="state"). K/V are
recomputed for every canvas token from its current layer input, so recomputed queries see every token's latest
state. Partial steps dequantize only the experts the selected tokens route to.

The whole canvas is recomputed, and each layer's clusters rebuilt, on the first step of every canvas and every
`refresh_every` steps. Selection runs on rank 0 and is broadcast, so expert-parallel ranks always process the
same tokens.

Usage::

    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, EntropyBoundParallelDecoder, KVCacheFactory
    from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

    cache = CanvasPolestarCache(len(model.layers), topk=2, refresh_every=8)
    llm = CanvasDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
                             canvas_cache=cache)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from polestar.models.llada.clustering import k_means
from polestar.models.llada.system_cache import compute_cluster_drift_kl, incremental_centroids


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class CanvasPolestarCache:
    """Polestar-Cache state for one DiffusionGemma generator; `CanvasDiffusionLLM` resets it at every canvas.

    Parameters
    ----------
    num_layers : int
    clusters : int
        K-means clusters per layer.
    topk : int or sequence of int
        Highest-drift clusters recomputed per layer on partial steps; a sequence gives one value per layer.
    refresh_every : int
        Recompute the whole canvas every this many steps of a canvas (step 0 is always full). 0 refreshes only on
        step 0.
    reuse : "delta" or "state"
    polestar_seed : int
        Seeds the k-means initialization.
    """

    def __init__(
        self,
        num_layers: int,
        *,
        clusters: int = 8,
        topk: int | list[int] | tuple[int, ...] = 4,
        refresh_every: int = 4,
        reuse: str = "delta",
        polestar_seed: int = 42,
    ) -> None:
        if reuse not in {"delta", "state"}:
            raise ValueError("reuse must be 'delta' or 'state'")
        if clusters <= 0:
            raise ValueError("clusters must be positive")
        if refresh_every < 0:
            raise ValueError("refresh_every must be non-negative")
        topk = list(topk) if isinstance(topk, (list, tuple)) else [topk] * num_layers
        if len(topk) != num_layers or any(k < 0 for k in topk):
            raise ValueError("topk must be a non-negative int or one value per layer")
        self.num_layers = num_layers
        self.clusters = clusters
        self.topk = topk
        self.refresh_every = refresh_every
        self.reuse = reuse
        self.polestar_seed = polestar_seed
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Forget all states; called at the start of every canvas."""
        L = self.num_layers
        self.step = 0
        self.full = True
        self.dirty = None
        self._prev_canvas = None
        self.h_in = [None] * L        # [C, H]: each token's layer input from the last step it was computed
        self.h_out = [None] * L       # [C, H]: the matching layer output
        self.centroids = [None] * L   # [1, K, H]
        self.counts = [None] * L      # [1, K, 1]
        self.labels = [None] * L      # [C]
        self.prev_proxy = [None] * L  # [1, heads, K, S]
        self.step_fracs = []
        self._fracs = []

    def stats(self) -> dict:
        partial = len(self.step_fracs)
        return {"full_steps": self.step - partial, "partial_steps": partial, "recompute_frac": list(self.step_fracs)}

    def begin_step(self, canvas: torch.Tensor) -> None:
        """canvas: [1, C] input token ids of this denoising pass."""
        if canvas.shape[0] != 1:
            raise ValueError("the Polestar canvas cache supports batch size 1")
        canvas = canvas[0]
        self.full = (
            self._prev_canvas is None
            or self._prev_canvas.shape != canvas.shape
            or (self.refresh_every > 0 and self.step % self.refresh_every == 0)
        )
        self.dirty = None if self.full else canvas != self._prev_canvas
        self._prev_canvas = canvas.clone()
        self.step += 1
        self._fracs = []

    def end_step(self) -> None:
        if not self.full:
            self.step_fracs.append(sum(self._fracs) / max(len(self._fracs), 1))

    def run_layer(self, layer, h, cos, sin, cache, positions):
        """Called by DiffusionGemmaModelLM.denoise for every layer. h: [1, C, H] layer input."""
        if self.full:
            return self._run_full(layer, h, cos, sin, cache, positions)
        return self._run_partial(layer, h, cos, sin, cache)

    # ------------------------------------------------------------------ full steps
    def _run_full(self, layer, h, cos, sin, cache, positions):
        out = layer(h, cos, sin, cache, "decode", positions)
        i = layer.layer_idx
        self.h_in[i] = h[0].detach().clone()
        self.h_out[i] = out[0].detach().clone()
        if not _distributed() or dist.get_rank() == 0:
            keys, _ = layer.self_attn.keys_values(layer.input_norm(h), cos, sin, cache)
            self._cluster(i, layer, keys)
        return out

    def _cluster(self, i, layer, keys):
        hidden = self.h_in[i]
        devices = [hidden.device] if hidden.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.polestar_seed + 7919 * i)
            centroids, labels = k_means(hidden[None, None], min(self.clusters, hidden.shape[0]))
        centroids, labels = centroids[:, 0], labels[0, 0]                      # [1, K, H], [C]
        K = centroids.shape[1]
        self.counts[i] = F.one_hot(labels, K).sum(0).to(hidden.dtype).view(1, K, 1)
        self.centroids[i] = centroids
        self.labels[i] = labels
        self.prev_proxy[i] = layer.proxy_attention(centroids[0], keys)[None]

    # ------------------------------------------------------------------ partial steps
    def _select(self, i, layer, keys):
        """Sorted indices of the canvas tokens to recompute in layer `i`."""
        C = self.h_in[i].shape[0]
        mask = torch.zeros(C, dtype=torch.uint8, device=self.h_in[i].device)
        if not _distributed() or dist.get_rank() == 0:
            selected = self.dirty.clone()
            k = min(self.topk[i], self.centroids[i].shape[1])
            if k > 0:
                proxy = layer.proxy_attention(self.centroids[i][0], keys)[None]  # [1, heads, K, S]
                drift = compute_cluster_drift_kl(proxy, self.prev_proxy[i])[0]   # [K]
                self.prev_proxy[i] = proxy
                selected |= torch.isin(self.labels[i], drift.topk(k).indices)
            mask = selected.to(torch.uint8)
        if _distributed():
            dist.broadcast(mask, src=0)
        idx = mask.bool().nonzero(as_tuple=True)[0]
        self._fracs.append(idx.numel() / C)
        return idx

    def _reused(self, i, h_now, scalar):
        if self.reuse == "state":
            return self.h_out[i].clone()
        return h_now * scalar + (self.h_out[i] - self.h_in[i] * scalar)

    def _run_partial(self, layer, h, cos, sin, cache):
        i = layer.layer_idx
        x = layer.input_norm(h)
        keys, values = layer.self_attn.keys_values(x, cos, sin, cache)
        idx = self._select(i, layer, keys)
        out = self._reused(i, h[0], layer.layer_scalar)[None]
        if idx.numel() == 0:
            return out
        h_sel = h[:, idx]
        attention = layer.self_attn.attend(x[:, idx], cos[idx], sin[idx], keys, values)
        layer.experts.sparse_dequant = True
        try:
            out_sel = layer._ffn(layer.post_attention(h_sel, attention))
        finally:
            layer.experts.sparse_dequant = False
        out[:, idx] = out_sel
        if not _distributed() or dist.get_rank() == 0:
            self.centroids[i] = incremental_centroids(
                self.centroids[i], self.labels[i][idx][None], self.counts[i],
                self.h_in[i][idx][None], h_sel,
            )
        self.h_in[i][idx] = h_sel[0]
        self.h_out[i][idx] = out_sel[0]
        return out
