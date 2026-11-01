"""System cache windows must match the release decoder's periodic local prefill."""

import torch
import torch.nn.functional as F
import pytest

from polestar.models.llada.system_cache import (
    apply_state_patching,
    build_cluster_update_packet,
    compute_cluster_drift_kl,
    initialize_clustered_hidden_state,
    initialize_clustered_hidden_state_with_offload,
    load_hidden_window_from_offload,
    load_selected_hidden_from_offload,
    quantize_hidden_for_offload,
    update_quantized_offload_state,
)
from polestar.models.llada.system_clustering import (
    dequantize_residual_codes,
    quantize_residual_codes,
    quantize_residual_given_clusters,
    unpack_codes,
)


def test_local_prefill_uses_global_labels_and_matching_centroids():
    hidden = torch.randn(1, 160, 32)
    active = torch.zeros(1, 160, dtype=torch.bool)
    active[:, 96:128] = True
    kwargs = {
        "x": hidden,
        "replace_position": active,
        "window_size": 32,
        "target_total": 8,
    }

    torch.manual_seed(17)
    full_centroids, full_labels, full_counts = initialize_clustered_hidden_state(
        current_hidden_state=hidden.clone(), **kwargs
    )
    torch.manual_seed(17)
    local_centroids, local_labels, local_counts = initialize_clustered_hidden_state(
        current_hidden_state=hidden[:, 32:160].clone(), **kwargs
    )

    torch.testing.assert_close(local_centroids, full_centroids)
    torch.testing.assert_close(local_labels, full_labels)
    torch.testing.assert_close(local_counts, full_counts)


def test_proxy_scores_preserve_attention_heads():
    centroids = torch.randn(1, 3, 8)
    labels = torch.tensor([[0, 1, 2, -1]])
    key = torch.randn(1, 2, 4, 4)
    value = torch.randn_like(key)
    _, scores = build_cluster_update_packet(
        clustered_hidden_state=(centroids, labels, torch.ones(1, 3, 1)),
        cache=(key, value),
        hidden_state_past=None,
        replace_position=torch.tensor([[False, False, False, True]]),
        topk=2,
        rope_k=key,
        attn_norm=torch.nn.Identity(),
        q_proj=torch.nn.Identity(),
        ff_norm=torch.nn.Identity(),
        ff_proj=torch.nn.Identity(),
        up_proj=torch.nn.Identity(),
        act=torch.nn.Identity(),
        ff_out=torch.nn.Identity(),
        attn_out=torch.nn.Identity(),
        dropout=torch.nn.Identity(),
        rotary_emb=None,
        scaled_dot_product_attention_fn=None,
        d_model=8,
        n_heads=2,
        prev_attn_scores=None,
    )
    query = centroids.view(1, 3, 2, 4).transpose(1, 2)
    expected = F.softmax(query @ key.transpose(-2, -1) / 2, dim=-1)
    torch.testing.assert_close(scores, expected)


def test_cluster_drift_reduces_each_head_before_averaging():
    current = torch.tensor(
        [[[[0.8, 0.2], [0.4, 0.6]], [[0.2, 0.8], [0.9, 0.1]]]]
    )
    previous = torch.tensor(
        [[[[0.5, 0.5], [0.7, 0.3]], [[0.6, 0.4], [0.3, 0.7]]]]
    )
    expected = (
        current
        * (torch.log(current + 1e-12) - torch.log(previous + 1e-12))
    ).sum(-1).clamp_min(0).mean(1)
    torch.testing.assert_close(compute_cluster_drift_kl(current, previous), expected)


def test_nvfp4_offload_uses_packed_centroid_residuals_and_local_indices():
    torch.manual_seed(7)
    centroids = torch.randn(1, 2, 32) * 4
    labels = torch.tensor([[-1, -1, 0, 1, 0, 1, -1]])
    hidden = centroids[:, labels[:, 2:6].squeeze(0), :] + torch.randn(1, 4, 32) * 0.01
    state = quantize_hidden_for_offload(
        hidden, centroids=centroids, labels=labels, start=2
    )
    assert state.q_cpu.shape == (1, 4, 16)
    assert state.scales_cpu.dtype == torch.uint8  # FP8 E4M3 bit pattern
    restored = load_selected_hidden_from_offload(
        state, torch.tensor([2, 4]), device=torch.device("cpu"), dtype=hidden.dtype
    )
    torch.testing.assert_close(restored, hidden[:, [0, 2], :], atol=0.01, rtol=0.01)

    updated = hidden[:, [0], :] + 0.05
    update_quantized_offload_state(state, torch.tensor([2]), updated)
    restored = load_selected_hidden_from_offload(
        state, torch.tensor([2]), device=torch.device("cpu"), dtype=hidden.dtype
    )
    torch.testing.assert_close(restored, updated, atol=0.02, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_nvfp4_encoder_matches_reference():
    pytest.importorskip("triton")
    for seed, width, dim in ((17, 6, 32), (23, 96, 4096)):
        torch.manual_seed(seed)
        hidden = torch.randn(1, width, dim, device="cuda", dtype=torch.bfloat16)
        centroids = torch.randn(1, 8, dim, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 8, (1, width), device="cuda")
        reference = quantize_hidden_for_offload(
            hidden, centroids=centroids, labels=labels, start=0, use_triton=False
        )
        fused = quantize_hidden_for_offload(
            hidden, centroids=centroids, labels=labels, start=0, use_triton=True
        )
        reference.ready_event.synchronize()
        fused.ready_event.synchronize()
        torch.testing.assert_close(fused.q_cpu, reference.q_cpu, rtol=0, atol=0)
        torch.testing.assert_close(fused.scales_cpu, reference.scales_cpu, rtol=0, atol=0)
        restored = load_hidden_window_from_offload(
            fused, device=hidden.device, dtype=hidden.dtype
        )
        gathered = centroids.gather(1, labels.unsqueeze(-1).expand_as(hidden))
        residual = dequantize_residual_codes(
            unpack_codes(fused.q_cpu.to("cuda")),
            fused.scales_cpu.to("cuda").view(torch.float8_e4m3fn),
        )
        expected = (gathered.float() + residual).to(hidden.dtype)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_nvfp4_quantization_restores_cluster_centroid():
    torch.manual_seed(11)
    centroids = torch.randn(1, 1, 32) * 10
    hidden = centroids + torch.randn(1, 3, 32) * 0.01
    recovered = quantize_residual_given_clusters(
        hidden, centroids, torch.zeros(1, 3, dtype=torch.long)
    )
    torch.testing.assert_close(recovered, hidden, atol=0.01, rtol=0.01)


def test_offload_only_stores_the_active_local_window():
    torch.manual_seed(23)
    hidden = torch.randn(1, 192, 32)
    active = torch.zeros(1, 192, dtype=torch.bool)
    active[:, 96:128] = True
    _, offload = initialize_clustered_hidden_state_with_offload(
        x=hidden, current_hidden_state=hidden, replace_position=active, window_size=32
    )
    assert offload.start == 32
    assert offload.q_cpu.shape == (1, 128, 16)


def test_sparse_patch_uses_offloaded_hidden_without_a_gpu_hidden_cache():
    torch.manual_seed(29)
    hidden = torch.randn(1, 6, 32) * 0.01
    centroids = hidden.mean(dim=1, keepdim=True)
    labels = torch.zeros(1, 6, dtype=torch.long)
    clusters = (centroids, labels, torch.tensor([[[6.0]]]))
    offload = quantize_hidden_for_offload(
        hidden, centroids=centroids, labels=labels, start=0
    )
    key = torch.randn(1, 1, 6, 32)
    value = torch.randn_like(key)
    indices = torch.tensor([2, 4])
    updated = hidden[:, indices, :] + 0.1
    kwargs = dict(
        update_packet_prev=(indices, updated),
        clustered_hidden_state=clusters,
        layer_past=(key, value),
        attn_norm=torch.nn.Identity(),
        k_proj=torch.nn.Identity(),
        v_proj=torch.nn.Identity(),
        d_model=32,
        n_heads=1,
        effective_n_kv_heads=1,
    )
    cached_hidden, cached_clusters, cached_kv = apply_state_patching(
        hidden_state_past=hidden, **kwargs
    )
    offloaded_hidden, offloaded_clusters, offloaded_kv = apply_state_patching(
        hidden_state_past=None, quant_offload_state=offload, **kwargs
    )
    assert offloaded_hidden is None
    torch.testing.assert_close(offloaded_clusters[0], cached_clusters[0], atol=0.002, rtol=0)
    torch.testing.assert_close(offloaded_kv[0], cached_kv[0], rtol=0, atol=0)
    torch.testing.assert_close(offloaded_kv[1], cached_kv[1], rtol=0, atol=0)
    restored = load_selected_hidden_from_offload(
        offload, indices, device=hidden.device, dtype=hidden.dtype
    )
    codes, scales = quantize_residual_codes(updated - centroids)
    expected = centroids + dequantize_residual_codes(
        codes, scales, precision="fp4", group_size=16
    )
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
    torch.testing.assert_close(restored, cached_hidden[:, indices, :], atol=0.025, rtol=0)


def test_fixed_candidate_packet_only_changes_selected_rows():
    torch.manual_seed(31)
    hidden = torch.randn(1, 6, 32)
    indices = torch.arange(1, 5)
    selected = torch.tensor([False, True, False, True])
    updated = hidden[:, indices] + 0.25
    labels = torch.zeros(1, 6, dtype=torch.long)
    clusters = (hidden.mean(dim=1, keepdim=True), labels, torch.tensor([[[6.0]]]))
    key = torch.randn(1, 1, 6, 32)
    value = torch.randn_like(key)
    staged = hidden.clone()

    _, next_clusters, (next_key, next_value) = apply_state_patching(
        update_packet_prev=(indices, updated, selected),
        hidden_state_past=None,
        clustered_hidden_state=clusters,
        layer_past=(key, value),
        attn_norm=torch.nn.Identity(),
        k_proj=torch.nn.Identity(),
        v_proj=torch.nn.Identity(),
        d_model=32,
        n_heads=1,
        effective_n_kv_heads=1,
        cached_hidden_window=staged,
    )
    torch.testing.assert_close(staged[:, indices[~selected]], hidden[:, indices[~selected]])
    torch.testing.assert_close(next_key[:, :, indices[~selected]], key[:, :, indices[~selected]])
    torch.testing.assert_close(
        next_value[:, :, indices[~selected]], value[:, :, indices[~selected]]
    )
    torch.testing.assert_close(staged[:, indices[selected]], updated[:, selected])
    expected_centroid = clusters[0] + (
        (updated[:, selected] - hidden[:, indices[selected]]).sum(dim=1, keepdim=True) / 6
    )
    torch.testing.assert_close(next_clusters[0], expected_centroid, atol=1e-6, rtol=0)
