import torch

from polestar.models.llada.clustering import k_means


def test_empty_clusters_do_not_advance_random_stream():
    hidden = torch.zeros(1, 1, 16, 8)
    torch.manual_seed(23)
    torch.randperm(16)
    next_expected = torch.rand(1)

    torch.manual_seed(23)
    centroids, labels = k_means(hidden, num_clusters=4)
    next_actual = torch.rand(1)

    assert torch.equal(next_actual, next_expected)
    assert torch.isfinite(centroids).all()
    assert labels.shape == (1, 1, 16)
