import pytest
import torch

from polestar.models.llada.generation import get_transfer_index, get_pre_transfer_index


def test_commit_does_not_select_existing_tokens():
    logits = torch.tensor([[[0., 10.], [10., 0.], [0., 0.]]])
    masked = torch.tensor([[True, False, True]])
    current = torch.tensor([[7, 1, 7]])
    predicted, selected, _ = get_transfer_index(logits, 0., "low_confidence", masked, current, None, 0.9)
    assert selected.tolist() == [[True, False, False]]
    assert predicted[0, 1].item() == current[0, 1].item()


def test_current_block_makes_progress_below_confidence_threshold():
    logits = torch.zeros(1, 3, 4)
    masked = torch.ones(1, 3, dtype=torch.bool)
    _, selected, _ = get_transfer_index(logits, 0., "low_confidence", masked, torch.full((1, 3), 7), None, 0.9)
    assert selected.sum().item() == 1


def test_suffix_confidence_rule_can_select_no_tokens():
    logits = torch.zeros(1, 3, 4)
    masked = torch.ones(1, 3, dtype=torch.bool)
    _, selected = get_pre_transfer_index(logits, 0., "low_confidence", masked, torch.full((1, 3), 7), num_transfer_tokens=None, threshold=0.9)
    assert not selected.any()


def test_current_and_suffix_threshold_rules():
    logits = torch.tensor([[[9., 0.], [2., 0.], [6., 0.], [0., 9.]]])
    masked = torch.tensor([[True, True, True, False]])
    current = torch.full((1, 4), 7)
    _, current_selected, _ = get_transfer_index(
        logits, 0., "low_confidence", masked, current, None, 0.9
    )
    _, suffix_selected = get_pre_transfer_index(
        logits, 0., "low_confidence", masked, current, None, 0.9
    )
    assert current_selected.tolist() == [[True, False, True, False]]
    assert suffix_selected.tolist() == [[True, False, True, False]]

    high_threshold = 0.99999
    _, current_selected, _ = get_transfer_index(
        logits, 0., "low_confidence", masked, current, None, high_threshold
    )
    _, suffix_selected = get_pre_transfer_index(
        logits, 0., "low_confidence", masked, current, None, high_threshold
    )
    assert current_selected.tolist() == [[True, False, False, False]]
    assert not suffix_selected.any()
