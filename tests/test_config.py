import json

import pytest

from polestar import PolestarConfig


@pytest.mark.parametrize("filename", [
    "llada8b.json",
    "llada8b-cuda-graphs.json",
    "llada8b-stream-overlap.json",
    "llada8b-nvfp4.json",
    "llada8b-nvfp4-overlap.json",
    "llada15.json",
    "dream7b.json",
    "dream7b-cuda-graphs.json",
    "llada-v.json",
])
def test_model_presets(filename):
    config = PolestarConfig.from_json("configs/" + filename)
    assert config.generation["block_length"] == 32
    assert config.generation["threshold"] == 0.9
    assert config.generation["threshold_early"] == 0.7


def test_invalid_block_schedule():
    with pytest.raises(ValueError, match="divisible"):
        PolestarConfig("llada", "checkpoint", {"max_new_tokens": 33, "block_length": 32})


def test_invalid_model_type():
    with pytest.raises(ValueError, match="model_type"):
        PolestarConfig("unknown", "checkpoint")


def test_configuration_roundtrip(tmp_path):
    data = {"model_type": "dream", "model_id": "checkpoint", "generation": {"max_new_tokens": 512, "steps": 512, "block_length": 32}}
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(data))
    assert PolestarConfig.from_json(path).generation == data["generation"]


def test_dream_cuda_graphs_are_supported():
    config = PolestarConfig("dream", "checkpoint", {"enable_cuda_graphs": True, "enable_fused_rope": True})
    assert config.generation["enable_cuda_graphs"]
    assert config.generation["enable_fused_rope"]


def test_gpu_accelerations_default_on_and_can_be_disabled():
    llada = PolestarConfig("llada", "checkpoint")
    assert all(llada.generation_settings("cuda")[name] for name in (
        "enable_cuda_graphs", "enable_quantization", "enable_overlap"
    ))
    assert not any(llada.generation_settings("cpu")[name] for name in (
        "enable_cuda_graphs", "enable_quantization", "enable_overlap"
    ))
    llada.generation["enable_overlap"] = False
    assert llada.generation_settings("cuda")["enable_overlap"] is False

    dream = PolestarConfig("dream", "checkpoint", {"enable_fused_rope": False})
    assert dream.generation_settings("cuda")["enable_cuda_graphs"] is True
    assert dream.generation_settings("cuda")["enable_fused_rope"] is False
