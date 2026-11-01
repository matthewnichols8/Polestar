"""dInfer must select Polestar Cache and Commit together."""

import torch
import pytest

from polestar.integrations import dinfer_commit


class FakeModel:
    device = torch.device("cpu")


def test_dinfer_polestar_selects_cache_and_commit(monkeypatch):
    from dinfer.decoding.serving import SamplingParams, init_generator

    calls = []
    result = object()

    def fake_factory(model, **kwargs):
        calls.append((model, kwargs))
        return result

    monkeypatch.setattr(dinfer_commit, "make_polestar_drift_generator", fake_factory)
    model = FakeModel()
    params = SamplingParams(
        parallel_decoding="polestar", cache="polestar", mask_id=126336,
        eos_id=126081, use_bd=False, polestar_seed=13,
        polestar_refresh_after_tokens=24,
        polestar_refresh_layer_count=19,
    )
    assert init_generator(model, params) is result
    assert calls == [(model, {
        "threshold": params.threshold,
        "lower_bound": params.current_commit_lower_bound,
        "mask_id": 126336,
        "eos_id": 126081,
        "refresh_after_tokens": 24,
        "refresh_layer_count": 19,
        "request_seed": 13,
        "steps": None,
    })]


def test_dinfer_polestar_requires_its_cache():
    from dinfer.decoding.serving import SamplingParams, init_generator

    params = SamplingParams(
        parallel_decoding="polestar", cache="prefix", mask_id=126336,
        eos_id=126081, use_bd=False,
    )
    with pytest.raises(ValueError, match="Polestar Cache and Commit"):
        init_generator(FakeModel(), params)


def test_dinfer_polestar_requires_greedy_decoding():
    from dinfer.decoding.serving import SamplingParams, init_generator

    params = SamplingParams(
        parallel_decoding="polestar", cache="polestar", temperature=0.5
    )
    with pytest.raises(ValueError, match="temperature=0"):
        init_generator(FakeModel(), params)


def test_dinfer_prefix_baseline_uses_documented_block_decoder():
    from dinfer.decoding.generate_uniform import BlockWiseDiffusionLLM
    from dinfer.decoding.serving import SamplingParams, init_generator

    params = SamplingParams(
        parallel_decoding="threshold", cache="prefix",
        cont_weight=0.0, use_bd=False,
    )
    assert isinstance(init_generator(FakeModel(), params), BlockWiseDiffusionLLM)


def test_dinfer_serving_exposes_optional_worker_decode_time(monkeypatch):
    from dinfer.decoding import serving

    worker = object.__new__(serving.DiffusionLLMServing)
    worker.sample_params = serving.SamplingParams(
        parallel_decoding="polestar",
        cache="polestar",
        eos_id=126081,
        polestar_profile_timing=True,
    )
    worker.timeout = 1
    monkeypatch.setattr(serving.handle, "add_requests", lambda reqs: None)
    monkeypatch.setattr(
        serving.handle,
        "get_responses",
        lambda timeout=None: [(torch.tensor([[1, 2, 3]]), 7, 0.25)],
    )

    output = worker.generate(torch.tensor([[1, 2]]), gen_length=1, block_length=1)
    assert output.tolist() == [[1, 2, 3]]
    assert worker.num_forwards == 7
    assert worker.last_decode_seconds == 0.25


def test_dinfer_serving_flushes_stop_before_joining_worker():
    from dinfer.decoding.serving import ServerGroup

    events = []

    class Queue:
        def put(self, message):
            events.append(("put", message))

        def close(self):
            events.append(("close", None))

        def join_thread(self):
            events.append(("join_thread", None))

    class Process:
        def join(self, timeout=None):
            events.append(("join", timeout))

        def is_alive(self):
            return False

    group = ServerGroup()
    group.procs = [Process()]
    group.req_qs = [Queue()]
    group.error_qs = [Queue()]
    group.res_q = Queue()
    group.stop_running()

    assert events[0] == ("put", "stop")
    assert events[1:4] == [("close", None), ("join_thread", None), ("join", 20)]
    assert group.procs == []


def test_dinfer_serving_reports_worker_error_without_waiting_for_timeout():
    from queue import Queue
    from dinfer.decoding.serving import ServerGroup

    group = ServerGroup()
    group.res_q = Queue()
    group.error_qs = [Queue()]
    group.error_qs[0].put((ValueError("failed"), "worker traceback"))

    with pytest.raises(RuntimeError, match="worker traceback"):
        group.get_response(timeout=60)
