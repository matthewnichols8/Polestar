"""CUDA graph replay for fixed-shape, sequential LLaDA decode operations."""

from __future__ import annotations

from collections.abc import Callable

import torch

CapturedOutput = torch.Tensor | tuple[torch.Tensor, ...]


class CapturedCall:
    """Own static input/output buffers for a sequential, batch-one worker."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor], CapturedOutput],
        *,
        length: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        clone_output: bool = False,
    ) -> None:
        self.clone_output = clone_output
        self.input = torch.zeros((1, length, width), device=device, dtype=dtype)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                forward(self.input)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = forward(self.input)
        if clone_output and not isinstance(self.output, torch.Tensor):
            raise ValueError("cloned CUDA graph outputs must be a single tensor")

    def accepts(self, x: torch.Tensor) -> bool:
        return x.shape == self.input.shape and x.dtype == self.input.dtype and x.device == self.input.device

    def __call__(self, x: torch.Tensor) -> CapturedOutput:
        if not self.accepts(x):
            raise ValueError("CUDA graph input shape, dtype, and device must match capture")
        self.input.copy_(x)
        self.graph.replay()
        if self.clone_output:
            # The next layer can retain the FFN result in its hidden-state cache.
            return self.output.clone()
        return self.output
