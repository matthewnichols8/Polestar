import torch
import numpy as np
import torch.nn.functional as F

from .utils import add_gumbel_noise
import torch.distributed as dist


def broadcast_if_needed(x, src=0, group=None):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.broadcast(x, src=src)












class ParallelDecoder:
    """ This is a parallel decoder that decodes tokens in a block.
    """
    def __init__(
            self,
            temperature,
            remasking='low_confidence',
            mask_id=126336,
    ):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        pass

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.

        Parameters
        ----------
        logits : Tensor
            The logits in a block
        block_start : int
            The location of the starting token in the block
        block_end : int
            The location of the ending token in the block.
        x : Tensor
            The tensor where the decoded tokens are written to.
        """


# Parallel decoding only
@torch.compile(dynamic=True)
def get_transfer_index_threshold(
    logits,
    temperature,
    mask_index,
    x,
    mask_id,
    threshold,
    rm_mask=True,
    use_float64=False,
    **kwargs,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if use_float64:
        p = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        p = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

    # gurantee the denoised token will not be the mask_id
    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    actual_threshold = (
        torch.max(confidence, dim=1)[0] - 1e-5
    ).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index


class ThresholdParallelDecoder(ParallelDecoder):
    """Parallel decoding driven by a confidence threshold."""
    def __init__(
            self,
            temperature,
            threshold,
            remasking='low_confidence',
            mask_id=126336,
            eos_id=126081,
            use_float64=False,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.eos_id = eos_id
        self.use_float64 = use_float64

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in the same block of multiple samples.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index_threshold(
            logits,
            self.temperature,
            mask_index,
            curr_x,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )
        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        broadcast_if_needed(x.data)


class EntropyBoundParallelDecoder(ParallelDecoder):
    """Parallel decoder for uniform-noise diffusion models (DiffusionGemma).

    Unlike the mask-based decoders above, there is no mask token: a block (a "canvas") starts as uniformly random
    tokens and every position is re-predicted at every step. Mirrors HF transformers' DiffusionGemma sampler:

      * temperature schedule  T = t_min + (t_max - t_min) * cur_step / max_denoising_steps
        (`cur_step` counts the steps *remaining*, from max_denoising_steps down to 1);
      * sample every position from softmax(logits / T); the argmax canvas is the current best guess;
      * accept the lowest-entropy positions while  cumsum(H_sorted) - H_i <= entropy_bound
        (at least one per step) and re-noise every other position with fresh uniform tokens;
      * stop once the argmax canvas is unchanged for `stability_threshold` steps and the mean entropy is below
        `confidence_threshold`.

    `decode` writes the next canvas into `x[:, block_start:block_end]` (rank 0's random choices are broadcast so
    tensor/expert-parallel ranks stay identical) and exposes `argmax`, `sc_logits` (logits/T, which the model uses
    for self-conditioning on the next step) and `done`.
    """

    def __init__(self, vocab_size, max_denoising_steps=48, entropy_bound=0.1, t_min=0.4, t_max=0.8,
                 stability_threshold=1, confidence_threshold=0.005, eos_id=(1, 106, 50), pad_id=0, seed=None,
                 device='cuda'):
        super().__init__(temperature=t_max, remasking='entropy_bound', mask_id=None)
        assert t_max > t_min >= 0
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps
        self.entropy_bound = entropy_bound
        self.t_min, self.t_max = t_min, t_max
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.eos_id = tuple(eos_id)
        self.pad_id = pad_id
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()
        self.history = None
        self.argmax = self.sc_logits = self.done = None

    def block_init(self, block_x, block_id):
        self.history = None
        self.argmax, self.sc_logits, self.done = block_x.clone(), None, None

    def noise(self, shape, device):
        return torch.randint(0, self.vocab_size, shape, device=device, generator=self.generator)

    def init_block(self, x, block_start, block_end):
        """Fill the block with uniform noise (identical on every rank) and reset the per-block state.
        `x` may be a tensor or a dInfer TokenArray."""
        block = x[:, block_start:block_end]
        x[:, block_start:block_end] = self.noise(tuple(block.shape), block.device)
        broadcast_if_needed(x.data)
        self.block_init(x[:, block_start:block_end], None)

    def temperature_at(self, cur_step):
        return self.t_min + (self.t_max - self.t_min) * (cur_step / self.max_denoising_steps)

    def decode(self, logits, block_start, block_end, x, cur_step):
        """logits: [B, block, V] fp32. Returns `done` ([B] bool): the block has converged."""
        canvas = x[:, block_start:block_end]
        proc = logits / self.temperature_at(cur_step)
        logp = torch.log_softmax(proc, dim=-1)
        probs = logp.exp()
        B, C, V = proc.shape
        sampled = torch.multinomial(probs.view(-1, V), 1, generator=self.generator).view(B, C)
        argmax = proc.argmax(dim=-1)
        entropy = -(probs * logp).sum(-1)
        del logp, probs

        sorted_h, order = torch.sort(entropy, dim=-1)
        keep_sorted = (torch.cumsum(sorted_h, dim=-1) - sorted_h) <= self.entropy_bound
        accepted = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        x[:, block_start:block_end] = torch.where(accepted, sampled, self.noise((B, C), canvas.device))

        if self.stability_threshold == 0:
            stable = torch.ones(B, dtype=torch.bool, device=canvas.device)
        else:
            if self.history is None:
                self.history = torch.full((self.stability_threshold, B, C), -1, dtype=argmax.dtype, device=argmax.device)
            stable = (self.history == argmax[None]).all(dim=-1).all(dim=0)
            self.history = torch.roll(self.history, shifts=-1, dims=0)
            self.history[-1] = argmax
        done = (stable & (entropy.mean(dim=-1) < self.confidence_threshold)).to(torch.uint8)

        broadcast_if_needed(x.data)
        broadcast_if_needed(argmax)
        broadcast_if_needed(done)
        self.argmax, self.done = argmax, done.bool()
        self.sc_logits = proc.to(torch.bfloat16)  # HF feeds back the processed logits in the embedding dtype
        return self.done
