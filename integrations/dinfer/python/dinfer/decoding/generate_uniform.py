import torch
import logging

from .utils import TokenArray

logger = logging.getLogger(__name__)


class DiffusionLLM:
    """ Diffusion LLM inference
    """

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompt: Torch.Tensor
            A tensor of shape (1, L) that contains the input prompt.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (1, L') that contains the prompt tokens and the generated tokens.
            EOS and any tokens after EOS have been removed.
        '''

def select_undecoded(seq_idx, orig_x, x, block, block_loc, mask_id, writeback=False):
    if x.batch_size == 1:
        return seq_idx, x
    bool_idx = torch.all(block != mask_id, dim=1)

    if writeback:
        # Write the decoded tokens back
        finished_idx = seq_idx[bool_idx]
        orig_x[finished_idx, block_loc.start:block_loc.end] = block[bool_idx]

    # Select the undecoded sequences
    return seq_idx, x

class BlockRunner:
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : DiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf):
        self.diff_iteration = diff_iteration
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input tokens in the block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID

        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1))
            for unroll_i in range(unroll_k):
                self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id)

            # If there are more than one sequence, we should filter the sequences and only decode
            # on the sequences that still have masked tokens.
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break
            batch_size = x.batch_size

        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            # Find the first location of EOS and set all tokens after the location to EOS.
            # Here we assume that don't perform remasking.
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx


class DiffusionIteration:
    """ A diffusion iteration to decode tokens
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0

    def forward(self, model, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        pass

class BaseDiffusionIteration(DiffusionIteration):
    """ A base implementation of diffusion iteration to decode.
    """
    def __init__(self):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        cache_update_kv = None
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            cache_update_kv = output.past_key_values
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        if kv_cache is None:
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            logits = logits[:, :block_length]
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(block, past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits

        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return cache_update_kv, logits




class BlockWiseDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : KVCacheFactory (optional)
        The KV-cache factory that generates a kv-cache for LLM.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.diff_iteration = BaseDiffusionIteration()
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)


    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()
