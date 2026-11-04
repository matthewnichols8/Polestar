import math
import copy
import torch

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if math.isclose(temperature, 0.0):
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise



class TokenArray:
    """ A token array to support read, update and expansion.

    We need to access the tokens that have been generated and write new tokens to the array.
    Some algorithms require to expand the token array.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device):
        self.prompt = prompt.to(device)
        self.data = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.gen_length = gen_length
        self.eos_id = eos_id
        self.mask_id = mask_id

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def batch_size(self):
        return self.prompt.shape[0]

    @property
    def device(self):
        return self.data.device

    def expand(self, new_len):
        pass

    def get_generated_tokens(self):
        if self.batch_size == 1:
            return self.data[self.data != self.eos_id].unsqueeze(0)
        else:
            self.data[self.data == self.mask_id] = self.eos_id
            return self.data

    def select_seqs(self, idx):
        arr = copy.copy(self)
        arr.prompt = self.prompt[idx]
        arr.data = self.data[idx]
        return arr

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vals):
        self.data[idx] = vals


class BlockLoc:
    """ The location of the block in the token array.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end



class BlockIterator:
    """ Block iterator

    This performs block-wise iteration on the input token array for diffusion decoding.

    Parameters
    ----------
    x : TokenArray
        The token array that contains decoded tokens and stores the new generated tokens
    block_length : int
        The length of the block
    start_block_align : bool
        Align the first decoding block to the block size. The first block may overlap with the prompt.
    """
    def __init__(self, x, block_length, start_block_align=False):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.start_block_align = start_block_align
        if start_block_align:
            self.first_block_start = self._get_first_block_start()
        else:
            self.first_block_start = self.x.prompt.shape[1]

    def _get_first_block_start(self):
        prompt = self.x.prompt
        non_mask_number = (prompt != self.x.mask_id).sum(dim=-1).min().item()
        start = ((non_mask_number) // self.block_length) * self.block_length
        return start

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        current_block_start = self.first_block_start + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = min(current_block_start + self.block_length, self.x.total_length)
        assert current_block_end <= self.x.total_length
        self.iter += 1
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]



class BlockIteratorFactory:
    """ Iterator factory

    This generates iterators for DiffusionLLM to iterate over a sequence.

    Parameters
    ----------
    start_block_align : bool
        Align the first decoding block to the block size. The first block may overlap with the prompt.

    Returns
    -------
    BlockIterator : the block iterator.
    """
    def __init__(self, start_block_align=False):
        self._start_block_align = start_block_align

    def create(self, x, block_length):
        return BlockIterator(x, block_length, start_block_align=self._start_block_align)


class KVCache:
    """ The KV-cache

    Parameters
    ----------
    past_key_values : List[torch.Tensor]
        The keys and values of each transformer layer.
    """
    def __init__(self, past_key_values, backend='vllm', length=2048, cache_align_size=128):
        if backend == 'vllm':
            assert len(past_key_values) % 2 == 0
            self._data = past_key_values
        else:
            self.cache_align_size = cache_align_size
            assert len(past_key_values) % 2 == 0
            self._raw_data = past_key_values
            self._consolidate_raw()

            n = -(-(self._raw_data.shape[4] + 64) // self.cache_align_size)
            next_pow2 = 1 << (n - 1).bit_length() if n > 1 else 1
            self.length = next_pow2 * self.cache_align_size

            device = self._raw_data.device
            num_layer, _, batch_size, num_heads, seq_len, hidden_dim = self._raw_data.shape
            self._data = torch.zeros(num_layer, 2, batch_size, num_heads, self.length, hidden_dim, device=device, dtype=torch.bfloat16)
            self._data[:, :, :, :, :seq_len] = self._raw_data
    def consolidate(self):
        if isinstance(self._data, torch.Tensor):
            return

        num_layers = len(self._data) // 2
        inner_shape = self._data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._data = torch.stack(self._data, dim=0).reshape(num_layers, 2, *inner_shape)

    def _consolidate_raw(self):
        if isinstance(self._raw_data, torch.Tensor):
            return

        num_layers = len(self._raw_data) // 2
        inner_shape = self._raw_data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._raw_data = torch.stack(self._raw_data, dim=0).reshape(num_layers, 2, *inner_shape)

    @property
    def num_layers(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[0]

    @property
    def seq_len(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[4]

    def get_keys(self, layer_idx):
        """ Get the keys of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][0]

    def get_values(self, layer_idx):
        """ Get the values of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][1]

    def update(self, key_states, val_states, layer_idx, replace_position=None, backend='vllm'):
        """ Update the keys and values of a transformer layer.

        Parameters
        ----------
        key_states : torch.Tensor
            The keys in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        val_states : torch.Tensor
            The values in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        layer_idx : int
            The index of the transformer layer
        replace_position : Tuple[int]
            The start and the end position where keys and values should be updated.

        Returns
        -------
        torch.Tensor: the new keys for the entire sequence of the transformer layer.
        torch.Tensor: the new values for the entire sequence of the transformer layer.
        """
        if backend ==  'vllm':
            # This is dual cache.
            if replace_position is not None:
                keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=replace_position[0], end=replace_position[1])
                values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=replace_position[0], end=replace_position[1])
            else:
                # This is prefix cache.
                keys = torch.cat([self.get_keys(layer_idx), key_states], dim=2)
                values = torch.cat([self.get_values(layer_idx), val_states], dim=2)
        else:
            cache_length = self.get_keys(layer_idx).shape[2]
            block_length = key_states.shape[2]
            keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=cache_length - block_length, end=cache_length)
            values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=cache_length - block_length, end=cache_length)
        return keys, values


class DiffusionKVCacheManager:
    """ KV-cache for diffusion LLM.

    The KV-cache caches the KV of the tokens before and after the block that is being decoded.
    Because diffusion LLM uses bidirectional attention, the KV-cache has to be updated frequently in the diffusion iterations.
    This class basically defines the KV-cache update policy in the diffusion iterations. This includes the locations where
    keys and values can be updated and the frequency of the keys and values can be updated.

    """
    def __init__(self, cache_update_freq=None, cache_type='prefix', backend='vllm', max_length=2048):
        self.past_key_values = None
        self.block_start = None
        self.block_end = None
        self.cache_update_freq = cache_update_freq
        assert cache_type in ['prefix', 'dual']
        self.cache_type = cache_type
        self.backend=backend
        self.max_length = max_length

    def require_update(self, iter_no, block_start, block_end):
        """ require to update the kv-cache.

        Parameters
        ----------
        iter_no : int
            The diffusion iteration number
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.
        """
        if self.past_key_values is None:
            _require_update = True
        # If self.cache_update_freq is not specified, the KV-cache is updated when we enter a new block.
        if self.cache_update_freq is None:
            _require_update = self.block_start != block_start or self.block_end != block_end
        else:
            # Otherwise, we update the KV-cache when we enter a new block or the specified number of
            # diffusion iterations is reached.
            _require_update = iter_no % self.cache_update_freq == 0 \
                    or (self.block_start != block_start or self.block_end != block_end)
        # TODO(zhengda) change update logic to block idx
        self.block_start = block_start
        self.block_end = block_end
        return _require_update

    def update(self, past_key_values, range_start=None, range_end=None):
        """ update the KV-cache

        Parameters
        ----------
        past_key_values : List[torch.Tensor]
            The key values in all transformer layers.
        range_start : int
            The start of the range that is being updated.
        range_end : int
            The end of the range that is being updated.
        """
        if isinstance(past_key_values, KVCache):
            self.past_key_values = past_key_values
        else:
            self.past_key_values = KVCache(past_key_values, self.backend, self.max_length)
        # We should make sure the kv-cache in all layers are converted into a tensor.
        self.past_key_values.consolidate()


    def range_update(self, past_key_values, range_start=0, range_end=0, block_length=32):
        """ update the KV-cache

        Parameters
        ----------
        past_key_values : List[torch.Tensor]
            The key values in all transformer layers.
        range_start : int
            The start of the range that is being updated.
        range_end : int
            The end of the range that is being updated.
        """
        if isinstance(past_key_values, KVCache):
            # raise ValueError("past_key_values should be a list of tensors")
            self.past_key_values = past_key_values
        else:
            num_layers = len(past_key_values) // 2
            inner_shape = past_key_values[0].shape
            compact_kvcache = torch.stack(past_key_values, dim=0).reshape(num_layers, 2, *inner_shape)
            if block_length>0:
                self.past_key_values = KVCache(torch.cat((compact_kvcache[:, :, :, :, range_start:range_end-block_length], compact_kvcache[:, :, :, :, -block_length:]), dim=4), self.backend, self.max_length)
            else:
                self.past_key_values = KVCache(compact_kvcache[:, :, :, :, range_start:range_end], self.backend, self.max_length)
        # We should make sure the kv-cache in all layers are converted into a tensor.
        self.past_key_values.consolidate()

    def get_key_values(self, block_start, block_end):
        """ Get the key-values given the block that is being decoded.

        Parameters
        ----------
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.

        Returns
        -------
        List[List[torch.Tensor]] : the key-values required to decode the specified block.
        torch.Tensor : the tensor indicates the valid locations in the returned key-values.
        """
        # The key-value cache cannot be empty.
        assert self.past_key_values is not None
        if self.cache_type == 'prefix':
            replace_position = (int(block_start), int(self.past_key_values.seq_len))
        else:
            replace_position = (int(block_start), int(block_end))
        return self.past_key_values, replace_position



class PerLayerKVCache:
    """ Prefix KV-cache stored per layer, for models whose layers have different K/V shapes.

    `KVCache` stacks all layers into one tensor, which requires identical shapes. DiffusionGemma mixes
    sliding-window layers (8 KV heads x 256) and full-attention layers (2 KV heads x 512), so each layer keeps
    its own [B, H_kv, S, D] tensors here. Layers whose type is 'sliding_attention' keep only the last
    `sliding_window - 1` tokens (as HF's DynamicSlidingWindowLayer); other layers keep everything.
    Absolute key positions are tracked per layer type for building causal/sliding masks.
    """
    def __init__(self, num_layers, layer_types, sliding_window=None):
        self.layer_types = list(layer_types)
        self.sliding_window = sliding_window
        self.k = [None] * num_layers
        self.v = [None] * num_layers
        self.pos = {lt: None for lt in set(self.layer_types)}

    def _keep(self, layer_type):
        return self.sliding_window - 1 if layer_type == 'sliding_attention' and self.sliding_window else None

    def seq_length(self):
        """Number of tokens written so far (positions are contiguous from 0)."""
        p = next(iter(self.pos.values()))
        return 0 if p is None else int(p[-1].item()) + 1

    def get(self, layer_idx):
        return self.k[layer_idx], self.v[layer_idx]

    def key_positions(self, layer_type):
        return self.pos[layer_type]

    def append(self, layer_idx, layer_type, k, v):
        """k, v: [B, H_kv, L, D] for the new tokens of one layer."""
        if self.k[layer_idx] is not None:
            k = torch.cat([self.k[layer_idx], k], dim=2)
            v = torch.cat([self.v[layer_idx], v], dim=2)
        keep = self._keep(layer_type)
        if keep:
            k, v = k[:, :, -keep:], v[:, :, -keep:]
        self.k[layer_idx], self.v[layer_idx] = k.contiguous(), v.contiguous()

    def append_positions(self, positions):
        """Record the absolute positions of the tokens just appended (call once per forward, after all layers)."""
        for lt in self.pos:
            p = positions if self.pos[lt] is None else torch.cat([self.pos[lt], positions])
            keep = self._keep(lt)
            self.pos[lt] = p[-keep:] if keep else p


class PerLayerPrefixCacheManager:
    """ Prefix KV-cache manager for uniform-noise canvas diffusion (DiffusionGemma).

    The cache holds the prompt and every committed canvas; it is written only when committed tokens are encoded
    (one causal pass per canvas), and the denoising passes over the current canvas read it without writing.
    """
    def __init__(self, num_layers, layer_types, sliding_window=None):
        self.past_key_values = PerLayerKVCache(num_layers, layer_types, sliding_window)

    def get_key_values(self, block_start=None, block_end=None):
        return self.past_key_values

    @property
    def seq_len(self):
        return self.past_key_values.seq_length()


class KVCacheFactory:
    """ KV-cache factory.

    This class generates KV-cache for the diffusion LLM when it runs diffusion iterations.
    `per_layer=True` creates the per-layer prefix cache used by DiffusionGemma.
    """
    def __init__(self, cache_type, cache_update_freq=None, backend='vllm', max_length=2048, per_layer=False):
        self.cache_type = cache_type
        self.cache_update_freq = cache_update_freq
        self.backend = backend
        self.max_length = max_length
        self.per_layer = per_layer

    def create(self, config=None):
        """ `config` (the model config) is required when `per_layer=True`, to size the per-layer cache. """
        if self.per_layer:
            if self.cache_type != 'prefix':
                raise ValueError('the per-layer cache only supports prefix caching')
            return PerLayerPrefixCacheManager(config.num_hidden_layers, config.layer_types,
                                              getattr(config, 'sliding_window', None))
        return DiffusionKVCacheManager(cache_update_freq=self.cache_update_freq, cache_type=self.cache_type)
