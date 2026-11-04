"""Tokenizer for DiffusionGemma without transformers 5.

`AutoTokenizer.from_pretrained` fails on this checkpoint under transformers 4.x (its tokenizer_config.json uses the
transformers-5 format), so build the fast tokenizer from tokenizer.json directly. The special tokens must be set
explicitly: the chat template inserts `bos_token`, and without it the prompt would silently lose `<bos>`.
The result tokenizes chat prompts identically to HF transformers 5's processor.
"""
import json
import os

from transformers import PreTrainedTokenizerFast


def load_diffusion_gemma_tokenizer(model_path):
    cfg = json.load(open(os.path.join(model_path, 'tokenizer_config.json')))
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path, 'tokenizer.json'),
                                  **{k: cfg[k] for k in ('bos_token', 'eos_token', 'pad_token', 'mask_token')})
    tok.chat_template = open(os.path.join(model_path, 'chat_template.jinja')).read()
    return tok
