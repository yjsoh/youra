"""
Default generator for RAGS
"""

import json
import logging
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ...config import Config
from ...abstract import AbsGenerator


class TransparentGenerator(AbsGenerator):
    """
    Transparent KVCache managing generator, returns the list[chunk + query]

    Must be decoded for human-readable output

    https://huggingface.co/docs/transformers/v4.37.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput

    - sequences (torch.LongTensor of shape (batch_size, sequence_length)) — The generated sequences. The second dimension (sequence_length) is either equal to max_length or shorter if all batches finished early due to the eos_token_id.
    - scores (tuple(torch.FloatTensor) optional, returned when output_scores=True is passed or when config.output_scores=True) — Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. Tuple of torch.FloatTensor with up to max_new_tokens elements (one element for each generated token), with each tensor of shape (batch_size, config.vocab_size).
    - attentions (tuple(tuple(torch.FloatTensor)), optional, returned when output_attentions=True is passed or config.output_attentions=True) — Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor of shape (batch_size, num_heads, generated_length, sequence_length).
    - hidden_states (utple(tuple(torch.FloatTensor)), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of torch.FloatTensor of shape (batch_size, generated_length, hidden_size).
    - past_key_values (tuple(tuple(torch.FloatTensor))), optional, returned when use_cache=True is passed or when config.use_cache=True) — NOTE: some models have a different past_key_values format, confirm with the model’s documentation. Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value tensor). The first Tuple is of length config.n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)) and optionally if config.is_encoder_decoder=True 2 additional tensors of shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
    """

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
        llm: AutoModelForCausalLM,
        log_level: int = logging.WARNING,
        **kwargs,
    ):
        super().__init__(config, tokenizer, llm)
        fh = logging.FileHandler(f"{__name__}.log", mode="a")
        fh.setLevel(logging.DEBUG)
        self.log.addHandler(fh)
        self.log.setLevel(log_level)

        with open("config/dataset2prompt.json", "r", encoding="utf-8") as f:
            self.dataset2prompt = json.load(f)

        self.encode_options = {"return_tensors": "pt", "add_special_tokens": True}
        self.decdoe_options = {"skip_special_tokens": True}
        self.generate_options = {
            "num_beams": 1,
            "top_p": 1.0,
            "temperature": 1.0,
            "return_dict_in_generate": True,
            "do_sample": False,
        }
        if "Llama-3" in self.model_name:
            self.generate_options.update({"pad_token_id": self.tokenizer.eos_token_id})

    @torch.no_grad()
    def _generate(self, augmented_str: str, **kwargs) -> dict[str, Union[str, list]]:
        ds_name = kwargs.get("ds_name", None)

        prompt = self.add_instuct_tokens(augmented_str)

        input_ids = self.tokenizer.encode(prompt, **self.encode_options).to(
            self.llm.device
        )

        input_len = input_ids.shape[1]
        maxgenlen = self.dataset2maxlen[ds_name]

        generated = self.llm.generate(
            input_ids, max_new_tokens=maxgenlen, **self.generate_options
        )
        generated_len = generated.sequences.shape[1]

        generated_str = self.tokenizer.decode(
            generated.sequences[0], skip_special_tokens=True
        )

        purely_generated = generated.sequences[:, input_len:]
        purely_generated_str = self.tokenizer.decode(
            purely_generated[0], skip_special_tokens=True
        )
        purely_generated_len = purely_generated.shape[1]

        return {
            "generated_str": generated_str,
            "generated_ntokens": generated_len,
            "purely_generated": purely_generated,
            "purely_generated_str": purely_generated_str,
            "purely_generated_ntokens": purely_generated_len,
        }
