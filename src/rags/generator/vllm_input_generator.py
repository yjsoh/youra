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


class VLLMInputGenerator(AbsGenerator):
    """
    Does not actually generate anything, just returns the input to the VLLM.
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

        prompt = self.add_instuct_tokens(augmented_str)

        generated_len = -1
        generated_str = prompt
        purely_generated = torch.ones([0, 0])
        purely_generated_str = ""
        purely_generated_len = 0

        return {
            "generated_str": generated_str,
            "generated_ntokens": generated_len,
            "purely_generated": purely_generated,
            "purely_generated_str": purely_generated_str,
            "purely_generated_ntokens": purely_generated_len,
        }
