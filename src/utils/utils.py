import random

import stanza
import numpy as np
import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def init_tokenizer(config, **kwargs):
    torch_dtype = config.torch_dtype
    device_map = config.device_map

    if (
        "meta-llama-3" in config.model_name.lower()
        or "mistral" in config.model_name.lower()
    ):
        _tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    else:
        _tokenizer = LlamaTokenizer.from_pretrained(
            config.model_name,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    return _tokenizer


def init_llm(config, **kwargs):
    torch_dtype = config.torch_dtype
    device_map = config.device_map

    load_options = {
        "local_files_only": True,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }

    if "llama-2" in config.model_name.lower():
        load_options["attn_implementation"] = "eager"
        _llm = LlamaForCausalLM.from_pretrained(
            config.model_name,
            **load_options,
        )
    elif "meta-llama-3" in config.model_name.lower():
        load_options["attn_implementation"] = "eager"
        load_options["pad_token_id"] = 128009
        _llm = LlamaForCausalLM.from_pretrained(
            config.model_name,
            **load_options,
        )
    elif "mistral" in config.model_name.lower():
        # default attn_implementation, otherwise OOM
        _llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_options,
        )
    else:
        _llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **load_options,
        )
    _llm = _llm.eval()  # set to evaluation mode,
    # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    return _llm


def init_nlp(config, **kwargs):
    """Initialize the NLP pipeline"""
    # initialize based on the config
    if config.config_dict["nlp"] == "stanza":
        nlp = stanza.Pipeline(
            "en",
            processors="tokenize,mwt,ner",
            download_method=None,
            logging_level="WARN",
        )
    else:
        raise ValueError("Unknown NLP")

    return nlp
