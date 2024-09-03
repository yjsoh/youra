"""Attention Helper functions for Transformers."""

import os
import time
import logging
from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import scipy.stats as stats
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..abstract import AbsChunks, AbsPKVManager

FREEMEM_ADJUST_FACTOR = 0.3


@torch.no_grad()
def mout2attn(mout, attn_vec_agg):
    """
    Extracts the last layer attention vector from the ModelOutput.

    Parameters
        mout: ModelOutput
        attn_vec_agg: str
            "mean" or "sum"

    Returns
        last_layer_attn: torch.Tensor

    """
    if attn_vec_agg == "mean":
        last_layer_attn = mout.attentions[-1].mean(dim=(0, 1, 2))
    elif attn_vec_agg == "sum":
        last_layer_attn = mout.attentions[-1].sum(dim=(0, 1, 2))
    else:
        raise ValueError(f"attn_vec_agg={attn_vec_agg} is not supported.")
    return last_layer_attn


@torch.no_grad()
def get_query_ids(
    tokenizer: AutoTokenizer,
    query: Union[str, torch.Tensor, None],
    input_ids: torch.Tensor,
    append_prompt_str: str,
    attention_method: str,
    log: logging.Logger,
    **kwargs,
):
    """
    Get the query ids from the query string or tensor.

    Parameters
        tokenizer: AutoTokenizer
        query: Union[str, torch.Tensor, None]
        chunk: AbsChunks
        **kwargs: should be empty

    Returns
        query_ids: torch.Tensor
    """

    # sanity check for kwargs
    if len(kwargs) > 0:
        raise ValueError(f"Unexpected kwargs: {kwargs}")

    encode_options = {"return_tensors": "pt", "add_special_tokens": False}

    log.debug("|-query: %s", query)
    log.debug("|-append_prompt_str: %s", append_prompt_str.replace("\n", "<newline>"))
    log.debug("|-attention_method: %s", attention_method)

    if query is None or query == "":
        if attention_method == "last_token":
            return input_ids[:, -1:]
        elif attention_method == "append_prompt":
            return tokenizer.encode(append_prompt_str, **encode_options)
        # elif attention_method == "random_token":
        #     vocab_size = tokenizer.vocab_size
        #     while True:
        #         random_token = torch.randint(
        #             0, vocab_size, (1, chunk.tokens.shape[1]))

        #         if random_token is not tokenizer.eos_token_id:
        #             break
        #     return tokenizer.encode(random_token, **encode_options)
        # elif attention_method == "preset_tokens":
        #     return tokenizer.encode(PRESET_TOKENS, **encode_options)
        else:
            raise ValueError(f"Invalid ATTENTION_METHOD: {attention_method}")

    elif isinstance(query, str):
        if attention_method == "last_token":
            return tokenizer.encode(query, **encode_options)
        elif attention_method == "append_prompt":
            appended_prompt = append_prompt_str + query + "\n\nAnswer: "
            return tokenizer.encode(appended_prompt, **encode_options)
        else:
            raise ValueError(f"Invalid ATTENTION_METHOD: {attention_method}")
    elif isinstance(query, torch.Tensor):
        return query
    else:
        raise ValueError(f"Invalid query type: {type(query)}")


@torch.no_grad()
def get_attention_vec(
    tokenizer: AutoTokenizer,
    llm: AutoModelForCausalLM,
    pkv_manager: AbsPKVManager,
    ctxt_ids: torch.Tensor,
    query_ids: torch.Tensor,
    chunk: AbsChunks,
    attention_mask: torch.Tensor = None,
    attn_agg: str = "mean",
    ds_name: str = "",
    q_index: int = -2,
    log: logging.Logger = logging.getLogger(__name__),
    **kwargs,
) -> torch.Tensor:
    """
    Get the attention vector from the last layer of the model.

    kwargs:
        append_prompt_str: str
        attention_method: str
    """

    log.debug("get_attention_vec(%s)", chunk.lr_index)
    query_ids = get_query_ids(
        tokenizer,
        query_ids,
        ctxt_ids,
        log=log,
        **kwargs,
    ).to(llm.device)
    log.debug(
        "|- query_ids shape=%s (decoded: %s)",
        query_ids.shape,
        tokenizer.decode(query_ids[0], skip_special_tokens=False).replace(
            "\n", "<newline>"
        ),
    )
    query_len = query_ids.shape[1]

    pkv = pkv_manager.get_or_create_past_key_values(
        tokenizer, ctxt_ids, chunk, ds_name, q_index
    )

    # if pkv_len + query_len > llm.config.max_position_embeddings:
    # raise ValueError(
    #     "pkv_len + query_len > llm.config.max_position_embeddings:" +
    #     f" {pkv_len} + {query_len} > {llm.config.max_position_embeddings}")

    log.debug("|- past_key_values: %s", pkv[-1][-1].shape)
    if attention_mask is not None:
        # _arlog.debug("|- attention_mask: %s", attention_mask.shape)

        # extend attention mask
        attention_mask = torch.cat(
            [attention_mask, torch.ones([1, query_len])], dim=-1
        ).to(llm.device)

        # _arlog.debug("|- extened attention_mask: %s", attention_mask.shape)

    model_input = {
        "input_ids": query_ids,
        "past_key_values": pkv,
    }

    if attention_mask is not None:
        pmout = llm(
            **model_input,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
    else:
        pmout = llm(**model_input, output_attentions=True, return_dict=True)

    log.debug("|- pmout.attentions[-1] shape=%s", pmout.attentions[-1].shape)
    last_layer_attn = mout2attn(pmout, attn_agg)
    log.debug("|- before stripping query_ids: %s", last_layer_attn.shape)
    last_layer_attn = last_layer_attn[:-query_len]
    log.debug("|- after stripping query_ids: %s", last_layer_attn.shape)

    last_layer_attn = last_layer_attn.detach().to(
        torch.device("cpu"), non_blocking=False
    )

    pkv_manager.dump_kvcache_to_dram(chunk.index.lr_index, pkv)

    return last_layer_attn


@torch.no_grad()
def get_reaction_vec(
    tokenizer: AutoTokenizer,
    llm: AutoModelForCausalLM,
    pkvm: AbsPKVManager,
    ctxt_ids: torch.Tensor,
    query: Union[str, torch.Tensor, None],
    chunk: AbsChunks,
    attention_mask: torch.Tensor = None,
    log: logging.Logger = logging.getLogger(__name__),
    **kwargs,
) -> torch.Tensor:
    """
    Get reaction vector

    kwargs: to be passed to get_attention_vec
     + attn_agg: str = "mean",
     + ds_name: str = "",
     + q_index: int = -2,
    """

    log.debug("|- get reaction vector for chunk(%s)", chunk.lr_index)

    before_attn_vec = get_attention_vec(
        tokenizer, llm, pkvm, ctxt_ids, None, chunk, attention_mask, **kwargs
    )
    after_attn_vec = get_attention_vec(
        tokenizer, llm, pkvm, ctxt_ids, query, chunk, attention_mask, **kwargs
    )

    # sanity check
    if isinstance(after_attn_vec, list):
        print(after_attn_vec)
    assert before_attn_vec.shape == after_attn_vec.shape
    assert (before_attn_vec != 0).any()

    relative = abs(after_attn_vec - before_attn_vec).cpu()

    log.debug(
        "|- reaction vec gmean: %f, len: %d", stats.gmean(relative), len(relative)
    )

    return before_attn_vec, after_attn_vec, relative


@torch.no_grad()
def attend_n_by_n_with_past(
    llm, input_ids, n, past_key_values, log=logging.getLogger(__name__)
):
    """
    Attend n-by-n with past key values.
    """
    assert n > 0
    assert isinstance(input_ids, torch.Tensor)
    assert len(input_ids.shape) == 2
    log.debug(
        "attend_n_by_n_with_past(input_ids(shape=[%d,%d]), n=%d)",
        input_ids.shape[0],
        input_ids.shape[1],
        n,
    )
    time_rec = []
    mout = None
    ntokens = 0
    _, max_seq_len_in_batch = input_ids.shape
    _s = time.time()
    for i in range(0, max_seq_len_in_batch, n):
        _ntoken = min(i + n, max_seq_len_in_batch) - i
        # attn_logger.debug("attend_n_by_n_with_past input_ids[%d:%d] pkv[, , %d, ]",
        #                   i,
        #                   i+_ntoken,
        #                   past_key_values[0][0].shape[2] if past_key_values is not None else -1)
        _input_ids = input_ids[:, i : i + _ntoken]
        model_input = {
            "input_ids": _input_ids,
            "past_key_values": past_key_values,
            "output_attentions": (i + n) >= max_seq_len_in_batch,
            "use_cache": True,
        }

        mout = llm(**model_input)
        past_key_values = mout.past_key_values
        torch.cuda.empty_cache()
        ntokens += _ntoken
        _e = time.time()
        time_rec.append(_e - _s)
        _s = _e

    if len(time_rec) > 0:
        log.debug(
            "|- %f(%f,%f,%f) seconds %f tps (ntoken=%d) executed on %s",
            np.sum(time_rec),
            np.min(time_rec),
            np.mean(time_rec),
            np.max(time_rec),
            ntokens / np.sum(time_rec),
            ntokens,
            llm.device,
        )
    else:
        log.debug("|- No time recorded")
    return mout


@torch.no_grad()
def auto_attend_n_by_n_with_past(
    llm,
    input_ids,
    past_key_values,
    free_mem=None,
    torch_dtype=torch.float32,
    log=logging.getLogger(__name__),
):
    """
    Automatically determine the n-by-n size and execute the attention.
    """

    def _estimate_attention_memory(_iter, nbyn):
        return (
            nlayer * batch_size * nhead * nbyn * (init_npkv + _iter * nbyn) * precision
        )

    def _estimate_pkv_memory(_iter, nbyn):
        return (
            nlayer
            * 2
            * batch_size
            * nhead
            * (init_npkv + _iter * nbyn)
            * nembd
            * precision
        )

    if free_mem is None:
        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0]
    # peak reaches x3 of the estimated memory
    adj_free_mem = free_mem * FREEMEM_ADJUST_FACTOR
    log.debug(
        "auto_attend_n_by_n_with_past: free_mem=%f, adj_free_mem=%f",
        free_mem / (2**20),
        adj_free_mem / (2**20),
    )
    free_mem = adj_free_mem

    batch_size, max_seq_len_in_batch = input_ids.shape
    nlayer = llm.config.num_hidden_layers
    nhead = llm.config.num_attention_heads
    nembd = llm.config.hidden_size // nhead
    init_npkv = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    seq_len = max_seq_len_in_batch + init_npkv
    precision = 4 if torch_dtype == torch.float32 else 2

    log.debug(
        "auto_attend_n_by_n_with_past: "
        + "batch_size=%d, "
        + "max_seq_len_in_batch=%d, "
        + "nlayer=%d, nhead=%d, nembd=%d, init_npkv=%d, precision=%d",
        batch_size,
        max_seq_len_in_batch,
        nlayer,
        nhead,
        nembd,
        init_npkv,
        precision,
    )

    # estimate nbyn
    bnbync = free_mem // (nlayer * batch_size * nhead * precision)
    log.debug("|- Assumed free memory: %f", free_mem / (2**20))
    log.debug(
        "|- NLayer * Batch Size * N head * Precision: %d",
        nlayer * batch_size * nhead * precision,
    )
    log.debug("|- Estimated bnbync: %d", bnbync)

    sub_constant = seq_len * nembd
    log.debug("|- Subtract constants: %d", sub_constant)

    divider = seq_len + init_npkv
    log.debug("|- Divided by seq_len + init_npkv: %d", divider)

    nbyn = (bnbync - sub_constant) // divider
    nbyn = int(nbyn)
    if nbyn < 1:
        nbyn = 1
        log.warning(
            "nbyn is less than 1. Likely to OOM and abort early. Setting nbyn=1"
        )
        # raise MemoryError(
        #     "nbyn is less than 1. Likely to OOM and abort early.")

    adj_nbyn = min(nbyn, max_seq_len_in_batch)
    niter = max(1, max_seq_len_in_batch // nbyn)
    log.debug("|- Estimated nbyn: %d, adj_nbyn: %d, niter=%d", nbyn, adj_nbyn, niter)

    # estimate memory usage
    mem_attn = _estimate_attention_memory(niter, adj_nbyn)
    mem_pkv = _estimate_pkv_memory(niter, adj_nbyn)
    mem_est = mem_pkv + mem_attn
    mem_est = mem_est / FREEMEM_ADJUST_FACTOR

    # get before max memory
    torch.cuda.reset_peak_memory_stats()
    mem_max_old = torch.cuda.max_memory_allocated() / 2**20

    # execute n-by-n
    mout = attend_n_by_n_with_past(llm, input_ids, adj_nbyn, past_key_values)

    # get after max memory
    mem_max = torch.cuda.max_memory_allocated() / 2**20
    mem_delta = mem_max - mem_max_old
    mem_loss = mem_delta - (mem_est / 2**20)
    log.debug(
        "auto_attend_n_by_n_with_past: mem_max_old=%f, mem_max=%f, delta=%f, loss=%f",
        mem_max_old,
        mem_max,
        mem_delta,
        mem_loss,
    )

    # get actual memory usage
    mout_attn_mem = (
        nlayer
        * mout["attentions"][-1].element_size()
        * mout["attentions"][-1].nelement()
        / 2**20
    )

    mout_pkv_mem = (
        2
        * nlayer
        * mout["past_key_values"][0][0].element_size()
        * mout["past_key_values"][0][0].nelement()
        / 2**20
    )

    log.debug(
        "auto_attend_n_by_n_with_past: attn_mem=%f, pkv_mem=%f, "
        + "total_mem=%f vs est_mem=%f (attn:%f, pkv:%f)",
        mout_attn_mem,
        mout_pkv_mem,
        mout_attn_mem + mout_pkv_mem,
        mem_est / 2**20,
        mem_attn / 2**20,
        mem_pkv / 2**20,
    )

    return mout


@torch.no_grad()
def attend_n_by_n(
    llm,
    input_ids,
    n,
    log=logging.getLogger(__name__),
):
    """
    Attend n-by-n.
    """

    log.debug(
        "attend_n_by_n(input_ids(shape=[%d,%d]), n=%d)",
        input_ids.shape[0],
        input_ids.shape[1],
        n,
    )
    assert n > 0
    assert isinstance(input_ids, torch.Tensor)
    assert len(input_ids.shape) == 2
    mout = None
    for i in range(0, input_ids.shape[1], n):
        _input_ids = input_ids[:, i : i + n]
        model_input = {
            "input_ids": _input_ids,
            "past_key_values": mout.past_key_values if mout is not None else None,
        }

        mout = llm(**model_input, output_attentions=True, use_cache=True)
        torch.cuda.empty_cache()
    return mout


@torch.no_grad()
def attend_one_by_one(llm, input_tokens):
    """
    Attend one-by-one. Wrapper for attend_n_by_n.
    """

    return attend_n_by_n(llm, input_tokens, 1)
