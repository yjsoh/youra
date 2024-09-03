import json
import logging
from typing import Union

import torch
import numpy as np
import scipy.stats as stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...config import Config
from ...dag import ChunkDAG
from ...utils.attention_helper import get_reaction_vec
from ...utils.retrieve_helper import merge_retrieved
from ...abstract import AbsRetriever, AbsChunks, AbsChunkDAG, AbsPKVManager

###################################################################################################
# # Setup loggers
# for handler in logging.getLogger().handlers:
#     logging.getLogger().removeHandler(handler)

# logging.getLogger().addHandler(logging.NullHandler())
# logging.basicConfig(level=logging.WARN, stream=None)


# t = logging.getLogger("attention_retriever_time_keeper")
# tk_fh = logging.FileHandler(f"{__name__}.time_keeper.log", mode="a")
# tk_fh.setLevel(logging.INFO)
# t.addHandler(tk_fh)
# t.setLevel(logging.INFO)


# m = logging.getLogger("attention_retriever_memory_keeper")
# fh = logging.FileHandler(f"{__name__}.memory_keeper.log", mode="a")
# fh.setLevel(logging.INFO)
# m.addHandler(fh)
# m.setLevel(logging.INFO)
###################################################################################################


class AttentionFlatRetriever(AbsRetriever):
    """
    AttentionFlatRetriever
    """

    def __init__(
        self,
        config: Config = None,
        tokenizer: AutoTokenizer = None,
        llm: AutoModelForCausalLM = None,
        context_window_size: int = -1,
        attn_agg: str = "mean",
        reaction_agg: str = "gmean",
        append_prompt_file: str = "config/dataset2appendprompt.json",
        prepend_prompt_file: str = "config/dataset2prependprompt.json",
        compression_ratio: float = 1.2,
        retain_ctxt_order: bool = True,
        postprocess_merge_retrieved: bool = True,
        add_prepend_prompt: bool = False,
        retrieval_budget: int = -1,
        concat_str: str = None,  # needed for retrieval budget adjustment
        attention_method: str = "append_prompt",
        log_level: int = logging.WARNING,
    ):
        super().__init__(config, tokenizer, llm)

        self.encode_options = {"return_tensors": "pt", "add_special_tokens": False}

        self.log = logging.getLogger(__name__)
        _fh = logging.FileHandler(f"{__name__}.log", mode="w")
        _fh.setLevel(logging.DEBUG)
        self.log.addHandler(_fh)
        self.log.setLevel(log_level)

        self.context_window_size = context_window_size
        if self.context_window_size == -1:
            self.context_window_size = self.llm.config.max_position_embeddings
        self.attn_agg = attn_agg
        self.reaction_agg = reaction_agg
        self.append_prompt_file = append_prompt_file
        self.prepend_prompt_file = prepend_prompt_file
        self.compression_ratio = compression_ratio
        self.retain_ctxt_order = retain_ctxt_order
        self.postprocess_merge_retrieved = postprocess_merge_retrieved
        self.add_prepend_prompt = add_prepend_prompt
        self.retrieval_budget = retrieval_budget
        if self.retrieval_budget == -1:
            self.retrieval_budget = self.context_window_size
        self.concat_str = concat_str
        if self.concat_str is None:
            concat_len = 0
        else:
            concat_len = self.tokenizer(self.concat_str, **self.encode_options)[
                "input_ids"
            ].shape[-1]
        self.concat_len = concat_len
        self.attention_method = attention_method

        with open(self.append_prompt_file, "r", encoding="utf-8") as f:
            self.append_prompt_dict = json.load(f)

        with open(self.prepend_prompt_file, "r", encoding="utf-8") as f:
            self.prepend_prompt_dict = json.load(f)

        self.prepend_prompt_len = {}
        for k, v in self.append_prompt_dict.items():
            self.prepend_prompt_len[k] = self.tokenizer(v, **self.encode_options)[
                "input_ids"
            ].shape[1]

    def _reaction_agg(self, reaction_vec: torch.Tensor) -> float:
        if self.reaction_agg == "gmean":
            return stats.gmean(reaction_vec)
        elif self.reaction_agg == "mean":
            return np.mean(reaction_vec)
        elif self.reaction_agg == "max":
            return np.max(reaction_vec)
        elif self.reaction_agg == "min":
            return np.min(reaction_vec)
        else:
            raise ValueError(f"Invalid reaction_agg: {self.reaction_agg}")

    def _retrieve(
        self, query: str, dag: AbsChunkDAG, pkvm: AbsPKVManager, **kwargs
    ) -> dict[str, Union[list[AbsChunks], list[list[AbsChunks]], str]]:

        chunks = dag.h2nodes[0]

        assert (
            len(chunks) > 0
        ), "DAG.chunks is empty. Either construct or load the DAG first."
        assert len(dag.h2nodes.keys()) == 2, "DAG is not flat."

        ds_name = dag.ds_name
        q_index = dag.q_index

        # get ctxt_ids
        ctxt_ids = self.tokenizer(dag.context, **self.encode_options)["input_ids"]
        self.log.debug("|- ctxt_ids: %s", ctxt_ids.shape)
        ctxt_len = ctxt_ids.shape[1]
        ctxt_prepend_prompt_len = (
            self.prepend_prompt_len[ds_name]
            if self.add_prepend_prompt
            else 1 if self.encode_options["add_special_tokens"] else 0
        )

        # get reaction vector
        reaction_vec_list = []
        reaction_vec_len_list = []
        self.log.info("Compute Reaction Vector")
        for root_chunk in dag.h2nodes[ChunkDAG.ROOT_LEVEL]:
            self.log.debug("  Computing reaction vector for root_chunk: %s", root_chunk)

            baseline_attn_vec, response_attn_vec, _reaction_vec = get_reaction_vec(
                self.tokenizer,
                self.llm,
                pkvm,
                ctxt_ids,
                query,
                root_chunk,
                attn_agg=self.attn_agg,
                ds_name=ds_name,
                q_index=q_index,
                log=self.log,
                append_prompt_str=self.append_prompt_dict[ds_name],
                attention_method=self.attention_method,
                **kwargs,
            )
            baseline_attn_vec = baseline_attn_vec.cpu()
            response_attn_vec = response_attn_vec.cpu()
            _reaction_vec = _reaction_vec.cpu()

            # sanity check
            assert (
                baseline_attn_vec.shape == response_attn_vec.shape
            ), "Baseline and response attn_vec shape mismatch."
            assert (baseline_attn_vec != 0.0).any()
            assert (response_attn_vec != 0.0).any()

            # log reaction vector
            # self.log.debug("  |- baseline attention_vec: %s", baseline_attn_vec)
            self.log.debug(
                "  |- baseline attention_vec stats: %s %f/%f",
                baseline_attn_vec.shape,
                np.mean(baseline_attn_vec.numpy()),
                np.std(baseline_attn_vec.numpy()),
            )
            # self.log.debug("  |- response attention vec: %s", response_attn_vec)
            self.log.debug(
                "  |- response attention vec stats: %s %f,%f",
                response_attn_vec.shape,
                np.mean(response_attn_vec.numpy()),
                np.std(response_attn_vec.numpy()),
            )
            # self.log.debug("  |- partial reaction_vec: %s", _reaction_vec)
            self.log.debug(
                "  |- partial reaction_vec stats: %s %f/%f",
                _reaction_vec.shape,
                np.mean(_reaction_vec.numpy()),
                np.std(_reaction_vec.numpy()),
            )

            # clean-up partial reaction vector
            # prepend_prompt_len = (
            #     self.prepend_prompt_len[ds_name]
            #     if self.add_prepend_prompt
            #     else 1 if self.encode_options["add_special_tokens"] else 0
            # )
            # _reaction_vec = _reaction_vec[prepend_prompt_len:]
            # self.log.debug(
            #     "  |- prepend-prompt cleaned partial reaction_vec len: %s",
            #     _reaction_vec.shape,
            # )

            reaction_vec_list.append(_reaction_vec)
            reaction_vec_len_list.append(_reaction_vec.shape[0])

        # concatenate reaction vectors
        reaction_vec = torch.cat(reaction_vec_list, dim=0).cpu().numpy()
        reaction_vec_len = sum(reaction_vec_len_list)
        # self.log.info("|- rection_vec: %s", reaction_vec)
        self.log.info("|- reaction_vec_len: %s", reaction_vec_len)
        self.log.info("|- reaction_vec shape: %s", reaction_vec.shape)
        assert reaction_vec.shape[0] == reaction_vec_len, "reaction_vec shape mismatch."

        # clean-up reaction vector
        reaction_vec = reaction_vec[ctxt_prepend_prompt_len:]
        assert (
            reaction_vec_len + ctxt_prepend_prompt_len == ctxt_len
        ), f"reaction: {reaction_vec_len} + prepend_len: {ctxt_prepend_prompt_len} vs. ctxt:{ctxt_len} mismatch."

        ### We now have a full reaction vector
        # We need to sort each chunk by reaction score in descending order
        # Then create a cumulative sum of the number of tokens in each chunk
        # Then we need to find the index where the context window size is reached

        # get ntokens per chunk
        for chunk in chunks:
            lrind = chunk.lr_index
            self.log.debug("Calculating the reaction_score for %s", lrind)

            # ntokens
            if hasattr(chunk, "ntokens"):
                continue
            else:
                setattr(chunk, "ntokens", -1)
            chunk.ntokens = lrind[1] - lrind[0]

            if lrind[1] < lrind[0]:
                self.log.warning(
                    "Warning: lrind[1] < lrind[0] for chunk: %s", chunk.index
                )
                setattr(chunk, "reaction_score", 0.0)
                continue
            elif lrind[1] == lrind[0]:
                setattr(chunk, "reaction_score", 0.0)
                continue

            # reaction score
            reaction_vec_slice = reaction_vec[lrind[0] : lrind[1]]
            reaction_score = self._reaction_agg(reaction_vec_slice)
            self.log.debug(
                "|- Creating slice... %s (shape=%s)", lrind, len(reaction_vec_slice)
            )

            if np.sum(reaction_vec_slice) == 0.0:
                self.log.warning(
                    "Warning: Zero reaction score for chunk: %s", chunk.index
                )
            elif np.isnan(reaction_score):
                self.log.warning(
                    "Warning: NaN reaction score for chunk, overwrite with 0.0: %s",
                    chunk.index,
                )
                reaction_score = 0.0

            setattr(chunk, "reaction_score", reaction_score)

        # get overall chunk2rs stats
        rs_list = [chunk.reaction_score for chunk in chunks]
        if np.sum(rs_list) == 0.0:
            _mean_rs = 0.0
            _std_rs = 0.0
        else:
            _mean_rs = np.mean(rs_list)
            _std_rs = np.std(rs_list)
        self.log.debug("loc: %.6f, scale: %.6f", _mean_rs, _std_rs)

        # get ranking based on reaction score
        sorted_chunks = sorted(chunks, key=lambda x: x.reaction_score, reverse=True)

        # log & sanity checks
        for chunk in chunks:
            chunk_len = chunk.ntokens
            reaction_score = chunk.reaction_score
            _rank = sorted_chunks.index(chunk)
            _cdf = stats.norm.cdf(reaction_score, loc=_mean_rs, scale=_std_rs)
            setattr(chunk, "rank", _rank)
            setattr(chunk, "cdf", _cdf)
            self.log.debug(
                "[chunk %15s rs:%.9f (norm=%.4f/rank=%d(%.2f)) chunk_len: %d] %s",
                chunk.index,
                reaction_score,
                _cdf,
                _rank,
                _rank / len(sorted_chunks),
                chunk_len,
                chunk.passage.replace("\n", "<newline>"),
            )

        # initially, take the compression rate # of sentences
        # where,
        #     compression ratio = n_sentences / n_sentences_to_pick
        n_sentences = len(chunks)
        init_n_sentences_to_pick = int(n_sentences / self.compression_ratio)
        self.log.debug("Initial n_sentences_to_pick: %d", init_n_sentences_to_pick)

        # create prefix sum
        prefix_sum = np.cumsum([chunk.ntokens for chunk in sorted_chunks])

        # find the index where the context window size is reached
        adj_retrieval_budget = (
            self.retrieval_budget - self.concat_len * init_n_sentences_to_pick
        )
        n_sentences_to_pick = min(
            init_n_sentences_to_pick,
            np.searchsorted(prefix_sum, adj_retrieval_budget),
        )
        self.log.debug(
            "After capping with adjusted_retrieval_budget (%d) n_sentences_to_pick: %d -> %d",
            adj_retrieval_budget,
            init_n_sentences_to_pick,
            n_sentences_to_pick,
        )

        retrieved_chunklist = sorted_chunks[:n_sentences_to_pick]
        retrieved_chunklist_len = sum([chunk.ntokens for chunk in retrieved_chunklist])

        for chunk in retrieved_chunklist:
            self.log.debug(
                "Retrieved! [chunk %15s rs:%.9f chunk_len: %d rank(0-based): %d cdf:%.4f] %s",
                chunk.index,
                chunk.reaction_score,
                chunk.ntokens,
                chunk.rank,
                chunk.cdf,
                chunk.passage.replace("\n", "<newline>"),
            )
        self.log.debug("retrieved_chunklist_len: %s", retrieved_chunklist_len)

        # sort the retrieved chunks by their position in the document
        if self.retain_ctxt_order:
            retrieved_chunklist = sorted(
                retrieved_chunklist, key=lambda x: x.index.lr_index
            )

            for chunk in retrieved_chunklist:
                self.log.debug(
                    "Retained! [chunk %15s rs:%.9f chunk_len: %d] %s ",
                    chunk.index,
                    chunk.reaction_score,
                    chunk.ntokens,
                    chunk.passage.replace("\n", "\\n"),
                )

        if self.postprocess_merge_retrieved:
            cleaned_strlist = merge_retrieved(retrieved_chunklist, self.log)
        else:
            cleaned_strlist = [chunk.passage for chunk in retrieved_chunklist]

        self.log.debug("Summary")
        self.log.debug(
            "|- Token-level    Compression Rate (%d/%d) = %.2f%%",
            reaction_vec_len,
            retrieved_chunklist_len,
            (
                100 * reaction_vec_len / retrieved_chunklist_len
                if retrieved_chunklist_len > 0
                else float("inf")
            ),
        )
        self.log.debug(
            "|- Sentence-level Compression Rate (%d/%d) = %.2f%%",
            len(chunks),
            len(retrieved_chunklist),
            (
                100 * len(chunks) / len(retrieved_chunklist)
                if len(retrieved_chunklist) > 0
                else float("inf")
            ),
        )

        return {
            "ntokens": reaction_vec_len,
            "retrieved_ntokens": retrieved_chunklist_len,
            "token_level_compression_rate": (
                reaction_vec_len / retrieved_chunklist_len
                if retrieved_chunklist_len > 0
                else float("inf")
            ),
            "retrieved_chunklist": retrieved_chunklist,
            "retrieval_history": [retrieved_chunklist],
            "cleaned_strlist": cleaned_strlist,
        }
