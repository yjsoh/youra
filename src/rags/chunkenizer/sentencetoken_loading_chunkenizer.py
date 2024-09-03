import os
import logging
import json

from ...config import Config
from ...abstract.abs_rag import AbsChunkDAG, AbsChunkenizer
from ...dag import ChunkDAG
from ...utils.chunkenize_helper import root_handling
from .sentencetokens_chunkenizer import (
    SentenceTokenSequenceChunk,
    SentenceTokenSequenceIndex,
)


MODELPATH2MODELNAME = {
    "meta-llama/Llama-2-7b-chat-hf": "llama2",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistralai",
}


class SentenceTokenSequenceLoadingChunkenizer(AbsChunkenizer):
    """
    Dataset specific chunkenizer

    e.g., dataset: "THUDM/LongBench", '2wikimqa', "paraphrase-MiniLM-L6-v2"

    Unlike SentenceTokenSequenceChunkenizer, which computes from scratch,
    this chunkenizer loads from the pre-computed file.

    e.g., Loads from data/easy.llama2.json
    """

    def __init__(
        self,
        config: Config,
        tokenizer,
        llm,
        how_root,
        context_window_size,
        root_split_offset,
        how_quant,
        quant,
        data_path="data",
        force_construct_dag=True,
        add_prepend_prompt=False,
        update_dag=False,
        log_level: int = logging.WARNING,
    ):
        """
        Why get ds as parameter? Prior filtering maybe applied before passing on the data.
        """
        super().__init__(config, tokenizer, llm)

        self.log = logging.getLogger(__name__)
        self.log_level = log_level
        fh = logging.FileHandler(f"{__name__}.log", mode="w")
        fh.setLevel(logging.DEBUG)
        self.log.addHandler(fh)
        self.log.setLevel(log_level)

        self.force_construct_dag = force_construct_dag
        self.add_prepend_prompt = add_prepend_prompt
        self.update_dag = update_dag

        with open("config/dataset2prependprompt.json", "r", encoding="utf-8") as f:
            self.dataset2prependprompt = json.load(f)

        self.encode_options = {"return_tensors": "pt", "add_special_tokens": False}

        self.how_root = how_root
        self.context_window_size = context_window_size
        if self.context_window_size == -1:
            self.context_window_size = llm.config.max_position_embeddings
        self.root_split_offset = root_split_offset
        self.how_quant = how_quant
        self.quant = quant

        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path ({data_path}) not found.")

        model_name = MODELPATH2MODELNAME[config.config_dict["model_name"]]
        with open(f"{data_path}/easy.{model_name}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        self.easy_data = {_data["ds_name"]: {} for _data in data}
        for _data in data:
            self.easy_data[_data["ds_name"]][_data["q_index"]] = _data

    def load_or_create_dag(self, **kwargs) -> AbsChunkDAG:
        """
        Load or create chunks
        """
        ds_path = kwargs.pop("ds_path", None)
        ds_name = kwargs.get("ds_name", None)
        q_index = kwargs.get("q_index", None)
        context = kwargs.get("context", None)

        model_name = kwargs.get("model_name", None)

        if not self.force_construct_dag:
            pass
        else:
            # don't even bother to check if the DAG exists
            dag = None

        if dag is None:
            if self.add_prepend_prompt:
                prepend_prompt = self.dataset2prependprompt[ds_name]
                context = prepend_prompt + context
            else:
                prepend_prompt = ""

            dag = self.create_dag(
                ds_path,
                ds_name,
                q_index,
                context,
                prepend_prompt,
                self.how_root,
                self.context_window_size,
                self.root_split_offset,
                self.how_quant,
                self.quant,
                -1,
            )

        if self.update_dag:
            self.save_dag(dag, ds_path, ds_name, q_index, model_name)

        return dag

    def load_dag(
        self, data_path: str, ds_path: str, ds_name: str, q_index: int, model_name: str
    ) -> AbsChunkDAG:
        """
        Load the DAG from the pre-computed file

        Return None if not found
        """

    def save_dag(self, dag: AbsChunkDAG, ds_path, ds_name, q_index, model_name):
        """
        Save the DAG to the database
        """

    def create_dag(
        self,
        ds_path,
        ds_name,
        q_index,
        context,
        prepend_prompt,
        how_root,
        context_window_size,
        root_split_offset,
        how_quant,
        quant,
        passage_index,
    ) -> AbsChunkDAG:
        """
        Create a DAG
        """

        ### Preprocess context
        context = prepend_prompt + context

        self.log.debug("Added prepend prompt to context.")

        c2s = self.easy_data[ds_name][q_index]
        quant_indices = c2s["sentence_quant_indices"]
        sentences = c2s["sentences"]
        target_sentences = c2s["target_sentences"]
        assert len(quant_indices) == len(sentences) + 1
        self.log.debug("Quant indices (%d): %s", len(quant_indices), quant_indices)

        ### Construct DAG object
        dag = ChunkDAG(self.config, ds_path, ds_name, q_index, context)
        root_lrindices = root_handling(
            quant_indices[-1],
            quant_indices,
            how_root,
            context_window_size,
            root_split_offset,
            how_quant,
            quant,
            log=self.log,
        )
        self.log.debug("Root lrindices: %s", root_lrindices)

        roots = []
        slice_from = 0
        for root_lrind in root_lrindices:
            slice_to = quant_indices.index(root_lrind[1])
            self.log.debug(
                "Root lrind: %s sentences[%d:%d]", root_lrind, slice_from, slice_to
            )
            root = SentenceTokenSequenceChunk(
                None,
                SentenceTokenSequenceIndex(root_lrind, -1, passage_index, -1),
                sentences[slice_from:slice_to],
                target_sentences[slice_from:slice_to],
            )
            slice_from = slice_to
            roots.append(root)
            dag.add_chunk(root, None)
        self.log.debug("Roots (n=%d): %s", len(roots), roots)

        root_index = 0
        for pair_index, lrind in enumerate(zip(quant_indices[:-1], quant_indices[1:])):
            self.log.debug("Current lrind: index=%d, %s", pair_index, lrind)
            if lrind[0] >= roots[root_index].index.lr_index[1]:
                self.log.debug("|-Moving to the next root node")
                root_index += 1

            if lrind[0] > lrind[1]:
                self.log.warning("lrind[1] > lrind[0]: %s", lrind)

            sent_index = SentenceTokenSequenceIndex(lrind, 0, passage_index, pair_index)

            chunk = SentenceTokenSequenceChunk(
                None,
                sent_index,
                [sentences[pair_index]],
                [target_sentences[pair_index]],
            )

            dag.add_chunk(chunk, roots[root_index])

        return dag
