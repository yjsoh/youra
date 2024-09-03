import os
import logging
import json

import pymongo
import stanza
import torch

from ...config import Config
from ...abstract.abs_rag import AbsChunkIndex, AbsChunks, AbsChunkDAG, AbsChunkenizer
from ...dag import ChunkDAG
from ...utils.quantize_helper import context2sentences
from ...utils.chunkenize_helper import root_handling


class SentenceTokenSequenceIndex(AbsChunkIndex):
    """
    Index for representing a node divided by sentence
    """

    def __init__(
        self,
        lr_index: tuple[int, int],
        hierarchy_index: int,
        passage_index: int,
        sent_index: int,
    ):
        super().__init__(lr_index[0], lr_index[1])
        self.hierarchy_index = hierarchy_index
        self.passage_index = passage_index
        self.sent_index = sent_index

    def __str__(self) -> str:
        return f"[SentenceTokenSequenceIndex]({self.lr_index}, {self.hierarchy_index}, {self.passage_index}, {self.sent_index})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o):
        if not isinstance(o, SentenceTokenSequenceIndex):
            return False

        if not isinstance(o, AbsChunkIndex):
            return False

        return (
            super().__eq__(o)
            and self.hierarchy_index == o.hierarchy_index
            and self.passage_index == o.passage_index
            and self.sent_index == o.sent_index
        )

    def __ne__(self, o):
        return not self.__eq__(o)

    def __hash__(self):
        return super().__hash__()

    def to_dict(self) -> dict:
        return super().to_dict() | {
            "type": str(self.__class__.__name__),
            "hierarchy_index": self.hierarchy_index,
            "passage_index": self.passage_index,
            "sent_index": self.sent_index,
        }

    @classmethod
    def from_dict(cls, obj_dict: dict):
        return cls(
            (obj_dict["lr_index"][0], obj_dict["lr_index"][1]),
            obj_dict["hierarchy_index"],
            obj_dict["passage_index"],
            obj_dict["sent_index"],
        )


class SentenceTokenSequenceChunk(AbsChunks):
    """
    Chunk for representing a node divided by sentence
    """

    def __init__(
        self,
        model,
        sentence_index: SentenceTokenSequenceIndex,
        sentences: list[str],
        target_sentences: list[str],
    ) -> None:
        super().__init__(sentence_index)

        self.sentences = sentences
        self.target_sentences = target_sentences

        # caching purposes
        self.hierarchy_index = sentence_index.hierarchy_index
        self.passage_index = sentence_index.passage_index
        self.sent_index = sentence_index.sent_index
        self.lr_index = sentence_index.lr_index
        self.passage = " ".join(sentences)

        # self.chunk_embedding = model.encode(self.sentences)

    def __str__(self) -> str:
        return f"[SentenceTokenSequenceChunk]({self.hierarchy_index}, {self.passage_index}, {self.sent_index}, {self.index} |parent_keys|={len(self.parent_keys)} |children_keys|={len(self.children_keys)} |parents|={len(self.parents)} |children|={len(self.children)} |sentences|={len(self.sentences)})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o) -> bool:
        if not isinstance(o, SentenceTokenSequenceChunk):
            return False

        return self.index.__eq__(o.index) and self.sentences == o.sentences

    def __hash__(self) -> int:
        return super().__hash__()

    def to_dict(self) -> dict:
        """
        Convert to dictionary
        """
        # super to_dict: index, parent_key, children_keys
        return super().to_dict() | {
            "type": str(self.__class__.__name__),
            "sentences": self.sentences,
            "target_sentences": self.target_sentences,
        }

    @classmethod
    def from_dict(cls, obj_dict: dict):
        """
        Load from dictionary
        """
        chunk_index = SentenceTokenSequenceIndex.from_dict(obj_dict["index"])
        return cls(
            None, chunk_index, obj_dict["sentences"], obj_dict["target_sentences"]
        )


class SentenceTokenSequenceChunkenizer(AbsChunkenizer):
    """
    Dataset specific chunkenizer

    e.g., dataset: "THUDM/LongBench", '2wikimqa', "paraphrase-MiniLM-L6-v2"

    Precedence for searching mongo uri
    1. 'mongouri' parameter
    2. os.getenv('MONGO_URI')
    3. 'mongodb://localhost:27017'
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
        nlp_model_id="en",
        mongouri="",
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

        self.nlp = stanza.Pipeline(
            nlp_model_id,  # "en"
            processors="tokenize,mwt",
            dir="/mnt/data/yj/.cache/stanza",
            download_method=None,
            logging_level="WARN",
        )
        mongouri = (
            mongouri
            if mongouri != ""
            else os.getenv("MONGO_URI", "mongodb://localhost:27017")
        )
        self.db = pymongo.MongoClient(mongouri)["heterollm"]

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
            dag = self.load_dag(ds_path, ds_name, q_index, model_name)
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
        self, ds_path: str, ds_name: str, q_index: int, model_name: str
    ) -> AbsChunkDAG:
        """
        Load the DAG from the database

        Return None if not found
        """

        docs = self.db[ds_name].find(
            {
                "ds_path": ds_path,
                "ds_name": ds_name,
                "q_index": q_index,
                # quant_indices exists
                # f'{model_name}.{ssmodel}.sentence_quant_indices': {'$exists': 1}
                # f'{model_name}.sentence_quant_indices': {'$exists': 1}
            },
            {
                "context": 1,
                # f'{model_name}.{ssmodel}.sentences': 1,
                # f'{model_name}.{ssmodel}.sentence_quant_indices': 1
                f"{model_name}.dag": 1,
            },
        )

        # sanity check
        doc_list = list(docs)
        if len(doc_list) == 0:
            self.log.info("Cannot find the specified query: %s, %s", ds_name, q_index)
            return None
        elif len(doc_list) > 1:
            raise ValueError(f"Multiple queries found: {ds_name}, {q_index}")

        doc_dict = doc_list[0]
        if "dag" not in doc_dict[f"{model_name}"]:
            return None

        dag_dict = doc_dict[f"{model_name}"]["dag"]
        return ChunkDAG.from_dict(dag_dict)

    def save_dag(self, dag: AbsChunkDAG, ds_path, ds_name, q_index, model_name):
        """
        Save the DAG to the database
        """

        dag_dict = dag.to_dict()

        self.db[ds_name].update_one(
            {
                "ds_path": ds_path,
                "ds_name": ds_name,
                "q_index": q_index,
            },
            {
                "$set": {f"{model_name}.dag": dag_dict},
            },
            upsert=True,
        )

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

        if prepend_prompt != "":
            prepend_prompt_ids = self.tokenizer.encode(
                prepend_prompt, **self.encode_options
            )
            prepend_prompt_len = prepend_prompt_ids.shape[1]
        else:
            prepend_prompt_ids = torch.tensor([], dtype=torch.long)
            prepend_prompt_len = 0

        self.log.setLevel(logging.ERROR)
        c2s = context2sentences(self.tokenizer, self.nlp, context, log=self.log)
        self.log.setLevel(self.log_level)

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
