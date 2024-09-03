"""Module for HeteroLLM abstraction."""

import os
import time
import json
import logging
from typing import Union

import stanza
import torch
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForCausalLM


from ..config import Config


class AbsConfigModule:
    """
    Abstract class for all configuration objects
    """

    def __init__(self, config: Config):
        self.config = config

    def to_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return self.config.to_dict()

    def from_dict(self, obj_dict: dict):
        """
        Load from dictionary
        """
        self.config = Config.from_dict(obj_dict)


class AbsLLMModule(AbsConfigModule):
    """
    Abstract class for all HeteroLLM objects
    """

    def __init__(
        self, config: Config, tokenizer: AutoTokenizer, llm: AutoModelForCausalLM
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.llm = llm


class SerializableInterface:
    """
    Interface for serializable objects
    """

    def to_dict(self) -> dict:
        """
        Convert to dictionary
        """
        raise NotImplementedError("Method not implemented")

    @classmethod
    def from_dict(cls, obj_dict: dict):
        """
        Load from dictionary
        """
        raise NotImplementedError("Method not implemented")


class AbsChunkIndex(SerializableInterface):
    """
    Left-right index for chunks. Left is inclusive, right is exclusive
    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, left: int, right: int) -> None:
        self.lr_index = (left, right)
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"[AbsChunkIndex]({self.left}, {self.right})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbsChunkIndex):
            return False
        return self.left == other.left and self.right == other.right

    def __ne__(self, o):
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __lt__(self, other) -> bool:
        if self.left == other.left:
            return self.right < other.right
        return self.left < other.left

    def __le__(self, other) -> bool:
        if self.left == other.left:
            return self.right <= other.right
        return self.left <= other.left

    def __gt__(self, other) -> bool:
        if self.left == other.left:
            return self.right > other.right
        return self.left > other.left

    def __ge__(self, other) -> bool:
        if self.left == other.left:
            return self.right >= other.right
        return self.left >= other.left

    def to_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {
            "type": str(self.__class__.__name__),
            "lr_index": [self.left, self.right],
        }

    @classmethod
    def from_dict(cls, obj_dict: dict):
        """
        Load from dictionary
        """
        type_ = obj_dict["type"]
        if not type_ in cls._registry:
            raise ValueError("Unknown type")

        subclass = cls._registry[type_]
        if not subclass:
            raise ValueError("Unknown type")
        return subclass.from_dict(obj_dict)


class AbsChunks(SerializableInterface):
    """
    Abstract chunks of a long context
    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(self, index: AbsChunkIndex) -> None:
        #######################################################
        ## Part of serialization. #############################
        self.index: AbsChunkIndex = index  # works as key
        self.parent_keys: list[AbsChunkIndex] = []
        self.children_keys: list[AbsChunkIndex] = []
        #######################################################
        ## Not part of serialization. Debugging purposes ######
        self.passage: str = ""
        #######################################################
        ## Not part of serialization. Caching purposes ########
        self.tokens: torch.Tensor = None  # cached tokens

        # for DAG traversal, not part of the serialization
        self.parents: list[AbsChunks] = []
        self.children: list[AbsChunks] = []
        #######################################################

    def __eq__(self, other) -> bool:
        return (
            self.index == other.index
            and self.parent_keys == other.parent_keys
            and self.children_keys == other.children_keys
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return self.index.__hash__()

    def __str__(self) -> str:
        passage_hint = self.passage[:10].replace("\n", " ")
        return f"[{self.__class__.__name__} {self.index} {passage_hint}]"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {
            "type": str(self.__class__.__name__),
            "index": self.index.to_dict(),
            "parent_keys": self.parent_keys,
            "children_keys": self.children_keys,
        }

    @classmethod
    def from_dict(cls, obj_dict: dict):
        """
        Load from dictionary
        """
        type_ = obj_dict["type"]
        if not type_ in cls._registry:
            print(cls._registry)
            raise ValueError("Unknown type")

        subclass = cls._registry[type_]
        if not subclass:
            raise ValueError("Unknown type")
        obj = subclass.from_dict(obj_dict)

        obj.index = AbsChunkIndex.from_dict(obj_dict["index"])
        obj.parent_keys = obj_dict["parent_keys"]
        obj.children_keys = obj_dict["children_keys"]

        return obj


class AbsChunkDAG(SerializableInterface):
    """
    Abstract class for chunk DAG
    """

    ROOT_LEVEL = -1
    _registry = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __init__(
        self,
        config: Config,
        ds_path: str,
        ds_name: str,
        q_index: int,
        context: str,
        chunks: list[AbsChunks],
        h2indices: dict[int, list[AbsChunkIndex]],
        p2c_edge: dict[AbsChunkIndex, list[AbsChunkIndex]],
        c2p_edge: dict[AbsChunkIndex, list[AbsChunkIndex]],
    ) -> None:
        #######################################################
        ## Part of serialization. #############################
        self.config: Config = config
        self.ds_path: str = ds_path
        self.ds_name: str = ds_name
        self.q_index: int = q_index
        self.context: str = context

        # hierarchy index to indices
        self.h2indices: dict[int, list[AbsChunkIndex]] = h2indices

        # nodes
        self.chunks: list[AbsChunks] = chunks

        # parent to children edges
        self.p2c_edge: dict[AbsChunkIndex, AbsChunkIndex] = p2c_edge

        # children to parent edges
        self.c2p_edge: dict[AbsChunkIndex, AbsChunkIndex] = c2p_edge
        #######################################################
        ## Not part of serialization. Caching purposes ########
        self.is_constructed = False

        self.quant_indices: list[int] = []

        # Index to node
        self.i2node: dict[AbsChunkIndex, AbsChunks] = {}

        # Hierarchy to list of chunks
        self.h2nodes: dict[int, list[AbsChunks]] = {}

        # parent to children chunks
        self.p2c_node: dict[AbsChunks, AbsChunks] = {}

        # children to parent chunks
        self.c2p_node: dict[AbsChunks, AbsChunks] = {}

        #######################################################
        # initialize based on the config
        # circular import if using utils.init_nlp
        if self.config.config_dict["nlp"] == "stanza":
            self.nlp = stanza.Pipeline(
                "en",
                processors="tokenize,mwt,ner",
                download_method=None,
                logging_level="WARN",
            )
        else:
            raise ValueError("Unknown NLP")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.config_dict["model_name"]
        )

    def __eq__(self, other) -> bool:
        return (
            self.config == other.config
            and self.ds_path == other.ds_path
            and self.ds_name == other.ds_name
            and self.q_index == other.q_index
            and self.context == other.context
            and self.h2indices == other.h2indices
            and self.p2c_edge == other.p2c_edge
            and self.c2p_edge == other.c2p_edge
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (
            f"[{self.__class__.__name__}"
            + f" {self.ds_path}/{self.ds_name}{self.q_index:03}"
            + f" height={len(self.h2indices)}"
            + f" edges(p2c, c2p)=({len(self.p2c_edge)}, {len(self.c2p_edge)})"
            + f" quant_indicies={len(self.quant_indices)}"
            + f" chunks={len(self.chunks)}"
            + f" h2nodes={len(self.h2nodes)}"
            + f" p2c_node={len(self.p2c_node)}"
            + f" c2p_node={len(self.c2p_node)}"
            + "]"
        )

    def construct(self, **kwargs) -> None:
        """
        Construct the caching purpose fields
        """
        raise NotImplementedError("Method not implemented")

    def to_dict(self) -> dict:
        """
        Convert the DAG to a dictionary
        """
        return {
            "type": str(self.__class__.__name__),
            "config": self.config.to_dict(),
            "ds_path": self.ds_path,
            "ds_name": self.ds_name,
            "q_index": self.q_index,
            "context": self.context,
            "chunks": [c.to_dict() for c in self.chunks],
            "h2indices": {
                k: [v.to_dict() for v in vs] for k, vs in self.h2indices.items()
            },
            "p2c_edge": {
                str(k): [v.to_dict() for v in vs] for k, vs in self.p2c_edge.items()
            },
            "c2p_edge": {
                str(k): [v.to_dict() for v in vs] for k, vs in self.c2p_edge.items()
            },
            "p2c_edge_str2keyobj": {str(k): k.to_dict() for k in self.p2c_edge},
            "c2p_edge_str2keyobj": {str(k): k.to_dict() for k in self.c2p_edge},
        }

    @classmethod
    def from_dict(cls, obj_dict: dict):
        """
        Load the DAG from a dictionary
        """
        type_ = obj_dict["type"]
        if not type_ in cls._registry:
            raise ValueError("Unknown type")

        subclass = cls._registry[type_]
        if not subclass:
            raise ValueError("Unknown type")
        return subclass.from_dict(obj_dict)


class AbsPKVManager(AbsConfigModule):
    """
    Abstract PKV Manager class
    """

    def __init__(self, config: Config, **kwargs):
        super().__init__(config)

    def get(self, key: str) -> Union[None, str]:
        """
        Get value from the key
        """
        raise NotImplementedError("Method not implemented")

    def put(self, key: str, value: str) -> bool:
        """
        Put value to the key
        """
        raise NotImplementedError("Method not implemented")

    def delete(self, key: str) -> bool:
        """
        Delete the key
        """
        raise NotImplementedError("Method not implemented")


class AbsChunkenizer(AbsLLMModule):
    """
    Abstract chunkenizer class
    """

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
        llm: AutoModelForCausalLM,
        **kwargs,
    ):
        super().__init__(config, tokenizer, llm)

    def load_or_create_dag(self, **kwargs) -> AbsChunkDAG:
        """
        Load or create chunks
        """
        # kwargs: ds_path, ds_name, q_index, additional configuration parameters to single out a DAG
        raise NotImplementedError("Method not implemented")

    def chunkenize(
        self, ds_path: str, ds_name: str, q_index: int, context: str, **kwargs
    ) -> dict[str, Union[str, list]]:
        """
        Chunkenize the context
        """
        dag = self.load_or_create_dag(
            ds_path=ds_path, ds_name=ds_name, q_index=q_index, context=context, **kwargs
        )
        return {
            "config": self.config.to_dict(),
            "ds_path": dag.ds_path,
            "ds_name": dag.ds_name,
            "q_index": dag.q_index,
            "dag": dag,
        }


class AbsRetriever(AbsLLMModule):
    """
    Abstract retriever class
    """

    def __init__(
        self,
        config: Config = None,
        tokenizer: AutoTokenizer = None,
        llm: AutoModelForCausalLM = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, llm)

    def _retrieve(
        self, query: str, dag: AbsChunkDAG, pkvm: AbsPKVManager, **kwargs
    ) -> dict[str, Union[list[AbsChunks], list[list[AbsChunks]], str]]:
        raise NotImplementedError("Should implement _retrieve method")

    def retrieve(
        self, query: str, dag: AbsChunkDAG, **kwargs
    ) -> dict[str, Union[str, list]]:
        """
        Retrieve relevant chunks for the query
        """
        retrieved = self._retrieve(query, dag, **kwargs)

        assert (
            "retrieved_chunklist" in retrieved
        ), "retrieved_chunklist not found in retrieved"
        assert "cleaned_strlist" in retrieved, "cleaned_strlist not found in retrieved"
        assert (
            "retrieval_history" in retrieved
        ), "retrieval_history not found in retrieved"
        assert "ntokens" in retrieved, "ntokens not found in retrieved"
        assert (
            "retrieved_ntokens" in retrieved
        ), "retrieved_ntokens not found in retrieved"
        assert (
            "token_level_compression_rate" in retrieved
        ), "token_level_compression_rate not found in retrieved"

        return retrieved | {
            "query": query,
        }


class AbsAugmenter(AbsLLMModule):
    """
    Abstract augmenter class
    """

    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config, tokenizer, None)
        log_name = kwargs.get("logger", "augmenter_logger")
        log_level = kwargs.get("log_level", logging.WARNING)

        self.config = config
        self.tokenizer = tokenizer

        self.log = logging.getLogger(log_name)
        self.log.setLevel(log_level)

    def _augment(
        self, query: str, retrieved: list[str], **kwargs
    ) -> dict[str, Union[str, list]]:
        raise NotImplementedError("Should implement _augment method")

    def augment(
        self, query: str, retrieved: list[str], **kwargs
    ) -> dict[str, Union[str, list]]:
        """
        Augment the query with the topk retrieved chunks
        """
        augmented = self._augment(query, retrieved, **kwargs)

        assert "augmented_str" in augmented, "augmented_str not found in augmented"

        return {"augmented_str": augmented["augmented_str"]}


class AbsGenerator(AbsLLMModule):
    """
    Abstract generator class
    """

    def __init__(self, config, tokenizer, llm, **kwargs):
        super().__init__(config, tokenizer, llm)
        log_name = kwargs.get("logger", "generator_logger")
        log_level = kwargs.get("log_level", logging.WARNING)

        with open("config/dataset2maxlen.json", "r", encoding="utf-8") as f:
            self.dataset2maxlen = json.load(f)

        self.model_name = self.config.config_dict["model_name"]
        self.log = logging.getLogger(log_name)
        self.log.setLevel(log_level)

    def add_instuct_tokens(self, content: str):
        """
        Add instruction tokens to the content
        """

        messages = [
            {"role": "user", "content": content},
        ]

        prompted_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return prompted_str

    def _generate(self, augmented_str: str, **kwargs) -> dict[str, Union[str, list]]:
        raise NotImplementedError("Should implement _generate method")

    @torch.no_grad()
    def generate(self, augmented_str: str, **kwargs) -> dict[str, Union[str, list]]:
        """
        Generate the output for the augmented query
        """
        generated = self._generate(augmented_str, **kwargs)

        assert "generated_str" in generated, "generated_str not found in generated"
        assert (
            "purely_generated" in generated
        ), "purely_generated not found in generated"
        assert (
            "purely_generated_str" in generated
        ), "purely_generated_str not found in generated"
        assert (
            "generated_ntokens" in generated
        ), "generated_ntokens not found in generated"
        assert (
            "purely_generated_ntokens" in generated
        ), "purely_generated_ntokens not found in generated"

        return {
            "generated_str": generated["generated_str"],
            "generated_ntokens": generated["generated_ntokens"],
            "purely_generated": generated["purely_generated"],
            "purely_generated_str": generated["purely_generated_str"],
            "purely_generated_ntokens": generated["purely_generated_ntokens"],
        }


class AbsGrader(AbsConfigModule):
    """
    Abstract grader class

    Wrapper for Chunkenizer, Retriever, Augmenter, Generator, Grader
    """

    def __init__(self, config: Config, grader_name: str, **kwargs):
        super().__init__(config)

        self.grader_name = grader_name
        self.log = logging.getLogger(__name__)

    def _grade(
        self, query: str, truth: list[str], generated: list[str], **kwargs
    ) -> float:
        raise NotImplementedError("Should implement _grade method")

    def grade(
        self, query: str, truth: list[str], generated: list[str], **kwargs
    ) -> dict[str, Union[str, list]]:
        """
        Grade the generated outputs
        """
        assert self.grader_name != "", "Grader name not set"

        scores = [self._grade(query, truth, [g]) for g in generated]

        max_score = max(scores) if len(scores) > 0 else 0.0

        return {
            "grader": f"{self.grader_name}",
            "scores": scores,
            "max_score": max_score,
        }


class AbsRetrievalGrader(AbsLLMModule):
    """
    Abstract retrieval grader class
    """

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
        llm: AutoModelForCausalLM,
        grader_name: str,
    ):
        super().__init__(config, tokenizer, llm)
        self.grader_name = grader_name
        self.evidence_folder = "evidence"

        self.log = logging.getLogger(__name__)

        # key: ds_name value: dict of (q_index, evidences: list[str])
        self.evidences = {}
        self.collect_evidences()

    def collect_evidences(self):
        """
        Collect evidences for the dataset
        """
        ds = os.listdir(self.evidence_folder)
        for evidence_file in ds:
            ds_name = os.path.basename(evidence_file).rstrip(".json")
            with open(
                f"{self.evidence_folder}/{ds_name}.json", "r", encoding="utf-8"
            ) as f:
                loaded = json.load(f)
                self.evidences[ds_name] = loaded
                # print(
                #     f"Loaded evidences for {ds_name}. {self.evidences[ds_name].keys()}")

    def get_evidences(self, ds_name, q_index) -> Union[None, list[str]]:
        """
        Get evidences for specific dataset and query index
        """
        try:
            q_index = str(q_index)
            iet = "<invalid_evidence>"
            evidence_candidates = self.evidences[ds_name][q_index]
            evidences = [e for e in evidence_candidates if iet not in e]
            return evidences
        except KeyError as e:
            print(f"Evidences for {ds_name} {q_index} not found. KeyError: {e}")
            return None

    def _grade(self, retrieved: list[str], evidences: list[str]) -> dict[str, float]:
        raise NotImplementedError("Should implement _grade method")

    def grade(self, retrieved: list[AbsChunks], evidences: list[str]) -> dict:
        """
        Grade the generated outputs
        """
        retrieved_strlist = [r.passage for r in retrieved]

        return self._grade(retrieved_strlist, evidences)


class AbsEvaluator(AbsLLMModule):
    """
    Abstract evaluator class
    """

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer,
        llm: AutoModelForCausalLM,
        chunkenizer: AbsChunkenizer,
        retriever: AbsRetriever,
        augmenter: AbsAugmenter,
        generator: AbsGenerator,
        grader: Union[AbsGrader, list[AbsGrader]],
        output_path: str = "output",
        log_level: int = logging.INFO,
    ):
        super().__init__(config, tokenizer, llm)
        self.config = config
        self.tokenizer = tokenizer
        self.llm = llm
        self.chunkenizer = chunkenizer
        self.retriever = retriever
        self.augmenter = augmenter
        self.generator = generator
        if isinstance(grader, list):
            self.grader_list = grader
        else:
            self.grader_list = [grader]

        self.model_name = self.config.model_name

        #######################################################
        # Logging
        self.log = logging.getLogger(__name__)
        self.log.addHandler(logging.FileHandler("eval.log", mode="w"))
        self.log.setLevel(log_level)

        _fh = logging.FileHandler(f"{output_path}/eval.log", mode="w")
        _fh.setLevel(logging.INFO)
        self.log.addHandler(_fh)

        _fh_debug = logging.FileHandler(f"{output_path}/eval.d.log", mode="w")
        _fh_debug.setLevel(logging.DEBUG)
        self.log.addHandler(_fh_debug)

        _fh_warning = logging.FileHandler("eval.warning.log", mode="a")
        _fh_warning.setLevel(logging.WARNING)
        self.log.addHandler(_fh_warning)
        #######################################################
        # MongoDB
        self.mongo_client = MongoClient("localhost", 27017)
        self.db = self.mongo_client["rag"]
        self.col = self.db["eval"]

    def get_query_truth(self) -> tuple:
        """
        Get query and truth from the dataset
        """
        raise NotImplementedError("Method not implemented")

    def add_grader(self, grader: AbsGrader):
        """
        Add a grader
        """
        self.grader_list.append(grader)

    def evaluate(
        self, ds_path, ds_name, q_index, context, query, truth, **kwargs
    ) -> Union[dict, None]:
        """
        End-to-end evaluation

        Args:
            ds_path: str
                Path to the dataset
            ds_name: str
                Name of the dataset
            q_index: int
                Query index
            context: str
                Context
            query: str
                Query
            truth: list[str]
                Ground truth

        Keyword Args:
            exp_name: str
                Experiment name
            exp_description: str
                Experiment description
            git_hash: str
                Git hash (git )
            dag: AbsChunkDAG
                DAG to reuse
            reuse_dag: bool
                Reuse DAG
            retrieved: list[AbsChunks]
                Retrieved chunks to reuse
            reuse_retrieved: bool
                Reuse retrieved
            augmented: dict
                Augmented to reuse
            reuse_augmented: bool
                Reuse augmented
            generated: dict
                Generated to reuse
            reuse_generated: bool
                Reuse generated

        Returns:
            dict: Evaluation results,
                None if invalid parameters(e.g., reuse_dag=True but no DAG found)
        """

        self.log.debug("Evaluating %s%03d...", ds_name, q_index)

        ###########################################################################################
        # Get parameters
        # col = kwargs.get("collection", None)
        dry_run = kwargs.pop("dry_run", False)

        exp_name = kwargs.pop("exp_name", "")
        exp_desc = kwargs.pop("exp_description", "")
        git_hash = kwargs.pop("git_hash", "")

        output_path = kwargs.pop("output_path", "output")

        dag = kwargs.get("dag", None)
        reuse_dag = kwargs.get("reuse_dag", False)

        retrieved = kwargs.get("retrieved", None)
        reuse_retrieved = kwargs.get("reuse_retrieved", False)

        augmented = kwargs.get("augmented", None)
        reuse_augmented = kwargs.get("reuse_augmented", False)

        generated = kwargs.get("generated", None)
        reuse_generated = kwargs.get("reuse_generated", False)

        to_ret = {
            "exp_name": exp_name,
            "exp_description": exp_desc,
            "git_hash": git_hash,
            "config": self.config.to_dict(),
            "ds_path": ds_path,
            "ds_name": ds_name,
            "q_index": q_index,
            "query": query,
            "truth": truth,
            "context": context,
            "reuse.dag": reuse_dag,  # bool
            "reuse.retrieved": reuse_retrieved,  # bool
            "reuse.augmented": reuse_augmented,  # bool
            "reuse.generated": reuse_generated,  # bool
            "dag": None,
            "retrieved_chunklist": None,  # list of LRIndex
            "retrieval_history": None,  # list of list of LRIndex
            "augmented_str": None,
            "generated_str": None,
            "purely_generated_str": None,
            "graded": None,
            "time_keeper": {},
            "status": "",
        }

        ###########################################################################################
        # Check if parameters are valid
        is_valid = True
        if reuse_dag and dag is None:
            self.log.warning("Reuse DAG set, but no DAG found")
            is_valid = False
        if reuse_retrieved and retrieved is None:
            self.log.warning("Reuse retrieved set, but no retrieved found")
            is_valid = False
        if reuse_augmented and augmented is None:
            self.log.warning("Reuse augmented set, but no augmented found")
            is_valid = False
        if reuse_generated and generated is None:
            self.log.warning("Reuse generated set, but no generated found")
            is_valid = False

        if not is_valid:
            return None

        if not dry_run:
            self.col.insert_one(to_ret)

            mongo_obj_key = {
                "exp_name": exp_name,
                "exp_description": exp_desc,
                "git_hash": git_hash,
                "config": self.config.to_dict(),
                "ds_path": ds_path,
                "ds_name": ds_name,
                "q_index": q_index,
            }

        ###########################################################################################
        # Chunkenize
        self.log.debug("Chunkenizing...")
        timer = time.time()
        if not reuse_dag:
            chunkenized = self.chunkenizer.chunkenize(
                ds_path,
                ds_name,
                q_index,
                context,
                query=query,
                model_name=self.model_name,
                **kwargs,
            )
            dag = chunkenized["dag"]
        to_ret["time_keeper"]["chunkenizer"] = time.time() - timer

        to_ret["dag"] = dag.to_dict()

        if not dry_run:
            self.col.update_one(
                mongo_obj_key,
                {
                    "$set": {
                        "dag": to_ret["dag"],
                        "time_keeper.chunkenizer": to_ret["time_keeper"]["chunkenizer"],
                    }
                },
            )

        ###########################################################################################
        # Retrieve
        self.log.debug("Retrieving...")
        timer = time.time()
        if not reuse_retrieved:
            retrieved = self.retriever.retrieve(query, dag, **kwargs)
            chunklist = retrieved["retrieved_chunklist"]
            ntokens = retrieved["ntokens"]
            retrieved_ntokens = retrieved["retrieved_ntokens"]
            token_level_compression_rate = retrieved["token_level_compression_rate"]
            retrieval_history = retrieved["retrieval_history"]
            cleaned_strlist = retrieved["cleaned_strlist"]
        to_ret["time_keeper"]["retriever"] = time.time() - timer
        to_ret["status"] = "retrieved"

        to_ret["ntokens"] = ntokens
        to_ret["retrieved_ntokens"] = retrieved_ntokens
        to_ret["token_level_compression_rate"] = token_level_compression_rate
        to_ret["retrieved_chunklist"] = [c.to_dict() for c in chunklist]
        to_ret["retrieval_history"] = [
            [c.to_dict() for c in h] for h in retrieval_history
        ]

        if not dry_run:
            self.col.update_one(
                mongo_obj_key,
                {
                    "$set": {
                        "ntokens": to_ret["ntokens"],
                        "retrieved_ntokens": to_ret["retrieved_ntokens"],
                        "token_level_compression_rate": to_ret[
                            "token_level_compression_rate"
                        ],
                        "retrieved_chunklist": to_ret["retrieved_chunklist"],
                        "retrieval_history": to_ret["retrieval_history"],
                        "time_keeper.retriever": to_ret["time_keeper"]["retriever"],
                    }
                },
            )

        ###########################################################################################
        # Augment
        self.log.debug("Augmenting...")
        timer = time.time()
        if not reuse_augmented:
            augmented = self.augmenter.augment(
                query, cleaned_strlist, ds_name=ds_name, **kwargs
            )
            augmented_str = augmented["augmented_str"]
        to_ret["time_keeper"]["augmenter"] = time.time() - timer
        to_ret["status"] = "augmented"

        to_ret["augmented_str"] = augmented_str

        if not dry_run:
            self.col.update_one(
                mongo_obj_key,
                {
                    "$set": {
                        "augmented_str": to_ret["augmented_str"],
                        "time_keeper.augmenter": to_ret["time_keeper"]["augmenter"],
                    }
                },
            )

        ###########################################################################################
        # Generate
        self.log.debug("Generating...")
        timer = time.time()
        if not reuse_generated:
            generated = self.generator.generate(
                augmented_str, ds_name=ds_name, **kwargs
            )
            generated_str = generated["generated_str"]
            generated_len = generated["generated_ntokens"]
            purely_generated_str = generated["purely_generated_str"]
            purely_generated_len = generated["purely_generated_ntokens"]
        to_ret["time_keeper"]["generator"] = time.time() - timer
        to_ret["status"] = "generated"

        to_ret["generated_str"] = generated_str
        to_ret["generated_ntokens"] = generated_len
        to_ret["purely_generated_str"] = purely_generated_str
        to_ret["purely_generated_ntokens"] = purely_generated_len

        if not dry_run:
            # storing the generated string is largely redundant as it includes augmented string
            self.col.update_one(
                mongo_obj_key,
                {
                    "$set": {
                        "generated_str": to_ret["generated_str"],
                        "generated_ntokens": to_ret["generated_ntokens"],
                        "purely_generated_str": to_ret["purely_generated_str"],
                        "purely_generated_ntokens": to_ret["purely_generated_ntokens"],
                        "time_keeper.generator": to_ret["time_keeper"]["generator"],
                    }
                },
            )

        ###########################################################################################
        # Grade
        # meaningless if nothing is graded, so don't cache and don't check for cached result
        graded_dict = {}
        self.log.debug("Grading...")

        timer = time.time()
        for grader in self.grader_list:
            assert isinstance(truth, list)
            assert isinstance(purely_generated_str, str)
            graded = grader.grade(query, truth, [purely_generated_str], **kwargs)

            grader = graded["grader"]
            scores = graded["scores"]

            graded_dict.update({grader: scores})
        to_ret["time_keeper"]["grader"] = time.time() - timer
        to_ret["status"] = "graded"

        to_ret["graded"] = graded_dict

        if not dry_run:
            self.col.update_one(
                mongo_obj_key,
                {
                    "$set": {
                        "graded": to_ret["graded"],
                        "time_keeper.grader": to_ret["time_keeper"]["grader"],
                    }
                },
            )
        self.log.info(
            "%s,%d,%s,%s,%s,%.4f",
            ds_name,
            q_index,
            query,
            truth,
            purely_generated_str,
            to_ret["graded"]["LongBenchGrader"][0],
        )

        with open(f"{output_path}/eval.jsonl", "a", encoding="utf-8") as f:
            json.dump(to_ret, f)
            f.write("\n")

        return to_ret

    # def oldeval(self, **kwarg):
    #     self.chunkenizer.load_or_create_chunks()

    #     all_retrieved = []
    #     all_augmented = []
    #     all_generated = []
    #     all_graded = []
    #     for query, truth in self.get_query_truth():

    #         retrieve_again = True
    #         if os.path.exists(f"{self.output_path}/all_retrieved.json"):
    #             print("Loading retrieved...")
    #             with open(f"{self.output_path}/all_retrieved.json", "r", encoding="utf-8") as f:
    #                 for line in f:
    #                     _retrieved = json.loads(line)
    #                     if _retrieved["ds_path"] == ds_path and\
    #                        _retrieved["ds_name"] == ds_name and _retrieved["q_index"] == q_index:
    #                         serializable_retrieved = _retrieved["retrieved"]
    #                         retrieved = [self.chunkenizer.lrind2chunk[tuple(
    #                             r)] for r in serializable_retrieved]
    #                         all_retrieved.append(retrieved)
    #                         retrieve_again = False
    #                         break

    #         if retrieve_again:
    #             print("Retrieving...")
    #             timer = time.time()
    #             retrieved = self.retriever.retrieve(query)
    #             self.tk["retriever"] = time.time() - timer
    #             all_retrieved.append(retrieved)
    #             serializable_retrieved = [r.lr_index for r in retrieved]
    #             with open(f"{self.output_path}/all_retrieved.json", "a", encoding="utf-8") as f:
    #                 # add additioanl information
    #                 _retrieved = {
    #                     "ds_path": ds_path,
    #                     "ds_name": ds_name,
    #                     "q_index": q_index,
    #                     "query": query,
    #                     "truth": truth,
    #                     "retrieved": serializable_retrieved,
    #                     "retrieval_history": self.retriever.retrieval_history
    #                 }
    #                 json.dump(_retrieved, f)
    #                 f.write("\n")

    #         print("Augmenting...")
    #         timer = time.time()
    #         augmented = self.augmenter.augment(query, retrieved)
    #         self.tk["augmenter"] = time.time() - timer
    #         all_augmented.append(augmented)
    #         serializable_augmented = {
    #             "augmented_strlist": augmented["augmented_strlist"]
    #         }
    #         with open(f"{self.output_path}/all_augmented.json", "a", encoding="utf-8") as f:
    #             # add additioanl information
    #             _augmented = {
    #                 "ds_path": ds_path,
    #                 "ds_name": ds_name,
    #                 "q_index": q_index,
    #                 "query": query,
    #                 "truth": truth,
    #                 "retrieved": serializable_retrieved,
    #                 "retrieval_history": self.retriever.retrieval_history,
    #                 "augmented": serializable_augmented
    #             }
    #             json.dump(_augmented, f)
    #             f.write("\n")

    #         print("Generating...")
    #         timer = time.time()
    #         generated = self.generator.generate(augmented)
    #         self.tk["generator"] = time.time() - timer
    #         all_generated.append(generated)
    #         serializable_generated = generated["generated_str"]
    #         with open(f"{self.output_path}/all_generated.json", "a", encoding="utf-8") as f:
    #             # add additioanl information
    #             _generated = {
    #                 "ds_path": ds_path,
    #                 "ds_name": ds_name,
    #                 "q_index": q_index,
    #                 "query": query,
    #                 "truth": truth,
    #                 "retrieved": serializable_retrieved,
    #                 "retrieval_history": self.retriever.retrieval_history,
    #                 "augmented": serializable_augmented,
    #                 "generated": serializable_generated
    #             }
    #             json.dump(_generated, f)
    #             f.write("\n")

    #         # for grader in self.grader_list:
    #         print("Grading...")
    #         print(f"Query: {query}")
    #         print(f"Truth: {truth}")
    #         print(f"Generated: {generated['generated_str']}")
    #         timer = time.time()
    #         decodable = [g[0] for g in generated["purely_generated"]]
    #         graded_list = []
    #         serializable_graded = []  # list of dict
    #         for grader in self.grader_list:
    #             graded = grader.grade(query, truth, decodable)
    #             graded_list.append(graded)
    #             serializable_graded.append({
    #                 "grader": graded["grader"],
    #                 "scores": graded["scores"]
    #             })

    #         for grader in self.retrieval_grader_list:
    #             evidences = grader.get_evidences(ds_name, q_index)
    #             graded = grader.grade(retrieved, evidences)
    #             graded_list.append(graded)
    #             serializable_graded.append(graded)

    #         self.tk["grader"] = time.time() - timer
    #         all_graded.append(graded_list)
    #         with open(f"{self.output_path}/all_graded.json", "a", encoding="utf-8") as f:
    #             # add additioanl information
    #             _graded = {
    #                 "config": self.config.to_dict(),
    #                 "ds_path": ds_path,
    #                 "ds_name": ds_name,
    #                 "q_index": q_index,
    #                 "total_tokens": self.chunkenizer.total_tokens,
    #                 "query": query,
    #                 "truth": truth,
    #                 "retrieved": serializable_retrieved,
    #                 "retrieval_history": self.retriever.retrieval_history,
    #                 "augmented": serializable_augmented,
    #                 "generated": serializable_generated,
    #                 "graded": serializable_graded
    #             }
    #             json.dump(_graded, f)
    #             f.write("\n")

    #         with open(f"{self.output_path}/time_keeper.json", "a", encoding="utf-8") as f:
    #             self.tk["ds_path"] = ds_path
    #             self.tk["ds_name"] = ds_name
    #             self.tk["q_index"] = q_index
    #             json.dump(self.tk, f)
    #             f.write("\n")

    #     return {
    #         "config": self.config.to_dict(),
    #         "ds_path": ds_path,
    #         "ds_name": ds_name,
    #         "q_index": q_index,
    #         "retrieved": all_retrieved,
    #         "augmented": all_augmented,
    #         "generated": all_generated,
    #         "graded": all_graded
    #     }
