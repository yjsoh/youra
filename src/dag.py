from .config import Config
from .abstract.abs_rag import AbsChunkIndex, AbsChunks, AbsChunkDAG
from .utils import get_or_create_quant_indices


class DefaultChunkIndex(AbsChunkIndex):

    def __eq__(self, o):
        if not isinstance(o, DefaultChunkIndex):
            return False

        return super().__eq__(o)

    def __ne__(self, o):
        return not self.__eq__(o)

    def __hash__(self):
        return super().__hash__()

    @classmethod
    def from_dict(cls, obj_dict: dict):
        return cls(left=obj_dict["lr_index"][0], right=obj_dict["lr_index"][1])

    def to_dict(self) -> dict:
        return super().to_dict() | {"type": str(self.__class__.__name__)}


class DefaultChunk(AbsChunks):
    def __init__(self, lr_index: DefaultChunkIndex) -> None:
        super().__init__(lr_index)

    def __eq__(self, o):
        if not isinstance(o, DefaultChunk):
            return False

        return super().__eq__(o)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return super().__hash__()

    @classmethod
    def from_dict(cls, obj_dict: dict):
        return cls((obj_dict["lr_index"][0], obj_dict["lr_index"][1]))

    def to_dict(self) -> dict:
        return super().to_dict() | {"type": str(self.__class__.__name__)}


class ChunkDAG(AbsChunkDAG):
    """
    ChunkDAG is a directed acyclic graph (DAG) that represents the relationship between chunks in a document.
    """

    ROOT = DefaultChunk(DefaultChunkIndex(-1, -1))

    def __init__(
        self,
        config: Config,
        ds_path: str,
        ds_name: str,
        q_index: int,
        context: str,
        chunks: list[AbsChunks] = None,
        h2indices: dict[int, list[AbsChunkIndex]] = None,
        p2c_edge: dict[AbsChunkIndex, list[AbsChunkIndex]] = None,
        c2p_edge: dict[AbsChunkIndex, list[AbsChunkIndex]] = None,
        quant_indices: list[int] = None,
    ):
        super().__init__(
            config,
            ds_path,
            ds_name,
            q_index,
            context,
            chunks,
            h2indices,
            p2c_edge,
            c2p_edge,
        )

        self.chunks = chunks if chunks is not None else []
        self.h2indices = h2indices if h2indices is not None else {}
        self.p2c_edge = p2c_edge if p2c_edge is not None else {}
        self.c2p_edge = c2p_edge if c2p_edge is not None else {}
        self.quant_indices = quant_indices if quant_indices is not None else []

        # cache-purpose fields
        self.i2node = {
            ChunkDAG.ROOT.index: ChunkDAG.ROOT,
        }

    def add_chunk(self, chunk: AbsChunks, parent: AbsChunks):
        """
        Add a chunk to the graph
        """
        is_parent_virt_root = False
        if parent is None:
            is_parent_virt_root = True
            parent = ChunkDAG.ROOT
            h = ChunkDAG.ROOT_LEVEL
        else:
            assert parent in self.chunks, f"Parent {parent} is not in the graph"
            h = parent.hierarchy_index + 1
            assert (
                h == chunk.hierarchy_index
            ), f"Chunk hierarchy index {chunk.hierarchy_index} is not equal to parent hierarchy index {h}"

        self.h2indices.setdefault(h, []).append(chunk.index)
        self.p2c_edge.setdefault(parent.index, []).append(chunk.index)
        if not is_parent_virt_root:
            self.c2p_edge.setdefault(chunk.index, []).append(parent.index)
        self.chunks.append(chunk)

        # caching purposes
        self.i2node[chunk.index] = chunk
        self.h2nodes.setdefault(h, []).append(chunk)
        self.p2c_node.setdefault(parent, []).append(chunk)
        if not is_parent_virt_root:
            self.c2p_node.setdefault(chunk, []).append(parent)

    def construct(self, **kwargs) -> None:
        """
        Construct the cache-purpose fields

        Chunk
          parent
          children

        DAG
          quant_indices
          h2nodes
          p2c_node
          c2p_node
        """
        if self.is_constructed:
            return

        tokenizer = kwargs.pop("tokenizer", None)
        nlp = kwargs.pop("nlp", None)
        model_name = kwargs.pop("model_name", None)
        force_reconstruct = kwargs.pop("force_reconstruct", False)
        save_quant_indices = kwargs.pop("save_quant_indices", False)
        log = kwargs.pop("log", None)

        assert not kwargs, f"Unprocessed arguments: {kwargs}"

        # build the graph
        self._build_graph()

        # quant indices
        self.quant_indices = get_or_create_quant_indices(
            tokenizer,
            nlp,
            model_name,
            self.context,
            self.ds_name,
            self.q_index,
            force_reconstruct,
            save_quant_indices,
            log,
        )

    def _build_graph(self):
        """
        Build the graph strucutre

        DAG.h2nodex, DAG.p2c_node, DAG.c2p_node, Chunk.parent, Chunk.children
        """

        self.i2node[ChunkDAG.ROOT.index] = ChunkDAG.ROOT
        for chunk in self.chunks:
            self.i2node[chunk.index] = chunk

        for c in self.p2c_edge[ChunkDAG.ROOT.index]:
            self.p2c_node.setdefault(ChunkDAG.ROOT, []).append(self.i2node[c])
            ChunkDAG.ROOT.children.append(self.i2node[c])

        for chunk in self.chunks:
            self.h2nodes.setdefault(chunk.hierarchy_index, []).append(chunk)

            if chunk.index not in self.p2c_edge:
                pass
            else:
                for c in self.p2c_edge[chunk.index]:
                    self.p2c_node.setdefault(chunk, []).append(self.i2node[c])
                    chunk.children.append(self.i2node[c])

            if chunk.index not in self.c2p_edge:
                pass
            else:
                for p in self.c2p_edge[chunk.index]:
                    self.c2p_node.setdefault(chunk, []).append(self.i2node[p])
                    chunk.parents.append(self.i2node[p])

        self.is_constructed = True

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, obj_dict: dict):
        config = Config.from_dict(obj_dict["config"])
        model_name = config.config_dict["model_name"]
        force_reconstruct = config.config_dict["QUANTIZE"]["force_reconstruct"]
        save_quant_indices = config.config_dict["QUANTIZE"]["save_quant_indices"]
        ds_path = obj_dict["ds_path"]
        ds_name = obj_dict["ds_name"]
        q_index = obj_dict["q_index"]
        context = obj_dict["context"]
        chunks = [AbsChunks.from_dict(c) for c in obj_dict["chunks"]]
        h2indices = {
            int(k): [AbsChunkIndex.from_dict(v) for v in vs]
            for k, vs in obj_dict["h2indices"].items()
        }
        # construct keys
        p2c_edge_str2keyobj: dict[str, AbsChunkIndex] = {
            k: AbsChunkIndex.from_dict(v)
            for k, v in obj_dict["p2c_edge_str2keyobj"].items()
        }
        p2c_edge = {
            p2c_edge_str2keyobj[k]: [AbsChunkIndex.from_dict(v) for v in vs]
            for k, vs in obj_dict["p2c_edge"].items()
        }
        c2p_edge_str2keyobj: dict[str, AbsChunkIndex] = {
            k: AbsChunkIndex.from_dict(v)
            for k, v in obj_dict["c2p_edge_str2keyobj"].items()
        }
        c2p_edge = {
            c2p_edge_str2keyobj[k]: [AbsChunkIndex.from_dict(v) for v in vs]
            for k, vs in obj_dict["c2p_edge"].items()
        }
        dag = cls(
            config,
            ds_path,
            ds_name,
            q_index,
            context,
            chunks,
            h2indices,
            p2c_edge,
            c2p_edge,
        )
        dag.construct(
            tokenizer=dag.tokenizer,
            nlp=dag.nlp,
            model_name=model_name,
            force_reconstruct=force_reconstruct,
            save_quant_indices=save_quant_indices,
        )
        return dag
