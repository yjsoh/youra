from typing import Union, Tuple
import os
import logging
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils.attention_helper import auto_attend_n_by_n_with_past
from .config import Config
from .abstract import AbsChunks, AbsPKVManager


class PKVManager(AbsPKVManager):
    ROOT_PKV_FOLDER = None
    USE_FULL_INDEX_FOR_KVCACHE_DICT = None

    def __init__(
        self,
        config: Config,
        tokenizer: AutoTokenizer = None,
        llm: AutoModelForCausalLM = None,
        log: logging.Logger = logging.getLogger(__name__),
        log_level: int = logging.WARNING,
        **kwargs,
    ):
        super().__init__(config)

        self.tokenizer = tokenizer
        self.llm = llm

        self.kvcache_dict = {}
        self.kvcache_trash_map = {}
        self.kvcache_trash_queue = []

        self.log = log
        self.log.addHandler(logging.FileHandler(f"{__name__}.log", mode="w"))
        self.log.setLevel(log_level)
        self.tk = {}

    # Interface methods
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

    def clear(self) -> bool:
        """
        Clear the cache
        """
        self.kvcache_dict.clear()
        self.kvcache_trash_map.clear()
        self.kvcache_trash_queue.clear()
        self.tk.clear()
        torch.cuda.empty_cache()
        time.sleep(1)
        return True

    def _get_root_pkv_path(self, ds_name, q_index, root_lrind: Tuple[int, int]) -> str:
        serializable_model_name = self.config.config_dict["model_name"].replace(
            "/", "."
        )

        how_root = self.config.config_dict["YOURA_CHUNKENIZER"]["ROOT_HANDLING_METHOD"]
        if how_root == "no_split":
            return f"{self.ROOT_PKV_FOLDER}/rootpkv.{ds_name}{q_index:03}.{serializable_model_name}.{self.config.torch_dtype}"
        elif how_root == "split_max_position_embeddings_no_overlap":
            # get root index from pivot_dict[-1]
            if root_lrind is None:
                raise ValueError("root_lrind is None")
            return f"{self.ROOT_PKV_FOLDER}/rootpkv.{ds_name}{q_index:03}.{serializable_model_name}.{self.config.torch_dtype}.{root_lrind[0]}-{root_lrind[1]}"
        else:
            raise ValueError(
                f"Invalid root handling method {how_root} when getting root pkv path"
            )

    def dump_kvcache_to_dram(
        self,
        lr_index: Tuple[int, int],
        kvcache: list[list[torch.Tensor]],
        device=torch.device("cpu"),
    ):
        """
        Dump the kvcache to DRAM
        """

        # caes 0 : use full index for kvcache_dict
        # case 1 : lr_index is already in the kvcache_dict
        # case 2 : lr_index is not in the kvcache_dict
        #   case 2-1: there is a enclosing kvcache
        #   case 2-2: current lr_index is the enclosing kvcache
        #   case 2-3: there is no kvcache

        if self.USE_FULL_INDEX_FOR_KVCACHE_DICT:
            self.kvcache_dict[lr_index] = self._dump_kvcache_to_dram(kvcache, device)
            return

        kvcache = self._dump_kvcache_to_dram(kvcache, device)

        self._set_kvcache(lr_index, kvcache)

    def _get_kvcache(self, lr_index):
        if self.USE_FULL_INDEX_FOR_KVCACHE_DICT:
            return self.kvcache_dict.get(lr_index, None)

        l, r = lr_index
        if l not in self.kvcache_dict:
            return None

        now_r = self.kvcache_dict[l]["r"]
        if now_r == r:
            return self.kvcache_dict[l]["kvcache"]
        else:
            return None

    def save_kvcache_to_file(self, kvcache, path):
        """
        Save the kvcache to file
        """
        # check if already saved
        if os.path.exists(f"{path}/complete.txt"):
            self.log.debug("Already saved to %s", path)
            return

        os.makedirs(path, exist_ok=True)

        assert isinstance(kvcache, list)
        assert isinstance(kvcache[0], list)
        assert isinstance(kvcache[0][0], torch.Tensor)

        # save to file
        for l, kv in enumerate(kvcache):
            torch.save(kv[0], f"{path}/layer{l:02}.k.pt")
            torch.save(kv[1], f"{path}/layer{l:02}.v.pt")
            with open(f"{path}/list.txt", "a", encoding="utf-8") as f:
                f.write(f"layer{l:02}\n")

        # mark as complete
        with open(f"{path}/complete.txt", "a", encoding="utf-8"):
            os.utime(f"{path}/complete.txt", None)

    def load_kvcache_from_file(self, path):
        """
        Load the kvcache from file
        """
        assert os.path.exists(f"{path}/complete.txt")
        assert os.path.exists(f"{path}/list.txt")

        kvcache = []
        with open(f"{path}/list.txt", "r", encoding="utf-8") as f:
            prefixes = f.readlines()
            prefixes = [p.strip() for p in prefixes]

            for prefix in prefixes:
                k = torch.load(f"{path}/{prefix}.k.pt")
                v = torch.load(f"{path}/{prefix}.v.pt")
                kvcache.append([k, v])
        assert len(kvcache) == self.llm.config.num_hidden_layers
        return kvcache

    def _set_kvcache(self, lr_index, kvcache):
        # save to file for future use
        # if lr_index in self.pivot_dict.get(-1, []):
        #     self.log.debug(
        #         "kv_cache[%s] is a root pivot, saving to file", lr_index)
        #     path = self._get_root_pkv_path(lr_index)
        #     self.save_kvcache_to_file(kvcache, path)

        if self.USE_FULL_INDEX_FOR_KVCACHE_DICT:
            self.kvcache_dict[lr_index] = kvcache
            return

        l, r = lr_index
        if l in self.kvcache_dict:
            now_r = self.kvcache_dict[l]["r"]
            if now_r == r:
                self.log.debug("kv_cache[%s] already exists", lr_index)
                return
            elif now_r > r:
                self.log.debug(
                    "Enclosing kv_cache[(%d, %d)] already exists for %s, aborting",
                    l,
                    now_r,
                    lr_index,
                )
                return
            else:
                self.log.debug("Current kvcache[%s] is the enclosing.", lr_index)

        self.log.debug(
            "Either current kvcache[%s] is enclosing or there is no kvcache starting from %d",
            lr_index,
            l,
        )

        now_d = self.kvcache_dict.get(l, {"r": -1, "kvcache": None})
        now_d["r"] = r
        now_d["kvcache"] = kvcache

        self.kvcache_dict[l] = now_d

    def _dump_kvcache_to_dram(self, kvcache, device=torch.device("cpu")):
        toargs = {
            "device": device,
            "non_blocking": True,
            # "dtype": torch.float32,
            # "copy": False,
            # "memory_format": torch.contiguous_format
        }
        # Note: if you unpack using k, v in kvcache, then k, v will not be moved to device
        if isinstance(kvcache[0], list):
            for kv in kvcache:
                # self.log.debug("vcache: %s", kv[1].shape)
                kv[0] = kv[0].to(**toargs).detach()
                kv[1] = kv[1].to(**toargs).detach()
            assert isinstance(kvcache, list)
            assert isinstance(kvcache[0], list)
            assert isinstance(kvcache[0][0], torch.Tensor)
            return kvcache
        else:
            dumped = list()
            for kv in kvcache:
                dumped.append(
                    [kv[0].to(**toargs).detach(), kv[1].to(**toargs).detach()]
                )
            del kvcache
            torch.cuda.empty_cache()
            return dumped

    def _truncated_kvcache_load_from_dram(self, lrind, target_lrind, device):
        """
        lrind: tuple of left and right indices of the kvcache
        target_lrind: tuple of left and right indices of the target segment
        """
        assert lrind[0] == target_lrind[0]
        assert lrind[1] >= target_lrind[1]
        if self.USE_FULL_INDEX_FOR_KVCACHE_DICT:
            assert lrind in self.kvcache_dict
        else:
            assert lrind[0] in self.kvcache_dict
        toargs = {
            "device": device,
            "non_blocking": True,
            # "dtype": torch.float32,
            # "copy": False,
            # "memory_format": torch.contiguous_format
        }
        new_kvcache_len = target_lrind[1] - target_lrind[0]

        list_kvcache = list()
        kvcache = self._get_kvcache(lrind)
        for kv in kvcache:
            kcache = kv[0]
            vcache = kv[1]
            list_kvcache.append(
                [
                    kcache[:, :, :new_kvcache_len, :].to(**toargs),
                    vcache[:, :, :new_kvcache_len, :].to(**toargs),
                ]
            )
        return list_kvcache

    def load_kvcache_from_dram(self, lrind, device):
        """
        Load the kvcache from DRAM
        """
        kvcache = self._get_kvcache(lrind)
        return self._load_kvcache_from_dram(kvcache, device)

    def _load_kvcache_from_dram(self, kvcache, device):
        toargs = {
            "device": device,
            "non_blocking": True,
            # "dtype": torch.float32,
            # "copy": False,
            # "memory_format": torch.contiguous_format
        }
        # assert isinstance(kvcache, list)
        # assert isinstance(kvcache[0], list)
        assert isinstance(kvcache[0][0], torch.Tensor)

        to_ret = list()
        for kv in kvcache:
            kcache = kv[0]
            vcache = kv[1]
            to_ret.append([kcache.to(**toargs), vcache.to(**toargs)])
        return to_ret

    def _find_candidate(self, lrind: tuple) -> Tuple[tuple, bool]:
        ans = (-1, -1)
        if self.USE_FULL_INDEX_FOR_KVCACHE_DICT:
            for k in self.kvcache_dict:
                if k[0] == lrind[0]:
                    if k[1] == lrind[1]:
                        return k, True
                    else:
                        ans = (k[0], max(ans[1], k[1]))
                else:
                    continue
            return ans, False

        l, r = lrind
        if l in self.kvcache_dict:
            now_r = self.kvcache_dict[l]["r"]
            if now_r == r:
                return (l, r), True
            else:
                return (l, now_r), False
        else:
            return ans, False

    def _get_or_create_past_key_values(self, lr_index, llm, root_input_ids):
        input_ids = root_input_ids.to(llm.device)
        self.log.debug(
            "_get_or_create_past_key_values(input_ids(shape=(%s)), %s)",
            input_ids.shape,
            lr_index,
        )
        self.log.debug("Current kvcache_dict's keys:")
        self.log.debug(self.kvcache_dict.keys())

        # check if there is a reusable KVCache
        # use the start index to find a segment with the same starting point
        build_upon_lrind, match = self._find_candidate(lr_index)

        if not match or build_upon_lrind == (-1, -1):
            self.log.debug(
                "_get_or_create_past_key_values(%s): No exact KVCache match. Searching trash.",
                lr_index,
            )
            # search trash
            for kvcache_in_trash in self.kvcache_trash_queue:
                if lr_index == kvcache_in_trash:
                    self.log.debug(
                        "_get_or_create_past_key_values(%s): Found in trash", lr_index
                    )
                    start_time = time.time()
                    kvcache = self.kvcache_trash_map[lr_index]
                    to_ret = self._load_kvcache_from_dram(kvcache, llm.device)
                    self.tk.setdefault("pkv_trash_hit", []).append(
                        time.time() - start_time
                    )
                    return to_ret

        if match:
            self.log.debug(
                "_get_or_create_past_key_values(%s): Found exact KVCache match",
                lr_index,
            )
            start_time = time.time()
            to_ret = self.load_kvcache_from_dram(lr_index, llm.device)
            self.tk.setdefault("pkv_hit", []).append(time.time() - start_time)
            return to_ret

        if build_upon_lrind[1] > lr_index[1]:  # truncate
            self.log.debug(
                "_get_or_create_past_key_values(%s): Truncate KVCache found", lr_index
            )
            start_time = time.time()
            to_ret = self._truncated_kvcache_load_from_dram(
                build_upon_lrind, lr_index, llm.device
            )
            self.tk.setdefault("pkv_truncate", []).append(time.time() - start_time)

            # move to trash
            self.kvcache_trash_queue.append(lr_index)
            self.kvcache_trash_map[lr_index] = to_ret
            if len(self.kvcache_trash_queue) > 10:
                self.kvcache_trash_map.pop(self.kvcache_trash_queue.pop(0))
            return to_ret

        if build_upon_lrind == (-1, -1):  # reconstruct
            self.log.debug(
                "_get_or_create_past_key_values(%s): Constructing KVCache from: %s",
                lr_index,
                build_upon_lrind,
            )
            build_upon_kvcache = None
            effective_lr_index = lr_index
        else:  # extend
            self.log.debug(
                "_get_or_create_past_key_values(%s): Extending KVCache from: %s",
                lr_index,
                build_upon_lrind,
            )
            build_upon_kvcache = self.load_kvcache_from_dram(
                build_upon_lrind, llm.device
            )
            effective_lr_index = (build_upon_lrind[1], lr_index[1])

        if (
            effective_lr_index[0] >= effective_lr_index[1]
            or effective_lr_index[0] < 0
            or effective_lr_index[1] < 0
        ):
            self.log.warning(
                "_get_or_create_past_key_values(%s): Invalid effective lr_index(%s) vs input_ids(shape=%s), aborting",
                lr_index,
                effective_lr_index,
                input_ids.shape,
            )
            return None

        _anbn_start_time = time.time()
        model_output = auto_attend_n_by_n_with_past(
            llm,
            input_ids[:, effective_lr_index[0] : effective_lr_index[1]],
            build_upon_kvcache,
            torch_dtype=self.config.torch_dtype,
        )
        if build_upon_lrind == (-1, -1):
            self.tk.setdefault("pkv_anbn_construct", []).append(
                time.time() - _anbn_start_time
            )
        else:
            self.tk.setdefault("pkv_anbn_extend", []).append(
                time.time() - _anbn_start_time
            )

        pkv = model_output.past_key_values if model_output is not None else None

        if isinstance(pkv, tuple):
            pkv = list(pkv)
            for layer, kv in enumerate(pkv):
                pkv[layer] = list(kv)

        if pkv is not None:
            self._set_kvcache(lr_index, pkv)

        return pkv

    def get_root_chunk(self, chunk: AbsChunks):
        root = chunk
        while len(root.parent_keys) != 0:
            root = root.parents[0]
        return root

    def get_or_create_past_key_values(
        self,
        tokenizer,
        ctxt_ids: torch.Tensor,
        chunk: AbsChunks,
        ds_name: str,
        q_index: int,
    ):
        root = self.get_root_chunk(chunk)
        self.log.debug("chunk: %s root: %s", chunk.lr_index, root.lr_index)

        lrind = root.lr_index
        rootpkv = self._get_kvcache(lrind)
        if rootpkv is None:
            # attempt loading from file
            path = self._get_root_pkv_path(ds_name, q_index, lrind)
            if os.path.exists(f"{path}/complete.txt"):
                self.log.debug("Loading past_key_values from %s", path)
                pkv = self.load_kvcache_from_file(path)
                self._set_kvcache(root.lr_index, pkv)
            else:
                self.log.debug("Creating past_key_values")

        if not isinstance(chunk.passage, str):
            raise ValueError(f"chunk.passage is not str: {type(chunk.passage)}")

        self.log.debug("ctxt_ids: %s", ctxt_ids.shape)

        return self._get_or_create_past_key_values(chunk.lr_index, self.llm, ctxt_ids)
