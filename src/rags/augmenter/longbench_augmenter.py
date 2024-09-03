"""
LongBenchAugmenter
"""

import json
import logging
from typing import Union

from ...abstract import AbsAugmenter


class LongBenchAugmenter(AbsAugmenter):
    """
    LongBenchAugmenter
    """

    def __init__(self, config, tokenizer, log_level: int = logging.WARNING, **kwargs):
        super().__init__(config, tokenizer, **kwargs)

        fh = logging.FileHandler(f"{__name__}.log", mode="a")
        fh.setLevel(logging.DEBUG)
        self.log.addHandler(fh)
        self.log.setLevel(log_level)

        with open("config/dataset2prompt.json", "r", encoding="utf-8") as f:
            self.dataset2prompt = json.load(f)

    def fill_prompt(self, query, context, ds_name) -> str:
        """
        Fill the prompt with the query and context, based on the dataset name
        """
        assert isinstance(query, str)
        assert isinstance(context, str)
        assert isinstance(ds_name, str)
        assert ds_name in self.dataset2prompt

        prompt = self.dataset2prompt[ds_name]
        input_dict = {"input": query, "context": context}
        return prompt.format(**input_dict)

    def _augment(
        self, query: str, retrieved: list[str], **kwargs
    ) -> dict[str, Union[str, list]]:
        self.log.debug("Augmenting %s", query)

        ds_name = kwargs.get("ds_name", None)
        self.log.debug("  ds_name: %s", ds_name)
        self.log.debug("  len(retrieved): %d", len(retrieved))

        assert ds_name in self.dataset2prompt
        assert isinstance(query, str)
        assert isinstance(retrieved, list)
        assert len(retrieved) > 0
        assert isinstance(retrieved[0], str)

        # get concat_str from config
        concat_str = self.config.config_dict["AUGMENTER"]["concat_str"]
        self.log.debug("  concat_str: %s", concat_str.replace("\n", "\\n"))

        # concat the retrieved passages
        retrieved_str = concat_str.join(retrieved)

        # augment the query
        augmented_str = self.fill_prompt(query, retrieved_str, ds_name)

        return {
            "query": query,
            "retrieved": retrieved,
            "concat_str": concat_str,
            "augmented_str": augmented_str,
        }
