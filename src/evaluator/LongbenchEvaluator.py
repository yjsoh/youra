import logging
from typing import List, Tuple, Generator


from ..abstract import (
    AbsChunkenizer,
    AbsRetriever,
    AbsAugmenter,
    AbsGenerator,
    AbsGrader,
    AbsEvaluator,
)
from ..config import Config
from transformers import LlamaTokenizer, LlamaPreTrainedModel


class LongBenchEvaluator(AbsEvaluator):
    def __init__(
        self,
        config: Config,
        tokenizer: LlamaTokenizer,
        llm: LlamaPreTrainedModel,
        chunkenizer: AbsChunkenizer,
        retriever: AbsRetriever,
        augmenter: AbsAugmenter,
        generator: AbsGenerator,
        grader: AbsGrader,
        output_path: str,
        log_level: int = logging.WARNING,
    ):
        super().__init__(
            config,
            tokenizer,
            llm,
            chunkenizer,
            retriever,
            augmenter,
            generator,
            grader,
            output_path,
            log_level=log_level,
        )

    def get_query_truth(self) -> Generator[Tuple[str, List[str]], None, None]:
        """
        Get the query and truth
        """
        q = self.chunkenizer.ds[0]["input"]
        a = self.chunkenizer.ds[0]["answers"]
        yield q, a
