"""
LongBenchGrader
"""

import re
import string
from collections import Counter

from ...abstract import AbsGrader
from ...config import Config


class LongBenchGrader(AbsGrader):
    """
    Grader using LongBench Grader
    """

    def __init__(self, config: Config, **kwargs):
        # Some grader may inherit from this class and override the grader_name
        grader_name = kwargs.get("grader_name", "LongBenchGrader")
        super().__init__(config, grader_name)

    def _normalize_answer(self, s: str):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _f1_score(self, prediction, ground_truth):
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def qa_f1_score(self, prediction, ground_truth, **kwargs):
        """
        Compute the QA F1 score
        """
        normalized_prediction = self._normalize_answer(prediction)
        normalized_ground_truth = self._normalize_answer(ground_truth)
        self.log.debug("normalized_ground_truth: %s", normalized_ground_truth)
        self.log.debug("normalized_prediction: %s", normalized_prediction)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        self.log.debug("normalized_ground_truth_tokens: %s", ground_truth_tokens)
        self.log.debug("normalized_prediction_tokens: %s", prediction_tokens)
        return self._f1_score(prediction_tokens, ground_truth_tokens)

    def _grade(
        self, query: str, truth: list[str], generated: list[str], **kwargs
    ) -> float:
        self.log.debug("truth: %s", truth)
        self.log.debug("generated: %s", generated)
        score = 0.0
        for g in generated:
            for t in truth:
                score = max(score, self.qa_f1_score(g, t))
        self.log.debug("LongBenchGrader Score: %f", score)
        return score
