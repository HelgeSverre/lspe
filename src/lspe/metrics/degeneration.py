"""Lexical degeneration metrics retained even for failed generations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def degeneration_metrics(token_ids: Iterable[int]) -> dict[str, float]:
    tokens = list(token_ids)
    if not tokens:
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "distinct_3": 0.0,
            "repeated_4gram_ratio": 0.0,
            "max_identical_run": 0.0,
        }
    metrics: dict[str, float] = {}
    for ngram_size in (1, 2, 3):
        grams = [
            tuple(tokens[index : index + ngram_size])
            for index in range(len(tokens) - ngram_size + 1)
        ]
        metrics[f"distinct_{ngram_size}"] = len(set(grams)) / len(grams) if grams else 0.0
    four_grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
    repeated = sum(count - 1 for count in Counter(four_grams).values() if count > 1)
    metrics["repeated_4gram_ratio"] = repeated / len(four_grams) if four_grams else 0.0
    longest = 1
    current = 1
    for previous, token in zip(tokens, tokens[1:], strict=False):
        current = current + 1 if previous == token else 1
        longest = max(longest, current)
    metrics["max_identical_run"] = float(longest)
    return metrics
