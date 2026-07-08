"""
Loader for the XQuAD Hindi dataset.

XQuAD is a cross-lingual extractive QA benchmark translated from SQuAD.
This loader supports the Hindi split (``xquad.hi``), normalizing each
example into a ``BenchmarkExample``.

Reference:
    https://huggingface.co/datasets/google/xquad
"""

from __future__ import annotations

import logging
from typing import Optional

from bharatrag.benchmarking.corruption import corrupt_answer
from bharatrag.benchmarking.models import BenchmarkExample

logger = logging.getLogger(__name__)


def load_xquad(
    language: str = "hindi",
    split: str = "validation",
    max_examples: Optional[int] = 100,
    corruption_seed: int = 42,
) -> list[BenchmarkExample]:
    """Load examples from the Google XQuAD dataset.

    Args:
        language: Target language.  Currently only ``"hindi"`` is
            supported (config ``xquad.hi``).
        split: HuggingFace dataset split (default ``"validation"`` —
            XQuAD only has a validation split).
        max_examples: Maximum number of examples to load.  ``None`` for
            the full split.
        corruption_seed: Seed for deterministic hallucinated-answer
            generation.

    Returns:
        List of ``BenchmarkExample`` instances.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
        ValueError: If the requested language is not supported.

    Example:
        >>> examples = load_xquad("hindi", max_examples=10)
        >>> len(examples)
        10
    """
    lang_configs = {"hindi": "xquad.hi"}
    config = lang_configs.get(language)
    if config is None:
        raise ValueError(
            f"XQuAD loader does not support language '{language}'. "
            f"Choose from: {list(lang_configs.keys())}"
        )

    try:
        import datasets  # type: ignore[import-untyped]
        from datasets import load_dataset  # type: ignore[import-untyped]
        datasets.disable_caching()
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for loading XQuAD. "
            "Install it with: pip install bharatrag[benchmarks]"
        )

    logger.info(
        "Loading XQuAD dataset (config=%s, split=%s, max=%s)",
        config,
        split,
        max_examples,
    )
    ds = load_dataset("google/xquad", config, split=split, trust_remote_code=True)

    examples: list[BenchmarkExample] = []
    for idx, row in enumerate(ds):
        if max_examples is not None and idx >= max_examples:
            break

        answer_texts = row.get("answers", {}).get("text", [])
        if not answer_texts or not answer_texts[0]:
            continue

        ground_truth = answer_texts[0]
        context = row.get("context", "")
        question = row.get("question", "")
        if not context or not question:
            continue

        hallucinated = corrupt_answer(
            ground_truth,
            language=language,
            seed=corruption_seed + idx,
        )

        examples.append(
            BenchmarkExample(
                question=question,
                contexts=[context],
                ground_truth_answer=ground_truth,
                hallucinated_answer=hallucinated,
                language=language,
                dataset_name="xquad",
                example_id=f"xquad-{language}-{idx:05d}",
            )
        )

    logger.info("Loaded %d XQuAD examples for %s", len(examples), language)
    return examples
