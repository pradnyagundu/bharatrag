"""
Loader for AI4Bharat IndicQA dataset.

IndicQA is an extractive QA dataset covering 11 Indian languages.
This loader supports Hindi (``indicqa.hi``) and Tamil (``indicqa.ta``)
splits, normalizing each example into a ``BenchmarkExample``.

The dataset is loaded via the HuggingFace ``datasets`` library and
cached locally.  Since the source data is extractive QA (no hallucinated
answer provided), we generate one using ``corruption.corrupt_answer()``.

Reference:
    https://huggingface.co/datasets/ai4bharat/IndicQA
"""

from __future__ import annotations

import logging
from typing import Optional

from bharatrag.benchmarking.corruption import corrupt_answer
from bharatrag.benchmarking.models import BenchmarkExample

logger = logging.getLogger(__name__)

# Map our language names to IndicQA config/subset names.
_LANG_TO_CONFIG: dict[str, str] = {
    "hindi": "indicqa.hi",
    "tamil": "indicqa.ta",
}


def load_indicqa(
    language: str = "hindi",
    split: str = "test",
    max_examples: Optional[int] = 100,
    corruption_seed: int = 42,
) -> list[BenchmarkExample]:
    """Load examples from the AI4Bharat IndicQA dataset.

    Args:
        language: Target language (``"hindi"`` or ``"tamil"``).
        split: HuggingFace dataset split (default ``"train"``).
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
        >>> examples = load_indicqa("hindi", max_examples=10)
        >>> len(examples)
        10
    """
    config = _LANG_TO_CONFIG.get(language)
    if config is None:
        raise ValueError(
            f"IndicQA loader does not support language '{language}'. "
            f"Choose from: {list(_LANG_TO_CONFIG.keys())}"
        )

    try:
        import datasets  # type: ignore[import-untyped]
        from datasets import load_dataset  # type: ignore[import-untyped]
        datasets.disable_caching()
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for loading IndicQA. "
            "Install it with: pip install bharatrag[benchmarks]"
        )

    logger.info(
        "Loading IndicQA dataset (config=%s, split=%s, max=%s)",
        config,
        split,
        max_examples,
    )
    ds = load_dataset("ai4bharat/IndicQA", config, split=split, trust_remote_code=True)

    examples: list[BenchmarkExample] = []
    for idx, row in enumerate(ds):
        if max_examples is not None and idx >= max_examples:
            break

        # IndicQA schema: context, question, answers.text[0]
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
                dataset_name="indicqa",
                example_id=f"indicqa-{language}-{idx:05d}",
            )
        )

    logger.info("Loaded %d IndicQA examples for %s", len(examples), language)
    return examples
