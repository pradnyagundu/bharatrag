"""Loading and evaluation helpers for BharatRAG's bundled benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Tuple

from bharatrag import evaluate

from dashboard.utils import MetricScores


@dataclass(frozen=True)
class BenchmarkExample:
    """A single correct-versus-hallucinated benchmark record."""

    identifier: str
    language: str
    category: str
    question: str
    contexts: Tuple[str, ...]
    correct_answer: str
    hallucinated_answer: str


@dataclass(frozen=True)
class BenchmarkDataset:
    """Validated contents of ``data/benchmark.json``."""

    languages: Tuple[str, ...]
    examples: Tuple[BenchmarkExample, ...]

    def for_language(self, language: str) -> Tuple[BenchmarkExample, ...]:
        """Return benchmark examples for one language."""
        return tuple(example for example in self.examples if example.language == language)

    def categories_for_language(self, language: str) -> Tuple[str, ...]:
        """Return sorted categories represented in one language subset."""
        return tuple(sorted({example.category for example in self.for_language(language)}))


@dataclass(frozen=True)
class BenchmarkComparison:
    """Aggregate scores for the correct and hallucinated answer variants."""

    language: str
    sample_count: int
    correct: MetricScores
    hallucinated: MetricScores


@lru_cache(maxsize=4)
def load_benchmark(path: str) -> BenchmarkDataset:
    """Load, validate, and cache a BharatRAG benchmark JSON file.

    Args:
        path: Filesystem location of a benchmark JSON document.

    Returns:
        A typed immutable benchmark dataset.

    Raises:
        FileNotFoundError: If the benchmark file does not exist.
        ValueError: If required benchmark fields are absent or invalid.
    """
    benchmark_path = Path(path)
    if not benchmark_path.is_file():
        raise FileNotFoundError("Benchmark file not found: {path}".format(path=path))

    try:
        with benchmark_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError("Benchmark JSON is invalid: {error}".format(error=error)) from error

    languages = _required_tuple(payload, "languages")
    raw_examples = payload.get("data")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("Benchmark field 'data' must be a non-empty list.")

    examples = tuple(_parse_example(raw_example, index) for index, raw_example in enumerate(raw_examples, 1))
    unknown_languages = sorted({example.language for example in examples} - set(languages))
    if unknown_languages:
        raise ValueError(
            "Benchmark examples use languages missing from 'languages': {languages}".format(
                languages=", ".join(unknown_languages)
            )
        )
    return BenchmarkDataset(languages=languages, examples=examples)


def evaluate_benchmark(
    dataset: BenchmarkDataset, language: str
) -> BenchmarkComparison:
    """Evaluate correct and hallucinated benchmark answers with BharatRAG.

    Args:
        dataset: Loaded benchmark dataset.
        language: Language subset to evaluate.

    Returns:
        Aggregate metric comparison for the selected language.

    Raises:
        ValueError: If no benchmark records exist for ``language``.
    """
    examples = dataset.for_language(language)
    if not examples:
        raise ValueError(
            "No benchmark examples are available for '{language}'.".format(
                language=language
            )
        )

    questions = [example.question for example in examples]
    contexts = [list(example.contexts) for example in examples]
    correct_answers = [example.correct_answer for example in examples]
    hallucinated_answers = [example.hallucinated_answer for example in examples]

    correct = MetricScores.from_evaluation(
        evaluate(
            questions=questions,
            contexts=contexts,
            answers=correct_answers,
            language=language,
        )
    )
    hallucinated = MetricScores.from_evaluation(
        evaluate(
            questions=questions,
            contexts=contexts,
            answers=hallucinated_answers,
            language=language,
        )
    )
    return BenchmarkComparison(
        language=language,
        sample_count=len(examples),
        correct=correct,
        hallucinated=hallucinated,
    )


def evaluate_all_languages(dataset: BenchmarkDataset) -> Dict[str, BenchmarkComparison]:
    """Evaluate every language in the bundled benchmark once."""
    return {
        language: evaluate_benchmark(dataset, language)
        for language in dataset.languages
    }


def comparison_rows(comparison: BenchmarkComparison) -> Tuple[Mapping[str, object], ...]:
    """Create display-ready rows for the benchmark comparison table."""
    return (
        _comparison_row("Correct Answer", comparison.correct),
        _comparison_row("Hallucinated Answer", comparison.hallucinated),
    )


def _comparison_row(answer_type: str, scores: MetricScores) -> Mapping[str, object]:
    return {
        "Type": answer_type,
        "Overall": scores.overall,
        "Context": scores.context_relevance,
        "Groundedness": scores.groundedness,
        "Answer Relevance": scores.answer_relevance,
    }


def _parse_example(value: object, index: int) -> BenchmarkExample:
    if not isinstance(value, dict):
        raise ValueError("Benchmark record {index} must be an object.".format(index=index))

    identifier = _required_text(value, "id", index)
    language = _required_text(value, "language", index)
    category = _required_text(value, "domain", index)
    question = _required_text(value, "question", index)
    correct_answer = _required_text(value, "ground_truth_answer", index)
    hallucinated_answer = _required_text(value, "hallucinated_answer", index)
    raw_contexts = value.get("context")
    if not isinstance(raw_contexts, list) or not raw_contexts or not all(
        isinstance(context, str) and context.strip() for context in raw_contexts
    ):
        raise ValueError(
            "Benchmark record {index} needs a non-empty 'context' list of strings.".format(
                index=index
            )
        )
    return BenchmarkExample(
        identifier=identifier,
        language=language,
        category=category,
        question=question,
        contexts=tuple(context.strip() for context in raw_contexts),
        correct_answer=correct_answer,
        hallucinated_answer=hallucinated_answer,
    )


def _required_tuple(payload: object, field: str) -> Tuple[str, ...]:
    if not isinstance(payload, dict):
        raise ValueError("Benchmark payload must be an object.")
    value = payload.get(field)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError("Benchmark field '{field}' must be a non-empty string list.".format(field=field))
    return tuple(item.strip() for item in value)


def _required_text(value: Mapping[str, object], field: str, index: int) -> str:
    field_value = value.get(field)
    if not isinstance(field_value, str) or not field_value.strip():
        raise ValueError(
            "Benchmark record {index} needs a non-empty '{field}' field.".format(
                index=index, field=field
            )
        )
    return field_value.strip()
