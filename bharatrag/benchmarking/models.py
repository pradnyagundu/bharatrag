"""
Data models for the benchmarking pipeline.

Every dataset loader normalizes its data into ``BenchmarkExample`` so that
the runner, stats, and report modules can treat all datasets uniformly.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class BenchmarkExample:
    """A single QA example normalized from any supported dataset.

    Attributes:
        question: The user question in the target language.
        contexts: Context passages that should support the answer.
        ground_truth_answer: The correct answer extracted from the dataset.
        hallucinated_answer: A synthetically corrupted answer (generated
            by ``corruption.py``).
        language: ISO-style language name (e.g. ``"hindi"``, ``"tamil"``).
        dataset_name: Source dataset identifier (e.g. ``"indicqa"``,
            ``"xquad"``).
        example_id: Unique ID within the source dataset.
        domain: Optional domain/category tag (e.g. ``"government_scheme"``).
    """

    question: str
    contexts: list[str]
    ground_truth_answer: str
    hallucinated_answer: str
    language: str
    dataset_name: str
    example_id: str
    domain: Optional[str] = None


@dataclasses.dataclass
class QuestionResult:
    """Per-question evaluation result with timing information.

    Attributes:
        example: The original benchmark example that was evaluated.
        correct_scores: Metric scores when using the ground-truth answer.
        hallucinated_scores: Metric scores when using the hallucinated answer.
        correct_wall_clock_s: Wall-clock seconds for the correct-answer eval.
        hallucinated_wall_clock_s: Wall-clock seconds for the hallucinated eval.
        correct_peak_memory_bytes: Peak memory delta for correct eval, or
            ``None`` if tracemalloc was unavailable.
        hallucinated_peak_memory_bytes: Peak memory delta for hallucinated
            eval, or ``None`` if tracemalloc was unavailable.
    """

    example: BenchmarkExample
    correct_scores: dict[str, float] = dataclasses.field(default_factory=dict)
    hallucinated_scores: dict[str, float] = dataclasses.field(
        default_factory=dict
    )
    correct_wall_clock_s: float = 0.0
    hallucinated_wall_clock_s: float = 0.0
    correct_peak_memory_bytes: Optional[int] = None
    hallucinated_peak_memory_bytes: Optional[int] = None


@dataclasses.dataclass
class DatasetResults:
    """Aggregated benchmark results for one dataset × language combination.

    Attributes:
        dataset_name: Source dataset identifier.
        language: Language that was evaluated.
        question_results: Per-question detailed results.
        correct_aggregate: Aggregated metric means for correct answers.
        hallucinated_aggregate: Aggregated metric means for hallucinated
            answers.
        total_wall_clock_s: Total wall-clock time for all evaluations.
        total_peak_memory_bytes: Peak memory usage across all evaluations,
            or ``None`` if tracemalloc was unavailable.
    """

    dataset_name: str
    language: str
    question_results: list[QuestionResult] = dataclasses.field(
        default_factory=list
    )
    correct_aggregate: dict[str, float] = dataclasses.field(
        default_factory=dict
    )
    hallucinated_aggregate: dict[str, float] = dataclasses.field(
        default_factory=dict
    )
    total_wall_clock_s: float = 0.0
    total_peak_memory_bytes: Optional[int] = None


@dataclasses.dataclass
class BenchmarkResults:
    """Top-level container for all benchmark run outputs.

    Attributes:
        dataset_results: Per-dataset × language result sets.
        ragas_results: Optional RAGAS adapter results, keyed by
            ``"dataset:language"``.
        deepeval_results: Optional DeepEval adapter results, keyed by
            ``"dataset:language"``.
    """

    dataset_results: list[DatasetResults] = dataclasses.field(
        default_factory=list
    )
    ragas_results: Optional[dict[str, list[dict]]] = None
    deepeval_results: Optional[dict[str, list[dict]]] = None
