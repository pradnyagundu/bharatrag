"""Evaluation data models and orchestration helpers for the dashboard.

The dashboard deliberately delegates all scoring to BharatRAG's public API and
metric classes. This module only validates UI input and shapes the returned
scores for presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

from bharatrag import evaluate
from bharatrag.embeddings.indic_embeddings import IndicEmbedder
from bharatrag.metrics.answer_relevance import AnswerRelevance
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness


# Dynamic: always in sync with the core library — never hardcode here.
from bharatrag.embeddings.indic_embeddings import SUPPORTED_LANGUAGES


METRIC_LABELS = {
    "context_relevance": "Context Relevance",
    "groundedness": "Groundedness",
    "answer_relevance": "Answer Relevance",
    "overall": "Overall Score",
}


class InputValidationError(ValueError):
    """Raised when dashboard input cannot be evaluated safely."""


@dataclass(frozen=True)
class EvaluationSample:
    """One question, its retrieved context chunks, and a generated answer."""

    question: str
    contexts: Tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class MetricScores:
    """The four dashboard metrics returned by BharatRAG."""

    context_relevance: float
    groundedness: float
    answer_relevance: float
    overall: float

    @classmethod
    def from_evaluation(cls, result: Mapping[str, object]) -> "MetricScores":
        """Create a metric collection from a BharatRAG evaluation result."""
        return cls(
            context_relevance=float(result["context_relevance"]),
            groundedness=float(result["groundedness"]),
            answer_relevance=float(result["answer_relevance"]),
            overall=float(result["overall"]),
        )

    def as_dict(self) -> Mapping[str, float]:
        """Return scores keyed by BharatRAG's metric names."""
        return {
            "context_relevance": self.context_relevance,
            "groundedness": self.groundedness,
            "answer_relevance": self.answer_relevance,
            "overall": self.overall,
        }


@dataclass(frozen=True)
class SampleEvaluation:
    """A sample together with its individual metric breakdown."""

    sample: EvaluationSample
    scores: MetricScores


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate and per-sample results for one dashboard evaluation run."""

    language: str
    summary: MetricScores
    samples: Tuple[SampleEvaluation, ...]


def parse_contexts(value: object) -> Tuple[str, ...]:
    """Split a newline-delimited context field into non-empty chunks.

    Args:
        value: Raw text entered in the dashboard context field.

    Returns:
        A tuple containing one clean retrieved chunk per non-empty line.
    """
    if not isinstance(value, str):
        return ()
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def samples_from_rows(rows: Iterable[Mapping[str, object]]) -> Tuple[EvaluationSample, ...]:
    """Build evaluation samples from rows returned by Streamlit's data editor.

    Empty rows are ignored so users can leave spare table rows while preparing
    a batch. Partially filled rows remain errors because they could otherwise
    hide an incomplete evaluation.

    Args:
        rows: Mappings with ``question``, ``contexts``, and ``answer`` values.

    Returns:
        Validated evaluation samples.

    Raises:
        InputValidationError: If no complete samples are supplied.
    """
    samples = []
    for row_number, row in enumerate(rows, start=1):
        question = _clean_text(row.get("question"))
        contexts_value = _clean_text(row.get("contexts"))
        answer = _clean_text(row.get("answer"))

        if not any((question, contexts_value, answer)):
            continue

        contexts = parse_contexts(contexts_value)
        if not question or not contexts or not answer:
            raise InputValidationError(
                "Row {row_number} needs a question, at least one context "
                "chunk, and an answer.".format(row_number=row_number)
            )
        samples.append(
            EvaluationSample(question=question, contexts=contexts, answer=answer)
        )

    validate_samples(samples)
    return tuple(samples)


def run_evaluation(
    samples: Sequence[EvaluationSample], language: str
) -> EvaluationReport:
    """Evaluate a batch and produce its aggregate and individual scores.

    Aggregate metrics are computed with :func:`bharatrag.evaluate`. Individual
    rows use BharatRAG's existing metric classes with one shared embedder,
    avoiding any dashboard-owned scoring logic or repeated model loading.

    Args:
        samples: Complete RAG examples to evaluate.
        language: BharatRAG language identifier.

    Returns:
        An immutable report ready for dashboard rendering.

    Raises:
        InputValidationError: If the supplied samples or language are invalid.
    """
    validate_samples(samples, language)

    summary = MetricScores.from_evaluation(
        evaluate(
            questions=[sample.question for sample in samples],
            contexts=[list(sample.contexts) for sample in samples],
            answers=[sample.answer for sample in samples],
            language=language,
        )
    )

    embedder = IndicEmbedder(language=language)
    context_relevance = ContextRelevance(language=language, embedder=embedder)
    groundedness = Groundedness(language=language, embedder=embedder)
    answer_relevance = AnswerRelevance(language=language, embedder=embedder)

    detailed_samples = []
    for sample in samples:
        context_score = context_relevance.score(sample.question, list(sample.contexts))
        groundedness_score = groundedness.score(sample.answer, list(sample.contexts))
        answer_score = answer_relevance.score(sample.question, sample.answer)
        detailed_samples.append(
            SampleEvaluation(
                sample=sample,
                scores=MetricScores(
                    context_relevance=context_score,
                    groundedness=groundedness_score,
                    answer_relevance=answer_score,
                    overall=round(
                        (context_score + groundedness_score + answer_score) / 3,
                        4,
                    ),
                ),
            )
        )

    return EvaluationReport(
        language=language,
        summary=summary,
        samples=tuple(detailed_samples),
    )


def score_status(score: float) -> str:
    """Classify a metric score using the dashboard's visible thresholds."""
    if score < 0.4:
        return "Poor"
    if score < 0.7:
        return "Moderate"
    return "Good"


def validate_samples(
    samples: Sequence[EvaluationSample], language: str = "hindi"
) -> None:
    """Validate dashboard input before model loading begins."""
    if language not in SUPPORTED_LANGUAGES:
        raise InputValidationError(
            "Unsupported language '{language}'. Choose from: {languages}.".format(
                language=language,
                languages=", ".join(SUPPORTED_LANGUAGES),
            )
        )
    if not samples:
        raise InputValidationError("Add at least one complete evaluation sample.")

    for index, sample in enumerate(samples, start=1):
        if not sample.question.strip():
            raise InputValidationError("Sample {index} is missing a question.".format(index=index))
        if not sample.contexts or not all(chunk.strip() for chunk in sample.contexts):
            raise InputValidationError(
                "Sample {index} needs at least one retrieved context chunk.".format(
                    index=index
                )
            )
        if not sample.answer.strip():
            raise InputValidationError("Sample {index} is missing an answer.".format(index=index))


def _clean_text(value: object) -> str:
    """Normalise a text-editor cell without stringifying missing values."""
    return value.strip() if isinstance(value, str) else ""
