"""
BharatRAG — RAG Evaluation Library for Indian Languages
Author: Pradnya Gundu
"""

from bharatrag.embeddings.indic_embeddings import IndicEmbedder
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness
from bharatrag.metrics.answer_relevance import AnswerRelevance


__all__ = ["evaluate"]

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("bharatrag")
except Exception:
    # Fallback for source-directory / editable usage
    __version__ = "0.1.0"
__author__ = "Pradnya Gundu"

_SUPPORTED_LANGUAGES = ("hindi", "marathi", "english")


def evaluate(
    questions: list[str],
    contexts: list[list[str]],
    answers: list[str],
    language: str = "hindi",
) -> dict:
    """
    Evaluate a RAG system on Indian language data.

    Args:
        questions: list of questions (in Hindi/Marathi)
        contexts:  list of lists — retrieved context chunks per question
        answers:   list of generated answers
        language:  "hindi" or "marathi" (default: "hindi")

    Returns:
        dict with scores for each metric

    Raises:
        TypeError:  if questions/contexts/answers are not lists
        ValueError: if lengths mismatch, inputs empty, or language unsupported

    Example:
        >>> from bharatrag import evaluate
        >>> results = evaluate(
        ...     questions=["भारत की राजधानी क्या है?"],
        ...     contexts=[["भारत की राजधानी नई दिल्ली है।"]],
        ...     answers=["भारत की राजधानी नई दिल्ली है।"],
        ...     language="hindi"
        ... )
        >>> print(results)
    """
    # ── Input validation (fail fast before model load) ────────────
    if not (isinstance(questions, list) and isinstance(contexts, list) and isinstance(answers, list)):
        raise TypeError("questions, contexts, and answers must be lists")
    if contexts and not all(isinstance(c, list) for c in contexts):
        raise TypeError("each element of contexts must be a list")
    if not questions:
        raise ValueError("at least one question is required")
    if len(questions) != len(contexts) or len(questions) != len(answers):
        raise ValueError(
            f"length mismatch: questions={len(questions)}, "
            f"contexts={len(contexts)}, answers={len(answers)}"
        )
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported language: {language!r}. "
            f"choose from: {_SUPPORTED_LANGUAGES}"
        )

    print(f"\nLoading embedding model for {language}...")

    # Load embedder ONCE and share across all 3 metrics
    embedder = IndicEmbedder(language=language)

    # Pass the same embedder to all metrics — no reloading
    cr = ContextRelevance(language=language, embedder=embedder)
    gr = Groundedness(language=language, embedder=embedder)
    ar = AnswerRelevance(language=language, embedder=embedder)

    print(f"Evaluating {len(questions)} question(s) in {language}...")

    cr_scores = []
    gr_scores = []
    ar_scores = []

    for i, (question, context, answer) in enumerate(
        zip(questions, contexts, answers)
    ):
        print(f"  Scoring question {i+1}/{len(questions)}...")
        cr_scores.append(cr.score(question, context))
        gr_scores.append(gr.score(answer, context))
        ar_scores.append(ar.score(question, answer))

    results = {
        "context_relevance": round(sum(cr_scores) / len(cr_scores), 4),
        "groundedness":      round(sum(gr_scores) / len(gr_scores), 4),
        "answer_relevance":  round(sum(ar_scores) / len(ar_scores), 4),
        "language":          language,
        "num_questions":     len(questions),
    }

    results["overall"] = round(
        (results["context_relevance"] +
         results["groundedness"] +
         results["answer_relevance"]) / 3,
        4
    )

    return results
