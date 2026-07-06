"""
BharatRAG — RAG Evaluation Library for Indian Languages
Author: Pradnya Gundu
"""

import logging

from bharatrag.embeddings.indic_embeddings import IndicEmbedder
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness
from bharatrag.metrics.answer_relevance import AnswerRelevance

logger = logging.getLogger(__name__)

__all__ = ["evaluate"]

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("bharatrag")
except Exception:
    __version__ = "0.1.0"

__author__ = "Pradnya Gundu"

_SUPPORTED_LANGUAGES = ("hindi", "marathi", "english")


def evaluate(
    questions: list,
    contexts: list,
    answers: list,
    language: str = "hindi",
) -> dict:
    """
    Evaluate a RAG system on Indian language data.

    Args:
        questions: list of questions (in Hindi/Marathi)
        contexts:  list of lists — retrieved context chunks per question
        answers:   list of generated answers
        language:  "hindi", "marathi", or "english" (default: "hindi")

    Returns:
        dict with scores for each metric

    Raises:
        TypeError: if inputs are not lists
        ValueError: if lists have mismatched lengths or unsupported language

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
    # Input validation
    if not isinstance(questions, list):
        raise TypeError("questions must be a list")
    if not isinstance(contexts, list):
        raise TypeError("contexts must be a list")
    if not isinstance(answers, list):
        raise TypeError("answers must be a list")
    if len(questions) == 0:
        raise ValueError("questions list cannot be empty")
    if len(questions) != len(contexts) or len(questions) != len(answers):
        raise ValueError("questions, contexts, and answers must have the same length")
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language '{language}' not supported. "
            f"Choose from: {list(_SUPPORTED_LANGUAGES)}"
        )
    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, list):
            raise TypeError(f"contexts[{i}] must be a list of strings")

    logger.info(f"Loading embedding model for {language}...")

    # Load embedder ONCE and share across all 3 metrics
    embedder = IndicEmbedder(language=language)

    cr = ContextRelevance(language=language, embedder=embedder)
    gr = Groundedness(language=language, embedder=embedder)
    ar = AnswerRelevance(language=language, embedder=embedder)

    logger.info(f"Evaluating {len(questions)} question(s) in {language}...")

    cr_scores = []
    gr_scores = []
    ar_scores = []

    for i, (question, context, answer) in enumerate(
        zip(questions, contexts, answers)
    ):
        logger.debug(f"Scoring question {i+1}/{len(questions)}...")
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