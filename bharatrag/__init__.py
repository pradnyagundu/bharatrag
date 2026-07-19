"""
BharatRAG — RAG Evaluation Library for Indian Languages
Author: Pradnya Gundu
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Crucial for macOS: prevents transformers from auto-loading TensorFlow/gRPC which causes [mutex.cc : 452] deadlocks
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import logging
logger = logging.getLogger(__name__)

from bharatrag.embeddings.indic_embeddings import IndicEmbedder
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness
from bharatrag.metrics.answer_relevance import AnswerRelevance



__all__ = [
    "evaluate",
    "IndicEmbedder",
    "ContextRelevance",
    "Groundedness",
    "AnswerRelevance",
]

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("bharatrag")
except Exception:
    __version__ = "0.1.0"

__author__ = "Pradnya Gundu"

# Source of truth: indic_embeddings.INDIC_MODELS — never hardcode languages here.
from bharatrag.embeddings.indic_embeddings import SUPPORTED_LANGUAGES as _SUPPORTED_LANGUAGES  # noqa: E402



def evaluate(
    questions: list,
    contexts: list,
    answers: list,
    language: str = "hindi",
    groundedness_threshold=0.45 #Sensitivity threshold for claim context matching
) -> dict:
    """
    Evaluate a RAG system on Indian language data.

    Args:
        questions: list of questions in the target language
        contexts:  list of lists — retrieved context chunks per question
        answers:   list of generated answers
        language:  one of the supported languages (run 'bharatrag languages'
                   for the full list, or inspect bharatrag._SUPPORTED_LANGUAGES)

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
        raise ValueError("at least one question is required")
    if len(questions) != len(contexts) or len(questions) != len(answers):
        raise ValueError("length mismatch: questions, contexts, and answers must have the same length")
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported language '{language}'. "
            f"Choose from: {list(_SUPPORTED_LANGUAGES)}"
        )
    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, list):
            raise TypeError(f"contexts[{i}] must be a list of strings")

    logger.info(f"Loading embedding model for {language}...")

    # Load embedder ONCE and share across all 3 metrics
    embedder = IndicEmbedder(language=language)

    cr = ContextRelevance(language=language, embedder=embedder)
    gr = Groundedness(language=language, threshold=groundedness_threshold, embedder=embedder) #Initialized metric with the user defined sensitivity threshold
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