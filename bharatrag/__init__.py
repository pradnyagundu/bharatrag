"""
BharatRAG — RAG Evaluation Library for Indian Languages
Author: Pradnya Gundu
"""

from bharatrag.embeddings.indic_embeddings import IndicEmbedder
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness
from bharatrag.metrics.answer_relevance import AnswerRelevance

__version__ = "0.1.0"
__author__ = "Pradnya Gundu"


def evaluate(questions, contexts, answers, language="hindi"):
    """
    Evaluate a RAG system on Indian language data.

    Args:
        questions: list of questions (in Hindi/Marathi)
        contexts:  list of lists — retrieved context chunks per question
        answers:   list of generated answers
        language:  "hindi" or "marathi" (default: "hindi")

    Returns:
        dict with scores for each metric

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