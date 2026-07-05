"""
Answer Relevance Metric
Measures: Does the answer actually address the question?
Score 0-1. Higher = more relevant.
"""

from bharatrag.embeddings.indic_embeddings import IndicEmbedder


class AnswerRelevance:

    def __init__(self, language: str = "hindi", embedder=None):
        self.language = language
        self.embedder = embedder or IndicEmbedder(language=language)

    def score(self, question: str, answer: str) -> float:
        if not question or not answer:
            return 0.0
        return round(self.embedder.similarity(question, answer), 4)

    def score_detailed(self, question: str, answer: str) -> dict:
        if not question or not answer:
            return {"overall": 0.0}
        similarity = self.embedder.similarity(question, answer)
        if similarity >= 0.7:
            interpretation = "Highly relevant"
        elif similarity >= 0.45:
            interpretation = "Moderately relevant"
        elif similarity >= 0.25:
            interpretation = "Weakly relevant"
        else:
            interpretation = "Not relevant"
        return {
            "overall": round(similarity, 4),
            "interpretation": interpretation,
            "question": question,
            "answer": answer[:100] + "..." if len(answer) > 100 else answer,
            "language": self.language
        }