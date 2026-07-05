"""
Context Relevance Metric
Measures: Did we retrieve the right context for this question?
Score 0-1. Higher = more relevant context retrieved.
"""

from bharatrag.embeddings.indic_embeddings import IndicEmbedder


class ContextRelevance:

    def __init__(self, language: str = "hindi", embedder=None):
        self.language = language
        # Use shared embedder if provided, else create new one
        self.embedder = embedder or IndicEmbedder(language=language)

    def score(self, question: str, contexts: list) -> float:
        if not contexts:
            return 0.0
        similarities = self.embedder.similarity_one_to_many(
            question, contexts
        )
        return round(sum(similarities) / len(similarities), 4)

    def score_detailed(self, question: str, contexts: list) -> dict:
        if not contexts:
            return {"overall": 0.0, "chunks": []}
        similarities = self.embedder.similarity_one_to_many(
            question, contexts
        )
        chunks_detail = [
            {
                "chunk": ctx[:100] + "..." if len(ctx) > 100 else ctx,
                "score": round(sim, 4)
            }
            for ctx, sim in zip(contexts, similarities)
        ]
        return {
            "overall": round(sum(similarities) / len(similarities), 4),
            "chunks": chunks_detail,
            "language": self.language
        }