"""
Groundedness Metric
Measures: Is the answer grounded in context, or hallucinated?
Score 0-1. Higher = more grounded.
"""

import re
from bharatrag.embeddings.indic_embeddings import IndicEmbedder


class Groundedness:

    def __init__(self, language: str = "hindi",
                 threshold: float = 0.45, embedder=None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be between 0 and 1, got {threshold}"
            )
        self.language = language
        self.threshold = threshold
        self.embedder = embedder or IndicEmbedder(language=language)

    def _split_into_claims(self, text: str) -> list:
        # Protect decimal numbers from being split.
        text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL_POINT>\2', text)
        # Protect common abbreviations dots 
        abbreviations = [
            "डॉ", "पं", "प्रो", "श्री", "श्रीमती",
            "Dr", "Mr", "Mrs", "Ms", "Prof", "e.g", "i.e", "vs"
        ]
        for abbr in abbreviations:
            text = re.sub(rf'(^|\s){abbr}[\.।]', r'\1' + f'{abbr}<ABBR_DOT>', text)
         # Split into actual sentence terminators (। (purna viram), !, ?, or dot with space/ending)
        raw_sentences = re.split(r'(?:[।!?]\s*|\.\s+|\.$)', text)
        # Clean up white spaces and restore original abbreviation dots
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue 
            s = s.replace('<DECIMAL_POINT>', '.')
            s = s.replace('<ABBR_DOT>', '.')
            sentences.append(s)
            
        return sentences

    def score(self, answer: str, contexts: list) -> float:
        if not answer or not contexts:
            return 0.0
        claims = self._split_into_claims(answer)
        if not claims:
            return 0.0
        supported = 0
        for claim in claims:
            similarities = self.embedder.similarity_one_to_many(
                claim, contexts
            )
            if max(similarities) >= self.threshold:
                supported += 1
        return round(supported / len(claims), 4)

    def score_detailed(self, answer: str, contexts: list) -> dict:
        if not answer or not contexts:
            return {"overall": 0.0, "claims": []}
        claims = self._split_into_claims(answer)
        claims_detail = []
        for claim in claims:
            similarities = self.embedder.similarity_one_to_many(
                claim, contexts
            )
            best_score = max(similarities)
            claims_detail.append({
                "claim": claim,
                "best_similarity": round(best_score, 4),
                "supported": best_score >= self.threshold
            })
        supported_count = sum(
            1 for c in claims_detail if c["supported"]
        )
        return {
            "overall": round(supported_count / len(claims), 4),
            "supported": supported_count,
            "total_claims": len(claims),
            "claims": claims_detail,
            "language": self.language
        }