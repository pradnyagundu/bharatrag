"""
Indic Embeddings — loads multilingual embedding models
that actually understand Hindi and Marathi.
"""

import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


# Best free models for Indian languages
INDIC_MODELS = {
    "hindi": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "marathi": "l3cube-pune/marathi-sentence-bert-nli",
    "english": "sentence-transformers/all-MiniLM-L6-v2",
}

# Module-level cache: one SentenceTransformer instance per language
_model_cache = {}


class IndicEmbedder:
    """
    Loads the right embedding model for a given Indian language
    and computes sentence embeddings + similarity scores.

    Models are cached globally — re-creating an IndicEmbedder for the same
    language reuses the already-loaded model without downloading again.

    Example:
        >>> embedder = IndicEmbedder(language="hindi")
        >>> score = embedder.similarity(
        ...     "भारत की राजधानी क्या है?",
        ...     "भारत की राजधानी नई दिल्ली है।"
        ... )
        >>> print(score)  # 0.91
    """

    def __init__(self, language: str = "hindi"):
        """
        Args:
            language: "hindi", "marathi", or "english"
        """
        if language not in INDIC_MODELS:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Choose from: {list(INDIC_MODELS.keys())}"
            )

        self.language = language
        self.model_name = INDIC_MODELS[language]
        self.model = self._load_model(language, self.model_name)

    @staticmethod
    def _load_model(language: str, model_name: str) -> SentenceTransformer:
        """Load model from module-level cache, or download and cache it."""
        if language not in _model_cache:
            logger.info(
                "Loading embedding model for %s: %s", language, model_name
            )
            _model_cache[language] = SentenceTransformer(model_name)
            logger.info("Model loaded successfully!")
        else:
            logger.debug("Reusing cached model for %s", language)
        return _model_cache[language]

    def embed(self, text: str) -> np.ndarray:
        """
        Convert a single text string into a vector (embedding).

        Args:
            text: input string in Hindi/Marathi/English

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list) -> np.ndarray:
        """
        Convert a list of texts into embeddings all at once.
        Faster than calling embed() in a loop.

        Args:
            texts: list of strings

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        Returns a score between 0 and 1.
        1.0 = identical meaning, 0.0 = completely unrelated.

        Args:
            text1: first string
            text2: second string

        Returns:
            float between 0 and 1
        """
        emb1 = self.embed(text1).reshape(1, -1)
        emb2 = self.embed(text2).reshape(1, -1)
        score = cosine_similarity(emb1, emb2)[0][0]
        # Clip to [0, 1] — cosine can return tiny negatives
        return float(np.clip(score, 0.0, 1.0))

    def similarity_one_to_many(self, query: str, candidates: list) -> list:
        """
        Compare one query against many candidate texts.
        Used in Context Relevance: compare question vs all chunks.

        Args:
            query: the question
            candidates: list of context chunks

        Returns:
            list of similarity scores, one per candidate
        """
        query_emb = self.embed(query).reshape(1, -1)
        candidate_embs = self.embed_batch(candidates)
        scores = cosine_similarity(query_emb, candidate_embs)[0]
        return [float(np.clip(s, 0.0, 1.0)) for s in scores]