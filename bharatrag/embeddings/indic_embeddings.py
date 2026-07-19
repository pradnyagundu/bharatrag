"""
Indic Embeddings — loads multilingual embedding models
that understand Indian languages: Hindi, Marathi, Tamil,
Bengali, Telugu, Gujarati, Punjabi, and English.

Design: SentenceTransformer is imported lazily (inside _load_model) so that
importing this module is fast and safe — it does NOT load PyTorch at import
time. This lets the CLI, tests, and any other code read INDIC_MODELS freely
without triggering the macOS tokenizer deadlock.
"""

import logging
import numpy as np


logger = logging.getLogger(__name__)


# ── Language → Model registry ──────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH. Add a new language here and the entire library
# (CLI, evaluate(), tests, IndicEmbedder validation) picks it up automatically.
INDIC_MODELS: dict = {
    "hindi":    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "marathi":  "l3cube-pune/marathi-sentence-bert-nli",
    "tamil":    "l3cube-pune/tamil-sentence-bert-nli",
    "bengali":  "l3cube-pune/bengali-sentence-bert-nli",
    "telugu":   "l3cube-pune/telugu-sentence-bert-nli",
    "gujarati": "l3cube-pune/gujarati-sentence-bert-nli",
    "punjabi":  "l3cube-pune/punjabi-sentence-bert-nli",
    "english":  "sentence-transformers/all-MiniLM-L6-v2",
}

# Derived — always consistent, no duplication.
SUPPORTED_LANGUAGES: tuple = tuple(INDIC_MODELS.keys())

# Module-level cache: one SentenceTransformer instance per language
_model_cache: dict = {}


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row of a and every row of b.

    Equivalent to sklearn.metrics.pairwise.cosine_similarity but without
    pulling in scikit-learn as a dependency.
    """
    a_n = np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
    b_n = np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)
    a_norm = a / a_n
    b_norm = b / b_n
    return a_norm @ b_norm.T


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
            language: one of the languages in INDIC_MODELS
                      (run 'bharatrag languages' for the full list)
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
    def _load_model(language: str, model_name: str):
        """
        Load model from module-level cache, or download and cache it.
        SentenceTransformer is imported HERE (lazily) — not at the top of the
        module — so importing indic_embeddings never triggers PyTorch loading.
        """
        if language not in _model_cache:
            # Lazy import: PyTorch only loads when a model is actually needed.
            from sentence_transformers import SentenceTransformer
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
            text: input string in any supported Indic language or English

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
        score = float(_cosine_similarity_matrix(emb1, emb2)[0][0])
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
        scores = _cosine_similarity_matrix(query_emb, candidate_embs)[0]
        return [float(np.clip(s, 0.0, 1.0)) for s in scores]