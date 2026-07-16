"""
BharatRAG test suite.

Fast unit tests use deterministic fake embedders and do not download models.
Run model-loading checks explicitly with:
    pytest tests/ -m integration -v
"""

import sys
from unittest.mock import MagicMock

import pytest

import bharatrag
from bharatrag import evaluate
from bharatrag.embeddings.indic_embeddings import IndicEmbedder
from bharatrag.metrics.answer_relevance import AnswerRelevance
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness


class FakeEmbedder:
    """Deterministic embedder for fast metric tests."""

    def __init__(self, pair_scores=None):
        self.pair_scores = pair_scores or {}

    def similarity(self, text1: str, text2: str) -> float:
        return self.pair_scores.get((text1, text2), 0.1)

    def similarity_one_to_many(self, query: str, candidates: list) -> list:
        return [self.similarity(query, candidate) for candidate in candidates]


def fake_evaluate(
    questions: list,
    contexts: list,
    answers: list,
    language: str = "hindi",
    **kwargs,
) -> dict:
    return {
        "context_relevance": 1.0,
        "groundedness": 1.0,
        "answer_relevance": 1.0,
        "overall": 1.0,
        "language": language,
        "num_questions": len(questions),
    }


@pytest.fixture(scope="module")
def real_hindi_embedder():
    return IndicEmbedder(language="hindi")


# ── ContextRelevance unit tests ─────────────────────────────────
class TestContextRelevance:
    def test_relevant_context_scores_higher_than_irrelevant(self):
        embedder = FakeEmbedder(
            {
                ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"): 0.9,
                ("भारत की राजधानी क्या है?", "आज क्रिकेट मैच था।"): 0.2,
            }
        )
        cr = ContextRelevance(language="hindi", embedder=embedder)

        relevant_score = cr.score(
            question="भारत की राजधानी क्या है?",
            contexts=["भारत की राजधानी नई दिल्ली है।"],
        )
        irrelevant_score = cr.score(
            question="भारत की राजधानी क्या है?",
            contexts=["आज क्रिकेट मैच था।"],
        )

        assert relevant_score > irrelevant_score

    def test_empty_contexts_returns_zero(self):
        cr = ContextRelevance(language="hindi", embedder=FakeEmbedder())

        assert cr.score("कोई सवाल?", []) == 0.0

    def test_score_between_0_and_1(self):
        embedder = FakeEmbedder(
            {("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"): 0.9}
        )
        cr = ContextRelevance(language="hindi", embedder=embedder)

        score = cr.score(
            "भारत की राजधानी क्या है?",
            ["भारत की राजधानी नई दिल्ली है।"],
        )

        assert 0.0 <= score <= 1.0

    def test_score_detailed_returns_chunk_scores(self):
        embedder = FakeEmbedder(
            {
                ("भारत की राजधानी क्या है?", "नई दिल्ली राजधानी है।"): 0.8,
                ("भारत की राजधानी क्या है?", "मौसम अच्छा है।"): 0.2,
            }
        )
        cr = ContextRelevance(language="hindi", embedder=embedder)

        result = cr.score_detailed(
            "भारत की राजधानी क्या है?",
            ["नई दिल्ली राजधानी है।", "मौसम अच्छा है।"],
        )

        assert result["overall"] == 0.5
        assert result["chunks"][0]["score"] == 0.8
        assert result["language"] == "hindi"


# ── Groundedness unit tests ─────────────────────────────────────
class TestGroundedness:
    def test_grounded_answer_scores_high(self):
        answer = "भारत की राजधानी नई दिल्ली है।"
        claim = "भारत की राजधानी नई दिल्ली है"
        context = "भारत की राजधानी नई दिल्ली है।"
        embedder = FakeEmbedder({(claim, context): 0.9})
        gr = Groundedness(language="hindi", embedder=embedder)

        assert gr.score(answer=answer, contexts=[context]) == 1.0

    def test_empty_answer_returns_zero(self):
        gr = Groundedness(language="hindi", embedder=FakeEmbedder())

        assert gr.score(answer="", contexts=["कोई context।"]) == 0.0

    def test_empty_contexts_returns_zero(self):
        gr = Groundedness(language="hindi", embedder=FakeEmbedder())

        assert gr.score(answer="कोई answer।", contexts=[]) == 0.0

    def test_score_between_0_and_1(self):
        answer = "भारत की राजधानी नई दिल्ली है।"
        context = "भारत की राजधानी नई दिल्ली है।"
        embedder = FakeEmbedder({(answer, context): 0.9})
        gr = Groundedness(language="hindi", embedder=embedder)

        score = gr.score(answer=answer, contexts=[context])

        assert 0.0 <= score <= 1.0

    def test_detailed_has_claims(self):
        embedder = FakeEmbedder(
            {
                ("दिल्ली राजधानी है", "दिल्ली भारत की राजधानी है।"): 0.9,
                ("मुंबई बड़ा शहर है", "दिल्ली भारत की राजधानी है।"): 0.2,
            }
        )
        gr = Groundedness(language="hindi", embedder=embedder)

        result = gr.score_detailed(
            answer="दिल्ली राजधानी है। मुंबई बड़ा शहर है।",
            contexts=["दिल्ली भारत की राजधानी है।"],
        )

        assert result["overall"] == 0.5
        assert result["supported"] == 1
        assert result["total_claims"] == 2
        assert len(result["claims"]) == 2

    def test_detailed_empty_claims_no_zero_division(self):
        gr = Groundedness(language="hindi", embedder=FakeEmbedder())

        result = gr.score_detailed(
            answer=".",
            contexts=["दिल्ली भारत की राजधानी है।"],
        )

        assert result["overall"] == 0.0
        assert result["total_claims"] == 0
        assert result["claims"] == []

    def test_split_into_claims_decimals_and_abbreviations(self):
        gr = Groundedness(language="hindi", embedder=FakeEmbedder())
        text = (
            "पीएम किसान योजना के तहत किसानों को 1.5 लाख रुपये मिलते हैं। "
            "डॉ. राम ने कहा कि यह योजना अच्छी है।"
        )

        claims = gr._split_into_claims(text)

        assert len(claims) == 2
        assert "1.5" in claims[0]
        assert "डॉ. राम" in claims[1]

    def test_split_into_claims_latin_abbreviations(self):
        gr = Groundedness(language="hindi", embedder=FakeEmbedder())
        text = "यह योजना किसानों, महिलाओं etc. के लिए है। यह अच्छी है।"

        claims = gr._split_into_claims(text)

        assert len(claims) == 2
        assert "etc." in claims[0] or "etc" in claims[0]
        assert "यह अच्छी है" in claims[1]

    def test_split_into_claims_gujarati_abbreviations(self):
        gr = Groundedness(language="gujarati", embedder=FakeEmbedder())
        # "Dr. Patel says it is good. He is right."
        text = "ડૉ. પટેલ કહે છે કે આ સારું છે. તે સાચું છે."
        claims = gr._split_into_claims(text)
        assert len(claims) == 2
        assert "ડૉ. પટેલ" in claims[0]
        assert "તે સાચું છે" in claims[1]


# ── AnswerRelevance unit tests ──────────────────────────────────
class TestAnswerRelevance:
    def test_relevant_answer_scores_higher_than_irrelevant(self):
        question = "भारत की राजधानी क्या है?"
        embedder = FakeEmbedder(
            {
                (question, "भारत की राजधानी नई दिल्ली है।"): 0.9,
                (question, "आज मौसम बहुत अच्छा है।"): 0.1,
            }
        )
        ar = AnswerRelevance(language="hindi", embedder=embedder)

        relevant = ar.score(question=question, answer="भारत की राजधानी नई दिल्ली है।")
        irrelevant = ar.score(question=question, answer="आज मौसम बहुत अच्छा है।")

        assert relevant > irrelevant

    def test_empty_inputs_return_zero(self):
        ar = AnswerRelevance(language="hindi", embedder=FakeEmbedder())

        assert ar.score("", "कोई answer") == 0.0
        assert ar.score("कोई question", "") == 0.0

    def test_score_between_0_and_1(self):
        embedder = FakeEmbedder(
            {("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"): 0.9}
        )
        ar = AnswerRelevance(language="hindi", embedder=embedder)

        score = ar.score(
            "भारत की राजधानी क्या है?",
            "भारत की राजधानी नई दिल्ली है।",
        )

        assert 0.0 <= score <= 1.0

    def test_score_detailed_interpretation(self):
        embedder = FakeEmbedder(
            {("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"): 0.9}
        )
        ar = AnswerRelevance(language="hindi", embedder=embedder)

        result = ar.score_detailed(
            "भारत की राजधानी क्या है?",
            "भारत की राजधानी नई दिल्ली है।",
        )

        assert result["overall"] == 0.9
        assert result["interpretation"] == "Highly relevant"


# ── evaluate() unit tests ───────────────────────────────────────
class TestEvaluate:
    def test_evaluate_returns_all_keys(self, monkeypatch):
        fake_embedder = FakeEmbedder(
            {
                ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"): 0.9,
                ("भारत की राजधानी नई दिल्ली है", "भारत की राजधानी नई दिल्ली है।"): 0.9,
            }
        )
        monkeypatch.setattr(bharatrag, "IndicEmbedder", lambda language: fake_embedder)

        results = evaluate(
            questions=["भारत की राजधानी क्या है?"],
            contexts=[["भारत की राजधानी नई दिल्ली है।"]],
            answers=["भारत की राजधानी नई दिल्ली है।"],
            language="hindi",
        )

        assert "context_relevance" in results
        assert "groundedness" in results
        assert "answer_relevance" in results
        assert "overall" in results
        assert "language" in results
        assert "num_questions" in results

    def test_evaluate_scores_between_0_and_1(self, monkeypatch):
        fake_embedder = FakeEmbedder(
            {
                ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"): 0.9,
                ("भारत की राजधानी नई दिल्ली है", "भारत की राजधानी नई दिल्ली है।"): 0.9,
            }
        )
        monkeypatch.setattr(bharatrag, "IndicEmbedder", lambda language: fake_embedder)

        results = evaluate(
            questions=["भारत की राजधानी क्या है?"],
            contexts=[["भारत की राजधानी नई दिल्ली है।"]],
            answers=["भारत की राजधानी नई दिल्ली है।"],
            language="hindi",
        )

        assert 0.0 <= results["context_relevance"] <= 1.0
        assert 0.0 <= results["groundedness"] <= 1.0
        assert 0.0 <= results["answer_relevance"] <= 1.0
        assert 0.0 <= results["overall"] <= 1.0

    def test_evaluate_correct_question_count(self, monkeypatch):
        fake_embedder = FakeEmbedder(
            {
                ("सवाल १", "context १"): 0.8,
                ("सवाल २", "context २"): 0.6,
                ("जवाब १", "context १"): 0.9,
                ("जवाब २", "context २"): 0.9,
                ("सवाल १", "जवाब १"): 0.7,
                ("सवाल २", "जवाब २"): 0.7,
            }
        )
        monkeypatch.setattr(bharatrag, "IndicEmbedder", lambda language: fake_embedder)

        results = evaluate(
            questions=["सवाल १", "सवाल २"],
            contexts=[["context १"], ["context २"]],
            answers=["जवाब १", "जवाब २"],
            language="hindi",
        )

        assert results["num_questions"] == 2


# ── evaluate() input validation tests (fast — no model loading) ─
class TestEvaluateValidation:
    def test_empty_questions_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one question"):
            evaluate([], [[]], ["answer"])

    def test_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="length mismatch"):
            evaluate(["q1", "q2"], [["c1"]], ["a1", "a2"])

    def test_wrong_type_questions_raises_type_error(self):
        with pytest.raises(TypeError):
            evaluate("not-a-list", [[]], ["answer"])

    def test_wrong_type_contexts_raises_type_error(self):
        with pytest.raises(TypeError):
            evaluate(["q"], ["not-a-list"], ["a"])

    def test_unsupported_language_raises_value_error(self):
        with pytest.raises(ValueError, match="unsupported language"):
            evaluate(["q"], [["c"]], ["a"], language="klingon")


# ── Integration wrapper unit tests (fast, mocked optional deps) ─
class TestIntegrationWrappers:
    def test_llamaindex_evaluator_raises_importerror_without_dependency(self):
        original_modules = sys.modules.copy()
        sys.modules["llama_index"] = None
        sys.modules["llama_index.core.evaluation"] = None

        if "bharatrag.integrations.llamaindex" in sys.modules:
            del sys.modules["bharatrag.integrations.llamaindex"]

        try:
            with pytest.raises(ImportError, match="Could not import llama_index"):
                from bharatrag.integrations.llamaindex import (  # noqa: F401
                    BharatRAGLlamaIndexEvaluator,
                )
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_llamaindex_evaluator_mocked(self, monkeypatch):
        original_modules = sys.modules.copy()
        try:
            mock_eval_module = MagicMock()

            class DummyBaseEvaluator:
                pass

            class DummyEvaluationResult:
                def __init__(
                    self,
                    query=None,
                    contexts=None,
                    response=None,
                    score=None,
                    passing=None,
                    feedback=None,
                ):
                    self.query = query
                    self.contexts = contexts
                    self.response = response
                    self.score = score
                    self.passing = passing
                    self.feedback = feedback

            mock_eval_module.BaseEvaluator = DummyBaseEvaluator
            mock_eval_module.EvaluationResult = DummyEvaluationResult

            sys.modules["llama_index"] = MagicMock()
            sys.modules["llama_index.core"] = MagicMock()
            sys.modules["llama_index.core.evaluation"] = mock_eval_module

            if "bharatrag.integrations.llamaindex" in sys.modules:
                del sys.modules["bharatrag.integrations.llamaindex"]

            from bharatrag.integrations import llamaindex as llamaindex_integration
            from bharatrag.integrations.llamaindex import BharatRAGLlamaIndexEvaluator

            monkeypatch.setattr(llamaindex_integration, "evaluate", fake_evaluate)
            evaluator = BharatRAGLlamaIndexEvaluator(
                metric="groundedness",
                language="hindi",
            )

            result = evaluator.evaluate(
                query="पीएम किसान योजना में कितने रुपये मिलते हैं?",
                contexts=[
                    "प्रधानमंत्री किसान सम्मान निधि योजना के तहत किसानों को 6000 रुपये मिलते हैं।"
                ],
                response="पीएम किसान योजना में 6000 रुपये मिलते हैं।",
            )

            assert result.score == 1.0
            assert result.passing is True
            assert "groundedness score: 1.0" in result.feedback
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_langchain_evaluator_mocked(self, monkeypatch):
        original_modules = sys.modules.copy()
        try:
            mock_langchain_core = MagicMock()

            class DummyStringEvaluator:
                pass

            mock_langchain_core.evaluation.StringEvaluator = DummyStringEvaluator

            sys.modules["langchain_core"] = mock_langchain_core
            sys.modules["langchain_core.evaluation"] = mock_langchain_core.evaluation

            if "bharatrag.integrations.langchain" in sys.modules:
                del sys.modules["bharatrag.integrations.langchain"]

            from bharatrag.integrations import langchain as langchain_integration

            monkeypatch.setattr(langchain_integration, "evaluate", fake_evaluate)
            evaluator = langchain_integration.BharatRAGLangChainEvaluator(
                metric="groundedness",
                language="hindi",
            )

            result = evaluator._evaluate_strings(
                prediction="पीएम किसान योजना में 6000 रुपये मिलते हैं।",
                reference="पीएम किसान योजना के तहत किसानों को 6000 रुपये मिलते हैं।",
                input="पीएम किसान योजना में कितने रुपये मिलते हैं?",
            )
            result_missing = evaluator._evaluate_strings(
                prediction="पीएम किसान योजना में 6000 रुपये मिलते हैं।",
                reference=None,
                input="पीएम किसान योजना में कितने रुपये मिलते हैं?",
            )

            assert result["score"] == 1.0
            assert result_missing["score"] == 0.0
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


# ── Real model integration tests ────────────────────────────────
@pytest.mark.integration
class TestIndicEmbedderIntegration:
    def test_supported_language_hindi(self):
        embedder = IndicEmbedder(language="hindi")

        assert embedder is not None

    def test_supported_language_punjabi(self):
        embedder = IndicEmbedder(language="punjabi")

        assert embedder is not None

    def test_unsupported_language_raises_error(self):
        with pytest.raises(ValueError):
            IndicEmbedder(language="klingon")

    def test_similarity_returns_float(self, real_hindi_embedder):
        score = real_hindi_embedder.similarity(
            "भारत की राजधानी क्या है?",
            "भारत की राजधानी नई दिल्ली है।",
        )

        assert isinstance(score, float)

    def test_similarity_between_0_and_1(self, real_hindi_embedder):
        score = real_hindi_embedder.similarity(
            "भारत की राजधानी क्या है?",
            "भारत की राजधानी नई दिल्ली है।",
        )

        assert 0.0 <= score <= 1.0

    def test_similar_sentences_score_higher_than_unrelated(self, real_hindi_embedder):
        similar_score = real_hindi_embedder.similarity(
            "भारत की राजधानी क्या है?",
            "भारत की राजधानी नई दिल्ली है।",
        )
        unrelated_score = real_hindi_embedder.similarity(
            "भारत की राजधानी क्या है?",
            "आज मौसम बहुत अच्छा है।",
        )

        assert similar_score > unrelated_score

    def test_similarity_one_to_many_returns_list(self, real_hindi_embedder):
        scores = real_hindi_embedder.similarity_one_to_many(
            "भारत की राजधानी क्या है?",
            ["नई दिल्ली राजधानी है।", "मौसम अच्छा है।"],
        )

        assert isinstance(scores, list)
        assert len(scores) == 2


@pytest.mark.integration
class TestRealModelEvaluateIntegration:
    def test_evaluate_returns_all_keys_with_real_model(self):
        results = evaluate(
            questions=["भारत की राजधानी क्या है?"],
            contexts=[["भारत की राजधानी नई दिल्ली है।"]],
            answers=["भारत की राजधानी नई दिल्ली है।"],
            language="hindi",
        )

        assert "overall" in results
        assert results["language"] == "hindi"
