"""
BharatRAG Test Suite
Run with: pytest tests/ -v
"""

import pytest
from bharatrag.embeddings.indic_embeddings import IndicEmbedder
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness
from bharatrag.metrics.answer_relevance import AnswerRelevance
from bharatrag import evaluate


# ── shared embedder for all tests (loads once) ──────────────────
@pytest.fixture(scope="module")
def hindi_embedder():
    return IndicEmbedder(language="hindi")


# ── IndicEmbedder tests ─────────────────────────────────────────
class TestIndicEmbedder:
    def test_supported_language_hindi(self):
        embedder = IndicEmbedder(language="hindi")
        assert embedder is not None

    def test_unsupported_language_raises_error(self):
        with pytest.raises(ValueError):
            IndicEmbedder(language="klingon")

    def test_similarity_returns_float(self, hindi_embedder):
        score = hindi_embedder.similarity(
            "भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"
        )
        assert isinstance(score, float)

    def test_similarity_between_0_and_1(self, hindi_embedder):
        score = hindi_embedder.similarity(
            "भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"
        )
        assert 0.0 <= score <= 1.0

    def test_similar_sentences_score_higher_than_unrelated(self, hindi_embedder):
        similar_score = hindi_embedder.similarity(
            "भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"
        )
        unrelated_score = hindi_embedder.similarity(
            "भारत की राजधानी क्या है?", "आज मौसम बहुत अच्छा है।"
        )
        assert similar_score > unrelated_score

    def test_similarity_one_to_many_returns_list(self, hindi_embedder):
        scores = hindi_embedder.similarity_one_to_many(
            "भारत की राजधानी क्या है?", ["नई दिल्ली राजधानी है।", "मौसम अच्छा है।"]
        )
        assert isinstance(scores, list)
        assert len(scores) == 2


# ── ContextRelevance tests ──────────────────────────────────────
class TestContextRelevance:
    def test_relevant_context_scores_higher_than_irrelevant(self, hindi_embedder):
        cr = ContextRelevance(language="hindi", embedder=hindi_embedder)

        relevant_score = cr.score(
            question="भारत की राजधानी क्या है?", contexts=["भारत की राजधानी नई दिल्ली है।"]
        )
        irrelevant_score = cr.score(
            question="भारत की राजधानी क्या है?", contexts=["आज क्रिकेट मैच था।"]
        )
        assert relevant_score > irrelevant_score

    def test_empty_contexts_returns_zero(self, hindi_embedder):
        cr = ContextRelevance(language="hindi", embedder=hindi_embedder)
        score = cr.score("कोई सवाल?", [])
        assert score == 0.0

    def test_score_between_0_and_1(self, hindi_embedder):
        cr = ContextRelevance(language="hindi", embedder=hindi_embedder)
        score = cr.score("भारत की राजधानी क्या है?", ["भारत की राजधानी नई दिल्ली है।"])
        assert 0.0 <= score <= 1.0

    def test_score_detailed_returns_dict(self, hindi_embedder):
        cr = ContextRelevance(language="hindi", embedder=hindi_embedder)
        result = cr.score_detailed("भारत की राजधानी क्या है?", ["नई दिल्ली राजधानी है।"])
        assert "overall" in result
        assert "chunks" in result


# ── Groundedness tests ──────────────────────────────────────────
class TestGroundedness:
    def test_grounded_answer_scores_high(self, hindi_embedder):
        gr = Groundedness(language="hindi", embedder=hindi_embedder)
        score = gr.score(
            answer="भारत की राजधानी नई दिल्ली है।",
            contexts=["भारत की राजधानी नई दिल्ली है।"],
        )
        assert score >= 0.5

    def test_empty_answer_returns_zero(self, hindi_embedder):
        gr = Groundedness(language="hindi", embedder=hindi_embedder)
        score = gr.score(answer="", contexts=["कोई context।"])
        assert score == 0.0

    def test_empty_contexts_returns_zero(self, hindi_embedder):
        gr = Groundedness(language="hindi", embedder=hindi_embedder)
        score = gr.score(answer="कोई answer।", contexts=[])
        assert score == 0.0

    def test_score_between_0_and_1(self, hindi_embedder):
        gr = Groundedness(language="hindi", embedder=hindi_embedder)
        score = gr.score(
            answer="भारत की राजधानी नई दिल्ली है।",
            contexts=["भारत की राजधानी नई दिल्ली है।"],
        )
        assert 0.0 <= score <= 1.0

    def test_detailed_has_claims(self, hindi_embedder):
        gr = Groundedness(language="hindi", embedder=hindi_embedder)
        result = gr.score_detailed(
            answer="दिल्ली राजधानी है। मुंबई बड़ा शहर है।",
            contexts=["दिल्ली भारत की राजधानी है।"],
        )
        assert "claims" in result
        assert "total_claims" in result
        assert result["total_claims"] == 2

    # Verify sentences split correctly without breaking on decimals or abbreviations
    def test_split_into_claims_decimals_and_abbreviations(self, hindi_embedder):
        gr = Groundedness(language="hindi", embedder=hindi_embedder)
        text = "पीएम किसान योजना के तहत किसानों को 1.5 लाख रुपये मिलते हैं। डॉ. राम ने कहा कि यह योजना अच्छी है।"
        claims = gr._split_into_claims(text)
        assert len(claims) == 2
        assert "1.5" in claims[0]
        assert "डॉ. राम" in claims[1]


# ── AnswerRelevance tests ───────────────────────────────────────
class TestAnswerRelevance:
    def test_relevant_answer_scores_higher_than_irrelevant(self, hindi_embedder):
        ar = AnswerRelevance(language="hindi", embedder=hindi_embedder)

        relevant = ar.score(
            question="भारत की राजधानी क्या है?", answer="भारत की राजधानी नई दिल्ली है।"
        )
        irrelevant = ar.score(
            question="भारत की राजधानी क्या है?", answer="आज मौसम बहुत अच्छा है।"
        )
        assert relevant > irrelevant

    def test_empty_inputs_return_zero(self, hindi_embedder):
        ar = AnswerRelevance(language="hindi", embedder=hindi_embedder)
        assert ar.score("", "कोई answer") == 0.0
        assert ar.score("कोई question", "") == 0.0

    def test_score_between_0_and_1(self, hindi_embedder):
        ar = AnswerRelevance(language="hindi", embedder=hindi_embedder)
        score = ar.score("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।")
        assert 0.0 <= score <= 1.0


# ── Full evaluate() tests ───────────────────────────────────────
class TestEvaluate:
    def test_evaluate_returns_all_keys(self):
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

    def test_evaluate_scores_between_0_and_1(self):
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

    def test_evaluate_correct_question_count(self):
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


# ── Integration tests ────────────────────────
class TestIntegrations:
    # To test the langchain dependency and evaluator
    def test_langchain_evaluator(self, hindi_embedder):
        try:
            from bharatrag.integrations.langchain import BharatRAGLangChainEvaluator
        except ImportError:
            # Kept skipped for now as it is an optional dependency
            pytest.skip(
                "Langchain dependencies are not installed. Skipping integration test."
            )

        # Test basic initialization
        evaluator = BharatRAGLangChainEvaluator(metric="groundedness", language="hindi")
        assert evaluator is not None
        assert evaluator.requires_input is True
        assert evaluator.requires_reference is True

        # Test evaluating strings
        result = evaluator._evaluate_strings(
            prediction="पीएम किसान योजना के तहत किसानों को 1.5 लाख रुपये मिलते हैं। डॉ. राम ने कहा कि यह योजना अच्छी है।",
            reference="पीएम किसान योजना के तहत किसानों को 1.5 लाख रुपये मिलते हैं। डॉ. राम ने कहा कि यह योजना अच्छी है।",
            input="पीएम किसान योजना में कितने रुपये मिलते हैं?",
        )
        assert "score" in result
        assert result["score"] == 1.0

    def test_llamaindex_evaluator_raises_importerror_without_dependency(self):
        with pytest.raises(ImportError, match="Could not import llama_index"):
            from bharatrag.integrations.llamaindex import BharatRAGLlamaIndexEvaluator

    # Mock integration test for llamaindex evaluator
    def test_llamaindex_evaluator_mocked(self):
        import sys
        from unittest.mock import MagicMock

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

            from bharatrag.integrations.llamaindex import BharatRAGLlamaIndexEvaluator

            evaluator = BharatRAGLlamaIndexEvaluator(
                metric="groundedness", language="hindi"
            )
            assert evaluator is not None

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

    # Mock Unit integration test for langchain evaluator
    def test_langchain_evaluator_mocked(self):
        import sys
        from unittest.mock import MagicMock

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

            from bharatrag.integrations.langchain import BharatRAGLangChainEvaluator

            evaluator = BharatRAGLangChainEvaluator(
                metric="groundedness", language="hindi"
            )
            assert evaluator is not None

            result = evaluator._evaluate_strings(
                prediction="पीएम किसान योजना के तहत किसानों को 1.5 लाख रुपये मिलते हैं। डॉ. राम ने कहा कि यह योजना अच्छी है।",
                reference="पीएम किसान योजना के तहत किसानों को 1.5 लाख रुपये मिलते हैं। डॉ. राम ने कहा कि यह योजना अच्छी है।",
                input="पीएम किसान योजना में कितने रुपये मिलते हैं?",
            )
            assert result["score"] == 1.0
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)
