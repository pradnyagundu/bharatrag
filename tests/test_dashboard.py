"""Fast unit tests for dashboard input and benchmark orchestration."""

from dashboard import utils
from dashboard.benchmark import evaluate_benchmark, load_benchmark
from dashboard.utils import InputValidationError, samples_from_rows


def test_samples_from_rows_splits_context_chunks_and_ignores_empty_rows():
    samples = samples_from_rows(
        [
            {
                "question": "What is BharatRAG?",
                "contexts": "First chunk\n\nSecond chunk",
                "answer": "It evaluates RAG systems.",
            },
            {"question": "", "contexts": "", "answer": ""},
        ]
    )

    assert len(samples) == 1
    assert samples[0].contexts == ("First chunk", "Second chunk")


def test_samples_from_rows_rejects_partial_rows():
    try:
        samples_from_rows(
            [{"question": "Question", "contexts": "Context", "answer": ""}]
        )
    except InputValidationError as error:
        assert "Row 1" in str(error)
    else:
        raise AssertionError("Expected a partial dashboard row to be rejected")


def test_load_benchmark_exposes_languages_and_categories():
    dataset = load_benchmark("data/benchmark.json")

    assert "hindi" in dataset.languages
    assert len(dataset.for_language("hindi")) == 30
    assert "government_scheme" in dataset.categories_for_language("hindi")


def test_evaluate_benchmark_uses_correct_and_hallucinated_answers(monkeypatch):
    dataset = load_benchmark("data/benchmark.json")
    calls = []

    def fake_evaluate(questions, contexts, answers, language):
        calls.append(answers)
        score = 0.8 if answers[0] == dataset.for_language(language)[0].correct_answer else 0.2
        return {
            "context_relevance": score,
            "groundedness": score,
            "answer_relevance": score,
            "overall": score,
        }

    monkeypatch.setattr("dashboard.benchmark.evaluate", fake_evaluate)
    comparison = evaluate_benchmark(dataset, "hindi")

    assert len(calls) == 2
    assert comparison.correct.overall == 0.8
    assert comparison.hallucinated.overall == 0.2


def test_run_evaluation_uses_bharatrag_batch_and_metric_apis(monkeypatch):
    sample = utils.EvaluationSample(
        question="Question",
        contexts=("Context",),
        answer="Answer",
    )

    monkeypatch.setattr(
        utils,
        "evaluate",
        lambda **kwargs: {
            "context_relevance": 0.8,
            "groundedness": 0.7,
            "answer_relevance": 0.6,
            "overall": 0.7,
        },
    )
    monkeypatch.setattr(utils, "IndicEmbedder", lambda language: object())

    class FakeMetric:
        def __init__(self, language, embedder):
            self.language = language
            self.embedder = embedder

    class FakeContextRelevance(FakeMetric):
        def score(self, question, contexts):
            return 0.8

    class FakeGroundedness(FakeMetric):
        def score(self, answer, contexts):
            return 0.7

    class FakeAnswerRelevance(FakeMetric):
        def score(self, question, answer):
            return 0.6

    monkeypatch.setattr(utils, "ContextRelevance", FakeContextRelevance)
    monkeypatch.setattr(utils, "Groundedness", FakeGroundedness)
    monkeypatch.setattr(utils, "AnswerRelevance", FakeAnswerRelevance)

    report = utils.run_evaluation([sample], "english")

    assert report.summary.overall == 0.7
    assert report.samples[0].scores.overall == 0.7
