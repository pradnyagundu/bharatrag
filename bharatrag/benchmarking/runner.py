"""
Benchmark runner — drives BharatRAG evaluation over loaded datasets.

Generalizes the pattern from ``examples/run_benchmark.py``: for each
dataset × language combination, evaluates both correct and hallucinated
answers, wraps each call with instrumentation, and optionally invokes
RAGAS / DeepEval adapters.
"""

from __future__ import annotations

import logging
from typing import Optional

from bharatrag import evaluate
from bharatrag.benchmarking.datasets import load_indicqa, load_xquad
from bharatrag.benchmarking.instrumentation import run_instrumented
from bharatrag.benchmarking.models import (
    BenchmarkExample,
    BenchmarkResults,
    DatasetResults,
    QuestionResult,
)

logger = logging.getLogger(__name__)

# Registry mapping dataset names to their loader functions.
_DATASET_LOADERS = {
    "indicqa": load_indicqa,
    "xquad": load_xquad,
}


class BenchmarkRunner:
    """Runs BharatRAG evaluation over public Indic QA datasets.

    This is the main entry point for the benchmarking pipeline.  It loads
    datasets, evaluates correct and hallucinated answers, records timing
    and memory, and optionally runs RAGAS/DeepEval for comparison.

    Args:
        datasets: Dataset names to evaluate (e.g. ``["indicqa", "xquad"]``).
        languages: Languages to evaluate (e.g. ``["hindi", "tamil"]``).
        max_examples: Maximum examples per dataset × language.
        corruption_seed: Seed for hallucinated-answer generation.

    Example:
        >>> runner = BenchmarkRunner(
        ...     datasets=["indicqa"],
        ...     languages=["hindi"],
        ...     max_examples=10,
        ... )
        >>> results = runner.run()
        >>> print(results.dataset_results[0].correct_aggregate)
    """

    def __init__(
        self,
        datasets: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
        max_examples: int = 100,
        corruption_seed: int = 42,
    ) -> None:
        self.datasets = datasets or ["indicqa", "xquad"]
        self.languages = languages or ["hindi"]
        self.max_examples = max_examples
        self.corruption_seed = corruption_seed

    def load_data(self) -> dict[str, list[BenchmarkExample]]:
        """Load all configured datasets, keyed by ``"dataset:language"``.

        Returns:
            Dict mapping ``"dataset:language"`` to lists of examples.

        Raises:
            ValueError: If an unknown dataset name is requested.
        """
        all_examples: dict[str, list[BenchmarkExample]] = {}
        for ds_name in self.datasets:
            loader = _DATASET_LOADERS.get(ds_name)
            if loader is None:
                raise ValueError(
                    f"Unknown dataset '{ds_name}'. "
                    f"Choose from: {list(_DATASET_LOADERS.keys())}"
                )
            for lang in self.languages:
                key = f"{ds_name}:{lang}"
                try:
                    examples = loader(
                        language=lang,
                        max_examples=self.max_examples,
                        corruption_seed=self.corruption_seed,
                    )
                    if not examples:
                        # Nothing usable came back (e.g. every row was
                        # filtered out for missing answers/context). Skip
                        # rather than pass an empty batch downstream: that
                        # would divide by zero in _evaluate_examples() and
                        # evaluate() itself rejects zero-length input.
                        logger.warning(
                            "Skipping %s: loader returned 0 usable examples",
                            key,
                        )
                        continue
                    all_examples[key] = examples
                    logger.info(
                        "Loaded %d examples for %s", len(examples), key
                    )
                except ValueError as exc:
                    logger.warning(
                        "Skipping %s: %s", key, exc
                    )
        return all_examples

    def _evaluate_examples(
        self,
        examples: list[BenchmarkExample],
        language: str,
        use_hallucinated: bool = False,
    ) -> list[QuestionResult]:
        """Evaluate a list of examples and return per-question results.

        Args:
            examples: The benchmark examples to evaluate.
            language: Language for BharatRAG's embedder.
            use_hallucinated: If ``True``, evaluate hallucinated answers
                instead of ground-truth answers.

        Returns:
            List of ``QuestionResult`` with scores and timing filled in.
        """
        questions = [e.question for e in examples]
        contexts = [e.contexts for e in examples]
        answers = [
            e.hallucinated_answer if use_hallucinated
            else e.ground_truth_answer
            for e in examples
        ]

        instrumented = run_instrumented(
            evaluate,
            questions=questions,
            contexts=contexts,
            answers=answers,
            language=language,
        )

        scores = instrumented.result
        # BharatRAG.evaluate() returns aggregate scores, not per-question.
        # We store them as batch-level aggregates.
        return scores, instrumented

    def run(
        self,
        include_ragas: bool = False,
        include_deepeval: bool = False,
    ) -> BenchmarkResults:
        """Run the full benchmark pipeline.

        Args:
            include_ragas: If ``True``, also run the RAGAS adapter.
            include_deepeval: If ``True``, also run the DeepEval adapter.

        Returns:
            ``BenchmarkResults`` with all evaluation data.
        """
        all_data = self.load_data()
        results = BenchmarkResults()

        ragas_results: dict[str, list[dict]] = {}
        deepeval_results: dict[str, list[dict]] = {}

        for key, examples in all_data.items():
            ds_name, lang = key.split(":")
            logger.info(
                "Running BharatRAG evaluation: %s (%d examples)",
                key,
                len(examples),
            )

            # ── Correct answers ──────────────────────────────────
            correct_scores, correct_instr = self._evaluate_examples(
                examples, lang, use_hallucinated=False
            )

            # ── Hallucinated answers ─────────────────────────────
            halluc_scores, halluc_instr = self._evaluate_examples(
                examples, lang, use_hallucinated=True
            )

            # Build per-question results from the newly added per_question evaluate() output
            question_results = []
            for i, ex in enumerate(examples):
                q_correct_scores = correct_scores.get("per_question", [correct_scores] * len(examples))[i]
                q_halluc_scores = halluc_scores.get("per_question", [halluc_scores] * len(examples))[i]
                
                question_results.append(
                    QuestionResult(
                        example=ex,
                        correct_scores=q_correct_scores,
                        hallucinated_scores=q_halluc_scores,
                        correct_wall_clock_s=(
                            correct_instr.wall_clock_seconds / len(examples)
                        ),
                        hallucinated_wall_clock_s=(
                            halluc_instr.wall_clock_seconds / len(examples)
                        ),
                        correct_peak_memory_bytes=(
                            correct_instr.peak_memory_bytes
                        ),
                        hallucinated_peak_memory_bytes=(
                            halluc_instr.peak_memory_bytes
                        ),
                    )
                )

            ds_result = DatasetResults(
                dataset_name=ds_name,
                language=lang,
                question_results=question_results,
                correct_aggregate={
                    k: v
                    for k, v in correct_scores.items()
                    if isinstance(v, (int, float))
                },
                hallucinated_aggregate={
                    k: v
                    for k, v in halluc_scores.items()
                    if isinstance(v, (int, float))
                },
                total_wall_clock_s=(
                    correct_instr.wall_clock_seconds
                    + halluc_instr.wall_clock_seconds
                ),
                total_peak_memory_bytes=max(
                    correct_instr.peak_memory_bytes or 0,
                    halluc_instr.peak_memory_bytes or 0,
                ) or None,
            )
            results.dataset_results.append(ds_result)

            # ── Optional RAGAS comparison ────────────────────────
            if include_ragas:
                try:
                    from bharatrag.benchmarking.comparison.ragas_adapter import (
                        RagasAdapter,
                    )

                    adapter = RagasAdapter()
                    ragas_out = adapter.evaluate(examples)
                    ragas_results[key] = ragas_out
                except (ImportError, EnvironmentError) as exc:
                    logger.warning("RAGAS comparison skipped: %s", exc)

            # ── Optional DeepEval comparison ─────────────────────
            if include_deepeval:
                try:
                    from bharatrag.benchmarking.comparison.deepeval_adapter import (
                        DeepEvalAdapter,
                    )

                    adapter = DeepEvalAdapter()
                    de_out = adapter.evaluate(examples)
                    deepeval_results[key] = de_out
                except (ImportError, EnvironmentError) as exc:
                    logger.warning("DeepEval comparison skipped: %s", exc)

        if ragas_results:
            results.ragas_results = ragas_results
        if deepeval_results:
            results.deepeval_results = deepeval_results

        return results
