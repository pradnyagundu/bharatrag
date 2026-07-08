"""
DeepEval adapter for benchmark comparison.

DeepEval uses an LLM judge to evaluate RAG quality.  This means:
  - Scores are **non-deterministic**.
  - An **API key** is required (``OPENAI_API_KEY`` by default).
  - Each evaluation call **costs money**.

By contrast, BharatRAG's metrics use embedding cosine similarity — they
are **free, offline, and deterministic**.  The two approaches measure
different things and should NOT be treated as directly interchangeable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from bharatrag.benchmarking.instrumentation import run_instrumented
from bharatrag.benchmarking.models import BenchmarkExample

logger = logging.getLogger(__name__)


class DeepEvalAdapter:
    """Adapter that runs DeepEval evaluation for comparison with BharatRAG.

    Raises:
        ImportError: If ``deepeval`` is not installed.
        EnvironmentError: If ``OPENAI_API_KEY`` is not set.
    """

    def __init__(self) -> None:
        # ── Check dependency ─────────────────────────────────────
        try:
            import deepeval  # noqa: F401  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "DeepEval is not installed.  Install it with:\n"
                "    pip install bharatrag[deepeval]\n"
                "DeepEval requires an LLM API key (see below)."
            )

        # ── Check API key ────────────────────────────────────────
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "DeepEval requires an OpenAI API key for its LLM judge.\n"
                "Set the OPENAI_API_KEY environment variable:\n"
                "    export OPENAI_API_KEY='sk-...'\n"
                "Note: DeepEval evaluation costs money and is "
                "non-deterministic."
            )

        logger.info(
            "DeepEval adapter initialised (LLM-based, non-deterministic)"
        )

    def _evaluate_single(
        self, example: BenchmarkExample
    ) -> dict[str, Any]:
        """Evaluate a single example with DeepEval metrics.

        Args:
            example: A single benchmark example.

        Returns:
            Dict with metric scores for this example.
        """
        from deepeval.metrics import (  # type: ignore[import-untyped]
            AnswerRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase  # type: ignore[import-untyped]

        test_case = LLMTestCase(
            input=example.question,
            actual_output=example.ground_truth_answer,
            retrieval_context=example.contexts,
        )

        faithfulness = FaithfulnessMetric()
        answer_rel = AnswerRelevancyMetric()

        faithfulness.measure(test_case)
        answer_rel.measure(test_case)

        return {
            "example_id": example.example_id,
            "faithfulness": faithfulness.score,
            "answer_relevancy": answer_rel.score,
            # NOTE: DeepEval scores are LLM-judge-based and
            # non-deterministic. They should not be directly
            # compared to BharatRAG's embedding-similarity scores.
            "_methodology": "llm_judge",
        }

    def evaluate(
        self, examples: list[BenchmarkExample]
    ) -> list[dict[str, Any]]:
        """Run DeepEval metrics on a list of benchmark examples.

        Args:
            examples: Normalized benchmark examples to evaluate.

        Returns:
            List of dicts, one per example, with DeepEval metric scores
            and timing information.
        """
        logger.info(
            "Running DeepEval evaluation on %d examples...", len(examples)
        )

        per_example: list[dict[str, Any]] = []
        for ex in examples:
            instrumented = run_instrumented(self._evaluate_single, ex)
            result = instrumented.result
            result["wall_clock_seconds"] = instrumented.wall_clock_seconds
            per_example.append(result)

        total_time = sum(r["wall_clock_seconds"] for r in per_example)
        logger.info(
            "DeepEval evaluation complete (%.1fs total)", total_time
        )
        return per_example
