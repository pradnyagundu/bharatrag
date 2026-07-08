"""
RAGAS adapter for benchmark comparison.

RAGAS uses an LLM judge (typically GPT-4 or similar) to evaluate RAG
quality.  This means:
  - Scores are **non-deterministic** — different runs may yield different
    results even on the same data.
  - An **API key** is required (``OPENAI_API_KEY`` by default).
  - Each evaluation call **costs money** — proportional to the number of
    examples and the token cost of the chosen model.

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


class RagasAdapter:
    """Adapter that runs RAGAS evaluation for comparison with BharatRAG.

    Raises:
        ImportError: If ``ragas`` is not installed.
        EnvironmentError: If ``OPENAI_API_KEY`` is not set.
    """

    def __init__(self) -> None:
        # ── Check dependency ─────────────────────────────────────
        try:
            import ragas  # noqa: F401  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "RAGAS is not installed.  Install it with:\n"
                "    pip install bharatrag[ragas]\n"
                "RAGAS requires an LLM API key (see below)."
            )

        # ── Check API key ────────────────────────────────────────
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "RAGAS requires an OpenAI API key for its LLM judge.\n"
                "Set the OPENAI_API_KEY environment variable:\n"
                "    export OPENAI_API_KEY='sk-...'\n"
                "Note: RAGAS evaluation costs money and is non-deterministic."
            )

        logger.info("RAGAS adapter initialised (LLM-based, non-deterministic)")

    def evaluate(
        self, examples: list[BenchmarkExample]
    ) -> list[dict[str, Any]]:
        """Run RAGAS metrics on a list of benchmark examples.

        Args:
            examples: Normalized benchmark examples to evaluate.

        Returns:
            List of dicts, one per example, with RAGAS metric scores
            and timing information.
        """
        from ragas import evaluate as ragas_evaluate  # type: ignore[import-untyped]
        from ragas.metrics import (  # type: ignore[import-untyped]
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        try:
            from datasets import Dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for RAGAS evaluation. "
                "Install it with: pip install bharatrag[benchmarks]"
            )

        # Build a HuggingFace Dataset from our examples.
        data = {
            "question": [e.question for e in examples],
            "answer": [e.ground_truth_answer for e in examples],
            "contexts": [e.contexts for e in examples],
            "ground_truth": [e.ground_truth_answer for e in examples],
        }
        hf_dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation on %d examples...", len(examples))
        instrumented = run_instrumented(
            ragas_evaluate,
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        ragas_result = instrumented.result
        scores_df = ragas_result.to_pandas()

        per_example: list[dict[str, Any]] = []
        for idx in range(len(examples)):
            row = scores_df.iloc[idx] if idx < len(scores_df) else {}
            per_example.append(
                {
                    "example_id": examples[idx].example_id,
                    "faithfulness": float(row.get("faithfulness", 0.0)),
                    "answer_relevancy": float(
                        row.get("answer_relevancy", 0.0)
                    ),
                    "context_precision": float(
                        row.get("context_precision", 0.0)
                    ),
                    "wall_clock_seconds": (
                        instrumented.wall_clock_seconds / len(examples)
                    ),
                    # NOTE: RAGAS scores are LLM-judge-based and
                    # non-deterministic. They should not be directly
                    # compared to BharatRAG's embedding-similarity scores.
                    "_methodology": "llm_judge",
                }
            )

        logger.info(
            "RAGAS evaluation complete (%.1fs total)",
            instrumented.wall_clock_seconds,
        )
        return per_example
