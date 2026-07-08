"""
Visualization — matplotlib charts for benchmark results.

Generates:
1. Grouped bar chart of RAG triad metrics by language (correct vs hallucinated).
2. Runtime/latency comparison bar chart (BharatRAG vs RAGAS/DeepEval).

Charts are saved as static PNGs under ``docs/benchmarks/assets/``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from bharatrag.benchmarking.models import BenchmarkResults

logger = logging.getLogger(__name__)


def _ensure_matplotlib():
    """Import and return matplotlib.pyplot, or raise a clear error."""
    try:
        import matplotlib  # type: ignore[import-untyped]

        matplotlib.use("Agg")  # Non-interactive backend for PNG export.
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install it with: pip install bharatrag[benchmarks]"
        )


def plot_triad_metrics(
    results: BenchmarkResults,
    output_dir: str = "docs/benchmarks/assets",
) -> str:
    """Generate a grouped bar chart of RAG triad metrics by language.

    Creates side-by-side bars for correct and hallucinated answers for
    each metric (Context Relevance, Groundedness, Answer Relevance).

    Args:
        results: The benchmark results to visualize.
        output_dir: Directory to save the PNG file.

    Returns:
        Absolute path to the saved PNG.
    """
    plt = _ensure_matplotlib()
    import numpy as np  # Already a core dependency.

    os.makedirs(output_dir, exist_ok=True)

    metrics = ["context_relevance", "groundedness", "answer_relevance"]
    metric_labels = ["Context Relevance", "Groundedness", "Answer Relevance"]

    # Collect data per dataset:language
    group_labels = []
    correct_vals: dict[str, list[float]] = {m: [] for m in metrics}
    halluc_vals: dict[str, list[float]] = {m: [] for m in metrics}

    for ds_result in results.dataset_results:
        label = f"{ds_result.dataset_name}\n({ds_result.language})"
        group_labels.append(label)
        for m in metrics:
            correct_vals[m].append(
                ds_result.correct_aggregate.get(m, 0.0)
            )
            halluc_vals[m].append(
                ds_result.hallucinated_aggregate.get(m, 0.0)
            )

    if not group_labels:
        logger.warning("No results to plot for triad metrics")
        return ""

    n_groups = len(group_labels)
    n_metrics = len(metrics)
    bar_width = 0.12
    x = np.arange(n_groups)

    # Color palette — professional, accessible.
    correct_colors = ["#2ecc71", "#3498db", "#9b59b6"]
    halluc_colors = ["#e74c3c", "#e67e22", "#f39c12"]

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 3), 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        offset_correct = (i * 2) * bar_width
        offset_halluc = (i * 2 + 1) * bar_width
        ax.bar(
            x + offset_correct,
            correct_vals[m],
            bar_width,
            label=f"{label} (correct)",
            color=correct_colors[i],
            alpha=0.9,
        )
        ax.bar(
            x + offset_halluc,
            halluc_vals[m],
            bar_width,
            label=f"{label} (hallucinated)",
            color=halluc_colors[i],
            alpha=0.9,
        )

    ax.set_xlabel("Dataset (Language)", color="white", fontsize=12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title(
        "BharatRAG — RAG Triad Metrics by Language",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + bar_width * (n_metrics - 0.5))
    ax.set_xticklabels(group_labels, color="white", fontsize=10)
    ax.tick_params(axis="y", colors="white")
    ax.set_ylim(0, 1.15)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8,
        facecolor="#1a1a2e",
        edgecolor="white",
        labelcolor="white",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

    output_path = os.path.join(output_dir, "triad_metrics.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved triad metrics chart to %s", output_path)
    return os.path.abspath(output_path)


def plot_latency_comparison(
    results: BenchmarkResults,
    output_dir: str = "docs/benchmarks/assets",
) -> str:
    """Generate a bar chart comparing evaluation latency across frameworks.

    Compares BharatRAG wall-clock time against RAGAS and DeepEval (where
    available) per dataset.

    Args:
        results: The benchmark results to visualize.
        output_dir: Directory to save the PNG file.

    Returns:
        Absolute path to the saved PNG.
    """
    plt = _ensure_matplotlib()
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    group_labels = []
    bharatrag_times: list[float] = []
    ragas_times: list[Optional[float]] = []
    deepeval_times: list[Optional[float]] = []

    for ds_result in results.dataset_results:
        key = f"{ds_result.dataset_name}:{ds_result.language}"
        label = f"{ds_result.dataset_name}\n({ds_result.language})"
        group_labels.append(label)
        bharatrag_times.append(ds_result.total_wall_clock_s)

        # RAGAS timing
        if results.ragas_results and key in results.ragas_results:
            ragas_total = sum(
                r.get("wall_clock_seconds", 0.0)
                for r in results.ragas_results[key]
            )
            ragas_times.append(ragas_total)
        else:
            ragas_times.append(None)

        # DeepEval timing
        if results.deepeval_results and key in results.deepeval_results:
            de_total = sum(
                r.get("wall_clock_seconds", 0.0)
                for r in results.deepeval_results[key]
            )
            deepeval_times.append(de_total)
        else:
            deepeval_times.append(None)

    if not group_labels:
        logger.warning("No results to plot for latency comparison")
        return ""

    n_groups = len(group_labels)
    bar_width = 0.25
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2.5), 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.bar(x, bharatrag_times, bar_width, label="BharatRAG", color="#2ecc71")

    if any(t is not None for t in ragas_times):
        ragas_plot = [t if t is not None else 0 for t in ragas_times]
        ax.bar(
            x + bar_width, ragas_plot, bar_width,
            label="RAGAS", color="#3498db",
        )

    if any(t is not None for t in deepeval_times):
        de_plot = [t if t is not None else 0 for t in deepeval_times]
        ax.bar(
            x + 2 * bar_width, de_plot, bar_width,
            label="DeepEval", color="#e74c3c",
        )

    ax.set_xlabel("Dataset (Language)", color="white", fontsize=12)
    ax.set_ylabel("Wall-Clock Time (seconds)", color="white", fontsize=12)
    ax.set_title(
        "Evaluation Latency Comparison",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(group_labels, color="white", fontsize=10)
    ax.tick_params(axis="y", colors="white")
    ax.legend(
        facecolor="#1a1a2e",
        edgecolor="white",
        labelcolor="white",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")

    output_path = os.path.join(output_dir, "latency_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved latency comparison chart to %s", output_path)
    return os.path.abspath(output_path)
