"""
Markdown report generator for benchmark results.

Produces a GitHub-renderable Markdown file with statistical tables,
embedded chart images, and methodology notes.
"""

from __future__ import annotations

import datetime
import logging
import os
from bharatrag.benchmarking.models import BenchmarkResults

logger = logging.getLogger(__name__)


def generate_report(
    results: BenchmarkResults,
    output_path: str = "docs/benchmarks/latest_report.md",
    chart_dir: str = "docs/benchmarks/assets",
) -> str:
    """Generate a Markdown benchmark report.

    Args:
        results: The benchmark results to report.
        output_path: Path for the output Markdown file.
        chart_dir: Directory containing chart PNGs (relative paths
            are used in the Markdown so it renders on GitHub).

    Returns:
        Absolute path to the generated report.

    Example:
        >>> generate_report(results)
        '/path/to/docs/benchmarks/latest_report.md'
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────
    lines.append("# BharatRAG Benchmark Report")
    lines.append("")
    lines.append(
        f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    )
    lines.append("")

    # ── Methodology note ─────────────────────────────────────────
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "BharatRAG evaluates RAG quality using **embedding cosine similarity** "
        "with language-specific sentence-transformer models.  This approach is:"
    )
    lines.append("")
    lines.append("- ✅ **Free** — no API key or paid LLM calls")
    lines.append("- ✅ **Offline** — runs entirely on the local machine")
    lines.append("- ✅ **Deterministic** — same input always produces the same score")
    lines.append("")
    lines.append(
        "RAGAS and DeepEval (where included) use an **LLM judge** "
        "(typically GPT-4).  Their scores are:"
    )
    lines.append("")
    lines.append("- ❌ **Paid** — each call costs money via the OpenAI API")
    lines.append("- ❌ **Non-deterministic** — scores vary between runs")
    lines.append("- ❌ **Online** — requires internet and an API key")
    lines.append("")
    lines.append(
        "> **⚠️ These two approaches measure different things and should "
        "NOT be treated as directly interchangeable.**  BharatRAG measures "
        "semantic similarity; RAGAS/DeepEval perform LLM-based reasoning."
    )
    lines.append("")

    # ── Summary table ────────────────────────────────────────────
    lines.append("## Results Summary")
    lines.append("")
    lines.append(
        "| Dataset | Language | Metric | Correct | Hallucinated |"
    )
    lines.append("|---------|----------|--------|---------|--------------|")

    metrics = ["context_relevance", "groundedness", "answer_relevance", "overall"]
    for ds_result in results.dataset_results:
        for m in metrics:
            correct_val = ds_result.correct_aggregate.get(m, "—")
            halluc_val = ds_result.hallucinated_aggregate.get(m, "—")
            lines.append(
                f"| {ds_result.dataset_name} | {ds_result.language} "
                f"| {m} | {correct_val} | {halluc_val} |"
            )

    lines.append("")

    # ── Timing summary ───────────────────────────────────────────
    lines.append("## Runtime Performance")
    lines.append("")
    lines.append("| Dataset | Language | Total Time (s) | Peak Memory |")
    lines.append("|---------|----------|---------------|-------------|")

    for ds_result in results.dataset_results:
        mem_str = (
            f"{ds_result.total_peak_memory_bytes / 1024 / 1024:.1f} MB"
            if ds_result.total_peak_memory_bytes
            else "N/A"
        )
        lines.append(
            f"| {ds_result.dataset_name} | {ds_result.language} "
            f"| {ds_result.total_wall_clock_s:.2f} | {mem_str} |"
        )

    lines.append("")

    # ── Charts ───────────────────────────────────────────────────
    # Use relative paths so they render on GitHub.
    triad_path = os.path.join("assets", "triad_metrics.png")
    latency_path = os.path.join("assets", "latency_comparison.png")

    triad_abs = os.path.join(chart_dir, "triad_metrics.png")
    latency_abs = os.path.join(chart_dir, "latency_comparison.png")

    if os.path.exists(triad_abs):
        lines.append("## RAG Triad Metrics")
        lines.append("")
        lines.append(f"![Triad Metrics]({triad_path})")
        lines.append("")

    if os.path.exists(latency_abs):
        lines.append("## Latency Comparison")
        lines.append("")
        lines.append(f"![Latency Comparison]({latency_path})")
        lines.append("")

    # ── RAGAS comparison ─────────────────────────────────────────
    if results.ragas_results:
        lines.append("## RAGAS Comparison")
        lines.append("")
        lines.append(
            "> Note: RAGAS scores are LLM-judge-based (non-deterministic, "
            "paid).  Shown for reference only."
        )
        lines.append("")
        lines.append(
            "| Dataset:Language | Faithfulness (mean) | "
            "Answer Relevancy (mean) | Context Precision (mean) |"
        )
        lines.append(
            "|-----------------|--------------------|-"
            "----------------------|------------------------|"
        )
        for key, scores_list in results.ragas_results.items():
            if not scores_list:
                continue
            faith = [s.get("faithfulness", 0) for s in scores_list]
            ans_rel = [s.get("answer_relevancy", 0) for s in scores_list]
            ctx_prec = [s.get("context_precision", 0) for s in scores_list]
            lines.append(
                f"| {key} "
                f"| {sum(faith)/len(faith):.4f} "
                f"| {sum(ans_rel)/len(ans_rel):.4f} "
                f"| {sum(ctx_prec)/len(ctx_prec):.4f} |"
            )
        lines.append("")

    # ── DeepEval comparison ──────────────────────────────────────
    if results.deepeval_results:
        lines.append("## DeepEval Comparison")
        lines.append("")
        lines.append(
            "> Note: DeepEval scores are LLM-judge-based "
            "(non-deterministic, paid).  Shown for reference only."
        )
        lines.append("")
        lines.append(
            "| Dataset:Language | Faithfulness (mean) | "
            "Answer Relevancy (mean) |"
        )
        lines.append(
            "|-----------------|--------------------|-"
            "----------------------|"
        )
        for key, scores_list in results.deepeval_results.items():
            if not scores_list:
                continue
            faith = [s.get("faithfulness", 0) for s in scores_list]
            ans_rel = [s.get("answer_relevancy", 0) for s in scores_list]
            lines.append(
                f"| {key} "
                f"| {sum(faith)/len(faith):.4f} "
                f"| {sum(ans_rel)/len(ans_rel):.4f} |"
            )
        lines.append("")

    # ── Footer ───────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        "*Report generated by "
        "[BharatRAG](https://github.com/pradnyagundu/bharatrag) "
        "benchmarking pipeline.*"
    )
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    abs_path = os.path.abspath(output_path)
    logger.info("Report written to %s", abs_path)
    return abs_path
