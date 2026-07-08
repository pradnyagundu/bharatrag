#!/usr/bin/env python3
"""
BharatRAG Public Dataset Benchmark Runner (CLI)

Runs the full benchmarking pipeline end to end:
  load datasets → evaluate → instrument → compare → stats → plot → report

Usage:
    python scripts/benchmark_public_datasets.py \\
        --datasets indicqa xquad \\
        --languages hindi tamil \\
        --max-examples 100 \\
        --include-ragas \\
        --include-deepeval \\
        --output docs/benchmarks/latest_report.md

Dependencies:
    pip install bharatrag[benchmarks]
    # For RAGAS/DeepEval comparison (optional):
    pip install bharatrag[ragas]     # + set OPENAI_API_KEY
    pip install bharatrag[deepeval]  # + set OPENAI_API_KEY
"""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BharatRAG — Benchmark against public Indic QA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["indicqa", "xquad"],
        choices=["indicqa", "xquad"],
        help="Datasets to benchmark (default: indicqa xquad)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["hindi"],
        help="Languages to evaluate (default: hindi)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Max examples per dataset×language (default: 100)",
    )
    parser.add_argument(
        "--corruption-seed",
        type=int,
        default=42,
        help="Random seed for hallucinated answer generation (default: 42)",
    )
    parser.add_argument(
        "--include-ragas",
        action="store_true",
        help="Also run RAGAS evaluation (requires pip install bharatrag[ragas] and OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--include-deepeval",
        action="store_true",
        help="Also run DeepEval evaluation (requires pip install bharatrag[deepeval] and OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmarks/latest_report.md",
        help="Output path for the Markdown report (default: docs/benchmarks/latest_report.md)",
    )
    parser.add_argument(
        "--chart-dir",
        default="docs/benchmarks/assets",
        help="Directory for chart PNGs (default: docs/benchmarks/assets)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # ── Configure logging ────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("bharatrag.benchmarking")

    # ── Lazy imports so the script starts fast even if deps are
    #    missing (gives a clear error instead of a traceback) ─────
    try:
        from bharatrag.benchmarking.runner import BenchmarkRunner
        from bharatrag.benchmarking.plots import (
            plot_latency_comparison,
            plot_triad_metrics,
        )
        from bharatrag.benchmarking.report import generate_report
    except ImportError as exc:
        print(
            f"Error: {exc}\n\n"
            "Install benchmark dependencies with:\n"
            "    pip install bharatrag[benchmarks]",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Run pipeline ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BharatRAG Public Dataset Benchmark")
    logger.info("=" * 60)
    logger.info("Datasets:      %s", args.datasets)
    logger.info("Languages:     %s", args.languages)
    logger.info("Max examples:  %d", args.max_examples)
    logger.info("Include RAGAS: %s", args.include_ragas)
    logger.info("Include DeepEval: %s", args.include_deepeval)

    runner = BenchmarkRunner(
        datasets=args.datasets,
        languages=args.languages,
        max_examples=args.max_examples,
        corruption_seed=args.corruption_seed,
    )

    results = runner.run(
        include_ragas=args.include_ragas,
        include_deepeval=args.include_deepeval,
    )

    # ── Generate charts ──────────────────────────────────────────
    logger.info("Generating charts...")
    try:
        plot_triad_metrics(results, output_dir=args.chart_dir)
        plot_latency_comparison(results, output_dir=args.chart_dir)
    except ImportError as exc:
        logger.warning("Chart generation skipped: %s", exc)

    # ── Generate report ──────────────────────────────────────────
    report_path = generate_report(
        results,
        output_path=args.output,
        chart_dir=args.chart_dir,
    )
    logger.info("Report written to %s", report_path)

    # ── Print summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    for ds_result in results.dataset_results:
        print(f"\n{ds_result.dataset_name} ({ds_result.language}):")
        print(f"  ✅ Correct:      {ds_result.correct_aggregate}")
        print(f"  ❌ Hallucinated: {ds_result.hallucinated_aggregate}")
        print(f"  ⏱  Time: {ds_result.total_wall_clock_s:.2f}s")
    print(f"\n📊 Report: {report_path}")


if __name__ == "__main__":
    main()
