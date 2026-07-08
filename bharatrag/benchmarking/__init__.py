"""
BharatRAG Benchmarking — reproducible evaluation pipeline for Indic QA datasets.

Provides dataset loaders, a benchmark runner, instrumentation, statistical
summaries, visualizations, and an auto-generated Markdown report.
"""

from bharatrag.benchmarking.models import BenchmarkExample
from bharatrag.benchmarking.corruption import corrupt_answer
from bharatrag.benchmarking.runner import BenchmarkRunner

__all__ = [
    "BenchmarkExample",
    "BenchmarkRunner",
    "corrupt_answer",
]
