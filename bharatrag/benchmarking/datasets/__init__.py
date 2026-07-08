"""
Dataset loaders for the benchmarking pipeline.

Each loader normalizes its source into ``list[BenchmarkExample]``.
"""

from bharatrag.benchmarking.datasets.indicqa import load_indicqa
from bharatrag.benchmarking.datasets.xquad import load_xquad

__all__ = ["load_indicqa", "load_xquad"]
