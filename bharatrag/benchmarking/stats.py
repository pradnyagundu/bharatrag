"""
Statistical summaries for benchmark results.

Computes mean, standard deviation, 95 % confidence intervals per
metric/language/dataset, and Pearson + Spearman correlations between
BharatRAG's groundedness and RAGAS's faithfulness where both are
available.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StatsSummary:
    """Descriptive statistics for a list of scores.

    Attributes:
        mean: Arithmetic mean.
        std: Sample standard deviation.
        ci_95_lower: Lower bound of the 95 % confidence interval.
        ci_95_upper: Upper bound of the 95 % confidence interval.
        n: Number of observations.
    """

    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    n: int


def compute_summary(scores: list[float]) -> StatsSummary:
    """Compute descriptive statistics for a list of scores.

    Uses a normal-approximation 95 % CI (z = 1.96).  Falls back to
    ``scipy.stats.t`` if the sample size is small (n < 30) and scipy
    is available; otherwise uses the normal approximation regardless.

    Args:
        scores: List of numeric scores.

    Returns:
        A ``StatsSummary`` dataclass.

    Raises:
        ValueError: If the list is empty.

    Example:
        >>> s = compute_summary([0.8, 0.85, 0.9, 0.75])
        >>> round(s.mean, 2)
        0.82
    """
    if not scores:
        raise ValueError("Cannot compute statistics on an empty list")

    n = len(scores)
    mean = sum(scores) / n

    if n == 1:
        return StatsSummary(
            mean=mean, std=0.0, ci_95_lower=mean, ci_95_upper=mean, n=1
        )

    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    std = math.sqrt(variance)
    se = std / math.sqrt(n)

    # Try to use t-distribution for small samples.
    z = 1.96
    try:
        from scipy.stats import t as t_dist  # type: ignore[import-untyped]

        z = float(t_dist.ppf(0.975, df=n - 1))
    except ImportError:
        if n < 30:
            logger.debug(
                "scipy not available; using z=1.96 even for n=%d", n
            )

    ci_lower = mean - z * se
    ci_upper = mean + z * se

    return StatsSummary(
        mean=round(mean, 6),
        std=round(std, 6),
        ci_95_lower=round(ci_lower, 6),
        ci_95_upper=round(ci_upper, 6),
        n=n,
    )


def compute_correlations(
    bharatrag_scores: list[float],
    ragas_scores: list[float],
) -> Optional[dict[str, Any]]:
    """Compute Pearson and Spearman correlations between score vectors.

    Intended to compare BharatRAG groundedness against RAGAS faithfulness
    on the same set of examples.

    Args:
        bharatrag_scores: BharatRAG metric scores (one per example).
        ragas_scores: RAGAS metric scores (one per example, same order).

    Returns:
        Dict with ``pearson_r``, ``pearson_p``, ``spearman_r``,
        ``spearman_p``, and ``n``.  Returns ``None`` if scipy is not
        installed or if there are fewer than 3 paired observations.

    Example:
        >>> corr = compute_correlations([0.8, 0.6, 0.9], [0.75, 0.55, 0.85])
        >>> corr is not None
        True
    """
    if len(bharatrag_scores) < 3 or len(ragas_scores) < 3:
        logger.warning(
            "Need at least 3 paired observations for correlation; got %d",
            min(len(bharatrag_scores), len(ragas_scores)),
        )
        return None

    n = min(len(bharatrag_scores), len(ragas_scores))
    x = bharatrag_scores[:n]
    y = ragas_scores[:n]

    try:
        from scipy.stats import (  # type: ignore[import-untyped]
            pearsonr,
            spearmanr,
        )
    except ImportError:
        logger.warning(
            "scipy is required for correlation analysis. "
            "Install it with: pip install bharatrag[benchmarks]"
        )
        return None

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    return {
        "pearson_r": round(float(pr), 4),
        "pearson_p": round(float(pp), 6),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 6),
        "n": n,
    }
