"""
Instrumentation — wall-clock latency and peak-memory wrappers.

Provides context managers that measure ``time.perf_counter`` and
``tracemalloc`` around evaluation calls.  If ``tracemalloc`` is
unavailable (e.g. on a restricted interpreter), memory tracking is
silently skipped and ``None`` is returned for memory values.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# Try to import tracemalloc once at module level.
try:
    import tracemalloc as _tracemalloc

    _HAS_TRACEMALLOC = True
except ImportError:
    _tracemalloc = None  # type: ignore[assignment]
    _HAS_TRACEMALLOC = False
    logger.info(
        "tracemalloc is not available; memory tracking will be disabled"
    )


@dataclasses.dataclass
class InstrumentedResult:
    """Result wrapper that bundles a return value with resource measurements.

    Attributes:
        result: The original return value of the instrumented callable.
        wall_clock_seconds: Wall-clock time in seconds.
        peak_memory_bytes: Peak *additional* memory allocated during the
            call, or ``None`` if tracemalloc was unavailable.
    """

    result: Any
    wall_clock_seconds: float
    peak_memory_bytes: Optional[int] = None


@contextlib.contextmanager
def timed() -> Generator[dict[str, float], None, None]:
    """Context manager that records wall-clock time.

    Yields a mutable dict; after the block exits, ``elapsed_s`` is set.

    Example:
        >>> with timed() as t:
        ...     result = expensive_function()
        >>> print(t["elapsed_s"])
    """
    timing: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["elapsed_s"] = time.perf_counter() - start


@contextlib.contextmanager
def memory_tracked() -> Generator[dict[str, Optional[int]], None, None]:
    """Context manager that records peak memory delta via tracemalloc.

    If ``tracemalloc`` is not available, ``peak_bytes`` will be ``None``.

    Example:
        >>> with memory_tracked() as mem:
        ...     result = expensive_function()
        >>> print(mem["peak_bytes"])
    """
    mem: dict[str, Optional[int]] = {"peak_bytes": None}
    if not _HAS_TRACEMALLOC:
        yield mem
        return

    already_tracing = _tracemalloc.is_tracing()
    if not already_tracing:
        _tracemalloc.start()

    snapshot_before = _tracemalloc.take_snapshot()
    try:
        yield mem
    finally:
        _, peak = _tracemalloc.get_traced_memory()
        # Peak is absolute; we want delta from start.
        stats_before = sum(
            s.size for s in snapshot_before.statistics("filename")
        )
        mem["peak_bytes"] = max(0, peak - stats_before)
        if not already_tracing:
            _tracemalloc.stop()


def run_instrumented(fn: Any, *args: Any, **kwargs: Any) -> InstrumentedResult:
    """Call ``fn(*args, **kwargs)`` with timing and memory instrumentation.

    Args:
        fn: Callable to instrument.
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        An ``InstrumentedResult`` containing the return value and
        resource measurements.
    """
    with timed() as t, memory_tracked() as mem:
        result = fn(*args, **kwargs)
    return InstrumentedResult(
        result=result,
        wall_clock_seconds=t["elapsed_s"],
        peak_memory_bytes=mem["peak_bytes"],
    )
