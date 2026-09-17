import gc
import time

import numpy as np
import psutil

from srlib.constants import (
    PROFILE_REPEATS,
    PROFILE_WARMUP,
)

# ---------------------------------------------------- #
# Cost of one call, shared by the classic and DL paths #
# ---------------------------------------------------- #
def profile_algorithm(
        func, *args, repeats=PROFILE_REPEATS, warmup=PROFILE_WARMUP, **kwargs):
    """Profile wall-clock time and memory of a callable in a single pass.

    Time is the median of ``repeats`` timed runs after ``warmup`` untimed
    ones, so the first-call cache misses do not bias the result and a
    single outlier does not carry the estimate.

    Memory is the growth of the process resident set across the timed
    runs. The classic algorithms are OpenCV calls that allocate their
    buffers in C++, outside the Python allocator, so a Python-level
    tracer cannot see them.

    Parameters
    ----------
    func : callable
        Algorithm to profile.
    repeats : int
        Timed executions. Must be at least 1.
    warmup : int
        Untimed executions run first.

    Returns
    -------
    tuple
        ``(result, elapsed_seconds, memory_bytes)`` where result comes
        from the last timed run.
    """

    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}.")

    for _ in range(warmup):
        func(*args, **kwargs)

    process = psutil.Process()
    gc.collect()
    rss_before = process.memory_info().rss

    result = None
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        timings.append(time.perf_counter() - start)

    rss_after = process.memory_info().rss

    return result, float(np.median(timings)), max(0, rss_after - rss_before)

# -------------------------- #
# Compute summary statistics #
# -------------------------- #
def compute_summary_stats(values):
    """Compute basic descriptive statistics for a numeric 1D array.

    Parameters
    ----------
    values : np.ndarray
        1D array (or empty array) of numeric values already cast to float.

    Returns
    -------
    dict
        Keys:
        - mean   : arithmetic mean (NaN if empty)
        - median : median (NaN if empty)
        - max    : maximum (NaN if empty)
        - std    : sample std (ddof=1) if n>1 else 0.0 (NaN if empty)
        - var    : sample variance (ddof=1) if n>1 else 0.0 (NaN if empty)
        - count  : number of elements

    Notes
    -----
    Requires at least one value: ``np.mean`` and ``np.max`` raise on an empty
    sequence. Every caller feeds it one sample per processed pair, and the
    benchmark refuses to run on an empty dataset, so the case does not arise.
    """

    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'max': float(np.max(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'var': float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        'count': int(len(values))
    }
