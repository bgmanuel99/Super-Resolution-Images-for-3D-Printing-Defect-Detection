import time
import tracemalloc

def time_algorithm(func, *args, **kwargs):
    """Return (result, elapsed_seconds) for the function.

    Only the body execution time is measured.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def memory_algorithm(func, *args, **kwargs):
    """Return result measuring peak memory for the function body.

    The peak in bytes is computed via tracemalloc (discarded here).
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak