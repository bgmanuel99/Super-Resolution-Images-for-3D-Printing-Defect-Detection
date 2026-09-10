import time
from contextlib import contextmanager

_RULE = "=" * 64

def format_duration(seconds):
    """Render a duration so that long stages stay readable.

    Parameters
    ----------
    seconds : float
        Elapsed wall-clock time.

    Returns
    -------
    str
        ``'42.1s'`` below a minute, ``'3m 07.4s'`` above it.
    """

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(seconds, 60)

    return f"{int(minutes)}m {seconds:04.1f}s"

def step(message):
    """Print one indented detail line inside the current stage.

    Parameters
    ----------
    message : str
        Already formatted text. Callers align their own columns, since what
        is worth aligning differs from one stage to the next.
    """

    print(f"  {message}", flush=True)

@contextmanager
def stage(title):
    """Frame a long stage with a header and the time it took.

    On failure it reports how long the stage ran before raising, which is
    what tells whether a crash happened on the first image or after an hour.

    Parameters
    ----------
    title : str
        Short name of the stage, shown in the header.

    Yields
    ------
    callable
        The ``step`` helper, so callers need only import ``stage``.
    """

    print(f"\n{_RULE}\n {title}\n{_RULE}", flush=True)
    started = time.perf_counter()

    try:
        yield step
    except BaseException:
        elapsed = format_duration(time.perf_counter() - started)
        print(f"  FAILED after {elapsed}\n", flush=True)
        raise

    elapsed = format_duration(time.perf_counter() - started)
    print(f"  done in {elapsed}\n", flush=True)
