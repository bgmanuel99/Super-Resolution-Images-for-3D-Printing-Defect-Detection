import time
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback

def last_epoch_metrics(history, keys=("loss", "psnr", "ssim")):
    """Take the last-epoch value of each train/validation curve.

    Reported as a scalar per curve because the comparison across models is
    a table, not a set of curves. A key the model never recorded resolves
    to None rather than being dropped, so every model writes the same set
    of keys and the reporting notebook never has to guard for a missing one.

    Parameters
    ----------
    history : dict
        Mapping of curve name to per-epoch values.
    keys : sequence of str
        Curve stems to read, each looked up as ``k`` and ``val_k``.

    Returns
    -------
    dict
        ``final_train_<k>`` and ``final_val_<k>`` for every requested key.
    """

    def tail(name):
        values = history.get(name)

        return float(values[-1]) if values is not None and len(values) else None

    metrics = {}
    for key in keys:
        metrics[f"final_train_{key}"] = tail(key)
        metrics[f"final_val_{key}"] = tail(f"val_{key}")

    return metrics

def _bytes_to_mb(b):
    if b is None:
        return None
    return float(b) / (1024.0 * 1024.0)

def _reduce_finite(values, reducer):
    """
    Apply a reducer over the finite entries, or return NaN if there are none.

    Filtering before reducing avoids the empty-slice warnings that the NaN
    aware reducers emit on a CPU-only run, where every sample is NaN.
    """

    finite = [v for v in values if v is not None and np.isfinite(v)]

    return float(reducer(finite)) if finite else float("nan")

def read_gpu_memory_mb(device="GPU:0"):
    """
    Current and peak GPU memory in MB, or NaN when there is no GPU.

    NaN rather than None so that aggregating the samples cannot raise, and
    so that a CPU-only run still produces a well-formed metrics dict.

    Returns
    -------
    tuple of float
        ``(current_mb, peak_mb)``.
    """

    try:
        info = tf.config.experimental.get_memory_info(device)
    except Exception:
        return float("nan"), float("nan")

    return _bytes_to_mb(info["current"]), _bytes_to_mb(info["peak"])

def reset_gpu_memory_peak(device="GPU:0"):
    """
    Reset the peak counter so the next read covers only what follows.

    The counter is cumulative over the process, so without this the peak
    reported for every epoch is the highest value seen since the kernel
    started, including whatever ran before.
    """

    try:
        tf.config.experimental.reset_memory_stats(device)
    except Exception:
        pass

def profile_evaluation(evaluate, device="GPU:0"):
    """
    Time an evaluation over the test set and measure the memory it needed.

    The peak counter is reset first, so the value describes this call and
    not the highest allocation seen since the process started.

    Parameters
    ----------
    evaluate : callable
        Zero-argument callable running the evaluation.
    device : str
        Device whose memory is read.

    Returns
    -------
    tuple
        ``(result, elapsed_seconds, memory)``, memory keyed
        'gpu_mean_current_mb' and 'gpu_peak_mb' to match what the training
        trackers report. Both are NaN when there is no GPU.
    """

    reset_gpu_memory_peak(device)
    current_before, _ = read_gpu_memory_mb(device)

    start = time.perf_counter()
    result = evaluate()
    elapsed = time.perf_counter() - start

    current_after, peak = read_gpu_memory_mb(device)

    return result, float(elapsed), {
        "gpu_mean_current_mb": _reduce_finite(
            [current_before, current_after], np.mean
        ),
        "gpu_peak_mb": peak,
    }

class _GpuMemorySampler:
    """
    Accumulates GPU memory samples and reduces them per epoch.

    The mean is taken over samples collected during the epoch, so it
    describes the epoch rather than its two endpoints, and the peak is read
    once at the end from a counter reset at the start.
    """

    def __init__(self, track_gpu=True, gpu_device="GPU:0"):
        self.track_gpu = track_gpu
        self.gpu_device = gpu_device
        self.gpu_mean_current_mb = []
        self.gpu_peak_mb = []
        self._samples = []

    def begin_epoch(self):
        self._samples = []
        if not self.track_gpu:
            return
        reset_gpu_memory_peak(self.gpu_device)
        self.sample()

    def sample(self):
        """Record one reading of the current allocation."""

        if not self.track_gpu:
            return
        current, _ = read_gpu_memory_mb(self.gpu_device)
        self._samples.append(current)

    def end_epoch(self):
        """Reduce this epoch's samples and return ``(mean_mb, peak_mb)``."""

        if not self.track_gpu:
            mean_mb = peak_mb = float("nan")
        else:
            self.sample()
            _, peak_mb = read_gpu_memory_mb(self.gpu_device)
            mean_mb = _reduce_finite(self._samples, np.mean)

        self.gpu_mean_current_mb.append(mean_mb)
        self.gpu_peak_mb.append(peak_mb)
        self._samples = []

        return mean_mb, peak_mb

    def as_dict(self):
        """Mean of the per-epoch means and the highest per-epoch peak."""

        return {
            "gpu_mean_current_mb": _reduce_finite(self.gpu_mean_current_mb, np.mean),
            "gpu_peak_mb": _reduce_finite(self.gpu_peak_mb, np.max),
        }

class EpochTimeCallback(Callback):
    """Keras Callback: records wall-clock time per epoch"""

    def __init__(self):
        super().__init__()
        self._t0 = None
        self.epoch_times_sec = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._t0 is None:
            return
        elapsed = time.perf_counter() - self._t0
        self.epoch_times_sec.append(elapsed)
        if isinstance(logs, dict):
            logs["epoch_time_sec"] = elapsed

    # Convenience accessors
    def mean_time_value(self):
        return float(np.mean(self.epoch_times_sec))

class EpochMemoryCallback(Callback):
    """Keras Callback: records GPU memory per epoch"""

    def __init__(self, track_gpu=True, gpu_device="GPU:0"):
        super().__init__()
        self._sampler = _GpuMemorySampler(track_gpu, gpu_device)

    @property
    def gpu_mean_current_mb(self):
        return self._sampler.gpu_mean_current_mb

    @property
    def gpu_peak_mb(self):
        return self._sampler.gpu_peak_mb

    def on_epoch_begin(self, epoch, logs=None):
        self._sampler.begin_epoch()

    def on_train_batch_end(self, batch, logs=None):
        self._sampler.sample()

    def on_epoch_end(self, epoch, logs=None):
        mean_mb, peak_mb = self._sampler.end_epoch()

        if isinstance(logs, dict):
            logs["gpu_mean_current_mb"] = mean_mb
            logs["gpu_peak_mb"] = peak_mb

    def as_dict(self):
        return self._sampler.as_dict()

class EpochTimeTracker:
    """Manual epoch time tracker for custom loops (e.g., ESRGAN)"""

    def __init__(self):
        self._t0 = None
        self.epoch_times_sec = []

    def begin_epoch(self):
        self._t0 = time.perf_counter()

    def end_epoch(self):
        if self._t0 is None:
            return
        self.epoch_times_sec.append(time.perf_counter() - self._t0)
        self._t0 = None

    def mean_time_value(self):
        return float(np.mean(self.epoch_times_sec))

class EpochMemoryTracker:
    """Manual epoch memory tracker for custom loops (e.g., ESRGAN)"""

    def __init__(self, track_gpu=True, gpu_device="GPU:0"):
        self._sampler = _GpuMemorySampler(track_gpu, gpu_device)

    @property
    def gpu_mean_current_mb(self):
        return self._sampler.gpu_mean_current_mb

    @property
    def gpu_peak_mb(self):
        return self._sampler.gpu_peak_mb

    def begin_epoch(self):
        self._sampler.begin_epoch()

    def sample_step(self):
        """Record one reading, to be called once per training step."""

        self._sampler.sample()

    def end_epoch(self):
        self._sampler.end_epoch()

    def as_dict(self):
        return self._sampler.as_dict()