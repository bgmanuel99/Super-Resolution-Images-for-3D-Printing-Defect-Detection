"""
Time and memory tracking utilities for training.

Includes:
- EpochTimeCallback: Keras Callback to record per-epoch training time.
- EpochMemoryCallback: Keras Callback to record per-epoch GPU memory stats.
- EpochTimeTracker: Manual tracker for custom loops (e.g., ESRGAN).
- EpochMemoryTracker: Manual tracker for custom loops (e.g., ESRGAN).
"""

import time
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback

def _bytes_to_mb(b):
	if b is None:
		return None
	return float(b) / (1024.0 * 1024.0)

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

	def __init__(self, track_cpu=True, track_gpu=True, gpu_device="GPU:0"):
		super().__init__()
		self.track_cpu = track_cpu
		self.track_gpu = track_gpu
		self.gpu_device = gpu_device

		# Per-epoch recorded metrics (only mean current and peak GPU memory)
		self.gpu_mean_current_mb = []
		self.gpu_peak_mb = []
  
		# Internals for epoch-begin baselines
		self._gpu_begin = None

	def _read_gpu_info(self):
		if not self.track_gpu or tf is None:
			return None
		try:
			# Returns {"current": int, "peak": int} in bytes
			return tf.config.experimental.get_memory_info(self.gpu_device)
		except Exception:
			return None

	def on_epoch_begin(self, epoch, logs=None):
		# Capture baselines
		self._gpu_begin = self._read_gpu_info()

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		# GPU memory
		gpu_begin = self._gpu_begin
		gpu_end = self._read_gpu_info()

		cur_begin = gpu_begin.get("current") if isinstance(gpu_begin, dict) else None
		cur_end = gpu_end.get("current") if isinstance(gpu_end, dict) else None
		if cur_begin is not None and cur_end is not None:
			mean_current_bytes = (cur_begin + cur_end) / 2.0
			gpu_mean_current_mb = _bytes_to_mb(mean_current_bytes)
		else:
			gpu_mean_current_mb = _bytes_to_mb(cur_end) if cur_end is not None else None
		self.gpu_mean_current_mb.append(gpu_mean_current_mb)

		peak_begin = gpu_begin.get("peak") if isinstance(gpu_begin, dict) else None
		peak_end = gpu_end.get("peak") if isinstance(gpu_end, dict) else None
		if peak_begin is not None and peak_end is not None:
			gpu_peak_mb = _bytes_to_mb(max(peak_begin, peak_end))
		elif peak_end is not None:
			gpu_peak_mb = _bytes_to_mb(peak_end)
		else:
			gpu_peak_mb = None
		self.gpu_peak_mb.append(gpu_peak_mb)

		# Expose values in logs for Keras history
		logs["gpu_mean_current_mb"] = gpu_mean_current_mb
		logs["gpu_peak_mb"] = gpu_peak_mb

	def as_dict(self):
		return {
			"gpu_mean_current_mb": float(np.mean(self.gpu_mean_current_mb)),
			"gpu_peak_mb": float(np.max(self.gpu_peak_mb)),
		}

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

	def __init__(self, track_cpu=True, track_gpu=True, gpu_device="GPU:0"):
		self.track_cpu = track_cpu
		self.track_gpu = track_gpu
		self.gpu_device = gpu_device

		self.gpu_mean_current_mb = []
		self.gpu_peak_mb = []
		self._gpu_begin = None

	def _read_gpu_info(self):
		if not self.track_gpu or tf is None:
			return None
		try:
			return tf.config.experimental.get_memory_info(self.gpu_device)  # type: ignore[attr-defined]
		except Exception:
			return None

	def begin_epoch(self):
		self._gpu_begin = self._read_gpu_info()

	def end_epoch(self):
		# GPU
		gpu_begin = self._gpu_begin
		gpu_end = self._read_gpu_info()

		cur_begin = gpu_begin.get("current") if isinstance(gpu_begin, dict) else None
		cur_end = gpu_end.get("current") if isinstance(gpu_end, dict) else None
		if cur_begin is not None and cur_end is not None:
			mean_current_bytes = (cur_begin + cur_end) / 2.0
			gpu_mean_current_mb = _bytes_to_mb(mean_current_bytes)
		else:
			gpu_mean_current_mb = _bytes_to_mb(cur_end) if cur_end is not None else None
		self.gpu_mean_current_mb.append(gpu_mean_current_mb)

		peak_begin = gpu_begin.get("peak") if isinstance(gpu_begin, dict) else None
		peak_end = gpu_end.get("peak") if isinstance(gpu_end, dict) else None
		if peak_begin is not None and peak_end is not None:
			gpu_peak_mb = _bytes_to_mb(max(peak_begin, peak_end))
		elif peak_end is not None:
			gpu_peak_mb = _bytes_to_mb(peak_end)
		else:
			gpu_peak_mb = None
		self.gpu_peak_mb.append(gpu_peak_mb)

		# Reset baselines
		self._gpu_begin = None

	def as_dict(self):
		return {
			"gpu_mean_current_mb": float(np.mean(self.gpu_mean_current_mb)),
			"gpu_peak_mb": float(np.max(self.gpu_peak_mb)),
		}