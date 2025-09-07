"""
Time and memory tracking utilities for training.

Includes:
- EpochTimeCallback: Keras Callback to record per-epoch training time.
- EpochMemoryCallback: Keras Callback to record per-epoch CPU/GPU memory stats.
- EpochTimeTracker: Manual tracker for custom loops (e.g., ESRGAN).
- EpochMemoryTracker: Manual tracker for custom loops (e.g., ESRGAN).
"""

import time
import psutil
import tensorflow as tf
from __future__ import annotations
from keras.callbacks import Callback

def _bytes_to_mb(b):
	if b is None:
		return None
	return float(b) / (1024.0 * 1024.0)

class EpochTimeCallback(Callback):
	"""Keras Callback: records wall-clock time per epoch."""

	def __init__(self) -> None:
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
	def as_dict(self):
		return {"epoch_time_sec": list(self.epoch_times_sec)}

class EpochMemoryCallback(Callback):
	"""Keras Callback: records CPU and GPU memory per epoch.

	Notes:
	- CPU memory is process RSS via psutil if available.
	- GPU memory uses tf.config.experimental.get_memory_info if available.
	  Metrics recorded as deltas between epoch begin and end, plus end values.
	"""

	def __init__(self, track_cpu: bool = True, track_gpu: bool = True, gpu_device: str = "GPU:0") -> None:
		super().__init__()
		self.track_cpu = track_cpu
		self.track_gpu = track_gpu
		self.gpu_device = gpu_device

		# Per-epoch recorded metrics
		self.cpu_rss_begin_mb = []
		self.cpu_rss_end_mb = []
		self.cpu_rss_delta_mb = []

		self.gpu_current_begin_mb = []
		self.gpu_current_end_mb = []
		self.gpu_current_delta_mb = []

		self.gpu_peak_begin_mb = []
		self.gpu_peak_end_mb = []
		self.gpu_peak_delta_mb = []

		# Internals for epoch-begin baselines
		self._cpu_begin_bytes = None
		self._gpu_begin = None

	def _read_cpu_bytes(self):
		if not self.track_cpu or psutil is None:
			return None
		try:
			return psutil.Process().memory_info().rss
		except Exception:
			return None

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
		self._cpu_begin_bytes = self._read_cpu_bytes()
		self._gpu_begin = self._read_gpu_info()

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		# CPU memory
		cpu_begin = self._cpu_begin_bytes
		cpu_end = self._read_cpu_bytes()
		self.cpu_rss_begin_mb.append(_bytes_to_mb(cpu_begin))
		self.cpu_rss_end_mb.append(_bytes_to_mb(cpu_end))
		if cpu_begin is not None and cpu_end is not None:
			self.cpu_rss_delta_mb.append(_bytes_to_mb(cpu_end - cpu_begin))
		else:
			self.cpu_rss_delta_mb.append(None)

		# GPU memory
		gpu_begin = self._gpu_begin
		gpu_end = self._read_gpu_info()

		# Current
		cur_begin = gpu_begin.get("current") if isinstance(gpu_begin, dict) else None
		cur_end = gpu_end.get("current") if isinstance(gpu_end, dict) else None
		self.gpu_current_begin_mb.append(_bytes_to_mb(cur_begin))
		self.gpu_current_end_mb.append(_bytes_to_mb(cur_end))
		if cur_begin is not None and cur_end is not None:
			self.gpu_current_delta_mb.append(_bytes_to_mb(cur_end - cur_begin))
		else:
			self.gpu_current_delta_mb.append(None)

		# Peak
		peak_begin = gpu_begin.get("peak") if isinstance(gpu_begin, dict) else None
		peak_end = gpu_end.get("peak") if isinstance(gpu_end, dict) else None
		self.gpu_peak_begin_mb.append(_bytes_to_mb(peak_begin))
		self.gpu_peak_end_mb.append(_bytes_to_mb(peak_end))
		if peak_begin is not None and peak_end is not None:
			self.gpu_peak_delta_mb.append(_bytes_to_mb(peak_end - peak_begin))
		else:
			self.gpu_peak_delta_mb.append(None)

		# Expose some values in logs for Keras history
		logs["cpu_rss_end_mb"] = self.cpu_rss_end_mb[-1]
		logs["cpu_rss_delta_mb"] = self.cpu_rss_delta_mb[-1]
		logs["gpu_peak_end_mb"] = self.gpu_peak_end_mb[-1]
		logs["gpu_peak_delta_mb"] = self.gpu_peak_delta_mb[-1]

	def as_dict(self):
		return {
			"cpu_rss_begin_mb": list(self.cpu_rss_begin_mb),
			"cpu_rss_end_mb": list(self.cpu_rss_end_mb),
			"cpu_rss_delta_mb": list(self.cpu_rss_delta_mb),
			"gpu_current_begin_mb": list(self.gpu_current_begin_mb),
			"gpu_current_end_mb": list(self.gpu_current_end_mb),
			"gpu_current_delta_mb": list(self.gpu_current_delta_mb),
			"gpu_peak_begin_mb": list(self.gpu_peak_begin_mb),
			"gpu_peak_end_mb": list(self.gpu_peak_end_mb),
			"gpu_peak_delta_mb": list(self.gpu_peak_delta_mb),
		}

class EpochTimeTracker:
	"""Manual epoch time tracker for custom loops (e.g., ESRGAN).

	Usage:
		t = EpochTimeTracker()
		for epoch in range(epochs):
			t.begin_epoch()
			...  # training work
			t.end_epoch()
		print(t.epoch_times_sec)
	"""

	def __init__(self) -> None:
		self._t0 = None
		self.epoch_times_sec = []

	def begin_epoch(self) -> None:
		self._t0 = time.perf_counter()

	def end_epoch(self) -> None:
		if self._t0 is None:
			return
		self.epoch_times_sec.append(time.perf_counter() - self._t0)
		self._t0 = None

	def as_dict(self):
		return {"epoch_time_sec": list(self.epoch_times_sec)}

class EpochMemoryTracker:
	"""Manual epoch memory tracker for custom loops (e.g., ESRGAN).

	Records CPU RSS and GPU current/peak deltas per epoch.

	Usage:
		m = EpochMemoryTracker(track_cpu=True, track_gpu=True, gpu_device="GPU:0")
		for epoch in range(epochs):
			m.begin_epoch()
			...  # training work
			m.end_epoch()
		data = m.as_dict()
	"""

	def __init__(self, track_cpu: bool = True, track_gpu: bool = True, gpu_device: str = "GPU:0") -> None:
		self.track_cpu = track_cpu
		self.track_gpu = track_gpu
		self.gpu_device = gpu_device

		self.cpu_rss_begin_mb = []
		self.cpu_rss_end_mb = []
		self.cpu_rss_delta_mb = []

		self.gpu_current_begin_mb = []
		self.gpu_current_end_mb = []
		self.gpu_current_delta_mb = []

		self.gpu_peak_begin_mb = []
		self.gpu_peak_end_mb = []
		self.gpu_peak_delta_mb = []

		self._cpu_begin_bytes = None
		self._gpu_begin = None

	def _read_cpu_bytes(self):
		if not self.track_cpu or psutil is None:
			return None
		try:
			return psutil.Process().memory_info().rss
		except Exception:
			return None

	def _read_gpu_info(self):
		if not self.track_gpu or tf is None:
			return None
		try:
			return tf.config.experimental.get_memory_info(self.gpu_device)  # type: ignore[attr-defined]
		except Exception:
			return None

	def begin_epoch(self) -> None:
		self._cpu_begin_bytes = self._read_cpu_bytes()
		self._gpu_begin = self._read_gpu_info()

	def end_epoch(self) -> None:
		# CPU
		cpu_begin = self._cpu_begin_bytes
		cpu_end = self._read_cpu_bytes()
		self.cpu_rss_begin_mb.append(_bytes_to_mb(cpu_begin))
		self.cpu_rss_end_mb.append(_bytes_to_mb(cpu_end))
		if cpu_begin is not None and cpu_end is not None:
			self.cpu_rss_delta_mb.append(_bytes_to_mb(cpu_end - cpu_begin))
		else:
			self.cpu_rss_delta_mb.append(None)

		# GPU
		gpu_begin = self._gpu_begin
		gpu_end = self._read_gpu_info()

		cur_begin = gpu_begin.get("current") if isinstance(gpu_begin, dict) else None
		cur_end = gpu_end.get("current") if isinstance(gpu_end, dict) else None
		self.gpu_current_begin_mb.append(_bytes_to_mb(cur_begin))
		self.gpu_current_end_mb.append(_bytes_to_mb(cur_end))
		if cur_begin is not None and cur_end is not None:
			self.gpu_current_delta_mb.append(_bytes_to_mb(cur_end - cur_begin))
		else:
			self.gpu_current_delta_mb.append(None)

		peak_begin = gpu_begin.get("peak") if isinstance(gpu_begin, dict) else None
		peak_end = gpu_end.get("peak") if isinstance(gpu_end, dict) else None
		self.gpu_peak_begin_mb.append(_bytes_to_mb(peak_begin))
		self.gpu_peak_end_mb.append(_bytes_to_mb(peak_end))
		if peak_begin is not None and peak_end is not None:
			self.gpu_peak_delta_mb.append(_bytes_to_mb(peak_end - peak_begin))
		else:
			self.gpu_peak_delta_mb.append(None)

		# Reset baselines
		self._cpu_begin_bytes = None
		self._gpu_begin = None

	def as_dict(self):
		return {
			"cpu_rss_begin_mb": list(self.cpu_rss_begin_mb),
			"cpu_rss_end_mb": list(self.cpu_rss_end_mb),
			"cpu_rss_delta_mb": list(self.cpu_rss_delta_mb),
			"gpu_current_begin_mb": list(self.gpu_current_begin_mb),
			"gpu_current_end_mb": list(self.gpu_current_end_mb),
			"gpu_current_delta_mb": list(self.gpu_current_delta_mb),
			"gpu_peak_begin_mb": list(self.gpu_peak_begin_mb),
			"gpu_peak_end_mb": list(self.gpu_peak_end_mb),
			"gpu_peak_delta_mb": list(self.gpu_peak_delta_mb),
		}