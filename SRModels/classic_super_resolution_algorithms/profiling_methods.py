import time
import tracemalloc
from math import sqrt

import cv2
import numpy as np

DEF_EPS = 1e-9

# Ranking utilities
MAXIMIZE_DEFAULT = ['psnr_mean', 'ssim_mean']
MINIMIZE_DEFAULT = ['time_mean', 'memory_mean', 'mae_mean', 'rmse_mean']

# --------------------- #
# Time / Memory metrics #
# --------------------- #
def time_algorithm(func, *args, **kwargs):
    """Return (result, elapsed_seconds) for the callable.

    Measures only the direct execution (wall clock) of the function body.
    """
    
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    
    return result, elapsed

def memory_algorithm(func, *args, **kwargs):
    """Return (result, peak_bytes) for the callable execution.

    Uses tracemalloc to capture peak allocated memory during the call.
    """
    
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, peak

# ------------- #
# Error metrics #
# ------------- #
def mae(a, b):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

def rmse(a, b):
    """Root Mean Squared Error."""
    
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(sqrt(np.mean(diff * diff) + DEF_EPS))

# ----------------------------- #
# Gradient / edge based metrics #
# ----------------------------- #
def _ensure_gray_f32(img):
    """Return grayscale float32 in [0,1]. Accepts RGB/gray uint8 or float."""
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    if img.max() > 1.5:
        img = img / 255.0
        
    return img

def sobel_mag(img):
    """Compute Sobel gradient magnitude for an image."""
    
    g = _ensure_gray_f32(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    
    return np.sqrt(gx * gx + gy * gy)

def gradient_mse(hr, sr):
    """MSE between Sobel magnitudes of HR and SR images."""
    
    hr_m = sobel_mag(hr)
    sr_m = sobel_mag(sr)
    
    return float(np.mean((hr_m - sr_m) ** 2))

def epi(hr, sr):
    """Edge Preservation Index: ratio of SR to HR gradient energy."""
    
    hr_m = sobel_mag(hr)
    sr_m = sobel_mag(sr)
    
    return float((sr_m.sum() + DEF_EPS) / (hr_m.sum() + DEF_EPS))

# -------------------------------- #
# Frequency / distribution metrics #
# -------------------------------- #
def hf_energy_ratio(hr, sr, radius_frac=0.6):
    """High-frequency energy ratio between SR and HR (grayscale)."""
    
    hr_f = hr.astype(np.float32)
    sr_f = sr.astype(np.float32)
    F_hr = np.fft.fftshift(np.fft.fft2(hr_f))
    F_sr = np.fft.fftshift(np.fft.fft2(sr_f))
    h, w = hr_f.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = r.max() + DEF_EPS
    mask = r > (radius_frac * r_max)
    num = np.abs(F_sr)[mask].sum() + DEF_EPS
    den = np.abs(F_hr)[mask].sum() + DEF_EPS
    
    return float(num / den)

def kl_divergence(p_img, q_img, bins=256):
    """KL divergence between grayscale histograms of p (HR) and q (SR)."""
    
    if p_img.dtype != np.uint8:
        p_arr = np.clip(p_img, 0, 1) * 255.0
    else:
        p_arr = p_img.astype(np.float32)
    if q_img.dtype != np.uint8:
        q_arr = np.clip(q_img, 0, 1) * 255.0
    else:
        q_arr = q_img.astype(np.float32)
    p_hist, _ = np.histogram(
        p_arr, bins=bins, range=(0, 255), density=True
    )
    q_hist, _ = np.histogram(
        q_arr, bins=bins, range=(0, 255), density=True
    )
    eps = 1e-12
    P = p_hist + eps
    Q = q_hist + eps
    
    return float(np.sum(P * np.log(P / Q)))

def kl_divergence_color(p_rgb, q_rgb, bins=64):
    """Average per-channel KL divergence for RGB images."""
    
    if p_rgb.dtype != np.uint8:
        p = np.clip(p_rgb, 0, 1) * 255.0
    else:
        p = p_rgb.astype(np.float32)
    if q_rgb.dtype != np.uint8:
        q = np.clip(q_rgb, 0, 1) * 255.0
    else:
        q = q_rgb.astype(np.float32)
    eps = 1e-12
    total = 0.0
    channels = p.shape[2]
    for c in range(channels):
        p_hist, _ = np.histogram(
            p[..., c], bins=bins, range=(0, 255), density=True
        )
        q_hist, _ = np.histogram(
            q[..., c], bins=bins, range=(0, 255), density=True
        )
        P = p_hist + eps
        Q = q_hist + eps
        total += np.sum(P * np.log(P / Q))

    return float(total / max(1, channels))

# -------------------- #
# Confidence intervals #
# -------------------- #
def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Estimate a mean confidence interval using bootstrap resampling.

    Parameters
    ----------
    values : np.ndarray
        1D array of numeric values. If length < 2 returns (NaN, NaN).
    n_boot : int, default 1000
        Number of bootstrap resamples.
    ci : float, default 0.95
        Central confidence level (e.g., 0.95 -> 2.5 and 97.5 percentiles).
    seed : int, default 42
        Seed for the NumPy Generator for reproducibility.

    Returns
    -------
    (float, float)
        Lower and upper bounds of the bootstrap percentile interval.

    Method
    ------
    For each bootstrap iteration a sample with replacement of size N is
    drawn; its mean is stored. After n_boot iterations the desired
    percentile bounds are taken from the distribution of bootstrapped means.
    """
    
    if values.size < 2:
        return (np.nan, np.nan)
    
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        stats.append(sample.mean())
    
    lower_p = (1.0 - ci) / 2.0 * 100.0
    upper_p = (1.0 + ci) / 2.0 * 100.0
    
    return (
        float(np.percentile(stats, lower_p)),
        float(np.percentile(stats, upper_p))
    )

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
    The function is resilient to empty input returning NaNs (except count=0)
    so downstream plotting logic can safely annotate missing data.
    """
    
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'max': float(values.max()),
        'std': float(values.std(ddof=1)) if values.size > 1 else 0.0,
        'var': float(values.var(ddof=1)) if values.size > 1 else 0.0,
        'count': int(values.size)
    }

def rank_algorithms(summary, maximize=None, minimize=None, weights=None):
    """Rank algorithms based on multiple metrics with normalization.

    Parameters
    ----------
    summary : dict
        Mapping algorithm_name -> metric dict. Each metric dict should
        contain the metrics referenced in `maximize` / `minimize`, e.g.
        'psnr_mean', 'time_mean', etc. Missing metrics are treated as NaN
        and contribute zero to the score.
    maximize : list[str] or None
        Metrics where higher values are better. Defaults to
        MAXIMIZE_DEFAULT if None.
    minimize : list[str] or None
        Metrics where lower values are better. Defaults to
        MINIMIZE_DEFAULT if None.
    weights : dict[str, float] or None
        Optional per-metric weights that should sum (approximately) to 1.
        If None, all selected metrics receive equal weight.

    Scoring Method
    --------------
    1. Collect all metrics referenced in maximize + minimize preserving
       order and uniqueness.
    2. For each metric compute (min, max) across available (non-NaN)
       algorithm values.
    3. Normalize each algorithm's metric into [0,1]:
       - For maximize metrics: (val - min) / (max - min)
       - For minimize metrics: (max - val) / (max - min)
       Invalid / constant-range metrics yield 0 contribution.
    4. Clip each normalized value to [0,1] for safety.
    5. Multiply by its weight and sum to obtain an aggregate score.

    Returns
    -------
    ranked : list[tuple[str, float]]
        Algorithms sorted descending by aggregate score.
    scores : dict[str, float]
        Raw aggregate score per algorithm.
    bounds : dict[str, tuple[float, float]]
        Per-metric (min, max) used for normalization (NaNs if undefined).

    Notes
    -----
    - Metrics with identical values across algorithms (max == min) have
      zero discriminative impact.
    - NaN metric values (or missing) are treated as zero contribution.
    - Adjust `weights` to emphasize metrics (e.g., PSNR vs time).
    - This function does not modify `summary`; it produces ranking
      artifacts for reporting / visualization.
    """
    
    maximize = maximize or MAXIMIZE_DEFAULT
    minimize = minimize or MINIMIZE_DEFAULT
    metrics_all = list(dict.fromkeys(maximize + minimize))

    values = {m: [] for m in metrics_all}
    for alg, stats in summary.items():
        for m in metrics_all:
            values[m].append(stats.get(m, np.nan))

    bounds = {}
    for m, vals in values.items():
        arr = np.array(vals, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            bounds[m] = ( np.nan, np.nan )
        else:
            bounds[m] = (float(valid.min()), float(valid.max()))

    if weights is None:
        w_each = 1.0 / len(metrics_all)
        weights = {m: w_each for m in metrics_all}

    scores = {}
    for alg, stats in summary.items():
        total = 0.0
        for m in metrics_all:
            val = stats.get(m, np.nan)
            lo, hi = bounds[m]
            if np.isnan(val) or np.isnan(lo) or np.isnan(hi) or hi - lo == 0:
                norm = 0.0
            else:
                if m in maximize:
                    norm = (val - lo) / (hi - lo)
                else:
                    norm = (hi - val) / (hi - lo)
                norm = float(np.clip(norm, 0.0, 1.0))
            total += weights.get(m, 0.0) * norm
        scores[alg] = total

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked, scores, bounds