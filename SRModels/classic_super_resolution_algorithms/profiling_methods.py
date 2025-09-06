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

    if len(values) < 2:
        return (np.nan, np.nan)
    
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
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
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'max': float(np.max(values)),
        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        'var': float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        'count': int(len(values))
    }
    
def build_metrics_summary(
    time_stats,
    memory_stats,
    psnr_stats,
    ssim_stats,
    mae_stats,
    rmse_stats,
    gradient_mse_stats,
    epi_stats,
    hf_energy_ratio_stats,
    kl_luma_stats,
    kl_color_stats,
):
    """
    Build aggregated per-algorithm summary metrics from provided stats dictionaries.
    Returns a dict keyed by algorithm with mean/max/var and other derived fields.
    This function is pure (does not mutate globals).
    """
    
    # Compute time/memory jitter & variance from collected stats
    tj, tv, mv = {}, {}, {}
    for alg in time_stats.keys():
        t_arr = time_stats.get(alg, [])
        if len(t_arr) > 1 and np.mean(t_arr) > 0:
            tj[alg] = float(np.std(t_arr, ddof=1) / np.mean(t_arr))
            tv[alg] = float(np.var(t_arr, ddof=1))
        else:
            tj[alg] = np.nan
            tv[alg] = np.nan

        m_arr = memory_stats.get(alg, [])
        if len(m_arr) > 1:
            mv[alg] = float(np.var(m_arr, ddof=1))
        else:
            mv[alg] = np.nan

    # Confidence intervals for PSNR/SSIM
    psnr_ci = {alg: bootstrap_ci(vals) for alg, vals in psnr_stats.items()}
    ssim_ci = {alg: bootstrap_ci(vals) for alg, vals in ssim_stats.items()}

    # Aggregate statistics per algorithm
    summary = {}
    for alg in time_stats.keys():
        time_stats_alg = compute_summary_stats(time_stats.get(alg, []))
        mem_stats_alg = compute_summary_stats(memory_stats.get(alg, []))
        psnr_stats_alg = compute_summary_stats(psnr_stats.get(alg, []))
        ssim_stats_alg = compute_summary_stats(ssim_stats.get(alg, []))
        mae_stats_alg = compute_summary_stats(mae_stats.get(alg, []))
        rmse_stats_alg = compute_summary_stats(rmse_stats.get(alg, []))
        grad_stats_alg = compute_summary_stats(gradient_mse_stats.get(alg, []))
        epi_stats_alg = compute_summary_stats(epi_stats.get(alg, []))
        hf_stats_alg = compute_summary_stats(hf_energy_ratio_stats.get(alg, []))
        kl_luma_stats_alg = compute_summary_stats(kl_luma_stats.get(alg, []))
        kl_color_stats_alg = compute_summary_stats(kl_color_stats.get(alg, []))

        summary[alg] = {
            'psnr_mean': psnr_stats_alg['mean'],
            'psnr_var': psnr_stats_alg['var'],
            'psnr_max': psnr_stats_alg['max'],
            'psnr_ci_low': psnr_ci[alg][0],
            'psnr_ci_high': psnr_ci[alg][1],
            'ssim_mean': ssim_stats_alg['mean'],
            'ssim_var': ssim_stats_alg['var'],
            'ssim_max': ssim_stats_alg['max'],
            'ssim_ci_low': ssim_ci[alg][0],
            'ssim_ci_high': ssim_ci[alg][1],
            'time_mean': time_stats_alg['mean'],
            'time_max': time_stats_alg['max'],
            'time_jitter': tj[alg],
            'time_var': tv[alg],
            'memory_mean': mem_stats_alg['mean'],
            'memory_max': mem_stats_alg['max'],
            'memory_var': mv[alg],
            'mae_mean': mae_stats_alg['mean'],
            'mae_max': mae_stats_alg['max'],
            'rmse_mean': rmse_stats_alg['mean'],
            'rmse_max': rmse_stats_alg['max'],
            'grad_mse_mean': grad_stats_alg['mean'],
            'epi_mean': epi_stats_alg['mean'],
            'hf_ratio_mean': hf_stats_alg['mean'],
            'kl_luma_mean': kl_luma_stats_alg['mean'],
            'kl_color_mean': kl_color_stats_alg['mean'],
        }
    return summary

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
        Metrics where higher values are better. If None, a comprehensive
        set is selected dynamically from available metrics
        (e.g., PSNR/SSIM means & maxima).
    minimize : list[str] or None
        Metrics where lower values are better. If None, a comprehensive
        set is selected dynamically from available metrics
        (e.g., time/memory stats, error metrics, variances, CI widths,
        and deviations-to-1 for EPI/HF ratio).
    weights : dict[str, float] or None
        Optional per-metric weights that should sum (approximately) to 1.
        If None, all selected metrics receive equal weight.

    Derived Metrics (auto when maximize/minimize are None)
    -----------------------------------------------------
    - psnr_ci_width = psnr_ci_high - psnr_ci_low (minimize)
    - ssim_ci_width = ssim_ci_high - ssim_ci_low (minimize)
    - epi_dev       = |epi_mean - 1| (minimize)
    - hf_ratio_dev  = |hf_ratio_mean - 1| (minimize)

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
    - When explicit maximize/minimize lists are provided, they are used
      as-is (no auto-augmentation). Defaults enable a richer composite
      score using all available metrics.
    - Metrics with identical values across algorithms (max == min) have
      zero discriminative impact.
    - NaN metric values (or missing) are treated as zero contribution.
    - Adjust `weights` to emphasize metrics (e.g., PSNR vs time).
    - This function does not modify `summary`; it produces ranking
      artifacts for reporting / visualization.
    """

    # Helper to read raw or derived metric value per algorithm
    def _get_metric_value(stats: dict, metric: str) -> float:
        if metric == 'psnr_ci_width':
            lo = stats.get('psnr_ci_low', np.nan)
            hi = stats.get('psnr_ci_high', np.nan)
            return float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
        if metric == 'ssim_ci_width':
            lo = stats.get('ssim_ci_low', np.nan)
            hi = stats.get('ssim_ci_high', np.nan)
            return float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
        if metric == 'epi_dev':
            v = stats.get('epi_mean', np.nan)
            return float(abs(v - 1.0)) if np.isfinite(v) else np.nan
        if metric == 'hf_ratio_dev':
            v = stats.get('hf_ratio_mean', np.nan)
            return float(abs(v - 1.0)) if np.isfinite(v) else np.nan
        return stats.get(metric, np.nan)

    # Build dynamic defaults if not provided
    if maximize is None and minimize is None:
        # Gather present metrics across all algorithms
        present = set()
        for _alg, st in summary.items():
            present.update(st.keys())

        # Maximize candidates (quality):
        maximize = [m for m in ['psnr_mean', 'psnr_max', 'ssim_mean', 'ssim_max'] if m in present]

        # Minimize candidates (cost/errors/variability):
        minimize_candidates = [
            'time_mean', 'time_max', 'time_jitter', 'time_var',
            'memory_mean', 'memory_max', 'memory_var',
            'mae_mean', 'mae_max', 'rmse_mean', 'rmse_max',
            'grad_mse_mean', 'kl_luma_mean', 'kl_color_mean',
            'psnr_var', 'ssim_var',
        ]
        minimize = [m for m in minimize_candidates if m in present]

        # Derived metrics if their components exist
        has_psnr_ci = ('psnr_ci_low' in present) and ('psnr_ci_high' in present)
        has_ssim_ci = ('ssim_ci_low' in present) and ('ssim_ci_high' in present)
        if has_psnr_ci:
            minimize.append('psnr_ci_width')
        if has_ssim_ci:
            minimize.append('ssim_ci_width')

        # Deviations-to-1 for EPI and HF ratio
        if 'epi_mean' in present:
            minimize.append('epi_dev')
        if 'hf_ratio_mean' in present:
            minimize.append('hf_ratio_dev')
    else:
        # Use explicit lists exactly as provided
        maximize = maximize or []
        minimize = minimize or []

    # Ordered unique union
    metrics_all = list(dict.fromkeys(list(maximize) + list(minimize)))

    # Compute bounds using raw/derived values
    bounds = {}
    for m in metrics_all:
        vals = []
        for _alg, st in summary.items():
            vals.append(_get_metric_value(st, m))
        arr = np.array(vals, dtype=float)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            bounds[m] = (np.nan, np.nan)
        else:
            bounds[m] = (float(valid.min()), float(valid.max()))

    # Default uniform weights if not provided
    if weights is None:
        w_each = 1.0 / max(1, len(metrics_all))
        weights = {m: w_each for m in metrics_all}

    # Score per algorithm
    scores = {}
    for alg, stats in summary.items():
        total = 0.0
        for m in metrics_all:
            val = _get_metric_value(stats, m)
            lo, hi = bounds[m]
            if not np.isfinite(val) or not np.isfinite(lo) or not np.isfinite(hi) or hi - lo == 0:
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