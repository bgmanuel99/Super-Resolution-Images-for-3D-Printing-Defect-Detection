import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_time_memory_panels(metric_summary, algorithms_order, colors_map, main_title, outfile, figsize=(18, 9)):
    """
    Plot Time & Memory stats and save to file.

    Panels included (row-major order):
        Row 1:
            1) Average Time (s)
            2) Max Time (s)
            3) Time Jitter (std/mean)
        Row 2:
            4) Average Peak Memory (MB)
            5) Max Peak Memory (MB)
            6) Memory Variance (MB^2)
    """

    means_time   = [metric_summary[a]['time_mean'] for a in algorithms_order]
    maxs_time    = [metric_summary[a]['time_max']  for a in algorithms_order]
    time_jitter  = [metric_summary[a].get('time_jitter', np.nan) for a in algorithms_order]
    mem_mean_mb  = [metric_summary[a]['memory_mean'] / (1024**2) for a in algorithms_order]
    mem_max_mb   = [metric_summary[a]['memory_max']  / (1024**2) for a in algorithms_order]
    mem_var_mb2  = [metric_summary[a].get('memory_var', np.nan) / ((1024**2) ** 2) for a in algorithms_order]

    # Order panels to match requested layout
    stat_groups = [
        (means_time,  'Average Time (s)', '{:.3g}'),
        (maxs_time,   'Max Time (s)', '{:.3g}'),
        (time_jitter, 'Time Jitter (std/mean)', '{:.3g}'),
        (mem_mean_mb, 'Average Peak Memory (MB)', '{:.6f}'),
        (mem_max_mb,  'Max Peak Memory (MB)', '{:.6f}'),
        (mem_var_mb2, 'Memory Variance (MB^2)', '{:.6g}'),
    ]

    x = np.arange(len(algorithms_order))

    # Choose a dynamic layout based on number of panels
    n_panels = len(stat_groups)
    if n_panels <= 4:
        rows, cols = 2, 2
    elif n_panels <= 6:
        rows, cols = 2, 3
    else:
        cols = 3
        rows = int(np.ceil(n_panels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes_arr = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    for idx, (data, subtitle, fmt) in enumerate(stat_groups):
        ax = axes_arr[idx]
        bars = ax.bar(x, data, color=[colors_map[a] for a in algorithms_order])
        ax.set_title(subtitle)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms_order, rotation=30, ha='right')
        cur_bottom, cur_top = ax.get_ylim()
        span = (cur_top - cur_bottom) if np.isfinite(cur_top - cur_bottom) and (cur_top - cur_bottom) > 0 else 1.0
        pad = 0.01 * span
        max_label_y = -np.inf

        for rect, val in zip(bars, data):
            if np.isnan(val):
                continue

            height = rect.get_height()
            label_y = height + pad
            ax.text(rect.get_x() + rect.get_width() / 2, label_y, fmt.format(val), ha='center', va='bottom', fontsize=8)

            if np.isfinite(label_y):
                max_label_y = max(max_label_y, label_y)

        if np.isfinite(max_label_y):
            cur_bottom, cur_top = ax.get_ylim()

            if max_label_y > cur_top:
                extra = max(0.02 * (max_label_y - cur_bottom), 0.02)
                ax.set_ylim(top=max_label_y + extra)
    
    for j in range(n_panels, len(axes_arr)):
        axes_arr[j].axis('off')

    fig.suptitle(main_title, fontsize=14)

    if outfile:
        fig.savefig(Path(outfile), dpi=150, bbox_inches='tight')

def plot_psnr_ssim_panels(metric_summary, algorithms_order, colors_map, main_title, outfile, figsize=(18, 9)):
    """
    Plot a 2x2 grid for PSNR & SSIM stats and save to file, with asymmetric
    confidence interval error bars on the mean panels (same behavior as before).
    Panels:
      1) PSNR Mean (dB) with CI
      2) PSNR Max (dB)
      3) SSIM Mean with CI
      4) SSIM Max
    """

    psnr_mean = [metric_summary[a]['psnr_mean'] for a in algorithms_order]
    psnr_max  = [metric_summary[a]['psnr_max']  for a in algorithms_order]
    ssim_mean = [metric_summary[a]['ssim_mean'] for a in algorithms_order]
    ssim_max  = [metric_summary[a]['ssim_max']  for a in algorithms_order]

    psnr_ci_low  = np.array([metric_summary[a]['psnr_ci_low'] for a in algorithms_order])
    psnr_ci_high = np.array([metric_summary[a]['psnr_ci_high'] for a in algorithms_order])
    psnr_err = np.vstack([
        np.clip(np.array(psnr_mean) - psnr_ci_low, 0, None),
        np.clip(psnr_ci_high - np.array(psnr_mean), 0, None)
    ])
    ssim_ci_low  = np.array([metric_summary[a]['ssim_ci_low'] for a in algorithms_order])
    ssim_ci_high = np.array([metric_summary[a]['ssim_ci_high'] for a in algorithms_order])
    ssim_err = np.vstack([
        np.clip(np.array(ssim_mean) - ssim_ci_low, 0, None),
        np.clip(ssim_ci_high - np.array(ssim_mean), 0, None)
    ])

    stat_groups = [
        (psnr_mean, 'PSNR Mean (dB)', '{:.2f}', psnr_err),
        (psnr_max,  'PSNR Max (dB)',  '{:.2f}', None),
        (ssim_mean, 'SSIM Mean',      '{:.4f}', ssim_err),
        (ssim_max,  'SSIM Max',       '{:.4f}', None),
    ]

    x = np.arange(len(algorithms_order))
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    for idx, (data, subtitle, fmt, err) in enumerate(stat_groups):
        ax = axes[idx // 2, idx % 2]
        yerr_arr = None

        if err is not None:
            yerr_arr = np.asarray(err, dtype=float)
            
            if yerr_arr.ndim == 1:
                yerr_arr = np.vstack([yerr_arr, yerr_arr])
            elif yerr_arr.ndim == 2:
                if yerr_arr.shape[0] == 2:
                    pass
                elif yerr_arr.shape[1] == 2:
                    yerr_arr = yerr_arr.T
                else:
                    yerr_arr = None
            else:
                yerr_arr = None

        bars = ax.bar(
            x,
            data,
            color=[colors_map[a] for a in algorithms_order],
            yerr=yerr_arr if yerr_arr is not None else None,
            ecolor='k' if yerr_arr is not None else None,
            capsize=3 if yerr_arr is not None else 0,
        )

        ax.set_title(subtitle)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms_order, rotation=30, ha='right')
        cur_bottom, cur_top = ax.get_ylim()
        span = (cur_top - cur_bottom) if np.isfinite(cur_top - cur_bottom) and (cur_top - cur_bottom) > 0 else 1.0
        pad = 0.01 * span
        max_label_y = -np.inf

        for i, (rect, val) in enumerate(zip(bars, data)):
            if np.isnan(val):
                continue

            height = rect.get_height()
            pos_err = 0.0

            if yerr_arr is not None and i < yerr_arr.shape[1]:
                pe = yerr_arr[1, i]

                if np.isfinite(pe):
                    pos_err = float(max(0.0, pe))

            label_y = height + pos_err + pad
            ax.text(rect.get_x() + rect.get_width() / 2, label_y, fmt.format(val), ha='center', va='bottom', fontsize=8)

            if np.isfinite(label_y):
                max_label_y = max(max_label_y, label_y)

        if np.isfinite(max_label_y):
            cur_bottom, cur_top = ax.get_ylim()

            if max_label_y > cur_top:
                extra = max(0.02 * (max_label_y - cur_bottom), 0.02)
                ax.set_ylim(top=max_label_y + extra)

    fig.suptitle(main_title, fontsize=14)

    if outfile:
        fig.savefig(Path(outfile), dpi=150, bbox_inches='tight')