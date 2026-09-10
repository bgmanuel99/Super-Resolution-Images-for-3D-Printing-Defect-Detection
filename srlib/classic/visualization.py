import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from srlib.classic.profiling import (
    rank_algorithms,
)

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
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
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
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outfile), dpi=150, bbox_inches='tight')
        
def plot_speed_quality_tradeoff_3d(metric_summary, algorithms, colors, results_dir=None,
                                   save=True, figsize=(10, 8), view=(22, -55)):
    """
    Plot a 3D Speed–Quality trade-off:
      - X: mean time (s)
      - Y: PSNR mean (dB)
      - Z: SSIM mean
      - Marker size ∝ mean memory (MB)

    Inputs:
      - metric_summary: dict keyed by algorithm with keys: time_mean, psnr_mean, ssim_mean, memory_mean
      - algorithms: list[str] order to plot
      - colors: dict[str, str] algorithm -> color
      - results_dir: directory to save figure (optional)
      - save: whether to save the figure to results_dir/speed_quality_tradeoff_3d.png
      - figsize: tuple for figure size
      - view: (elev, azim) for 3D view

    Returns nothing: the figure is shown and, if results_dir is given, saved.
    """
    # Prepare data
    x_time = [metric_summary[a]['time_mean'] for a in algorithms]
    y_psnr = [metric_summary[a]['psnr_mean'] for a in algorithms]
    z_ssim = [metric_summary[a]['ssim_mean'] for a in algorithms]
    mem_mb = [
        (metric_summary[a]['memory_mean'] or 0.0) / (1024**2)
        if metric_summary[a]['memory_mean'] is not None else np.nan
        for a in algorithms
    ]

    color_list = [colors[a] for a in algorithms]

    # Size mapping (memory -> marker size)
    mem_arr = np.array(mem_mb, dtype=float)
    mem_arr = np.nan_to_num(mem_arr, nan=0.0, posinf=0.0, neginf=0.0)
    m_min, m_max = float(mem_arr.min()), float(mem_arr.max())
    size_min, size_max = 40.0, 240.0
    den = (m_max - m_min) if (m_max - m_min) > 1e-12 else 1.0
    sizes = size_min + (size_max - size_min) * (mem_arr - m_min) / den

    # Plot 3D scatter
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_time, y_psnr, z_ssim, s=sizes, c=color_list, alpha=0.9, edgecolors='k', linewidth=0.6)

    # Axis labels and title
    ax.set_xlabel('Time Mean (s)')
    ax.set_ylabel('PSNR Mean (dB)')
    ax.set_zlabel('SSIM Mean')
    ax.set_title('Speed–Quality Trade-off (3D: Time–PSNR–SSIM)')

    # View and panes
    ax.view_init(elev=view[0], azim=view[1])
    ax.xaxis.pane.set_facecolor((0.95, 0.95, 1.0, 0.4))
    ax.yaxis.pane.set_facecolor((0.95, 1.0, 0.95, 0.4))
    ax.zaxis.pane.set_facecolor((1.0, 0.95, 0.95, 0.4))
    ax.grid(True, linestyle=':', alpha=0.6)

    # Padded limits
    def _pad(vs, pad_frac=0.05):
        vmin, vmax = float(np.min(vs)), float(np.max(vs))
        span = vmax - vmin
        if not np.isfinite(span) or span <= 0:
            return vmin - 1, vmax + 1
        pad = span * pad_frac
        return vmin - pad, vmax + pad

    ax.set_xlim(*_pad(x_time))
    ax.set_ylim(*_pad(y_psnr))
    ax.set_zlim(*_pad(z_ssim))

    # Projections onto planes for readability
    xz_yfloor = ax.get_ylim()[0]
    yz_xfloor = ax.get_xlim()[0]
    xy_zfloor = ax.get_zlim()[0]

    for x, y, z, c in zip(x_time, y_psnr, z_ssim, color_list):
        # Lines to planes
        ax.plot([x, x], [y, y], [xy_zfloor, z], linestyle='--', color=c, alpha=0.25, linewidth=0.8)
        ax.plot([x, x], [xz_yfloor, y], [z, z], linestyle='--', color=c, alpha=0.15, linewidth=0.8)
        ax.plot([yz_xfloor, x], [y, y], [z, z], linestyle='--', color=c, alpha=0.15, linewidth=0.8)

    # Shadow points on XY plane
    ax.scatter(x_time, y_psnr, [xy_zfloor]*len(x_time),
               s=np.maximum(20, sizes*0.35), c=color_list, alpha=0.2, edgecolors='none')

    # Annotate each point with algorithm name
    for a, x, y, z in zip(algorithms, x_time, y_psnr, z_ssim):
        ax.text(x, y, z, a, fontsize=8, ha='center', va='bottom', zorder=5, path_effects=None)

    # Legends
    # Color legend (algorithms)
    color_handles = [Patch(facecolor=colors[a], edgecolor='k', label=a) for a in algorithms]

    # Size legend (memory)
    # pick representative memory values
    reps = []
    if m_max > 0:
        positive = mem_arr[mem_arr > 0]
        if positive.size:
            reps = np.unique(np.round(np.quantile(positive, [0.25, 0.5, 0.75]), 3)).tolist()
    if not reps:
        reps = [max(0.1, m_min), max(0.5, (m_min + m_max)/2), max(1.0, m_max)]

    size_handles = []
    for r in reps:
        ms = size_min + (size_max - size_min) * ((r - m_min) / den)
        ms = max(10, ms)
        size_handles.append(Line2D([0], [0], marker='o', color='w', label=f'{r:.3f} MB',
                                   markerfacecolor='#777777', markersize=np.sqrt(ms), alpha=0.7, markeredgecolor='k'))

    legend1 = ax.legend(handles=color_handles, title='Algorithm', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.legend(handles=size_handles, title='Memory (mean, MB)', loc='upper left', bbox_to_anchor=(1.02, 0.55))
    ax.add_artist(legend1)

    plt.show()

    # Optional: save figure
    if save and results_dir is not None:
        try:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            out_fig = Path(results_dir) / 'speed_quality_tradeoff_3d.png'
            fig.savefig(out_fig, dpi=150, bbox_inches='tight')
        except Exception:
            pass


def plot_edge_metrics_grid(metric_summary, algorithms, colors, results_dir=None, figsize=(12, 5)):
    """
    Display and save a 1x2 grid with mean values of Gradient MSE and EPI across algorithms.

    Panels:
      [0] Gradient MSE Mean (lower is better)
      [1] Edge Preservation Index (EPI) Mean (~1 is ideal)
    """

    grad_mse_mean = [metric_summary[a].get('grad_mse_mean', np.nan) for a in algorithms]
    epi_mean = [metric_summary[a].get('epi_mean', np.nan) for a in algorithms]

    def _bar(ax, data, title, fmt='{:.4g}'):
        x = np.arange(len(algorithms))
        bars = ax.bar(x, data, color=[colors[a] for a in algorithms])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=30, ha='right')
        bottom, top = ax.get_ylim()
        span = top - bottom if np.isfinite(top - bottom) and (top - bottom) > 0 else 1.0
        pad = 0.01 * span
        ymax = -np.inf
        for rect, val in zip(bars, data):
            if not np.isfinite(val):
                continue
            y = rect.get_height() + pad
            ax.text(rect.get_x() + rect.get_width()/2, y, fmt.format(val), ha='center', va='bottom', fontsize=8)
            ymax = max(ymax, y)
        if np.isfinite(ymax) and ymax > ax.get_ylim()[1]:
            bottom, _ = ax.get_ylim()
            ax.set_ylim(top=ymax + max(0.02 * (ymax - bottom), 0.02))

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    _bar(axes[0], grad_mse_mean, 'Gradient MSE Mean') # Lower values are better
    _bar(axes[1], epi_mean, 'Edge Preservation Index (EPI) Mean') # Values closer to 1 are better

    fig.suptitle('Edge/Gradient Metrics: Mean Values')
    if results_dir is not None:
        try:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            out = Path(results_dir) / 'edge_gradient_metrics_mean.png'
            fig.savefig(out, dpi=150, bbox_inches='tight')
        except Exception:
            pass
    plt.show()

    
def plot_and_save_super_resolution_example(vis, ibp_example, nlm_example, egi_example, freq_example, results_dir):
    hr_img_v, lr_img_v, bilinear_v, bicubic_v, area_v, lanczos_v = vis
    *_, ibp_v = ibp_example
    *_, nlm_v = nlm_example
    *_, egi_v = egi_example
    *_, freq_v = freq_example

    images = [
        ('HR', hr_img_v),
        ('LR', lr_img_v),
        ('Bilinear', bilinear_v),
        ('Bicubic', bicubic_v),
        ('Area', area_v),
        ('Lanczos', lanczos_v),
        ('IBP', ibp_v),
        ('NLM', nlm_v),
        ('EGI', egi_v),
        ('FREQ', freq_v),
    ]

    plt.figure(figsize=(18, 7))
    for i, (title, img) in enumerate(images, start=1):
        plt.subplot(2, 5, i)
        cmap = 'gray' if img.ndim == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out_grid = Path(results_dir) / 'super_resolution_example.png'
    plt.savefig(out_grid, dpi=150)
    
    
def show_algorithm_ranking(
    metric_summary,
    maximize=None,
    minimize=None,
    weights=None,
    results_dir=None,
    filename='algorithm_ranking_panel.png',
    dpi=150,
    colors_map=None):
    """
    Compute and display ranking using rank_algorithms(summary, maximize, minimize, weights).

    Visual outputs:
      - Horizontal bar chart of aggregate scores (descending order).
      - Heatmap of per-metric normalized contributions (weight * normalized value)
        for each algorithm, using the same metrics selected by rank_algorithms.
        A metric that rank_algorithms skipped is drawn grey rather than as a
        zero contribution, since its weight was redistributed and not lost.
        Row sums therefore approximate, but do not equal, the plotted score.

        Saving:
            - If results_dir is provided, saves the two-subplot panel to
                results_dir/filename (default: algorithm_ranking_panel.png) with dpi.

        Returns (ranked, scores, bounds) as rank_algorithms does.
    """

    # Run ranking
    ranked, scores, bounds = rank_algorithms(
        summary=metric_summary, maximize=maximize, minimize=minimize, weights=weights
    )

    # Print ranking
    print('Ranking (best to worst):')
    for i, (alg, score) in enumerate(ranked, start=1):
        print(f'{i:2d}. {alg:10s}  score={score:.4f}')

    # Prepare visualization artifacts
    alg_order = [alg for alg, _ in ranked]
    score_vals = [scores[a] for a in alg_order]

    # Metrics used by rank_algorithms are the keys of bounds (order preserved)
    metrics_all = list(bounds.keys())

    # Direction sets: use provided maximize/minimize if given; else infer
    if maximize is not None or minimize is not None:
        max_set = set(maximize or [])
        min_set = set(minimize or [])
    else:
        # Infer defaults consistent with rank_algorithms auto selection
        max_candidates = {'psnr_mean', 'psnr_max', 'ssim_mean', 'ssim_max'}
        derived_minimize = {'psnr_ci_width', 'ssim_ci_width', 'epi_dev', 'hf_ratio_dev'}
        max_set = set(m for m in metrics_all if m in max_candidates)
        # Everything else minimized unless explicitly in max_set
        min_set = set(m for m in metrics_all if m not in max_set)
        # Ensure derived minimize metrics are included when present
        min_set.update(derived_minimize.intersection(metrics_all))

    # Helper to compute raw/derived metric values
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

    # Weights for contributions
    if weights is None:
        w_each = 1.0 / max(1, len(metrics_all))
        weights_used = {m: w_each for m in metrics_all}
    else:
        # Use provided weights; default 0 for missing
        weights_used = {m: float(weights.get(m, 0.0)) for m in metrics_all}

    # Build contributions matrix [alg, metric] = weight * normalized(metric).
    # A metric that rank_algorithms had to skip is left as NaN rather than 0:
    # the ranking redistributes its weight over the remaining metrics, so
    # drawing it as a zero contribution would show it as the worst possible
    # value and reinstate the very penalty the redistribution removes.
    contrib = np.full((len(alg_order), len(metrics_all)), np.nan, dtype=float)
    for j, m in enumerate(metrics_all):
        lo, hi = bounds[m]
        rng = hi - lo if (np.isfinite(hi) and np.isfinite(lo)) else np.nan
        for i, alg in enumerate(alg_order):
            val = _get_metric_value(metric_summary[alg], m)
            if not np.isfinite(val) or not np.isfinite(lo) or not np.isfinite(hi) or rng == 0:
                continue
            if m in max_set and m not in min_set:
                norm = (val - lo) / (hi - lo)
            else:
                norm = (hi - val) / (hi - lo)
            contrib[i, j] = weights_used[m] * float(np.clip(norm, 0.0, 1.0))

    # Plot: scores bar + contributions heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True, gridspec_kw={'width_ratios': [1, 1.6]})

    # Left: aggregate scores barh
    ax0 = axes[0]
    y = np.arange(len(alg_order))
    # Build per-algorithm colors
    if colors_map is not None:
        bar_colors = [colors_map.get(a, '#4c72b0') for a in alg_order]
    else:
        # Fallback to a matplotlib categorical palette with enough distinct colors
        cmap_name = 'tab20' if len(alg_order) > 10 else 'tab10'
        cmap = plt.get_cmap(cmap_name, len(alg_order))
        bar_colors = [cmap(i) for i in range(len(alg_order))]

    ax0.barh(y, score_vals, color=bar_colors, edgecolor='k', linewidth=0.6)
    ax0.set_yticks(y)
    ax0.set_yticklabels(alg_order)
    ax0.invert_yaxis()
    ax0.set_xlabel('Aggregate score')
    ax0.set_title('Ranking scores')

    # Right: heatmap of per-metric contributions
    ax1 = axes[1]
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('lightgrey')
    im = ax1.imshow(np.ma.masked_invalid(contrib), aspect='auto', cmap=cmap)
    ax1.set_yticks(np.arange(len(alg_order)))
    ax1.set_yticklabels(alg_order)
    ax1.set_xticks(np.arange(len(metrics_all)))
    ax1.set_xticklabels(metrics_all, rotation=45, ha='right', fontsize=8)
    ax1.set_title('Per-metric contribution (weight × normalized)\n'
                  'grey = skipped, its weight goes to the other metrics')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Contribution', rotation=90)

    # Optional: save panel
    if results_dir is not None:
        try:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(results_dir) / filename
            fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        except Exception:
            pass

    plt.show()

    return ranked, scores, bounds