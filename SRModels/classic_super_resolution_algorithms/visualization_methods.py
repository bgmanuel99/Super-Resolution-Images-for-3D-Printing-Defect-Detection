import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from skimage.metrics import structural_similarity as ssim

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

    Returns: (fig, ax)
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
    legend2 = ax.legend(handles=size_handles, title='Memory (mean, MB)', loc='upper left', bbox_to_anchor=(1.02, 0.55))
    ax.add_artist(legend1)

    plt.show()

    # Optional: save figure
    if save and results_dir is not None:
        try:
            out_fig = Path(results_dir) / 'speed_quality_tradeoff_3d.png'
            fig.savefig(out_fig, dpi=150, bbox_inches='tight')
        except Exception:
            pass

def plot_error_metrics_grid(metric_summary, algorithms, colors, results_dir=None, figsize=(14, 8)):
    """
    Display and save a 2x2 grid with MAE/RMSE mean and max across algorithms.

    Panels (row-major):
      [0,0] MAE Mean
      [0,1] MAE Max
      [1,0] RMSE Mean
      [1,1] RMSE Max
    """

    # Collect values strictly from metric_summary
    mae_mean = [metric_summary[a].get('mae_mean', np.nan) for a in algorithms]
    mae_max  = [metric_summary[a].get('mae_max',  np.nan) for a in algorithms]
    rmse_mean = [metric_summary[a].get('rmse_mean', np.nan) for a in algorithms]
    rmse_max  = [metric_summary[a].get('rmse_max',  np.nan) for a in algorithms]

    def _bar(ax, data, title, fmt='{:.4g}'):
        x = np.arange(len(algorithms))
        # Pre-compute dynamic limits with headroom for labels
        data_arr = np.array(data, dtype=float)
        valid = data_arr[np.isfinite(data_arr)]
        if valid.size:
            ymin = 0.0 if valid.min() >= 0 else float(valid.min())
            span = float(valid.max() - ymin)
            if not np.isfinite(span) or span <= 0:
                span = 1.0
            margin = 0.10 * span  # headroom for text above bars
            ax.set_ylim(ymin, float(valid.max()) + margin)
        bars = ax.bar(x, data, color=[colors[a] for a in algorithms])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=30, ha='right')
        # annotate
        bottom, top = ax.get_ylim()
        span = top - bottom if np.isfinite(top - bottom) and (top - bottom) > 0 else 1.0
        pad = 0.02 * span
        max_needed = -np.inf
        for rect, val in zip(bars, data):
            if not (isinstance(val, (int, float)) and np.isfinite(val)):
                continue
            label_y = rect.get_height() + pad * 0.6
            ax.text(rect.get_x() + rect.get_width()/2, label_y, fmt.format(val), ha='center', va='bottom', fontsize=8)
            max_needed = max(max_needed, label_y)
        # If labels would overflow, expand top with a small extra
        if np.isfinite(max_needed) and max_needed > top:
            extra = max(0.03 * (max_needed - bottom), 0.03)
            ax.set_ylim(bottom, max_needed + extra)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    _bar(axes[0,0], mae_mean, 'MAE Mean') # Lower is better
    _bar(axes[0,1], mae_max,  'MAE Max') # Lower is better
    _bar(axes[1,0], rmse_mean,'RMSE Mean') # Lower is better
    _bar(axes[1,1], rmse_max, 'RMSE Max') # Lower is better

    fig.suptitle('Error Metrics: MAE & RMSE (Mean/Max)')
    if results_dir is not None:
        try:
            out = Path(results_dir) / 'error_metrics_mae_rmse.png'
            fig.savefig(out, dpi=150, bbox_inches='tight')
        except Exception:
            pass
    plt.show()

def plot_edge_metrics_grid(metric_summary, algorithms, colors, results_dir=None, figsize=(12, 5)):
    """
    Display and save a 1x2 grid with mean values of Gradient MSE and EPI across algorithms.

    Panels:
      [0] Gradient MSE Mean (lower is better)
      [1] Edge Preservation Index (EPI) Mean (≈1 is ideal)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

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
            out = Path(results_dir) / 'edge_gradient_metrics_mean.png'
            fig.savefig(out, dpi=150, bbox_inches='tight')
        except Exception:
            pass
    plt.show()

def plot_frequency_distribution_metrics_grid(metric_summary, algorithms, colors, results_dir=None, figsize=(16, 5)):
    """
    Display and save a 1x3 grid with mean values of:
      - HF Energy Ratio mean (relative)
      - KL Luma mean
      - KL Color mean (may be NaN for grayscale-only methods)
    across algorithms.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    hf_ratio_mean = [metric_summary[a].get('hf_ratio_mean', np.nan) for a in algorithms]
    kl_luma_mean = [metric_summary[a].get('kl_luma_mean', np.nan) for a in algorithms]
    kl_color_mean = [metric_summary[a].get('kl_color_mean', np.nan) for a in algorithms]

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

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    _bar(axes[0], hf_ratio_mean, 'High-Frequency Energy Ratio Mean (relative)')
    _bar(axes[1], kl_luma_mean, 'KL Divergence (Luma) Mean') # Lower is better
    _bar(axes[2], kl_color_mean, 'KL Divergence (Color) Mean') # Lower is better

    fig.suptitle('Frequency/Distribution Metrics: Mean Values')
    if results_dir is not None:
        try:
            out = Path(results_dir) / 'freq_distribution_metrics_mean.png'
            fig.savefig(out, dpi=150, bbox_inches='tight')
        except Exception:
            pass
    plt.show()
    
def plot_and_save_super_resolution_example(vis, ibp_example, nlm_example, egi_example, freq_example, results_dir):
    hr_img_v, lr_img_v, bilinear_v, bicubic_v, area_v, lanczos_v = vis
    hr_g_v, lr_g_v, ibp_v = ibp_example
    hr_v, nlm_v = nlm_example
    hr_egi_v, lr_egi_v, egi_v = egi_example
    hr_freq_v, freq_v = freq_example

    def to_display(img):
        if img.ndim == 2:
            return img if img.dtype != np.float32 else np.clip(img, 0, 1)
        return img

    images = [
        ('HR', hr_img_v),
        ('LR', lr_img_v),
        ('Bilinear', bilinear_v),
        ('Bicubic', bicubic_v),
        ('Area', area_v),
        ('Lanczos', lanczos_v),
        ('IBP', ibp_v),
        ('NLM', nlm_v if nlm_v.ndim == 3 else nlm_v),
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
    out_grid = results_dir / 'super_resolution_example.png'
    plt.savefig(out_grid, dpi=150)
    
def plot_and_save_ssim_similarity_maps(vis, ibp_example, nlm_example, egi_example, freq_example, results_dir):
    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    hr_img_v, lr_img_v, bilinear_v, bicubic_v, area_v, lanczos_v = vis
    hr_g_v, lr_g_v, ibp_v = ibp_example
    hr_v, nlm_v = nlm_example
    hr_egi_v, lr_egi_v, egi_v = egi_example
    hr_freq_v, freq_v = freq_example

    # Lista de pares (nombre, HR, SR) en el orden requerido
    pairs = [
        ('Bilinear', to_gray(hr_img_v), to_gray(bilinear_v)),
        ('Bicubic',  to_gray(hr_img_v), to_gray(bicubic_v)),
        ('Area',     to_gray(hr_img_v), to_gray(area_v)),
        ('Lanczos',  to_gray(hr_img_v), to_gray(lanczos_v)),
        ('IBP',      hr_g_v, ibp_v),
        ('NLM',      hr_v, nlm_v if nlm_v.ndim == 2 else cv2.cvtColor(nlm_v, cv2.COLOR_RGB2GRAY)),
        ('EGI',      hr_egi_v, egi_v),
        ('FREQ',     hr_freq_v, freq_v),
    ]

    ssim_maps = []
    titles = []
    for name, hr_g, sr_g in pairs:
        data_range = 255 if hr_g.dtype != np.float32 else 1.0
        val, ssim_map = ssim(hr_g, sr_g, data_range=data_range, full=True)
        ssim_maps.append((ssim_map, val))
        titles.append(name)

    plt.figure(figsize=(20, 6))
    for i, ((ssim_map, val), name) in enumerate(zip(ssim_maps, titles), start=1):
        plt.subplot(2, 4, i)
        plt.imshow(ssim_map, cmap='gray', vmin=0, vmax=1)
        plt.title(f"{name}\nSSIM={val:.4f}")
        plt.axis('off')
    plt.tight_layout()
    out_diff = results_dir / 'ssim_similarity_maps.png'
    plt.savefig(out_diff, dpi=150)