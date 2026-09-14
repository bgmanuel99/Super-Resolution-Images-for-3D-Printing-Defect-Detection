import os
import numpy as np
import matplotlib.pyplot as plt

def plot_sr_metrics(
    srcnn_metrics: dict,
    edsr_metrics: dict,
    esrgan_metrics: dict,
    title: str = "SR models: Train / Validation / Evaluation metrics",
    figsize=(10, 10),
    save_path: str | None = None):
    """
    Create a 3x2 grid of subplots comparing SRCNN, EDSR, ESRGAN metrics.
    Rows: Train (PSNR/SSIM), Validation (PSNR/SSIM), Evaluation (PSNR/SSIM).
    Each subplot shows three bars (one per model).

    Loss is deliberately not plotted. The three models do not minimise the
    same quantity -- SRCNN uses MSE, EDSR uses MAE and ESRGAN a weighted sum
    of perceptual, adversarial, pixel and spectral terms -- so their losses
    differ by four orders of magnitude and share no axis. PSNR and SSIM are
    the two quantities that do mean the same thing for all three.
    """

    def _get(m: dict | None, key: str) -> float:
        try:
            v = None if m is None else m.get(key, None)
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    models = ["SRCNN", "EDSR", "ESRGAN"]
    data = {
        "SRCNN": srcnn_metrics,
        "EDSR": edsr_metrics,
        "ESRGAN": esrgan_metrics,
    }
    colors = {"SRCNN": "tab:blue", "EDSR": "tab:orange", "ESRGAN": "tab:green"}

    # Collect values per subplot
    train_psnr = [ _get(data[m], "final_train_psnr") for m in models ]
    train_ssim = [ _get(data[m], "final_train_ssim") for m in models ]

    val_psnr   = [ _get(data[m], "final_val_psnr") for m in models ]
    val_ssim   = [ _get(data[m], "final_val_ssim") for m in models ]

    eval_psnr  = [ _get(data[m], "eval_psnr") for m in models ]
    eval_ssim  = [ _get(data[m], "eval_ssim") for m in models ]

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    def _bar(ax, values, title_text, ylabel=None):
        ax.bar(models, values, color=[colors[m] for m in models])
        ax.set_title(title_text)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Row 1: Train
    _bar(axes[0, 0], train_psnr, "Train PSNR", ylabel="dB")
    _bar(axes[0, 1], train_ssim, "Train SSIM")

    # Row 2: Validation
    _bar(axes[1, 0], val_psnr, "Val PSNR", ylabel="dB")
    _bar(axes[1, 1], val_ssim, "Val SSIM")

    # Row 3: Evaluation
    _bar(axes[2, 0], eval_psnr, "Eval PSNR", ylabel="dB")
    _bar(axes[2, 1], eval_ssim, "Eval SSIM")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "sr_models_metrics.png"), dpi=150)
    return fig, axes

def plot_sr_time(
    srcnn_metrics: dict,
    edsr_metrics: dict,
    esrgan_metrics: dict,
    title: str = "SR models: training and evaluation time (s)",
    figsize=(12, 4),
    save_path: str | None = None):
    """
    Compare the time the three SR models spend training and evaluating.

    Training is the mean over epochs and evaluation is the single pass over
    the test set, so the two panels are not on the same scale and are drawn
    apart. Reconstructing a full frame is not included: that cost belongs
    to the pipeline that slices the image, not to the model.
    """

    def _get_time(m: dict | None, key: str) -> float:
        try:
            v = None if m is None else m.get(key, None)
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    models = ["SRCNN", "EDSR", "ESRGAN"]
    colors = {"SRCNN": "tab:blue", "EDSR": "tab:orange", "ESRGAN": "tab:green"}

    train_times = [
        _get_time(srcnn_metrics, "epoch_time_sec"),
        _get_time(edsr_metrics, "epoch_time_sec"),
        _get_time(esrgan_metrics, "epoch_time_sec"),
    ]
    eval_times = [
        _get_time(srcnn_metrics, "eval_time_sec"),
        _get_time(edsr_metrics, "eval_time_sec"),
        _get_time(esrgan_metrics, "eval_time_sec"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _bar(ax, values, title_text):
        ax.bar(models, values, color=[colors[m] for m in models])
        ax.set_title(title_text)
        ax.set_ylabel("Seconds")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    _bar(axes[0], train_times, "Training (mean per epoch)")
    _bar(axes[1], eval_times, "Evaluation (test set)")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "sr_models_time.png"), dpi=150)
    return fig, axes

def plot_sr_memory(
    srcnn_metrics: dict,
    edsr_metrics: dict,
    esrgan_metrics: dict,
    title: str = "SR models: training and evaluation GPU memory (MB)",
    figsize=(14, 8),
    save_path: str | None = None,):
    """
    Compare the mean and peak GPU memory over training and evaluation.

    Reconstructing a full frame is not included: that cost belongs to the
    pipeline that slices the image, not to the model.
    """

    def _get_mem(m: dict | None, section: str, key: str) -> float:
        try:
            if m is None:
                return np.nan
            return float(m.get(section, {}).get(key, np.nan))
        except Exception:
            return np.nan

    models = ["SRCNN", "EDSR", "ESRGAN"]
    colors = {"SRCNN": "tab:blue", "EDSR": "tab:orange", "ESRGAN": "tab:green"}
    metrics = [srcnn_metrics, edsr_metrics, esrgan_metrics]

    panels = [
        ("memory", "gpu_mean_current_mb", "Training (mean)"),
        ("memory", "gpu_peak_mb", "Training (peak)"),
        ("eval_memory", "gpu_mean_current_mb", "Evaluation (mean)"),
        ("eval_memory", "gpu_peak_mb", "Evaluation (peak)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, (section, key, panel_title) in zip(axes.ravel(), panels):
        values = [_get_mem(m, section, key) for m in metrics]
        ax.bar(models, values, color=[colors[m] for m in models])
        ax.set_title(panel_title)
        ax.set_ylabel("MB")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "sr_models_memory.png"), dpi=150)
    return fig, axes

def plot_vgg16_training_curves(
    metrics_by_source: dict,
    title: str = "VGG16 fine-tuning: loss and accuracy",
    figsize=(14, 5),
    save_path: str | None = None):
    """
    Plot the loss and accuracy curves of every VGG16 variant.

    Both variants are drawn on the same pair of axes so the resolution they
    were trained on can be compared directly. Each variant owns a hue, and
    within that hue the two splits are told apart by three cues at once:
    training is the dark shade, solid, with round markers, and validation is
    the light shade, dashed, with square markers. One cue alone is not
    enough on a sixty-epoch curve where the two splits overlap.

    A dotted horizontal line is the variant's test score. The vertical line
    is where its first phase ended and the backbone was unfrozen, which is
    what the step in every curve at that epoch comes from; both lines carry
    their own legend entry so the figure needs no external caption.

    The loss panel is logarithmic because the two phases live on different
    scales: the head starts above 0.5 and the fine-tuning settles below
    0.05, so on a linear axis the whole second phase collapses into the
    bottom tenth of the panel.

    Parameters
    ----------
    metrics_by_source : dict
        Maps a source tag such as ``'hr'`` or ``'lr'`` to the metrics dict
        saved by the training notebook.
    save_path : str, optional
        Directory the figure is written to. Nothing is saved when omitted.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """

    # Dark shade for training, light shade of the same hue for validation.
    palette = {
        "hr": {"train": "#14375e", "val": "#7aa6d2"},
        "lr": {"train": "#8a3c07", "val": "#f0a35e"},
    }
    fallback = {"train": "#333333", "val": "#999999"}
    panels = [
        ("loss", "final_train_loss", "final_val_loss", "eval_loss", "Loss"),
        ("accuracy", "final_train_accuracy", "final_val_accuracy",
         "eval_accuracy", "Accuracy"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, (name, train_key, val_key, eval_key, ylabel) in zip(axes, panels):
        for source, metrics in metrics_by_source.items():
            shades = palette.get(source, fallback)
            tag = source.upper()

            for key, split, style, marker in (
                    (train_key, "train", "-", "o"),
                    (val_key, "val", "--", "s")):
                curve = metrics.get(key)
                if not isinstance(curve, (list, tuple)) or len(curve) == 0:
                    continue
                ax.plot(
                    range(1, len(curve) + 1), curve,
                    style, color=shades[split], marker=marker, markersize=4,
                    # A marker on every epoch turns a long curve into a solid
                    # band and hides the shape it is there to show.
                    markevery=max(1, len(curve) // 12),
                    linewidth=1.6, zorder=3,
                    label=f"{tag} {split}",
                )

            score = metrics.get(eval_key)
            if score is not None:
                ax.axhline(
                    float(score), color=shades["train"], ls=":", linewidth=1.4,
                    alpha=0.9, zorder=2,
                    label=f"{tag} test = {float(score):.4f}",
                )

            # Phase 1 trains the head over a frozen backbone and phase 2
            # opens its last layers, so both the loss and the accuracy step
            # at this epoch by construction rather than by instability.
            head_epochs = metrics.get("head_epochs_run")
            if head_epochs:
                ax.axvline(
                    float(head_epochs) + 0.5, color=shades["train"],
                    ls="-.", linewidth=1.2, alpha=0.55, zorder=1,
                    label=(f"{tag} backbone unfrozen "
                           f"(phase 2 from epoch {int(head_epochs) + 1})"),
                )

        ax.set_title(name.capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if name == "loss":
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2, framealpha=0.9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(save_path, "vgg16_training_curves.png"), dpi=150
        )
    return fig, axes

