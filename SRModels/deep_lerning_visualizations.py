import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_sr_metrics(
    srcnn_metrics: dict,
    edsr_metrics: dict,
    esrgan_metrics: dict,
    title: str = "SR models: Train / Validation / Evaluation metrics",
    figsize=(14, 10),
    save_path: str | None = None):
    """
    Create a 3x3 grid of subplots comparing SRCNN, EDSR, ESRGAN metrics.
    Rows: Train (loss/PSNR/SSIM), Validation (loss/PSNR/SSIM), Evaluation (loss/PSNR/SSIM).
    Each subplot shows three bars (one per model).
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
    train_loss = [ _get(data[m], "final_train_loss") for m in models ]
    train_psnr = [ _get(data[m], "final_train_psnr") for m in models ]
    train_ssim = [ _get(data[m], "final_train_ssim") for m in models ]

    val_loss   = [ _get(data[m], "final_val_loss") for m in models ]
    val_psnr   = [ _get(data[m], "final_val_psnr") for m in models ]
    val_ssim   = [ _get(data[m], "final_val_ssim") for m in models ]

    eval_loss  = [ _get(data[m], "eval_loss") for m in models ]
    eval_psnr  = [ _get(data[m], "eval_psnr") for m in models ]
    eval_ssim  = [ _get(data[m], "eval_ssim") for m in models ]

    fig, axes = plt.subplots(3, 3, figsize=figsize)

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
    _bar(axes[0, 0], train_loss, "Train Loss", ylabel="Loss")
    _bar(axes[0, 1], train_psnr, "Train PSNR", ylabel="dB")
    _bar(axes[0, 2], train_ssim, "Train SSIM")

    # Row 2: Validation
    _bar(axes[1, 0], val_loss, "Val Loss", ylabel="Loss")
    _bar(axes[1, 1], val_psnr, "Val PSNR", ylabel="dB")
    _bar(axes[1, 2], val_ssim, "Val SSIM")

    # Row 3: Evaluation
    _bar(axes[2, 0], eval_loss, "Eval Loss", ylabel="Loss")
    _bar(axes[2, 1], eval_psnr, "Eval PSNR", ylabel="dB")
    _bar(axes[2, 2], eval_ssim, "Eval SSIM")

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
    srcnn_eval_time: float,
    edsr_eval_time: float,
    esrgan_eval_time: float,
    title: str = "SR models: Train vs Eval time (s)",
    figsize=(12, 4),
    save_path: str | None = None):
    def _get_time(m: dict | None, key: str) -> float:
        try:
            v = None if m is None else m.get(key, None)
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan

    models = ["SRCNN", "EDSR", "ESRGAN"]
    colors = {"SRCNN": "tab:blue", "EDSR": "tab:orange", "ESRGAN": "tab:green"}

    # Gather times
    train_times = [
        _get_time(srcnn_metrics, "epoch_time_sec"),
        _get_time(edsr_metrics, "epoch_time_sec"),
        _get_time(esrgan_metrics, "epoch_time_sec"),
    ]
    eval_times = [float(srcnn_eval_time), float(edsr_eval_time), float(esrgan_eval_time)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Train time subplot
    ax = axes[0]
    ax.bar(models, train_times, color=[colors[m] for m in models])
    ax.set_title("Tiempo entrenamiento (s)")
    ax.set_ylabel("Segundos")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(train_times):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Eval time subplot
    ax = axes[1]
    ax.bar(models, eval_times, color=[colors[m] for m in models])
    ax.set_title("Tiempo evaluación (s)")
    ax.set_ylabel("Segundos")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(eval_times):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

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
    srcnn_eval_mean_mb: float,
    edsr_eval_mean_mb: float,
    esrgan_eval_mean_mb: float,
    srcnn_eval_peak_mb: float,
    edsr_eval_peak_mb: float,
    esrgan_eval_peak_mb: float,
    title: str = "SR models: GPU memory (MB)",
    figsize=(14, 8),
    save_path: str | None = None,):
    def _get_train_mem_mean(m: dict | None) -> float:
        try:
            if m is None:
                return np.nan
            mem = m.get("memory", {})
            return float(mem.get("gpu_mean_current_mb", np.nan))
        except Exception:
            return np.nan

    def _get_train_mem_peak(m: dict | None) -> float:
        try:
            if m is None:
                return np.nan
            mem = m.get("memory", {})
            return float(mem.get("gpu_peak_mb", np.nan))
        except Exception:
            return np.nan

    models = ["SRCNN", "EDSR", "ESRGAN"]
    colors = {"SRCNN": "tab:blue", "EDSR": "tab:orange", "ESRGAN": "tab:green"}

    train_mean = [
        _get_train_mem_mean(srcnn_metrics),
        _get_train_mem_mean(edsr_metrics),
        _get_train_mem_mean(esrgan_metrics),
    ]
    train_peak = [
        _get_train_mem_peak(srcnn_metrics),
        _get_train_mem_peak(edsr_metrics),
        _get_train_mem_peak(esrgan_metrics),
    ]

    eval_mean = [float(srcnn_eval_mean_mb), float(edsr_eval_mean_mb), float(esrgan_eval_mean_mb)]
    eval_peak = [float(srcnn_eval_peak_mb), float(edsr_eval_peak_mb), float(esrgan_eval_peak_mb)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    def _bar(ax, values, title_text):
        ax.bar(models, values, color=[colors[m] for m in models])
        ax.set_title(title_text)
        ax.set_ylabel("MB")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    _bar(axes[0, 0], train_mean, "Entrenamiento (media)")
    _bar(axes[0, 1], train_peak, "Entrenamiento (pico)")
    _bar(axes[1, 0], eval_mean, "Evaluación (media)")
    _bar(axes[1, 1], eval_peak, "Evaluación (pico)")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, "sr_models_memory.png"), dpi=150)
    return fig, axes

def plot_confusion(ax, cm, classes, title):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    return im

def plot_2x2(images, titles=None, cmap='gray'):
    if len(images) != 4:
        raise ValueError("Debes pasar exactamente 4 imágenes.")

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap=cmap)
        if titles and len(titles) == 4:
            ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()