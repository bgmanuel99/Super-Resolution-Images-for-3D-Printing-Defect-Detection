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

def plot_classification_reports_panel(y_true, algo_names, preds_lists, class_names=None, save_dir='DL_results', prefix='cls_report'):
    """
    Genera una única figura con 3 filas x 2 columnas que contiene:
      (1,1) Accuracy global por algoritmo (barras)
      (1,2) Recall MACRO por algoritmo (barras)
      (2,1) Macro F1 por algoritmo (barras)
      (2,2) Weighted F1 por algoritmo (barras)
      (3,1) Heatmap de F1 por clase y algoritmo
      (3,2) Heatmap de Accuracy por clase y algoritmo (equivale a recall por clase)

    Parámetros:
    - y_true: array-like de labels reales.
    - algo_names: lista[str] con nombres de algoritmos (longitud N).
    - preds_lists: lista de predicciones (longitud N), cada una array-like del mismo largo que y_true.
    - class_names: nombres de clase; si None, se infieren de y_true.
    - save_dir: carpeta donde guardar.
    - prefix: prefijo del archivo (se guardará como f"{prefix}_panel.png").

    Devuelve: (fig, axes, metrics)
    metrics = {
      'accuracy': [...], 'macro_f1': [...], 'weighted_f1': [...],
      'macro_recall': [...], 'f1_per_class': (C,N), 'acc_per_class': (C,N)
    }
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report

    y_true = np.asarray(y_true)
    classes_sorted = sorted(np.unique(y_true))
    if class_names is None:
        class_names = [str(c) for c in classes_sorted]

    n_methods = len(algo_names)
    n_classes = len(class_names)

    accuracies = []
    macro_f1s = []
    weighted_f1s = []
    macro_recalls = []

    f1_per_class = np.full((n_classes, n_methods), np.nan, dtype=float)
    acc_per_class = np.full((n_classes, n_methods), np.nan, dtype=float)  # recall por clase

    for j, (name, y_pred) in enumerate(zip(algo_names, preds_lists)):
        y_pred = np.asarray(y_pred)
        n = int(min(len(y_true), len(y_pred)))
        if n == 0:
            accuracies.append(np.nan)
            macro_f1s.append(np.nan)
            weighted_f1s.append(np.nan)
            macro_recalls.append(np.nan)
            continue

        yt = y_true[:n]
        yp = y_pred[:n]

        report = classification_report(
            yt, yp,
            labels=classes_sorted,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        accuracies.append(float(report.get('accuracy', np.nan)))
        macro_f1s.append(float(report.get('macro avg', {}).get('f1-score', np.nan)))
        weighted_f1s.append(float(report.get('weighted avg', {}).get('f1-score', np.nan)))
        macro_recalls.append(float(report.get('macro avg', {}).get('recall', np.nan)))

        for i, cname in enumerate(class_names):
            # F1 por clase
            try:
                f1 = report[cname]['f1-score']
            except KeyError:
                f1 = np.nan
            f1_per_class[i, j] = float(f1) if f1 is not None else np.nan

            # Accuracy por clase (proporción de aciertos dentro de esa clase) = recall de la clase
            try:
                r = report[cname]['recall']
            except KeyError:
                r = np.nan
            acc_per_class[i, j] = float(r) if r is not None else np.nan

    # Figura única 3x2
    fig, axes = plt.subplots(3, 2, figsize=(22, 16))
    x = np.arange(n_methods)

    # (1,1) Accuracy global
    ax = axes[0, 0]
    bars = ax.bar(x, accuracies, color='tab:blue', alpha=0.88)
    ax.set_title('Accuracy global por algoritmo (↑ mejor)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.25)
    for b, v in zip(bars, accuracies):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # (1,2) Recall MACRO por algoritmo
    ax = axes[0, 1]
    bars = ax.bar(x, macro_recalls, color='tab:purple', alpha=0.88)
    ax.set_title('Recall (Macro) por algoritmo (↑ mejor)')
    ax.set_ylabel('Recall (Macro)')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.25)
    for b, v in zip(bars, macro_recalls):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # (2,1) Macro F1
    ax = axes[1, 0]
    bars = ax.bar(x, macro_f1s, color='tab:green', alpha=0.88)
    ax.set_title('Macro F1 por algoritmo (↑ mejor)')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.25)
    for b, v in zip(bars, macro_f1s):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # (2,2) Weighted F1
    ax = axes[1, 1]
    bars = ax.bar(x, weighted_f1s, color='tab:orange', alpha=0.88)
    ax.set_title('Weighted F1 por algoritmo (↑ mejor)')
    ax.set_ylabel('Weighted F1')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.25)
    for b, v in zip(bars, weighted_f1s):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # Ejes X para barras
    for r in [0, 1]:
        for c in [0, 1]:
            axes[r, c].set_xticks(x)
            axes[r, c].set_xticklabels(algo_names, rotation=30, ha='right')

    # (3,1) Heatmap F1 por clase
    ax = axes[2, 0]
    im = ax.imshow(f1_per_class, interpolation='nearest', cmap='YlGnBu', vmin=0.0, vmax=1.0)
    ax.set_title('F1-score por clase y algoritmo')
    ax.set_xlabel('Algoritmo / Método')
    ax.set_ylabel('Clase')
    ax.set_xticks(np.arange(n_methods))
    ax.set_xticklabels(algo_names, rotation=30, ha='right')
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(class_names)
    for i in range(n_classes):
        for j in range(n_methods):
            v = f1_per_class[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('F1-score')

    # (3,2) Heatmap Accuracy por clase (recall por clase)
    ax = axes[2, 1]
    im2 = ax.imshow(acc_per_class, interpolation='nearest', cmap='YlOrRd', vmin=0.0, vmax=1.0)
    ax.set_title('Accuracy por clase y algoritmo (≡ recall por clase)')
    ax.set_xlabel('Algoritmo / Método')
    ax.set_ylabel('Clase')
    ax.set_xticks(np.arange(n_methods))
    ax.set_xticklabels(algo_names, rotation=30, ha='right')
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(class_names)
    for i in range(n_classes):
        for j in range(n_methods):
            v = acc_per_class[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='black')
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Accuracy por clase')

    plt.tight_layout(rect=(0, 0, 1, 0.98))

    # Guardar una sola imagen con todo el panel
    try:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f'{prefix}_panel.png')
        fig.savefig(out_path, dpi=150)
    except Exception as e:
        print('Aviso: no se pudo guardar panel:', e)

    metrics = {
        'accuracy': accuracies,
        'macro_f1': macro_f1s,
        'weighted_f1': weighted_f1s,
        'macro_recall': macro_recalls,
        'f1_per_class': f1_per_class,
        'acc_per_class': acc_per_class,
    }
    return fig, axes, metrics

def plot_4x3(images, titles=None, cmap='gray'):
    """
    Plot a 4x3 panel (4 rows x 3 columns).
    - If fewer than 12 images are provided, remaining cells are left blank.
    - Titles are applied when provided (matched by index).
    """
    
    if not isinstance(images, (list, tuple)):
        raise ValueError("'images' debe ser una lista o tupla de imágenes (np.ndarray).")

    rows, cols = 4, 3
    total = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    n = len(images)
    for i in range(total):
        ax = axes[i]
        if i < n:
            ax.imshow(images[i], cmap=cmap)
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
def plot_confidence_panel(y, algo_names, label_lists, conf_lists, save_dir='DL_results', filename='sr_confidence_panel.png'):
    mean_conf_all = []
    mean_conf_correct = []
    mean_conf_wrong = []
    error_rates = []
    counts = []
    counts_correct = []
    counts_wrong = []

    yt = np.asarray(y, dtype=int)

    for preds, confs in zip(label_lists, conf_lists):
        yp = np.asarray(preds, dtype=int)
        cf = np.asarray(confs, dtype=float)
        n = int(min(len(yt), len(yp), len(cf)))
        if n == 0:
            mean_conf_all.append(np.nan)
            mean_conf_correct.append(np.nan)
            mean_conf_wrong.append(np.nan)
            error_rates.append(np.nan)
            counts.append(0)
            counts_correct.append(0)
            counts_wrong.append(0)
            continue
        y_true = yt[:n]
        y_pred = yp[:n]
        cfs = cf[:n]

        correct = (y_pred == y_true)
        err_rate = 1.0 - float(np.mean(correct))

        mean_all = float(np.nanmean(cfs)) if n > 0 else np.nan
        mean_corr = float(np.nanmean(cfs[correct])) if np.any(correct) else np.nan
        mean_wrong = float(np.nanmean(cfs[~correct])) if np.any(~correct) else np.nan

        mean_conf_all.append(mean_all)
        mean_conf_correct.append(mean_corr)
        mean_conf_wrong.append(mean_wrong)
        error_rates.append(err_rate)
        counts.append(n)
        counts_correct.append(int(np.sum(correct)))
        counts_wrong.append(int(n - np.sum(correct)))

    # Figura única con 3 subplots apilados
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    idx = np.arange(len(algo_names))

    # Subplot 1: confianza media global
    bars1 = axes[0].bar(idx, mean_conf_all, color='tab:blue', alpha=0.85)
    axes[0].set_ylabel('Confianza media')
    axes[0].set_title('Confianza media global por algoritmo / método de SR')
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis='y', alpha=0.25)
    for b, m, n in zip(bars1, mean_conf_all, counts):
        if np.isfinite(m):
            axes[0].text(b.get_x() + b.get_width()/2, m, f'{m:.2f}\n(n={n})', ha='center', va='bottom', fontsize=8)

    # Subplot 2: barras agrupadas (global/correctas/incorrectas)
    w = 0.25
    axes[1].bar(idx - w, mean_conf_all, width=w, label='Media', color='tab:blue', alpha=0.85)
    axes[1].bar(idx,      mean_conf_correct, width=w, label='Correctas', color='tab:green', alpha=0.85)
    axes[1].bar(idx + w,  mean_conf_wrong, width=w, label='Incorrectas', color='tab:red', alpha=0.75)
    axes[1].set_ylabel('Confianza')
    axes[1].set_title('Confianza media: global, aciertos, errores')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].legend(ncols=3, loc='upper center')
    for i in range(len(algo_names)):
        vals = [mean_conf_all[i], mean_conf_correct[i], mean_conf_wrong[i]]
        xs = [idx[i] - w, idx[i], idx[i] + w]
        for xv, v in zip(xs, vals):
            if np.isfinite(v):
                axes[1].text(xv, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # Subplot 3: tasa de error (1 - accuracy)
    bars3 = axes[2].bar(idx, error_rates, color='tab:red', alpha=0.8)
    axes[2].set_xticks(idx)
    axes[2].set_xticklabels(algo_names, rotation=30, ha='right')
    axes[2].set_ylabel('Tasa de error')
    axes[2].set_title('Error por algoritmo / método de SR (1 - accuracy)')
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(axis='y', alpha=0.25)
    for b, e, nc, nw in zip(bars3, error_rates, counts_correct, counts_wrong):
        if np.isfinite(e):
            axes[2].text(b.get_x() + b.get_width()/2, e, f'{e:.2f}\n(ok={nc}, err={nw})', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Guardar figura combinada
    try:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), dpi=150)
    except Exception as e:
        print('Aviso: no se pudo guardar', filename, ':', e)

    plt.show()