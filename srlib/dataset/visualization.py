import os

import matplotlib.pyplot as plt
import numpy as np

from srlib.constants import EDA_RESULTS_DIR, HR_ROOT, LR_ROOT

def _save(fig, output_path, **kwargs):
    """Write a figure, creating its directory when it is missing."""

    if not output_path:
        return

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig.savefig(output_path, bbox_inches="tight", **kwargs)

def _pick_sample_basenames(folder, count, indices=None):
    """
    Choose which images of a class folder to show.

    Without explicit indices the picks are spread evenly across the folder,
    which is more representative than taking the first few and does not
    break when the folder has fewer images than requested.
    """

    names = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".png")
    )
    if not names:
        raise ValueError(f"No PNG images found under {folder}")

    if indices is None:
        step = max(1, len(names) // (count + 1))
        chosen = [min((i + 1) * step, len(names) - 1) for i in range(count)]
    else:
        chosen = [min(int(i), len(names) - 1) for i in indices[:count]]

    return [names[i] for i in chosen]

def plot_hr_lr_samples(
        class_folders=("low_z_offset", "high_z_offset"),
        samples_per_folder=2,
        sample_indices=None,
        output_path=None,
        figsize=(8, 4)):
    """
    Show HR images next to their LR counterparts, one column per sample.

    HR on the top row and LR on the bottom, so the loss of detail the
    degradation introduces can be read column by column.

    Parameters
    ----------
    class_folders : sequence of str
        Subfolders of the HR and LR roots to sample from, in plotting order.
    samples_per_folder : int
        Columns to draw per folder.
    sample_indices : dict, optional
        ``{folder: [index, ...]}`` to pin specific images. Indices are into
        the sorted file list and are clamped to the folder size. Defaults to
        evenly spread picks.
    output_path : str, optional
        Where to write the figure. Defaults to ``hr_lr_samples.png`` under
        the EDA results directory. Pass an empty string to skip saving.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """

    sample_indices = sample_indices or {}
    if output_path is None:
        output_path = os.path.join(EDA_RESULTS_DIR, "hr_lr_samples.png")

    selected = [
        (folder, basename)
        for folder in class_folders
        for basename in _pick_sample_basenames(
            os.path.join(HR_ROOT, folder),
            samples_per_folder,
            sample_indices.get(folder),
        )
    ]

    fig, axes = plt.subplots(2, len(selected), figsize=figsize)
    for column, (folder, basename) in enumerate(selected):
        for row, root in enumerate((HR_ROOT, LR_ROOT)):
            axes[row, column].imshow(
                plt.imread(os.path.join(root, folder, basename))
            )
            axes[row, column].axis("off")

    fig.text(0.0, 0.75, "HR", va="center", ha="center", fontsize=12)
    fig.text(0.0, 0.25, "LR", va="center", ha="center", fontsize=12)
    plt.tight_layout()
    _save(fig, output_path, dpi=300)

    return fig, axes

def plot_patch_pairs(
        X, Y, count=3, seed=None, output_path=None, figsize=None):
    """
    Show random input/target patch pairs exactly as a model receives them.

    This is the check that the loader handed the model what it was meant
    to: the input on the left, its target on the right, with the shapes in
    the titles so a scale-factor mistake is visible at a glance.

    Parameters
    ----------
    X, Y : np.ndarray
        Input and target patch arrays of the same partition.
    count : int
        Rows to draw.
    seed : int, optional
        Seed of the row picks, so a figure can be reproduced.
    output_path : str, optional
        Where to write the figure. Nothing is written when omitted.
    figsize : tuple, optional
        Defaults to a size that scales with ``count``.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """

    if len(X) != len(Y):
        raise ValueError(
            f"X and Y must be aligned, got {len(X)} and {len(Y)} patches."
        )

    count = min(count, len(X))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(X), size=count, replace=False)
    title_font = {"family": "serif", "size": 10}

    fig, axes = plt.subplots(
        count, 2, figsize=figsize or (6, 2.6 * count), squeeze=False
    )
    for row, index in enumerate(picks):
        for column, (label, patches) in enumerate((("LR", X), ("HR", Y))):
            axes[row, column].imshow(np.clip(patches[index], 0.0, 1.0))
            axes[row, column].set_title(
                f"{label} {patches[index].shape}", fontdict=title_font
            )
            axes[row, column].axis("off")

    plt.tight_layout()
    _save(fig, output_path, dpi=300)

    return fig, axes

def plot_classification_patches(
        patches_by_source, labels_by_source, count=4, seed=None,
        output_path=None, figsize=None):
    """
    Show training patches of every classifier variant side by side.

    Patches have the same pixel size in every row, so what the figure makes
    visible is the object scale each variant learns to recognise, which is
    the only thing that separates the two classifiers.

    Parameters
    ----------
    patches_by_source : dict
        ``{source: X_train}`` for each variant, drawn in insertion order.
    labels_by_source : dict
        ``{source: y_train}`` aligned with ``patches_by_source``.
    count : int
        Columns to draw per variant.
    seed : int, optional
        Seed of the column picks.
    output_path : str, optional
        Where to write the figure. Nothing is written when omitted.
    figsize : tuple, optional
        Defaults to a size that scales with the number of variants.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """

    sources = list(patches_by_source)
    rng = np.random.default_rng(seed)
    title_font = {"family": "serif", "size": 9}

    fig, axes = plt.subplots(
        len(sources), count,
        figsize=figsize or (2 * count, 2.2 * len(sources)),
        squeeze=False,
    )
    for row, source in enumerate(sources):
        X, y = patches_by_source[source], labels_by_source[source]
        for column, index in enumerate(
                rng.choice(len(X), size=min(count, len(X)), replace=False)):
            axes[row, column].imshow(np.clip(X[index], 0.0, 1.0))
            axes[row, column].set_title(
                f"{source.upper()} - label {y[index]}", fontdict=title_font
            )
            axes[row, column].axis("off")

    plt.tight_layout()
    _save(fig, output_path, dpi=300)

    return fig, axes
