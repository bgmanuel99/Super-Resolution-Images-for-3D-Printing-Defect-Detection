import os
import pickle
import re
import shutil
from datetime import datetime

from srlib.constants import (
    ESRGAN_PREVIEW_STAGING,
    ESRGAN_PREVIEW_SUBDIR,
    MODEL_DEFAULT_SCALE_FACTORS,
    MODEL_FAMILY_ROOTS,
    MODELS,
    TIMESTAMP_FORMAT,
    TIMESTAMP_PATTERN,
)

_TIMESTAMP_RE = re.compile(TIMESTAMP_PATTERN)

def prepare_run_directory(model, run_name):
    """Create and return the directory a training run writes into.

    Owned by this module because it is the same path ``model_artifacts``
    resolves when the run is read back. Creating it here also means a
    deleted ``models/`` tree is rebuilt by the next run instead of raising.

    Parameters
    ----------
    model : str
        Model family, one of ``MODELS``.
    run_name : str
        Directory name, e.g. ``SRCNN_20260828_181500``.

    Returns
    -------
    str
        Absolute path of the created run directory.
    """

    run_dir = os.path.join(MODEL_FAMILY_ROOTS[_validate_model(model)], run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir

def save_model_summary(run_dir, run_name, models, line_length=110):
    """Write the architecture of every network of a run to a text file.

    Saved next to the weights because the weights alone do not say what
    shape they belong to: reading a checkpoint back needs the same code
    that built it, and this file is what lets a run be understood without
    re-running the notebook that produced it.

    Parameters
    ----------
    run_dir : str
        Directory returned by ``prepare_run_directory``.
    run_name : str
        Same name used for the directory, used as the filename stem.
    models : keras.Model or dict
        A single network, or ``{label: network}`` when the run trained
        more than one, as ESRGAN does with its generator and discriminator.
    line_length : int
        Width the summary is formatted to. Fixed rather than left to Keras
        so the file does not change with the terminal it was written from.

    Returns
    -------
    str
        Path of the written file.
    """

    if not isinstance(models, dict):
        models = {None: models}

    lines = []
    for label, model in models.items():
        if label is not None:
            lines += ["=" * line_length, f" {label}", "=" * line_length]
        model.summary(print_fn=lines.append, line_length=line_length)
        lines.append("")

    return _write_text(run_dir, f"{run_name}_summary.txt", lines)

def _format_metric(value):
    """Render one metric the way the Keras progress line does."""

    if value is None:
        return "None"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if value != value:
        return "nan"
    if value == 0.0 or 1e-3 <= abs(value) < 1e5:
        return f"{value:.4f}"

    return f"{value:.4e}"

def _ordered_curve_names(curves):
    """
    Order the curves so each validation one follows its training partner.

    Insertion order alone scatters them, and a line reading
    ``loss ... val_loss`` is what makes the two comparable at a glance.
    """

    names, seen = [], set()
    for name in curves:
        if name.startswith("val_"):
            continue
        names.append(name)
        seen.add(name)
        partner = f"val_{name}"
        if partner in curves:
            names.append(partner)
            seen.add(partner)

    return names + [name for name in curves if name not in seen]

def save_epoch_log(run_dir, run_name, history):
    """Write one line per epoch with every curve the run recorded.

    The metrics pickle keeps only the last value of each curve, and the
    console scrollback is gone as soon as the kernel restarts, so without
    this file the shape of a training run cannot be recovered afterwards.

    Parameters
    ----------
    run_dir : str
        Directory returned by ``prepare_run_directory``.
    run_name : str
        Same name used for the directory, used as the filename stem.
    history : dict
        Either ``{curve: per-epoch values}``, or ``{phase: {curve: ...}}``
        when the run had more than one phase, as VGG16 does with its head
        and fine-tuning passes.

    Returns
    -------
    str
        Path of the written file.
    """

    phased = bool(history) and all(
        isinstance(value, dict) for value in history.values()
    )
    phases = history if phased else {None: history}

    blocks = []
    for label, curves in phases.items():
        curves = {
            name: list(values) for name, values in curves.items()
            if values is not None
        }
        names = _ordered_curve_names(curves)
        epochs = max((len(curves[name]) for name in names), default=0)
        blocks.append((label, curves, names, epochs))

    total = sum(epochs for _, _, _, epochs in blocks)
    lines = [run_name, f"{total} epoch(s) recorded"]

    for label, curves, names, epochs in blocks:
        lines.append("")
        if label is not None:
            lines.append(label)
        for index in range(epochs):
            parts = [f"Epoch {index + 1}/{epochs}"]
            parts += [
                f"{name}: {_format_metric(curves[name][index])}"
                for name in names if index < len(curves[name])
            ]
            lines.append(" - ".join(parts))

    return _write_text(run_dir, f"{run_name}_epochs_data.txt", lines)

def _write_text(run_dir, filename, lines):
    """Write the lines of one run artefact and return its path."""

    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in lines) + "\n")

    return path

def stage_preview_directory(session):
    """Create and return the staging directory of one training session.

    ESRGAN writes its preview grids while training, before the run
    directory exists, so they land here first keyed by the session that
    produced them.

    Parameters
    ----------
    session : str
        Identifier of the training session, normally a timestamp.

    Returns
    -------
    str
        Path of the created staging directory.
    """

    path = os.path.join(ESRGAN_PREVIEW_STAGING, str(session))
    os.makedirs(path, exist_ok=True)

    return path

def collect_staged_previews(run_dir, subdir=ESRGAN_PREVIEW_SUBDIR):
    """Move every staged preview into a run directory.

    A session whose model is never saved leaves its previews staged, so the
    next run that does save carries them along. They keep their session
    identifier in that case: attributing images to a run that did not
    produce them is worse than an extra folder level. The usual case, one
    staged session, is flattened straight into ``subdir``.

    Parameters
    ----------
    run_dir : str
        Run directory returned by ``prepare_run_directory``.
    subdir : str
        Folder inside the run the previews are moved into.

    Returns
    -------
    dict
        ``{session: moved_file_count}`` for every session that was moved.
    """

    if not os.path.isdir(ESRGAN_PREVIEW_STAGING):
        return {}

    sessions = sorted(
        name for name in os.listdir(ESRGAN_PREVIEW_STAGING)
        if os.path.isdir(os.path.join(ESRGAN_PREVIEW_STAGING, name))
    )
    if not sessions:
        return {}

    target_root = os.path.join(run_dir, subdir)
    moved = {}

    for session in sessions:
        source = os.path.join(ESRGAN_PREVIEW_STAGING, session)
        target = (
            target_root if len(sessions) == 1
            else os.path.join(target_root, session)
        )
        os.makedirs(target, exist_ok=True)

        count = 0
        for name in sorted(os.listdir(source)):
            shutil.move(os.path.join(source, name), os.path.join(target, name))
            count += 1

        moved[session] = count
        os.rmdir(source)

    # Leaving an empty staging root behind would suggest there is something
    # pending when there is not.
    if not os.listdir(ESRGAN_PREVIEW_STAGING):
        os.rmdir(ESRGAN_PREVIEW_STAGING)

    return moved

def save_run_metrics(run_dir, run_name, metrics):
    """Persist a run's metrics under the name ``model_artifacts`` expects.

    Parameters
    ----------
    run_dir : str
        Directory returned by ``prepare_run_directory``.
    run_name : str
        Same name used for the directory, used as the filename stem.
    metrics : dict
        Values consumed by the reporting notebooks.

    Returns
    -------
    str
        Path of the written pickle.
    """

    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{run_name}_metrics.pkl")
    with open(path, "wb") as f:
        pickle.dump(metrics, f)

    return path

def _validate_model(model):
    """Normalise a model name and reject unknown ones."""

    if not isinstance(model, str):
        raise TypeError(f"model must be a string, got {type(model).__name__}")

    normalised = model.strip().upper()
    if normalised not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    return normalised

def _run_directories(root):
    """List the immediate subdirectories of a family root."""

    if not os.path.isdir(root):
        return []

    return [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]

def list_runs(model):
    """
    List the training runs available on disk for a model, newest first.

    Runs are ordered by the timestamp embedded in the directory name, not by
    file modification time: the embedded value is the moment the model was
    actually trained, and it survives copying the repository or restoring a
    backup, both of which rewrite mtime.

    Parameters
    ----------
    model : {'SRCNN', 'EDSR', 'ESRGAN', 'VGG16'}
        Model family to inspect. Case insensitive.

    Returns
    -------
    list of str
        Run timestamps formatted as ``%Y%m%d_%H%M%S``, newest first.
    """

    model = _validate_model(model)
    directories = _run_directories(MODEL_FAMILY_ROOTS[model])

    timestamps = []
    for name in directories:
        match = _TIMESTAMP_RE.search(name)
        if match is None:
            continue
        if name[:match.start()].rstrip("_").upper() == model:
            timestamps.append(match.group(1))

    def sort_key(timestamp):
        try:
            return datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        except ValueError:
            return datetime.min

    return sorted(set(timestamps), key=sort_key, reverse=True)

def resolve_timestamp(model, timestamp=None):
    """
    Resolve which run of a model to use.

    Parameters
    ----------
    model : {'SRCNN', 'EDSR', 'ESRGAN', 'VGG16'}
        Model family.
    timestamp : str, optional
        Run to select, formatted as ``%Y%m%d_%H%M%S``. None picks the most
        recent run found on disk, which is the right default after a
        retraining. Pass an explicit value to evaluate an older run, for
        instance to compare several hyperparameter settings.

    Returns
    -------
    str
        The resolved run timestamp.
    """

    model = _validate_model(model)
    available = list_runs(model)

    if not available:
        raise FileNotFoundError(
            f"No {model} run found under {MODEL_FAMILY_ROOTS[model]}. "
            "Train the model first, or place its artefacts there."
        )

    if timestamp is None:
        return available[0]

    if not isinstance(timestamp, str):
        raise TypeError(
            f"timestamp must be a string, got {type(timestamp).__name__}"
        )

    timestamp = timestamp.strip()
    if timestamp not in available:
        raise FileNotFoundError(
            f"{model} run {timestamp!r} not found. "
            f"Available runs (newest first): {available}"
        )

    return timestamp

def _srcnn_artifacts(root, timestamp, scale_factor):
    run_dir = os.path.join(root, f"SRCNN_{timestamp}")

    return {
        "run_dir": run_dir,
        "weights": os.path.join(run_dir, f"SRCNN_{timestamp}.h5"),
        "hr_dimensions": os.path.join(
            run_dir, f"SRCNN_{timestamp}_hrh_hrw.pkl"
        ),
        "metrics": os.path.join(run_dir, f"SRCNN_{timestamp}_metrics.pkl"),
    }

def _edsr_artifacts(root, timestamp, scale_factor):
    run_dir = os.path.join(root, f"EDSR_{timestamp}")

    return {
        "run_dir": run_dir,
        "weights": os.path.join(run_dir, f"EDSR_x{scale_factor}_{timestamp}.h5"),
        "metrics": os.path.join(run_dir, f"EDSR_{timestamp}_metrics.pkl"),
    }

def _esrgan_artifacts(root, timestamp, scale_factor):
    run_dir = os.path.join(root, f"ESRGAN_{timestamp}")

    return {
        "run_dir": run_dir,
        "generator": os.path.join(
            run_dir, f"ESRGAN_generator_x{scale_factor}_{timestamp}.h5"
        ),
        "discriminator": os.path.join(
            run_dir, f"ESRGAN_discriminator_x{scale_factor}_{timestamp}.h5"
        ),
        "metrics": os.path.join(run_dir, f"ESRGAN_{timestamp}_metrics.pkl"),
    }

def _vgg16_artifacts(root, timestamp, scale_factor):
    run_dir = os.path.join(root, f"VGG16_{timestamp}")

    return {
        "run_dir": run_dir,
        "weights": os.path.join(run_dir, f"VGG16_{timestamp}.h5"),
        "metrics": os.path.join(run_dir, f"VGG16_{timestamp}_metrics.pkl"),
    }

_ARTIFACT_BUILDERS = {
    "SRCNN": _srcnn_artifacts,
    "EDSR": _edsr_artifacts,
    "ESRGAN": _esrgan_artifacts,
    "VGG16": _vgg16_artifacts,
}

def model_artifacts(
    model,
    timestamp=None,
    scale_factor=None,
    require_existing=True):
    """
    Build the absolute paths of every artefact of one training run.

    Parameters
    ----------
    model : {'SRCNN', 'EDSR', 'ESRGAN', 'VGG16'}
        Model family.
    timestamp : str, optional
        Run to select. None picks the most recent one.
    scale_factor : int, optional
        Upscaling factor, part of the EDSR and ESRGAN weight filenames.
        Defaults to the value in ``constants``.
    require_existing : bool
        When True, check that every resolved file is on disk and fail with
        the list of missing ones. Turn it off to build the paths of a run
        that has not been trained yet.

    Returns
    -------
    dict
        ``timestamp`` plus one entry per artefact. SRCNN adds
        ``hr_dimensions``; ESRGAN exposes ``generator`` and
        ``discriminator`` instead of ``weights``.
    """

    model = _validate_model(model)
    timestamp = resolve_timestamp(model, timestamp)

    if scale_factor is None:
        scale_factor = MODEL_DEFAULT_SCALE_FACTORS.get(model)

    artifacts = _ARTIFACT_BUILDERS[model](
        MODEL_FAMILY_ROOTS[model], timestamp, scale_factor
    )
    artifacts["timestamp"] = timestamp

    if require_existing:
        missing = [
            path for key, path in artifacts.items()
            if key not in ("timestamp",)
            and not key.endswith("run_dir")
            and not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError(
                f"{model} run {timestamp} is incomplete, missing: {missing}"
            )

    return artifacts

def available_runs():
    """
    Report the runs on disk for every model.

    Returns
    -------
    dict
        Mapping of model name to its run timestamps, newest first.
    """

    return {model: list_runs(model) for model in MODELS}

def print_available_runs():
    """Print the runs on disk for every model, newest first."""

    for model, timestamps in available_runs().items():
        if timestamps:
            print(f"{model}: {len(timestamps)} run(s)")
            for position, timestamp in enumerate(timestamps):
                marker = " (latest)" if position == 0 else ""
                print(f"  - {timestamp}{marker}")
        else:
            print(f"{model}: no run found under {MODEL_FAMILY_ROOTS[model]}")
