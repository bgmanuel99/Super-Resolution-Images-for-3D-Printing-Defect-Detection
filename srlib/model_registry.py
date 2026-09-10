import os
import pickle
import re
from datetime import datetime

from srlib.constants import (
    MODEL_DEFAULT_SCALE_FACTORS,
    MODEL_FAMILY_ROOTS,
    MODELS,
    TIMESTAMP_FORMAT,
    TIMESTAMP_PATTERN,
    VGG16_SOURCES,
)

_TIMESTAMP_RE = re.compile(TIMESTAMP_PATTERN)

def vgg16_variant_name(source, timestamp):
    """
    Build the canonical name identifying a trained VGG16 variant.

    Both variants share every hyperparameter and differ only in the
    resolution of the patches they were trained on, so the source tag is
    the only thing that tells their checkpoints apart. Keeping the name in
    a single function stops the model file, its folder and its metrics
    pickle from drifting into different conventions.

    Lives here, next to the run resolution logic, because this name *is*
    the on-disk layout: the notebook that trains the classifiers and the
    one that runs the detection pipeline have to agree on it, and a second
    copy of the convention is exactly how they would stop agreeing.

    Parameters
    ----------
    source : {'hr', 'lr'}
        Which side of the LR/HR pair the model was trained on.
    timestamp : str
        Training timestamp, formatted as ``%Y%m%d_%H%M%S``.

    Returns
    -------
    str
        Name of the form ``VGG16_HR_20260828_181500``.
    """

    if source not in VGG16_SOURCES:
        raise ValueError(f"source must be one of {VGG16_SOURCES}, got {source!r}")
    if not timestamp or not isinstance(timestamp, str):
        raise ValueError("timestamp must be a non-empty string.")

    return f"VGG16_{source.upper()}_{timestamp}"

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

    A VGG16 run is only reported when *both* the HR and the LR variant are
    present, because the defect detection pipeline needs the pair.

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

    if model == "VGG16":
        expected = {source.upper() for source in VGG16_SOURCES}
        found = {}
        for name in directories:
            match = _TIMESTAMP_RE.search(name)
            if match is None:
                continue
            timestamp = match.group(1)
            for source in VGG16_SOURCES:
                if name == vgg16_variant_name(source, timestamp):
                    found.setdefault(timestamp, set()).add(source.upper())

        timestamps = [
            timestamp for timestamp, variants in found.items()
            if variants == expected
        ]
    else:
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
    artifacts = {}

    # The VGG16 notebook trains one classifier per input resolution in the
    # same run, so a single timestamp identifies both variants.
    for source in VGG16_SOURCES:
        variant = vgg16_variant_name(source, timestamp)
        run_dir = os.path.join(root, variant)
        artifacts[f"{source}_run_dir"] = run_dir
        artifacts[f"{source}_weights"] = os.path.join(run_dir, f"{variant}.h5")
        artifacts[f"{source}_metrics"] = os.path.join(
            run_dir, f"{variant}_metrics.pkl"
        )

    return artifacts

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
        ``discriminator`` instead of ``weights``; VGG16 prefixes its keys
        with ``hr_`` and ``lr_``.
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
