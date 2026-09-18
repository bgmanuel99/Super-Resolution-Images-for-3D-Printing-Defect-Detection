import os
from pathlib import Path

import cv2

# =====================================================================
# Repository layout
# =====================================================================
# Paths are resolved from this file, not from os.getcwd(), so a notebook
# sees the same paths whatever directory its kernel was started in.
SRLIB_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRLIB_ROOT)

DATA_ROOT = os.path.join(REPO_ROOT, "data")
VIDEOS_ROOT = os.path.join(DATA_ROOT, "videos")
IMAGES_ROOT = os.path.join(DATA_ROOT, "images")

# The only part of 'data' that is versioned.
METADATA_ROOT = os.path.join(DATA_ROOT, "metadata")

# Written by notebooks/1_dataset_creation.ipynb and read by every training
# notebook and by the detection pipeline.
HR_ROOT = os.path.join(IMAGES_ROOT, "HR")
LR_ROOT = os.path.join(IMAGES_ROOT, "LR")
CLASS_LABELS_PATH = os.path.join(METADATA_ROOT, "class_labels_map.pkl")
DEGRADATION_LOG_PATH = os.path.join(METADATA_ROOT, "degradation_log.json")

MODELS_ROOT = os.path.join(REPO_ROOT, "models")

RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
EDA_RESULTS_DIR = os.path.join(RESULTS_ROOT, "eda")

# PSNR above which the EDA reports an LR/HR pair as degenerate: that far
# above the ~27 dB of the dataset the frame held no content to degrade.
DEGENERATE_PAIR_PSNR = 45.0

# A Path because the classic visualisations build filenames with '/'.
CLASSIC_RESULTS_DIR = Path(RESULTS_ROOT) / "classic"
DL_RESULTS_DIR = os.path.join(RESULTS_ROOT, "deep_learning")

# =====================================================================
# Sliding window and upscaling factor per model
# =====================================================================
# SRCNN sees an already upscaled image, so its patch is in HR pixels. Its
# 9x9 'same' convolutions contaminate a 4 px border, which covers 31 % of
# a 48 px patch and 56 % of a 24 px one.
SRCNN_PATCH_SIZE = 48
SRCNN_STRIDE = 48

SRCNN_UPSCALE_INTERPOLATION = cv2.INTER_CUBIC

# EDSR and ESRGAN take LR patches, so the paired HR patch is this size
# times the scale factor.
EDSR_PATCH_SIZE = 48
EDSR_STRIDE = 48
EDSR_SCALE_FACTOR = 2

# Kept at 24 because the attention map after upsampling grows with the
# square of the patch area: at 32 px it took a gigabyte per batch.
ESRGAN_PATCH_SIZE = 24
ESRGAN_STRIDE = 24
ESRGAN_SCALE_FACTOR = 2

# The custom training loop has no plateau callback, so the knob is how many
# times the rate may halve over the WHOLE run; 'fit' turns that count into
# a decay interval once the number of optimiser steps is known. The
# discriminator starts lower so it does not overpower the generator early.
ESRGAN_GENERATOR_LR = 1e-4
ESRGAN_DISCRIMINATOR_LR = 1e-5
ESRGAN_LR_DECAY_HALVINGS = 4
ESRGAN_LR_DECAY_RATE = 0.5

# Epochs between two preview grids of generator outputs.
ESRGAN_PREVIEW_EVERY = 5

# Previews are staged here because the run directory only exists once the
# model is saved; they are moved into it at save time.
ESRGAN_PREVIEW_STAGING = os.path.join(MODELS_ROOT, "ESRGAN", "_staged_previews")
ESRGAN_PREVIEW_SUBDIR = "grid_figures"

VGG_PATCH_SIZE = 96
VGG_STRIDE = 48

# Weights of the four generator loss terms, set so the WEIGHTED terms stay
# within one order of magnitude of each other. The paper's values assume an
# L1 perceptual term over pre-activation features; this implementation uses
# an MSE over post-activation ones, which is orders of magnitude larger and
# would otherwise absorb the whole objective. Only the ratios matter under
# Adam, so the reported loss is not comparable with earlier runs.
ESRGAN_LOSS_WEIGHTS = {
    "perceptual": 1e-3,
    "adversarial": 2e-2,
    "pixel": 1.0,
    "spectral": 0.2,
}

# Untimed runs that warm the caches, and timed runs whose median is kept.
PROFILE_WARMUP = 1
PROFILE_REPEATS = 5

RANDOM_SEED = 42

# =====================================================================
# Shared experimental protocol
# =====================================================================
# Every dataset loader MUST use these values: they are what makes the five
# consumers derive the same image-level partition. Diverging on any of them
# silently breaks the disjointness between the pipeline test set and the
# training set of the models it evaluates.
DATASET_FRACTION = 1.0
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# =====================================================================
# Training run registry
# =====================================================================
MODELS = ("SRCNN", "EDSR", "ESRGAN", "VGG16")

# Runs are identified by the timestamp that ends their directory name,
# e.g. 'SRCNN_20250910_014014'.
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
TIMESTAMP_PATTERN = r"(\d{8}_\d{6})$"

MODEL_FAMILY_ROOTS = {
    model: os.path.join(MODELS_ROOT, model) for model in MODELS
}

# Only EDSR and ESRGAN carry the factor in their weight filenames.
MODEL_DEFAULT_SCALE_FACTORS = {
    "EDSR": EDSR_SCALE_FACTOR,
    "ESRGAN": ESRGAN_SCALE_FACTOR,
}

# =====================================================================
# VGG16 fine-tuning hyperparameters
# =====================================================================
# Declared here rather than in the notebook so the run that produced a
# checkpoint can be reconstructed from the repository alone.
VGG16_SETUP_PARAMS = dict(
    train_last_n_layers=6,
    dropout_rate=0.3,
    l2_reg=1e-4,
    learning_rate=1e-3,
    loss="sparse_categorical_crossentropy",
    from_pretrained=False,
    pretrained_path=None,
)

# Phase 1 trains the head with the backbone frozen; phase 2 opens its last
# layers at a rate two orders of magnitude lower, so the ImageNet filters
# are refined instead of overwritten. Phase 2 gets more patience because it
# improves in smaller steps.
VGG16_FIT_PARAMS = dict(
    batch_size=32,
    head_epochs=150,
    finetune_epochs=150,
    head_patience=10,
    finetune_patience=15,
    finetune_learning_rate=1e-5,
    use_augmentation=True,
)

# =====================================================================
# ESRGAN training
# =====================================================================
# Tune to the available GPU memory.
ESRGAN_BATCH_SIZE = 16

# Capacity of the RRDB generator: 4.7 M parameters, still well below the
# 23 blocks and 32 growth channels of the paper, which would give 16.8 M.
ESRGAN_RRDB_BLOCKS = 12
ESRGAN_GROWTH_CHANNELS = 16

# =====================================================================
# LR dataset degradation
# =====================================================================
# Bumped whenever the structure of the degradation log changes, so an old
# log is never reinterpreted under new semantics.
DEGRADATION_LOG_VERSION = 2

# Recorded in the degradation log and validated against it, so it is part
# of the dataset definition rather than a free parameter of one call.
DEGRADATION_SCALE_FACTOR = 0.5

# The seed picks WHICH dataset is drawn; the config below picks HOW
# degraded it is. A new seed redraws from the same distribution, so keep it
# fixed while comparing configs.
DEFAULT_DEGRADATION_SEED = 42

# Grayscale standard deviation below which a crop is discarded. Fade frames
# hold no object, and a degradation cannot alter a flat image, so the pair
# would score above 80 dB of PSNR.
BLANK_FRAME_MIN_STD = 1.0

DEFAULT_DEGRADATION_CONFIG = {
    "gaussian_blur": {
        "probability": 0.7,
        "ksize_choices": [3, 5, 7],
        "sigma_range": [0.8, 2.0],
    },
    "motion_blur": {
        "probability": 0.3,
        "size_choices": [5, 7, 9],
    },
    "downscale": {
        "interpolation_choices": [
            "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA", "INTER_LANCZOS4",
        ],
    },
    "gaussian_noise": {
        "probability": 0.7,
        "std_range": [2.0, 10.0],
    },
    "jpeg": {
        "probability": 0.7,
        "quality_range": [20, 60],
    },
}

# The log stores the interpolation by name, not by OpenCV code, so it stays
# readable and stable across OpenCV versions.
INTERPOLATION_NAME_TO_CODE = {
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}

# True rebuilds the exact dataset recorded in the log instead of sampling a
# new one, ignoring the seed and the config above.
REPLAY_DEGRADATION_FROM_LOG = False

# Per-subfolder extraction settings, keyed by the subfolder names under
# VIDEOS_ROOT. The intervals differ so both classes contribute a comparable
# number of frames from recordings of different length.
VIDEO_MAX_VIDEOS_PER_FOLDER = {
    "low_z_offset": 12,
    "high_z_offset": 12,
}

VIDEO_FRAME_INTERVAL_PER_FOLDER = {
    "low_z_offset": 39,
    "high_z_offset": 20,
}

VIDEO_CLASS_ID_PER_FOLDER = {
    "low_z_offset": 0,
    "high_z_offset": 1,
}

# =====================================================================
# Profiling of the classic algorithms
# =====================================================================
# Guard on denominators and sqrt arguments, so an all-zero image yields 0
# instead of a division by zero or a NaN.
DEF_EPS = 1e-9

# Fraction of the spectrum radius above which energy counts as high
# frequency.
HF_RADIUS_FRACTION = 0.6

# The two families are applied differently: the interpolations take the
# colour image at once, the advanced methods run channel by channel.
# CLASSIC_ALGORITHMS is the order every figure follows.
CLASSIC_INTERPOLATION_ALGORITHMS = ("bilinear", "bicubic", "area", "lanczos")
CLASSIC_ADVANCED_ALGORITHMS = ("ibp", "nlm", "egi", "freq")
CLASSIC_ALGORITHMS = (
    CLASSIC_INTERPOLATION_ALGORITHMS + CLASSIC_ADVANCED_ALGORITHMS
)

# One colour per algorithm, shared by every figure.
CLASSIC_ALGORITHM_COLORS = {
    "bilinear": "#4c72b0",
    "bicubic": "#55a868",
    "area": "#c44e52",
    "lanczos": "#8172b2",
    "ibp": "#ccb974",
    "nlm": "#64b5cd",
    "egi": "#8c8c8c",
    "freq": "#937860",
}

# One accumulator per metric, in the order 'build_metrics_summary' expects
# its arguments.
CLASSIC_BENCHMARK_METRICS = (
    "time", "memory", "psnr", "ssim", "mae", "rmse",
    "gradient_mse", "epi", "hf_energy_ratio", "kl_luma", "kl_color",
)

# Refinement steps of the iterative back-projection.
IBP_ITERATIONS = 10

# Position in the sorted pair list whose outputs illustrate the qualitative
# figures. Fixed rather than random so the figures are reproducible.
CLASSIC_EXAMPLE_INDEX = 0
