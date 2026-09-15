import os
from pathlib import Path

import cv2

# =====================================================================
# Repository layout
# =====================================================================
# Every path below is resolved from this file instead of from os.getcwd(),
# so a notebook yields the same paths no matter which directory its kernel
# was started in. This module lives in srlib/, one level below the root.
#
# The repository keeps one top-level directory per kind of artefact: 'data'
# holds inputs, 'models' trained weights, 'results' figures, 'notebooks' the
# entry points and 'srlib' the code. Nothing is written inside the code tree.
SRLIB_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRLIB_ROOT)

DATA_ROOT = os.path.join(REPO_ROOT, "data")
VIDEOS_ROOT = os.path.join(DATA_ROOT, "videos")
IMAGES_ROOT = os.path.join(DATA_ROOT, "images")

# Metadata sits apart from the images: it is small, it is what makes the
# dataset interpretable, and it is the only part of 'data' that is versioned.
METADATA_ROOT = os.path.join(DATA_ROOT, "metadata")

# The dataset written by notebooks/1_dataset_creation.ipynb and read by every
# training notebook and by the defect detection pipeline. Writer and readers
# share these four constants, so they cannot drift apart.
HR_ROOT = os.path.join(IMAGES_ROOT, "HR")
LR_ROOT = os.path.join(IMAGES_ROOT, "LR")
CLASS_LABELS_PATH = os.path.join(METADATA_ROOT, "class_labels_map.pkl")
DEGRADATION_LOG_PATH = os.path.join(METADATA_ROOT, "degradation_log.json")

# Single root under which every model family keeps one directory per run.
MODELS_ROOT = os.path.join(REPO_ROOT, "models")

# Figure output directories, one per experiment, all under the same root so
# that the results chapter is assembled from a single place.
# CLASSIC_RESULTS_DIR is a Path because the classic visualisations build their
# filenames with the '/' operator.
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
EDA_RESULTS_DIR = os.path.join(RESULTS_ROOT, "eda")

# PSNR above which an LR/HR pair is reported as degenerate by the EDA. The
# degradation leaves the bulk of the dataset near 27 dB, so a pair this far
# above it is one whose frame held no content for the degradation to alter.
DEGENERATE_PAIR_PSNR = 45.0
CLASSIC_RESULTS_DIR = Path(RESULTS_ROOT) / "classic"
DL_RESULTS_DIR = os.path.join(RESULTS_ROOT, "deep_learning")

# =====================================================================
# Sliding window and upscaling factor per model
# =====================================================================
# SRCNN sees an already upscaled image, so its patch is in HR pixels. Its
# 9x9 'same' convolutions contaminate a 4 px border with zero padding, and
# the patch has to be large enough for that border not to dominate: at
# 48 px it covers 31 % of the area, against 56 % at 24 px.
SRCNN_PATCH_SIZE = 48
SRCNN_STRIDE = 48

SRCNN_UPSCALE_INTERPOLATION = cv2.INTER_CUBIC

# EDSR and ESRGAN take LR patches, so the paired HR patch is this size
# times the scale factor. Both use the value of their own paper: the
# receptive field of these networks spans tens of LR pixels, and a patch
# smaller than it wastes their depth.
EDSR_PATCH_SIZE = 48
EDSR_STRIDE = 48
EDSR_SCALE_FACTOR = 2

# The generator applies self-attention after upsampling, whose cost grows
# with the square of the patch area. At 32 px the attention map alone took
# a gigabyte per batch, so the LR patch is kept at 24 and the map at a
# quarter of that.
ESRGAN_PATCH_SIZE = 24
ESRGAN_STRIDE = 24
ESRGAN_SCALE_FACTOR = 2

# ESRGAN trains in a custom loop, so it has no plateau callback to give
# patience to. Its equivalent knob is how many times the rate is allowed to
# halve over the WHOLE run; 'fit' turns that count into a decay interval
# once it knows how many optimiser steps the run will take. Declaring the
# interval directly is what kept going wrong: the same value that decays
# gently over fifty epochs freezes both networks over two hundred, and at
# 1,800 steps per epoch a 30,000 step interval halved the rate fifteen
# times across 250 epochs, ending at a billionth of the initial value.
# The discriminator starts an order of magnitude lower than the generator
# so that it does not overpower it early in training.
ESRGAN_GENERATOR_LR = 1e-4
ESRGAN_DISCRIMINATOR_LR = 1e-5
ESRGAN_LR_DECAY_HALVINGS = 4
ESRGAN_LR_DECAY_RATE = 0.5

# How often the training loop renders a grid of generator outputs, so the
# reconstructions can be followed over the run. Every epoch produced more
# images than anyone reads and cost a forward pass each time.
ESRGAN_PREVIEW_EVERY = 5

# Previews are written here while training runs, because the run directory
# is only created once the model is saved. They are moved into the run at
# save time; a training that is never saved leaves them behind on purpose,
# so an interrupted attempt can still be inspected.
ESRGAN_PREVIEW_STAGING = os.path.join(MODELS_ROOT, "ESRGAN", "_staged_previews")
ESRGAN_PREVIEW_SUBDIR = "grid_figures"

VGG_PATCH_SIZE = 96
VGG_STRIDE = 48

# Weights of the four terms of the ESRGAN generator loss, chosen so that
# the four WEIGHTED terms stay within one order of magnitude of each other.
# The paper's weights (1.0 / 5e-3 / 1e-2 / -) assume a perceptual term
# measured as an L1 distance over pre-activation VGG features. This
# implementation measures it as an MSE over post-activation block5_conv4,
# which is two to three orders of magnitude larger, so under the paper's
# numbers the perceptual term was 99.96 % of the objective: the pixel term
# contributed 0.002 % and the adversarial one 0.022 %, leaving PSNR
# effectively unoptimised and the discriminator without influence.
#
# Measured on a converged batch, these weights put the pixel term near half
# of the objective, the perceptual and adversarial terms around a fifth
# each, and the spectral term the rest. Absolute scale is irrelevant under
# Adam, which normalises by gradient magnitude, so only the ratios matter;
# note that the reported generator loss is now of order 0.2 instead of 20
# and is not comparable with runs made before this change.
ESRGAN_LOSS_WEIGHTS = {
    "perceptual": 1e-3,
    "adversarial": 2e-2,
    "pixel": 1.0,
    "spectral": 0.2,
}

# Profiling of the classic algorithms: untimed runs that warm the caches
# and timed runs whose median is reported.
PROFILE_WARMUP = 1
PROFILE_REPEATS = 5

RANDOM_SEED = 42

# =====================================================================
# Shared experimental protocol
# =====================================================================
# Every dataset loader MUST use these values so that all five consumers
# (SRCNN, EDSR, ESRGAN, VGG16 and the defect detection pipeline) derive
# the exact same image-level partition from the same seed. Diverging on
# any of them silently breaks the disjointness between the pipeline test
# set and the training set of every model it evaluates.
DATASET_FRACTION = 1.0
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# =====================================================================
# Training run registry
# =====================================================================
MODELS = ("SRCNN", "EDSR", "ESRGAN", "VGG16")

# Every training notebook writes its artefacts under a run directory whose
# name ends in the training timestamp, e.g. 'SRCNN_20250910_014014'. That
# timestamp is the run identifier used throughout model_registry.
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
# One classifier scores every row of the defect detection comparison, so
# these values are what the whole study is measured through. Declared here
# rather than in the notebook so the run that produced a checkpoint can be
# reconstructed from the repository alone.
VGG16_SETUP_PARAMS = dict(
    train_last_n_layers=6,
    dropout_rate=0.3,
    l2_reg=1e-4,
    learning_rate=1e-3,
    loss="sparse_categorical_crossentropy",
    from_pretrained=False,
    pretrained_path=None,
)

# Fine-tuning runs in two phases. The head is trained first at the rate above
# while the backbone stays frozen; only then are its last layers opened, at a
# rate two orders of magnitude lower, so the ImageNet filters get refined
# instead of overwritten by the gradients of a head that started at random.
# Phase 2 is given more patience because it improves in smaller steps and
# would otherwise stop before it settles. Both values leave room for the
# plateau callback to halve the rate a few times before early stopping
# fires, which is what a patience of 3 did not: the rate collapsed and the
# run ended within a couple of epochs of each other.
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

# Capacity of the RRDB generator. Far below the 23 blocks and 32 growth
# channels of the paper, and deliberately so: at these values the generator
# holds 1.16 M parameters against the 1.37 M of EDSR, so the two learned
# models are compared at matched capacity. Restoring the paper's values
# would give it 16.8 M, twelve times EDSR, and any gap would then be
# attributable to size rather than to the architecture.
ESRGAN_RRDB_BLOCKS = 12
ESRGAN_GROWTH_CHANNELS = 16

# =====================================================================
# LR dataset degradation
# =====================================================================
# Bumped whenever the structure of images/degradation_log.json changes, so an
# old log is never silently reinterpreted under new semantics.
DEGRADATION_LOG_VERSION = 2

# Downscaling factor applied to an HR frame to reach LR size. It is recorded
# in the degradation log and validated against it, so it is part of the
# dataset definition rather than a free parameter of a single call.
DEGRADATION_SCALE_FACTOR = 0.5

# Master seed of the LR dataset. DEFAULT_DEGRADATION_SEED picks WHICH dataset
# is drawn; DEFAULT_DEGRADATION_CONFIG picks HOW degraded it is. Searching for
# a realistic LR dataset means tuning the config: a new seed alone redraws
# from the same distribution, so the average severity does not change. Keep
# the seed fixed while comparing configs, and bump it only to check that a
# result is not an artefact of one particular draw.
#
# Every cycle: edit the knobs, delete images/HR, images/LR and the log, re-run.
DEFAULT_DEGRADATION_SEED = 42

# Grayscale standard deviation below which an extracted crop is discarded.
# Fade frames at the start and end of a recording hold no object, and a
# degradation cannot alter a flat image, so the pair would score above 80 dB
# of PSNR and enter the splits as a black square compared against itself.
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

# A degradation log stores the interpolation by name, not by OpenCV code, so
# that it stays readable and stable across OpenCV versions.
INTERPOLATION_NAME_TO_CODE = {
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}

# True rebuilds the exact LR dataset recorded in the log instead of sampling a
# new one, ignoring the two knobs above. Needs the same videos and the same
# extraction settings, which the log records under "metadata".
REPLAY_DEGRADATION_FROM_LOG = False

# Per-subfolder extraction settings of the source videos. The three maps are
# keyed by the same subfolder names under VIDEOS_ROOT.
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
# Guard added to denominators and to sqrt arguments so that an all-zero image
# yields 0 instead of a division by zero or a NaN.
DEF_EPS = 1e-9

# Fraction of the spectrum radius above which energy counts as high frequency.
HF_RADIUS_FRACTION = 0.6

# The two families are declared apart because they are applied differently:
# the interpolations take the colour image at once and the advanced methods
# run channel by channel. Both are scored in the same RGB domain.
# CLASSIC_ALGORITHMS is the order every figure of the comparison follows, so
# a row means the same algorithm across all of them.
CLASSIC_INTERPOLATION_ALGORITHMS = ("bilinear", "bicubic", "area", "lanczos")
CLASSIC_ADVANCED_ALGORITHMS = ("ibp", "nlm", "egi", "freq")
CLASSIC_ALGORITHMS = (
    CLASSIC_INTERPOLATION_ALGORITHMS + CLASSIC_ADVANCED_ALGORITHMS
)

# One colour per algorithm, shared by every figure so that the same method is
# never drawn in two different colours across the results chapter.
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

# Accumulators the benchmark fills, one list per algorithm each. The order is
# the one 'build_metrics_summary' expects its arguments in.
CLASSIC_BENCHMARK_METRICS = (
    "time", "memory", "psnr", "ssim", "mae", "rmse",
    "gradient_mse", "epi", "hf_energy_ratio", "kl_luma", "kl_color",
)

# Refinement steps of the iterative back-projection.
IBP_ITERATIONS = 10

# Position in the sorted pair list whose outputs illustrate the qualitative
# figures. Fixed rather than random so the figures are reproducible.
CLASSIC_EXAMPLE_INDEX = 0