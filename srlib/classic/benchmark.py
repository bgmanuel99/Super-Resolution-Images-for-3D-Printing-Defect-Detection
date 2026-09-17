from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
from tqdm import tqdm

from srlib.progress import stage
from srlib.profiling import profile_algorithm
from srlib.dataset.loading import select_dataset_basenames, split_stratified
from srlib.constants import (
    CLASS_LABELS_PATH,
    DATASET_FRACTION,
    CLASSIC_ADVANCED_ALGORITHMS,
    CLASSIC_ALGORITHM_COLORS,
    CLASSIC_ALGORITHMS,
    CLASSIC_BENCHMARK_METRICS,
    CLASSIC_EXAMPLE_INDEX,
    CLASSIC_INTERPOLATION_ALGORITHMS,
    HF_RADIUS_FRACTION,
    HR_ROOT,
    IBP_ITERATIONS,
    LR_ROOT,
)
from srlib.classic.algorithms import (
    apply_per_channel,
    back_projection,
    edge_guided_interpolation,
    frequency_extrapolation,
    interpolate_area,
    interpolate_bicubic,
    interpolate_bilinear,
    interpolate_lanczos,
    non_local_means,
)
from srlib.classic.profiling import (
    build_metrics_summary,
    epi,
    gradient_mse,
    hf_energy_ratio,
    kl_divergence,
    kl_divergence_color,
    mae,
    rmse,
)

@dataclass
class BenchmarkResults:
    """Everything the plotting functions need after a benchmark run.

    Carrying the algorithm order and the colour map alongside the metrics
    keeps every figure consistent without the notebook redeclaring them.

    Attributes
    ----------
    metric_summary : dict
        Per-algorithm aggregates from ``build_metrics_summary``.
    algorithms : tuple of str
        Algorithm order used by every figure.
    colors : dict
        Colour per algorithm.
    pair_count : int
        Image pairs the run went through.
    interpolation_example : tuple
        ``(hr, lr, bilinear, bicubic, area, lanczos)`` of the example image.
    ibp_example : tuple
        ``(hr, lr, ibp)`` of the example image, all RGB uint8.
    nlm_example : tuple
        ``(hr, nlm)`` of the example image, both RGB uint8.
    egi_example : tuple
        ``(hr, lr, egi)`` of the example image, all RGB uint8.
    freq_example : tuple
        ``(hr, freq)`` of the example image, both RGB uint8.
    """

    metric_summary: dict
    algorithms: tuple
    colors: dict
    pair_count: int
    interpolation_example: tuple = None
    ibp_example: tuple = None
    nlm_example: tuple = None
    egi_example: tuple = None
    freq_example: tuple = None

class ClassicSuperResolutionBenchmark:
    """Profiles the classic super-resolution algorithms over the test split.

    Runs the four interpolations and the four advanced algorithms on every
    LR/HR pair of the test partition, recording time, memory and nine
    quality metrics per algorithm, and keeps one image aside to illustrate
    the outputs.

    All eight produce RGB uint8 at the HR frame size and are scored in the
    same colour domain, which is what makes a single weighted ranking over
    the whole family meaningful. The advanced algorithms operate on one
    channel at a time, so they cost three times as much as a grayscale run.
    """

    # Only the name to implementation mapping lives here. The names, their
    # order and their colours are configuration and sit in 'constants'; this
    # is the dispatch table, which is code.
    INTERPOLATION_FUNCTIONS = {
        "bilinear": interpolate_bilinear,
        "bicubic": interpolate_bicubic,
        "area": interpolate_area,
        "lanczos": interpolate_lanczos,
    }

    def __init__(
            self,
            hr_root=HR_ROOT,
            lr_root=LR_ROOT,
            class_map_path=CLASS_LABELS_PATH,
            example_index=CLASSIC_EXAMPLE_INDEX,
            fraction=DATASET_FRACTION,
            ibp_iterations=IBP_ITERATIONS):
        """
        Parameters
        ----------
        hr_root, lr_root : str
            Roots of the HR and LR image trees. Pairs are matched by
            basename, so the defect subfolders line up.
        class_map_path : str
            Pickled ``{basename: class_id}`` mapping, used to stratify the
            subsample.
        example_index : int
            Position in the sorted pair list whose outputs are kept for the
            qualitative figures.
        fraction : float or None
            Fraction of the dataset kept before the split, shared with the
            model loaders.
        ibp_iterations : int
            Refinement steps of the iterative back-projection.
        """

        self.hr_root = hr_root
        self.lr_root = lr_root
        self.class_map_path = class_map_path
        self.example_index = example_index
        self.fraction = fraction
        self.ibp_iterations = ibp_iterations
        self._stats = self._empty_stats()
        self._examples = {}

    def _empty_stats(self):
        """Build the per-metric, per-algorithm accumulators."""

        return {
            metric: {algorithm: [] for algorithm in CLASSIC_ALGORITHMS}
            for metric in CLASSIC_BENCHMARK_METRICS
        }

    def _list_pairs(self):
        """
        List the LR/HR pairs to benchmark, in a stable order.

        Only the test partition is returned, resolved through the shared
        selector and the shared split, so the algorithms are scored on the
        very images the learned models are evaluated on. Scoring them over
        the whole dataset would compare the two families on different
        populations, and the reconstruction cost reported for each would
        then describe a different set of frames.

        Returns
        -------
        list of tuple
            ``(basename, hr_path, lr_path)`` sorted by basename.
        """

        basenames, pairs, labels = select_dataset_basenames(
            self.hr_root, self.lr_root, self.class_map_path,
            fraction=self.fraction,
        )
        _, _, (test_basenames, _) = split_stratified(basenames, labels)

        return [(name, *pairs[name]) for name in sorted(test_basenames)]

    @staticmethod
    def _read_rgb(path):
        """Read one image as RGB, failing loudly on an unreadable file."""

        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed reading {path}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _profile(func):
        """
        Run one algorithm, measuring elapsed time and memory together.

        Returns
        -------
        tuple
            ``(output, elapsed_seconds, memory_bytes)``.
        """

        return profile_algorithm(func)

    def _record_cost(self, algorithm, elapsed, memory):
        """Store the time and memory of one algorithm on one image."""

        self._stats["time"][algorithm].append(elapsed)
        self._stats["memory"][algorithm].append(memory)

    def _run_interpolations(self, lr_img, target_size):
        """
        Upscale one LR image with each interpolation, profiling every call.

        Parameters
        ----------
        target_size : tuple of int
            ``(width, height)``, the order ``cv2.resize`` expects.

        Returns
        -------
        dict
            Algorithm name to upscaled RGB image.
        """

        outputs = {}
        for name in CLASSIC_INTERPOLATION_ALGORITHMS:
            upscale = self.INTERPOLATION_FUNCTIONS[name]
            output, elapsed, memory = self._profile(
                lambda fn=upscale: fn(lr_img, target_size)
            )
            self._record_cost(name, elapsed, memory)
            outputs[name] = output

        return outputs

    def _run_advanced(self, lr_img, target_shape):
        """
        Reconstruct one LR image with each advanced algorithm.

        The algorithms are single-channel, so each one is applied to every
        RGB channel independently and the results stacked back together.

        Parameters
        ----------
        lr_img : np.ndarray
            (H, W, 3) RGB image.
        target_shape : tuple of int
            ``(height, width)``. Note the order differs from the
            interpolations, which take the cv2 ``(width, height)``.

        Returns
        -------
        dict
            Algorithm name to RGB reconstruction.
        """

        calls = {
            "ibp": lambda: apply_per_channel(
                back_projection, lr_img,
                target_shape=target_shape, iterations=self.ibp_iterations,
            ),
            "nlm": lambda: apply_per_channel(
                non_local_means, lr_img, target_shape=target_shape
            ),
            "egi": lambda: apply_per_channel(
                edge_guided_interpolation, lr_img, target_shape=target_shape
            ),
            "freq": lambda: apply_per_channel(
                frequency_extrapolation, lr_img, target_shape=target_shape
            ),
        }

        outputs = {}
        for name in CLASSIC_ADVANCED_ALGORITHMS:
            output, elapsed, memory = self._profile(calls[name])
            self._record_cost(name, elapsed, memory)
            outputs[name] = output

        return outputs

    def _record_quality_metrics(self, algorithm, hr_img, sr_img):
        """Score an RGB reconstruction against the HR reference."""

        hr_float = hr_img.astype(np.float32) / 255.0
        sr_float = sr_img.astype(np.float32) / 255.0
        hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)
        sr_gray = cv2.cvtColor(sr_img, cv2.COLOR_RGB2GRAY)

        self._stats["psnr"][algorithm].append(
            psnr(hr_float, sr_float, data_range=1.0)
        )
        self._stats["ssim"][algorithm].append(
            ssim(hr_float, sr_float, channel_axis=2, data_range=1.0)
        )
        self._stats["mae"][algorithm].append(mae(hr_img, sr_img))
        self._stats["rmse"][algorithm].append(rmse(hr_img, sr_img))
        self._stats["gradient_mse"][algorithm].append(
            gradient_mse(hr_img, sr_img)
        )
        self._stats["epi"][algorithm].append(epi(hr_img, sr_img))
        self._stats["hf_energy_ratio"][algorithm].append(
            hf_energy_ratio(hr_gray, sr_gray, radius_frac=HF_RADIUS_FRACTION)
        )
        self._stats["kl_luma"][algorithm].append(
            kl_divergence(hr_gray, sr_gray)
        )
        self._stats["kl_color"][algorithm].append(
            kl_divergence_color(hr_img, sr_img)
        )

    def _store_examples(self, hr_img, lr_img, interp, advanced):
        """Keep the outputs of one image for the qualitative figures."""

        self._examples = {
            "interpolation_example": (
                hr_img, lr_img,
                interp["bilinear"], interp["bicubic"],
                interp["area"], interp["lanczos"],
            ),
            "ibp_example": (hr_img, lr_img, advanced["ibp"]),
            "nlm_example": (hr_img, advanced["nlm"]),
            "egi_example": (hr_img, lr_img, advanced["egi"]),
            "freq_example": (hr_img, advanced["freq"]),
        }

    def _process_pair(self, index, hr_img, lr_img):
        """Run every algorithm on one LR/HR pair and record the results."""

        height, width = hr_img.shape[:2]

        interp = self._run_interpolations(lr_img, (width, height))
        advanced = self._run_advanced(lr_img, (height, width))

        for outputs in (interp, advanced):
            for name, sr_img in outputs.items():
                self._record_quality_metrics(name, hr_img, sr_img)

        if index == self.example_index:
            self._store_examples(hr_img, lr_img, interp, advanced)

    def run(self):
        """
        Profile every algorithm over the test partition.

        Pairs are read one at a time rather than preloaded: the dataset at
        full resolution is hundreds of megabytes and only one pair is needed
        at any moment.

        Returns
        -------
        BenchmarkResults
            Aggregated metrics plus the example outputs.
        """

        self._stats = self._empty_stats()
        self._examples = {}

        pairs = self._list_pairs()

        with stage(
                f"Classic SR benchmark | {len(CLASSIC_ALGORITHMS)} algorithms "
                f"x {len(pairs)} pairs") as step:
            step(f"algorithms {', '.join(CLASSIC_ALGORITHMS)}")
            step(
                f"test partition {len(pairs)} images at fraction "
                f"{self.fraction}"
            )

            for index, (_, hr_path, lr_path) in enumerate(
                    tqdm(pairs, desc="  pairs", unit="pair")):
                self._process_pair(
                    index, self._read_rgb(hr_path), self._read_rgb(lr_path)
                )

        return BenchmarkResults(
            metric_summary=self._build_summary(),
            algorithms=CLASSIC_ALGORITHMS,
            colors=dict(CLASSIC_ALGORITHM_COLORS),
            pair_count=len(pairs),
            **self._examples,
        )

    def _build_summary(self):
        """Aggregate the accumulated stats into the per-algorithm summary."""

        return build_metrics_summary(
            self._stats["time"],
            self._stats["memory"],
            self._stats["psnr"],
            self._stats["ssim"],
            self._stats["mae"],
            self._stats["rmse"],
            self._stats["gradient_mse"],
            self._stats["epi"],
            self._stats["hf_energy_ratio"],
            self._stats["kl_luma"],
            self._stats["kl_color"],
        )
