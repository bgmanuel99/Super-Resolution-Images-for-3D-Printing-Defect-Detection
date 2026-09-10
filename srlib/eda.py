import os

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
from tqdm import tqdm

from srlib.progress import stage
from srlib.constants import (
    EDA_RESULTS_DIR,
    HR_ROOT,
    LR_ROOT,
    SRCNN_UPSCALE_INTERPOLATION,
)

class ImagePairLoader:
    """Separated I/O utilities for iterating and aligning LR/HR pairs."""

    @staticmethod
    def iter_pairs(lr_base, hr_base):
        """Yield matching (lr_relpath, hr_relpath) pairs by scanning subfolders.

        Pairs are matched by their relative path from the base dirs. Only files
        present in both LR and HR trees are returned. Order is lexicographic.
        """

        exts = (".png", ".jpg", ".jpeg")

        def walk_relnames(base):
            rels = set()
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith(exts):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, base)
                        rels.add(rel)
            return rels

        lr_set = walk_relnames(lr_base)
        hr_set = walk_relnames(hr_base)

        common = sorted(lr_set & hr_set)
        if not common:
            raise ValueError(
                "No matching LR/HR image pairs were found under the provided "
                "directories."
            )

        for rel in common:
            # Use the same relative path for LR and HR; callers will join with bases
            yield rel, rel

    @staticmethod
    def load_and_align(
            lr_path, hr_path, upscale_interpolation=SRCNN_UPSCALE_INTERPOLATION):
        """Load two images and resize LR to HR size if required.

        Every pair is upscaled with the SAME declared interpolation. The
        per-image degradation kernel is deliberately not read back, even
        though the degradation log records it: an LR capture in the wild does
        not come with the filter that produced it, and measuring each image
        under a different kernel would also make the images incomparable with
        one another.

        The default is the interpolation SRCNN is fed at training and at
        inference time, so what this analysis reports as the LR baseline is
        exactly what the model receives.

        Parameters
        ----------
        upscale_interpolation : int
            OpenCV interpolation used to bring LR up to the HR frame size.
        """

        lr = cv2.imread(lr_path)
        hr = cv2.imread(hr_path)

        if lr is None or hr is None:
            raise ValueError(f"Failed reading {lr_path} or {hr_path}")

        if lr.shape[:2] != hr.shape[:2]:
            lr = cv2.resize(
                lr,
                (hr.shape[1], hr.shape[0]),
                interpolation=upscale_interpolation,
            )

        return lr, hr

class ImageDatasetAnalyzer:
    """Utility collection to analyze LR/HR image pairs.

    All methods are static so they can be called without
    instantiation.
    """

    @staticmethod
    def loss_fn():
        """Return (singleton) the loaded LPIPS model.

        Returns
        -------
        lpips.LPIPS
            Initialized LPIPS model instance.
        """

        if not hasattr(ImageDatasetAnalyzer, '_loss_fn'):
            ImageDatasetAnalyzer._loss_fn = lpips.LPIPS(net="alex")

        return ImageDatasetAnalyzer._loss_fn

    @staticmethod
    def lpips_score(lr_img, hr_img):
        """Compute LPIPS between two aligned BGR images.

        Parameters
        ----------
        lr_img : np.ndarray
            Aligned LR image (BGR).
        hr_img : np.ndarray
            HR image (BGR).

        Returns
        -------
        float
            LPIPS value.
        """

        def to_tensor(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = 2 * img - 1
            img = np.transpose(img, (2, 0, 1))
            return torch.from_numpy(img).unsqueeze(0).float()

        return ImageDatasetAnalyzer.loss_fn()(
            to_tensor(lr_img),
            to_tensor(hr_img)
        ).item()

    @staticmethod
    def laplacian_variance(gray):
        """Variance of Laplacian (sharpness proxy).

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image.

        Returns
        -------
        float
            Variance value.
        """

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def psnr_metric(lr_img, hr_img):
        """Compute PSNR between HR and LR images.

        Parameters
        ----------
        lr_img : np.ndarray
            Aligned LR image.
        hr_img : np.ndarray
            HR image.

        Returns
        -------
        float
            PSNR value.
        """

        return psnr(hr_img, lr_img, data_range=255)

    @staticmethod
    def ssim_metric(lr_img, hr_img):
        """Compute SSIM between HR and LR images.

        Parameters
        ----------
        lr_img : np.ndarray
            Aligned LR image.
        hr_img : np.ndarray
            HR image.

        Returns
        -------
        float
            SSIM value.
        """

        return ssim(hr_img, lr_img, channel_axis=2, data_range=255)

    @staticmethod
    def glcm_contrast(gray, angles=None, levels=32, multi_angle=False):
        """Extract the GLCM contrast of a grayscale image.

        Contrast is the only Haralick descriptor kept. Homogeneity and
        correlation are strongly correlated with it, so the three of them
        answered the same question about the texture.

        32 levels keep the co-occurrence matrix at 32x32. At 256 levels the
        matrix is 64 times larger per angle for a descriptor that does not
        become more informative on this texture.

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image.
        angles : iterable | None
            Angles in radians. If None uses [0] unless
            multi_angle True (then 4 angles).
        levels : int
            Quantization levels (default 32).
        multi_angle : bool
            Average the descriptor across four angles when True.

        Returns
        -------
        float
            GLCM contrast.
        """

        if angles is None:
            if multi_angle:
                angles = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
            else:
                angles = (0,)

        # Quantize to requested levels
        if gray.max() == 0:
            norm = np.zeros_like(gray, dtype=np.uint8)
        else:
            norm = (
                (gray.astype(np.float32) / 255.0) * (levels - 1)
            ).astype(np.uint8)
        glcm = graycomatrix(
            norm,
            [1],
            list(angles),
            levels,
            symmetric=True,
            normed=True
        )

        return float(graycoprops(glcm, "contrast").mean())

    @staticmethod
    def saturation_mean(hsv):
        """Mean HSV saturation.

        Saturation is the only photometric descriptor kept, because it is
        the only one the degradation moves appreciably: chroma subsampling
        and the colour interpolation shift it by roughly 18 % of the HR
        level, while mean brightness stays within 1 % and therefore
        describes the scene rather than the degradation.

        Parameters
        ----------
        hsv : np.ndarray
            Image converted to HSV.

        Returns
        -------
        float
            Mean of the saturation channel.
        """

        return float(np.mean(hsv[:, :, 1]))

    @staticmethod
    def ringing(gray):
        """Intensity spread over the band surrounding the detected edges.

        This is the single artefact descriptor of the EDA. Of the three that
        were measured it is the least redundant, reaching 0.43 against the
        retained fidelity and sharpness variables where the blocking score
        reached 0.80. Blocking was also dropped because it did not measure
        blocking: it samples a whole-image DCT every 8 rows instead of a
        block-wise one, so it tracked the JPEG quality that produced the
        artefact at 0.01 and its only response was to noise, which the
        Laplacian variance already reports.

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image.

        Returns
        -------
        float
            Standard deviation inside the edge band, 0.0 if no edge is found.
        """

        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel)

        # Canny and dilate return uint8 masks holding 0 or 255, so combining
        # them with bitwise operators yields 0/255 rather than True/False.
        # Indexing with that array selects ROWS 0 and 255 instead of the band,
        # which silently returned the spread of two arbitrary rows on any
        # image taller than 255 px and raised IndexError on any smaller one.
        edge_region = (dilated > 0) & (edges == 0)

        if not np.any(edge_region):
            return 0.0

        return float(np.std(gray[edge_region]))

    @staticmethod
    def color_noise(img):
        """Mean absolute deviation of an image from its Gaussian blur.

        Not a per-pair metric: it is a high-pass measure that correlated
        0.93 with the Laplacian variance already kept. It survives only to
        feed the noise figures, which read it as a spatial map and as a
        dataset-wide distribution rather than as a variable to correlate.

        Parameters
        ----------
        img : np.ndarray
            BGR image.

        Returns
        -------
        float
            Mean absolute high-pass response.
        """

        blur = cv2.GaussianBlur(img, (5, 5), 0)

        return float(np.mean(np.abs(img.astype(float) - blur.astype(float))))

class ImagePairMetrics:
    """Container of metrics computed for an LR/HR pair.

    Deliberately small. An earlier revision carried 33 variables whose
    families measured the same quantity several times over: the RMS noise
    correlated 0.99 with the Laplacian variance it sat next to, the twelve
    per-channel skew and kurtosis variables correlated above 0.99 both
    across the three channels and between LR and HR while tracking the
    degradation at 0.14 or less, and two of the three artefact scores were
    high-pass measures of what the Laplacian variance already reported.

    What is left is one descriptor per property the analysis reasons about:
    perceptual fidelity, sharpness, edge artefacts, texture and photometry.
    """

    def __init__(
        self,
        filename,
        lpips,
        psnr,
        ssim,
        lap_var_lr,
        lap_var_hr,
        ringing_lr,
        ringing_hr,
        glcm_contrast,
        saturation_mean_lr,
        saturation_mean_hr,
    ):
        self.filename = filename
        self.lpips = lpips
        self.psnr = psnr
        self.ssim = ssim
        self.lap_var_lr = lap_var_lr
        self.lap_var_hr = lap_var_hr
        self.ringing_lr = ringing_lr
        self.ringing_hr = ringing_hr
        self.glcm_contrast = glcm_contrast
        self.saturation_mean_lr = saturation_mean_lr
        self.saturation_mean_hr = saturation_mean_hr

    def as_dict(self):
        """Return metrics as a dict for DataFrame conversion."""

        return self.__dict__.copy()

class MetricsAggregator:
    """Orchestrates metric extraction for all image pairs."""

    @staticmethod
    def collect(
        lr_dir,
        hr_dir,
        glcm_multi_angle=False,
        glcm_levels=32,
        upscale_interpolation=SRCNN_UPSCALE_INTERPOLATION,
    ):
        """Compute metrics for each pair and accumulate global visual data.

        Returns a tuple (rows, global_data) where global_data holds the
        accumulators required to build the global advanced visualization panel.
        """

        rows = []
        sat_bins = np.linspace(0, 256, 51)  # 50 bins 0-255
        global_data = {
            'count': 0,
            'lr_fft_sum': None,
            'hr_fft_sum': None,
            'grad_hr_sum': None,
            'glcm_sum': None,  # shape (256, 256, 1, 1)
            'sat_lr_counts': np.zeros(len(sat_bins) - 1, dtype=np.float64),
            'sat_hr_counts': np.zeros(len(sat_bins) - 1, dtype=np.float64),
            'sat_bins': sat_bins,
            'noise_means_lr': [],
        }

        pairs = list(ImagePairLoader.iter_pairs(lr_dir, hr_dir))
        for lf, hf in tqdm(pairs, desc="Computing metrics", unit="img"):
            lr_img, hr_img = ImagePairLoader.load_and_align(
                os.path.join(lr_dir, lf),
                os.path.join(hr_dir, hf),
                upscale_interpolation=upscale_interpolation,
            )

            gray_lr = cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY)
            gray_hr = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
            hsv_lr = cv2.cvtColor(lr_img, cv2.COLOR_BGR2HSV)
            hsv_hr = cv2.cvtColor(hr_img, cv2.COLOR_BGR2HSV)
            lpips_val = ImageDatasetAnalyzer.lpips_score(lr_img, hr_img)
            psnr_val = ImageDatasetAnalyzer.psnr_metric(lr_img, hr_img)
            ssim_val = ImageDatasetAnalyzer.ssim_metric(lr_img, hr_img)
            glcm_contrast = ImageDatasetAnalyzer.glcm_contrast(
                gray_lr,
                levels=glcm_levels,
                multi_angle=glcm_multi_angle
            )
            color_noise_lr = ImageDatasetAnalyzer.color_noise(lr_img)

            rows.append(
                ImagePairMetrics(
                    filename=lf.replace('\\', '/'),
                    lpips=lpips_val,
                    psnr=psnr_val,
                    ssim=ssim_val,
                    lap_var_lr=ImageDatasetAnalyzer.laplacian_variance(gray_lr),
                    lap_var_hr=ImageDatasetAnalyzer.laplacian_variance(gray_hr),
                    ringing_lr=ImageDatasetAnalyzer.ringing(gray_lr),
                    ringing_hr=ImageDatasetAnalyzer.ringing(gray_hr),
                    glcm_contrast=glcm_contrast,
                    saturation_mean_lr=ImageDatasetAnalyzer.saturation_mean(
                        hsv_lr
                    ),
                    saturation_mean_hr=ImageDatasetAnalyzer.saturation_mean(
                        hsv_hr
                    ),
                )
            )

            sobelx = cv2.Sobel(gray_hr, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray_hr, cv2.CV_64F, 0, 1, ksize=5)
            grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
            lr_fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray_lr)))
            hr_fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray_hr)))
            lr_glcm_full = graycomatrix(
                gray_lr,
                [1],
                [0],
                256,
                symmetric=True,
                normed=True
            )

            if global_data['lr_fft_sum'] is None:
                global_data['lr_fft_sum'] = lr_fft_mag.astype(np.float64)
                global_data['hr_fft_sum'] = hr_fft_mag.astype(np.float64)
                global_data['grad_hr_sum'] = grad_mag
                global_data['glcm_sum'] = lr_glcm_full.astype(np.float64)
            else:
                global_data['lr_fft_sum'] += lr_fft_mag
                global_data['hr_fft_sum'] += hr_fft_mag
                global_data['grad_hr_sum'] += grad_mag
                global_data['glcm_sum'] += lr_glcm_full

            lr_counts, _ = np.histogram(
                hsv_lr[:, :, 1], bins=global_data['sat_bins']
            )
            hr_counts, _ = np.histogram(
                hsv_hr[:, :, 1], bins=global_data['sat_bins']
            )
            global_data['sat_lr_counts'] += lr_counts
            global_data['sat_hr_counts'] += hr_counts
            global_data['noise_means_lr'].append(color_noise_lr)
            global_data['count'] += 1

        return rows, global_data

class StatsReporter:
    """Utilities to convert and summarize metrics to a DataFrame."""

    @staticmethod
    def dataframe(rows):
        """Convert list of ImagePairMetrics into a DataFrame.

        Parameters
        ----------
        rows : list[ImagePairMetrics]
            List of metric objects.

        Returns
        -------
        pandas.DataFrame
            One row per image pair.
        """

        return pd.DataFrame([r.as_dict() for r in rows])

    @staticmethod
    def summary(df):
        """Return basic descriptive statistics.

        Parameters
        ----------
        df : pandas.DataFrame
            Metrics data.

        Returns
        -------
        pandas.DataFrame
            mean, std and quartiles.
        """

        return df.describe().T[['mean', 'std', '25%', '50%', '75%']]

class ImageDataVisualization:
    """Visualization utilities for exploratory analysis."""

    @staticmethod
    def save_visual_example(lr_img, hr_img, output_path, lpips_val):
        """Save comparison figure and a difference heatmap.

        Parameters
        ----------
        lr_img : np.ndarray
            LR image.
        hr_img : np.ndarray
            HR image.
        output_path : str
            Output PNG path.
        lpips_val : float
            LPIPS value for title.
        """

        lr_resized = cv2.resize(
            lr_img,
            (hr_img.shape[1], hr_img.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

        diff_map = cv2.absdiff(lr_resized, hr_img)
        diff_map_color = cv2.applyColorMap(
            cv2.convertScaleAbs(cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)),
            cv2.COLORMAP_JET
        )

        _, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(cv2.cvtColor(lr_resized, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Rescaled LR")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title("HR")
        axes[1].axis("off")

        axes[2].imshow(diff_map_color)
        axes[2].set_title(f"Difference map\nLPIPS: {lpips_val:.4f}")
        axes[2].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def create_advanced_visualizations(lr_img, hr_img, output_path):
        """Create per-pair advanced panel: spectra, gradients, GLCM, noise
        map, saturation distribution."""

        plt.figure(figsize=(20, 10))

        # 1. LR Spectrum
        plt.subplot(231)
        lr_fft = np.fft.fft2(cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY))
        plt.imshow(np.log(np.abs(np.fft.fftshift(lr_fft)) + 1e-8),
                   cmap="viridis")
        plt.title("LR Frequency Spectrum")
        plt.colorbar()

        # 2. HR Spectrum
        plt.subplot(232)
        hr_fft = np.fft.fft2(cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY))
        plt.imshow(np.log(np.abs(np.fft.fftshift(hr_fft)) + 1e-8),
                   cmap="viridis")
        plt.title("HR Frequency Spectrum")
        plt.colorbar()

        # 3. HR Gradient Magnitude
        plt.subplot(233)
        gray_hr = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_hr, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_hr, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        plt.imshow(gradient_magnitude, cmap="magma")
        plt.title("Gradient Magnitude")
        plt.colorbar()

        # 4. LR GLCM
        plt.subplot(234)
        lr_gray = cv2.cvtColor(lr_img, cv2.COLOR_BGR2GRAY)
        lr_glcm = graycomatrix(
            lr_gray, [1], [0], 256, symmetric=True, normed=True
        )
        lr_contrast = graycoprops(lr_glcm, "contrast")[0, 0]
        plt.imshow(lr_glcm[:, :, 0, 0], cmap="plasma")
        plt.title(f"LR GLCM (Contrast: {lr_contrast:.2f})")
        plt.colorbar()

        # 5. LR Color Noise Map
        plt.subplot(235)
        blur = cv2.GaussianBlur(lr_img, (5, 5), 0)
        noise_map = np.mean(
            np.abs(lr_img.astype(float) - blur.astype(float)), axis=2
        )
        color_noise_mean = ImageDatasetAnalyzer.color_noise(lr_img)
        plt.imshow(noise_map, cmap="hot")
        plt.title(f"Noise Map (Mean: {color_noise_mean:.2f})")
        plt.colorbar()

        # 6. Saturation Distribution LR vs HR
        plt.subplot(236)
        lr_hsv = cv2.cvtColor(lr_img, cv2.COLOR_BGR2HSV)[:, :, 1]
        hr_hsv = cv2.cvtColor(hr_img, cv2.COLOR_BGR2HSV)[:, :, 1]
        plt.hist(lr_hsv.ravel(), bins=50, alpha=0.5, density=True,
                 label="LR", color="steelblue")
        plt.hist(hr_hsv.ravel(), bins=50, alpha=0.5, density=True,
                 label="HR", color="orange")
        plt.title("Saturation Distribution")
        plt.legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def create_global_advanced_visualizations(global_data, output_path):
        """Create a global panel with averaged spectra, gradient, averaged
        GLCM, noise mean distribution, saturation densities."""

        if global_data is None or global_data.get('count', 0) == 0:
            print("No global data available to create advanced visualization.")
            return

        n = global_data['count']
        eps = 1e-8

        # Average spectra (compute log after averaging magnitudes)
        lr_fft_avg = global_data['lr_fft_sum'] / n
        hr_fft_avg = global_data['hr_fft_sum'] / n

        # Average gradient magnitude
        grad_hr_avg = global_data['grad_hr_sum'] / n

        # Average (unnormalized-sum) GLCM then renormalize
        glcm_sum = global_data['glcm_sum']
        glcm_avg = glcm_sum / glcm_sum.sum()
        glcm_contrast = graycoprops(glcm_avg, "contrast")[0, 0]

        # Saturation histograms (normalize to density)
        sat_bins = global_data['sat_bins']
        bin_width = sat_bins[1] - sat_bins[0]
        sat_lr_density = (
            global_data['sat_lr_counts'] /
            (global_data['sat_lr_counts'].sum() * bin_width + eps)
        )
        sat_hr_density = (
            global_data['sat_hr_counts'] /
            (global_data['sat_hr_counts'].sum() * bin_width + eps)
        )
        sat_centers = 0.5 * (sat_bins[:-1] + sat_bins[1:])

        noise_means = np.array(global_data['noise_means_lr'], dtype=np.float64)

        plt.figure(figsize=(20, 10))

        # 1 LR Spectrum
        plt.subplot(231)
        plt.imshow(np.log(lr_fft_avg + eps), cmap="viridis")
        plt.title("LR Avg Frequency Spectrum")
        plt.colorbar()

        # 2 HR Spectrum
        plt.subplot(232)
        plt.imshow(np.log(hr_fft_avg + eps), cmap="viridis")
        plt.title("HR Avg Frequency Spectrum")
        plt.colorbar()

        # 3 Gradient Magnitude
        plt.subplot(233)
        plt.imshow(grad_hr_avg, cmap="magma")
        plt.title("HR Avg Gradient Magnitude")
        plt.colorbar()

        # 4 GLCM
        plt.subplot(234)
        plt.imshow(glcm_avg[:, :, 0, 0], cmap="plasma")
        plt.title(f"LR Avg GLCM (Contrast: {glcm_contrast:.2f})")
        plt.colorbar()

        # 5 Noise mean distribution
        plt.subplot(235)
        plt.hist(
            noise_means, bins=30, color='tomato', edgecolor='black', alpha=0.8
        )
        plt.title(
            f"LR Color Noise Mean Dist\nMean={noise_means.mean():.2f} "
            f"Std={noise_means.std():.2f}"
        )
        plt.xlabel("Per-image color noise mean")
        plt.ylabel("Count")

        # 6 Saturation density
        plt.subplot(236)
        plt.plot(sat_centers, sat_lr_density, label='LR', color='steelblue')
        plt.plot(sat_centers, sat_hr_density, label='HR', color='orange')
        plt.title("Global Saturation Density")
        plt.xlabel("Saturation value")
        plt.ylabel("Density")

        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def basic_distributions(df, output_dir):
        """Save distributions.png with one histogram per unpaired metric.

        Shows the HR side of the paired variables; the LR/HR contrast is the
        subject of paired_histograms.png instead.
        """

        metrics = [
            'lpips', 'psnr', 'ssim', 'lap_var_hr', 'ringing_hr',
            'glcm_contrast'
        ]
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#6baed6"
        ]
        plt.figure(figsize=(18, 9))
        rows, cols = 2, 3
        for i, (m, c) in enumerate(zip(metrics, colors), 1):
            plt.subplot(rows, cols, i)
            plt.hist(df[m], bins=30, color=c, edgecolor='black', alpha=0.85)
            plt.title(m)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'distributions.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    @staticmethod
    def paired_histograms(df, output_dir):
        """Overlay the LR and HR histograms of every paired metric.

        These three pairs are what the degradation is read from: sharpness
        lost, edge artefacts gained and the saturation shift.
        """

        overlay_pairs = [
            ('lap_var_lr', 'lap_var_hr', 'Laplacian Variance'),
            ('ringing_lr', 'ringing_hr', 'Ringing Artifact'),
            ('saturation_mean_lr', 'saturation_mean_hr', 'Saturation Mean'),
        ]
        plt.figure(figsize=(16, 4.5))
        for i, (lr_col, hr_col, title) in enumerate(overlay_pairs, 1):
            plt.subplot(1, len(overlay_pairs), i)
            plt.hist(
                df[lr_col], bins=30, alpha=0.55, label='LR', color='#1f77b4',
                edgecolor='black', linewidth=0.4
            )
            plt.hist(
                df[hr_col], bins=30, alpha=0.55, label='HR', color='#ff7f0e',
                edgecolor='black', linewidth=0.4
            )
            plt.title(title)
            plt.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'paired_histograms.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    @staticmethod
    def paired_boxplots(df, output_dir):
        """Save paired_boxplots.png with the LR vs HR boxplots.

        Same three pairs as paired_histograms, read as medians and spread
        instead of as shapes.
        """

        plt.figure(figsize=(9, 6))
        groups = [
            ('lap_var_lr', 'lap_var_hr', 'LaplacianVar'),
            ('ringing_lr', 'ringing_hr', 'Ringing'),
            ('saturation_mean_lr', 'saturation_mean_hr', 'Saturation'),
        ]
        data, labels = [], []
        for lr_col, hr_col, name in groups:
            data.append(df[lr_col])
            labels.append(f'{name} LR')
            data.append(df[hr_col])
            labels.append(f'{name} HR')

        # Tick labels are set through xticks, which is spelled the same in
        # every Matplotlib version.
        box = plt.boxplot(data, patch_artist=True)
        palette = ['#1f77b4', '#ff7f0e'] * len(groups)
        for patch, col in zip(box['boxes'], palette):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)

        plt.xticks(
            range(1, len(labels) + 1), labels, rotation=25, ha='right'
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'paired_boxplots.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    @staticmethod
    def correlation_matrix(df, output_dir):
        """Generate a correlation heatmap over the whole retained set.

        Saves correlation_matrix.png. Every variable of the reduced set is
        included on purpose: this figure is what justifies the set, so it
        has to show the redundancy that is left rather than hide it.
        """

        metrics = [
            'lpips', 'psnr', 'ssim', 'lap_var_lr', 'lap_var_hr',
            'ringing_lr', 'ringing_hr', 'glcm_contrast',
            'saturation_mean_lr', 'saturation_mean_hr'
        ]

        available = [m for m in metrics if m in df.columns]
        if len(available) < 3:
            print('Not enough columns for correlation matrix.')
            return

        corr = df[available].corr()
        plt.figure(figsize=(1.2 * len(available), 0.9 * len(available)))
        sns.heatmap(
            corr, cmap='flare', annot=True, fmt='.2f', center=0, square=True,
            cbar_kws={'shrink': 0.75}
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'correlation_matrix.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    @staticmethod
    def scatter_relations(df, output_dir):
        """Scatter plots for metric relations.

        Saves scatter_relations.png with the fidelity metrics against each
        other, each paired variable against its own HR counterpart, and the
        sharpness and texture descriptors against PSNR.
        """

        pairs = [
            ('lpips', 'psnr', 'LPIPS vs PSNR', '#1f77b4'),
            ('lpips', 'ssim', 'LPIPS vs SSIM', '#ff7f0e'),
            ('lap_var_lr', 'lap_var_hr', 'LaplacianVar LR vs HR', '#2ca02c'),
            ('ringing_lr', 'ringing_hr', 'Ringing LR vs HR', '#9467bd'),
            ('saturation_mean_lr', 'saturation_mean_hr',
             'Saturation LR vs HR', '#8c564b'),
            ('lap_var_hr', 'psnr', 'LaplacianVar HR vs PSNR', '#d62728'),
            ('glcm_contrast', 'psnr', 'GLCM Contrast vs PSNR', '#e377c2'),
            ('ringing_lr', 'ssim', 'Ringing LR vs SSIM', '#7f7f7f')
        ]
        rows, cols = 4, 2
        plt.figure(figsize=(cols * 5, rows * 3.2))
        for i, (x, y, title, color) in enumerate(pairs, 1):
            if x not in df.columns or y not in df.columns:
                continue
            plt.subplot(rows, cols, i)
            plt.scatter(
                df[x], df[y], s=14, alpha=0.75, color=color,
                edgecolors='white', linewidths=0.4
            )
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(title, fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, 'scatter_relations.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
class EDAPipeline:
    """Runs the exploratory analysis of the LR/HR dataset end to end.

    Holds the run settings so that the steps can be called individually when
    experimenting in a notebook, while ``run`` performs the whole analysis.

    Every figure lands under ``output_dir``:

    - ``advanced_global_panel.png``: dataset-wide spectra, gradients, GLCM,
      noise and saturation.
    - ``distributions.png``, ``paired_histograms.png``,
      ``paired_boxplots.png``, ``correlation_matrix.png``,
      ``scatter_relations.png``: per-metric views.
    - ``LPIPS_Scenarios/``: the best and worst pairs by LPIPS.
    """

    def __init__(
            self,
            lr_dir=LR_ROOT,
            hr_dir=HR_ROOT,
            output_dir=EDA_RESULTS_DIR,
            top_k_examples=1,
            glcm_multi_angle=False,
            glcm_levels=32,
            upscale_interpolation=SRCNN_UPSCALE_INTERPOLATION):
        """
        Parameters
        ----------
        lr_dir, hr_dir : str
            Roots of the LR and HR trees. Pairs are matched by relative path.
        output_dir : str
            Directory the figures are written to.
        top_k_examples : int
            How many best and worst LPIPS pairs to render individually.
        glcm_multi_angle : bool
            Average the GLCM contrast over four angles instead of one.
        glcm_levels : int
            Quantisation levels of the per-image GLCM. 32 is enough for this
            texture and keeps the co-occurrence matrix 64 times smaller per
            angle than the 256 levels this used to run with.
        upscale_interpolation : int
            OpenCV interpolation used to bring LR up to the HR frame before
            comparing. Fixed for every image on purpose; see
            ``ImagePairLoader.load_and_align``.
        """

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.output_dir = output_dir
        self.top_k_examples = top_k_examples
        self.glcm_multi_angle = glcm_multi_angle
        self.glcm_levels = glcm_levels
        self.upscale_interpolation = upscale_interpolation

    def run(self):
        """Run the whole analysis.

        Returns
        -------
        pandas.DataFrame
            One row of metrics per image pair.
        """

        with stage(
                f"EDA | GLCM {self.glcm_levels} levels, "
                f"{'4 angles' if self.glcm_multi_angle else '1 angle'}") as step:
            self._prepare_output_dirs()

            step("computing per-pair metrics")
            rows, global_data = self._collect_metrics()
            df = StatsReporter.dataframe(rows)
            step(f"metrics   {len(df)} pairs x {len(df.columns) - 1} variables")

            # The panels are rendered at 300 dpi and take longer than the
            # metrics on a large dataset, so they get their own line.
            step("rendering dataset panels")
            self._save_dataset_panels(df, global_data)

            step(f"rendering {self.top_k_examples} best and worst LPIPS pairs")
            self._save_lpips_scenarios(df)

            step(f"figures   written to {self.output_dir}")

        return df

    def _scenario_dirs(self):
        """Return the directories holding the best and worst LPIPS pairs."""

        examples_dir = os.path.join(self.output_dir, "LPIPS_Scenarios")

        return (
            os.path.join(examples_dir, "best_scenarios"),
            os.path.join(examples_dir, "worst_scenarios"),
        )

    def _prepare_output_dirs(self):
        """Create the output tree before anything is written to it."""

        os.makedirs(self.output_dir, exist_ok=True)
        for directory in self._scenario_dirs():
            os.makedirs(directory, exist_ok=True)

    def _collect_metrics(self):
        """Compute the per-pair metrics and the dataset-wide accumulators."""

        return MetricsAggregator.collect(
            self.lr_dir,
            self.hr_dir,
            glcm_multi_angle=self.glcm_multi_angle,
            glcm_levels=self.glcm_levels,
            upscale_interpolation=self.upscale_interpolation,
        )

    def _save_dataset_panels(self, df, global_data):
        """Write the figures that describe the dataset as a whole."""

        ImageDataVisualization.create_global_advanced_visualizations(
            global_data,
            os.path.join(self.output_dir, "advanced_global_panel.png"),
        )
        ImageDataVisualization.basic_distributions(df, self.output_dir)
        ImageDataVisualization.paired_histograms(df, self.output_dir)
        ImageDataVisualization.paired_boxplots(df, self.output_dir)
        ImageDataVisualization.correlation_matrix(df, self.output_dir)
        ImageDataVisualization.scatter_relations(df, self.output_dir)

    def _save_lpips_scenarios(self, df):
        """Render the pairs that LPIPS rates as the best and the worst.

        These are the two ends of the degradation the dataset contains, which
        is what tells whether the LR images are plausibly bad rather than
        uniformly bad.
        """

        best_dir, worst_dir = self._scenario_dirs()
        df_sorted = df.sort_values("lpips")

        selections = (
            (df_sorted.head(self.top_k_examples), best_dir, "best"),
            (df_sorted.tail(self.top_k_examples), worst_dir, "worst"),
        )
        for subset, scenario_dir, label in selections:
            for rank, row in enumerate(subset.itertuples(), 1):
                self._save_pair_figures(
                    row.filename, row.lpips, scenario_dir, label, rank
                )

    def _save_pair_figures(
            self, rel_path, lpips_value, scenario_dir, label, rank):
        """Write the comparison and the advanced panel of a single pair.

        The subfolder structure of the dataset is preserved inside the
        scenario directory, so images of different defect classes cannot
        collide on the same filename.
        """

        lr_img, hr_img = ImagePairLoader.load_and_align(
            os.path.join(self.lr_dir, rel_path),
            os.path.join(self.hr_dir, rel_path),
            upscale_interpolation=self.upscale_interpolation,
        )

        rel_no_ext = os.path.splitext(rel_path.replace("\\", "/"))[0]
        parent = os.path.dirname(rel_no_ext)
        stem = os.path.basename(rel_no_ext)

        save_dir = (
            os.path.join(scenario_dir, parent) if parent else scenario_dir
        )
        os.makedirs(save_dir, exist_ok=True)

        ImageDataVisualization.save_visual_example(
            lr_img,
            hr_img,
            os.path.join(save_dir, f"{label}_{rank}_{stem}.png"),
            lpips_value,
        )
        ImageDataVisualization.create_advanced_visualizations(
            lr_img,
            hr_img,
            os.path.join(save_dir, f"{label}_{rank}_advanced_{stem}.png"),
        )
