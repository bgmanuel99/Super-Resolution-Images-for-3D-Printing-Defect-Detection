import itertools
import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

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
from srlib.constants import (
    DL_RESULTS_DIR,
    EDSR_PATCH_SIZE,
    EDSR_SCALE_FACTOR,
    EDSR_STRIDE,
    ESRGAN_PATCH_SIZE,
    ESRGAN_SCALE_FACTOR,
    ESRGAN_STRIDE,
    IBP_ITERATIONS,
    SRCNN_PATCH_SIZE,
    SRCNN_STRIDE,
    VGG_PATCH_SIZE,
    VGG_STRIDE,
)
from srlib.deep_learning.edsr import EDSR
from srlib.deep_learning.esrgan import ESRGAN
from srlib.deep_learning.srcnn import SRCNNModel
from srlib.defect_detection.vgg16 import FineTunedVGG16
from srlib.model_registry import model_artifacts

class DefectDetectionPipeline:
    """
    Runs the whole defect detection comparison over one test split.

    Every method reconstructs the same LR test images at the HR frame size,
    and every row, references included, is classified by the one fine-tuned
    VGG16. A single classifier is what makes the rows comparable: any
    difference between them is then a difference between the images, never
    between two models that were fitted separately.

    LR is the baseline and HR the ceiling, so the two bracket every
    reconstruction. Without the ceiling an accuracy cannot be read as close
    to or far from what the resolution allows.

    The baseline keeps its native resolution and is classified as it is,
    with no resampling: rescaling it would equalise the frame but would add
    an interpolation step that is not part of the capture, turning the
    comparison into super-resolution against interpolation. The cost is
    that it aggregates 16 voting patches against the 81 of every other row
    and shows the object at twice the apparent scale the classifier was
    trained on, so the gap between the baseline and an SR row is not due to
    resolution alone. That has to be declared when the row is read.

    All eleven reconstructions produce RGB float images in ``[0, 1]`` at the
    HR frame size, which is the input regime the classifier was trained on.
    """

    # Reading order of the table and of every figure: the LR baseline, the
    # classic interpolations, the remaining classic algorithms, the learned
    # models, and the HR ceiling last. The two references sit at the ends so
    # every figure is read as the band the middle rows have to fall in, and
    # the families in between are grouped by increasing sophistication.
    METHODS = (
        "LR",
        "Bilinear",
        "Bicubic",
        "Area",
        "Lanczos",
        "Back-Projection",
        "Non-Local Means",
        "Edge-guided",
        "Freq-extrapolation",
        "SRCNN",
        "EDSR",
        "ESRGAN",
        "HR",
    )

    # Short labels for the qualitative grid, where long names do not fit.
    SHORT_LABELS = {
        "Back-Projection": "IBP",
        "Non-Local Means": "NLM",
        "Edge-guided": "EGI",
        "Freq-extrapolation": "FDE",
    }

    INTERPOLATIONS = {
        "Bilinear": interpolate_bilinear,
        "Bicubic": interpolate_bicubic,
        "Area": interpolate_area,
        "Lanczos": interpolate_lanczos,
    }

    ADVANCED = {
        "Back-Projection": back_projection,
        "Non-Local Means": non_local_means,
        "Edge-guided": edge_guided_interpolation,
        "Freq-extrapolation": frequency_extrapolation,
    }

    def __init__(
            self,
            X_LR_test,
            X_HR_test,
            y_test,
            srcnn_run=None,
            edsr_run=None,
            esrgan_run=None,
            vgg16_run=None,
            edsr_scale_factor=EDSR_SCALE_FACTOR,
            esrgan_scale_factor=ESRGAN_SCALE_FACTOR,
            srcnn_patch_size=SRCNN_PATCH_SIZE,
            srcnn_stride=SRCNN_STRIDE,
            edsr_patch_size=EDSR_PATCH_SIZE,
            edsr_stride=EDSR_STRIDE,
            esrgan_patch_size=ESRGAN_PATCH_SIZE,
            esrgan_stride=ESRGAN_STRIDE,
            esrgan_batch_size=8,
            vgg_patch_size=VGG_PATCH_SIZE,
            vgg_stride=VGG_STRIDE,
            vgg_batch_size=64,
            ibp_iterations=IBP_ITERATIONS,
            results_dir=DL_RESULTS_DIR):
        """
        Resolve the training runs and keep the inference settings.

        Parameters
        ----------
        X_LR_test : np.ndarray
            (N, H, W, 3) LR test images as RGB floats in ``[0, 1]``.
        X_HR_test : np.ndarray
            (N, H*, W*, 3) original HR frames of the same test images, in the
            same order. They are the ceiling row, not an input to any
            reconstruction.
        y_test : np.ndarray
            Ground truth class of each test image.
        srcnn_run, edsr_run, esrgan_run, vgg16_run : str, optional
            Run timestamps to load. None picks the most recent run of that
            model, so a rerun of the training notebooks is enough to refresh
            the comparison.
        results_dir : str
            Directory every figure is written to.
        """

        self.X_LR_test = X_LR_test
        self.X_HR_test = X_HR_test
        self.y_test = np.asarray(y_test)
        self.results_dir = results_dir

        if len(X_HR_test) != len(X_LR_test):
            raise ValueError(
                "X_HR_test and X_LR_test must hold the same images in the "
                f"same order, got {len(X_HR_test)} and {len(X_LR_test)}."
            )

        self.srcnn_patch_size = srcnn_patch_size
        self.srcnn_stride = srcnn_stride
        self.edsr_patch_size = edsr_patch_size
        self.edsr_stride = edsr_stride
        self.esrgan_patch_size = esrgan_patch_size
        self.esrgan_stride = esrgan_stride
        self.esrgan_batch_size = esrgan_batch_size
        self.vgg_patch_size = vgg_patch_size
        self.vgg_stride = vgg_stride
        self.vgg_batch_size = vgg_batch_size
        self.ibp_iterations = ibp_iterations
        self.edsr_scale_factor = edsr_scale_factor
        self.esrgan_scale_factor = esrgan_scale_factor

        self.artifacts = {
            "SRCNN": model_artifacts("SRCNN", srcnn_run),
            "EDSR": model_artifacts(
                "EDSR", edsr_run, scale_factor=edsr_scale_factor
            ),
            "ESRGAN": model_artifacts(
                "ESRGAN", esrgan_run, scale_factor=esrgan_scale_factor
            ),
            "VGG16": model_artifacts("VGG16", vgg16_run),
        }

        with open(self.artifacts["SRCNN"]["hr_dimensions"], "rb") as f:
            self.hr_h, self.hr_w = pickle.load(f)

        self.models = {}
        self.sr_images = {}
        self.labels = {}
        self.confidences = {}

    def report_runs(self):
        """Print the run each model was loaded from."""

        print("Resolved runs")
        for model, artifacts in self.artifacts.items():
            print(f"  {model:<7} {artifacts['timestamp']}")
        print(f"  HR frame size: {self.hr_w} x {self.hr_h}")

    def setup_models(self):
        """Load the three SR generators and the VGG16 classifier."""

        print("Loading models")

        srcnn = SRCNNModel()
        srcnn.setup_model(
            from_pretrained=True,
            pretrained_path=self.artifacts["SRCNN"]["weights"],
        )

        edsr = EDSR()
        edsr.setup_model(
            scale_factor=self.edsr_scale_factor,
            from_pretrained=True,
            pretrained_path=self.artifacts["EDSR"]["weights"],
        )

        esrgan = ESRGAN()
        esrgan.setup_model(
            scale_factor=self.esrgan_scale_factor,
            from_trained=True,
            generator_pretrained_path=self.artifacts["ESRGAN"]["generator"],
            discriminator_pretrained_path=(
                self.artifacts["ESRGAN"]["discriminator"]
            ),
        )

        vgg16 = FineTunedVGG16()
        vgg16.setup_model(
            from_pretrained=True,
            pretrained_path=self.artifacts["VGG16"]["weights"],
        )

        self.models = {
            "SRCNN": srcnn,
            "EDSR": edsr,
            "ESRGAN": esrgan,
            "VGG16": vgg16,
        }

        return self.models

    # Width every per-row log line pads its label to, so the build section
    # and the classification section line their columns up with each other.
    _LABEL_WIDTH = 22

    def _map_images(self, name, transform):
        """
        Apply a per-image transform over the LR test set.

        The progress bar is transient and the summary line is not, so a
        finished row leaves one line behind instead of two: over thirteen
        rows that is the difference between a readable log and a wall.

        Returns
        -------
        list
            One output per test image, in input order.
        """

        started = time.perf_counter()
        outputs = [
            transform(lr_img)
            for lr_img in tqdm(
                self.X_LR_test,
                total=len(self.X_LR_test),
                desc=f"  {name}",
                leave=False,
            )
        ]
        print(
            f"  {name:<{self._LABEL_WIDTH}} "
            f"{time.perf_counter() - started:7.2f}s",
            flush=True,
        )

        return outputs

    def _run_deep_model(self, name, super_resolve):
        """Super-resolve the test set with one deep learning model."""

        return np.stack(self._map_images(name, super_resolve), axis=0)

    def build_sr_images(self):
        """
        Reconstruct the test set with every method.

        The two reference rows are stored unchanged: LR at its native
        resolution and HR as the ceiling. The remaining eleven methods
        output RGB floats in ``[0, 1]`` at the HR frame size.

        Returns
        -------
        dict
            Method name to its reconstructions. Built in whatever order is
            cheapest; every figure reads it back in ``METHODS`` order.
        """

        if not self.models:
            self.setup_models()

        print(
            f"\nBuilding super-resolved images from "
            f"{len(self.X_LR_test)} LR test frames"
        )
        started = time.perf_counter()

        # Neither reference is reconstructed, so both bypass _map_images.
        self.sr_images = {"LR": self.X_LR_test, "HR": self.X_HR_test}
        for name, note in (
                ("LR", "baseline, native resolution"),
                ("HR", "reference, used unchanged")):
            print(f"  {name:<{self._LABEL_WIDTH}} {note}")

        self.sr_images["SRCNN"] = self._run_deep_model(
            "SRCNN",
            lambda lr: self.models["SRCNN"].super_resolve_image(
                lr, hr_h=self.hr_h, hr_w=self.hr_w,
                patch_size=self.srcnn_patch_size, stride=self.srcnn_stride,
            ),
        )
        self.sr_images["EDSR"] = self._run_deep_model(
            "EDSR",
            lambda lr: self.models["EDSR"].super_resolve_image(
                lr, patch_size_lr=self.edsr_patch_size,
                stride=self.edsr_stride,
            ),
        )
        self.sr_images["ESRGAN"] = self._run_deep_model(
            "ESRGAN",
            lambda lr: self.models["ESRGAN"].super_resolve_image(
                lr, patch_size_lr=self.esrgan_patch_size,
                stride=self.esrgan_stride,
                batch_size=self.esrgan_batch_size,
            ),
        )

        for name, upscale in self.INTERPOLATIONS.items():
            self.sr_images[name] = self._map_images(
                name,
                lambda lr, fn=upscale: fn(
                    lr, target_shape=(self.hr_w, self.hr_h)
                ),
            )

        for name, algorithm in self.ADVANCED.items():
            self.sr_images[name] = self._map_images(
                name, lambda lr, fn=algorithm: self._advanced_sr(fn, lr)
            )

        print(f"SR images built in {time.perf_counter() - started:.2f}s")

        return self.sr_images

    def _advanced_sr(self, algorithm, lr_img):
        """
        Reconstruct one image with a single-channel classic algorithm.

        The algorithm runs on each RGB channel and the result is returned on
        the same scale as the interpolation family, so the classifier sees
        one single input regime.
        """

        kwargs = {"target_shape": (self.hr_h, self.hr_w)}
        if algorithm is back_projection:
            kwargs["iterations"] = self.ibp_iterations

        sr = apply_per_channel(
            algorithm, (lr_img * 255).astype(np.uint8), **kwargs
        )

        return sr.astype(np.float32) / 255.0

    def predict(self):
        """
        Classify every row with the fine-tuned VGG16.

        The same classifier scores the two references and the eleven
        reconstructions, so no row carries an advantage from having been
        matched with a model of its own.

        Returns
        -------
        tuple
            ``(labels, confidences)``, each a dict keyed by method name.
        """

        if not self.sr_images:
            raise RuntimeError(
                "No reconstructions available. Call build_sr_images first."
            )
        if not self.models:
            self.setup_models()

        print(f"\nClassifying {len(self.METHODS)} rows with VGG16")
        started = time.perf_counter()

        classifier = self.models["VGG16"]

        self.labels, self.confidences = {}, {}
        for name in self.METHODS:
            row_started = time.perf_counter()
            labels, confidences = self._classify(
                classifier, self.sr_images[name], name
            )
            self.labels[name] = labels
            self.confidences[name] = confidences

            accuracy = float(np.mean(np.asarray(labels) == self.y_test))
            print(
                f"  {name:<{self._LABEL_WIDTH}} "
                f"{time.perf_counter() - row_started:7.2f}s  "
                f"accuracy={accuracy:.4f}",
                flush=True,
            )

        print(f"Predictions done in {time.perf_counter() - started:.2f}s")

        return self.labels, self.confidences

    def _classify(self, classifier, images, name):
        """
        Classify a set of images by majority voting over patches.

        The progress bar is transient for the same reason it is in
        ``_map_images``: the caller writes the line that stays.

        Returns
        -------
        tuple
            ``(labels, confidences)`` as plain lists.
        """

        labels, confidences = [], []
        for image in tqdm(
                images, total=len(images), desc=f"  {name}", leave=False):
            label, confidence = classifier.classify_defects_method(
                image=image,
                patch_size=self.vgg_patch_size,
                stride=self.vgg_stride,
                batch_size=self.vgg_batch_size,
            )
            labels.append(int(label))
            confidences.append(float(confidence))

        return labels, confidences

    def run(self):
        """
        Set up the models, reconstruct the test set and classify it.

        Returns
        -------
        tuple
            ``(sr_images, labels, confidences)``.
        """

        started = time.perf_counter()

        self.report_runs()
        self.setup_models()
        self.build_sr_images()
        self.predict()

        print(f"\nPipeline finished in {time.perf_counter() - started:.2f}s")

        return self.sr_images, self.labels, self.confidences

    @property
    def class_names(self):
        """Class labels present in the ground truth, as strings."""

        return [str(c) for c in sorted(np.unique(self.y_test))]

    def _require_predictions(self):
        """Fail with an actionable message when there is nothing to plot."""

        if not self.labels:
            raise RuntimeError(
                "No predictions available. Call run or predict first."
            )

    def _save_figure(self, fig, filename, dpi=150, also_eps=False):
        """Write a figure to the results directory."""

        os.makedirs(self.results_dir, exist_ok=True)
        fig.savefig(
            os.path.join(self.results_dir, filename), dpi=dpi,
            bbox_inches="tight",
        )
        if also_eps:
            stem = os.path.splitext(filename)[0]
            fig.savefig(
                os.path.join(self.results_dir, f"{stem}.eps"), dpi=dpi,
                format="eps", bbox_inches="tight",
            )

    @staticmethod
    def _draw_confusion(ax, cm, classes, title):
        """Draw one confusion matrix with its counts annotated."""

        image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        threshold = cm.max() / 2.0
        for i, j in itertools.product(
                range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j, i, format(cm[i, j], "d"), horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
            )

        return image

    def _method_grid(self, cols=3):
        """Return the ``(rows, cols)`` grid that fits every method.

        Derived from ``METHODS`` rather than fixed, because a hardcoded grid
        smaller than the method count makes ``zip`` drop the last methods
        without raising.
        """

        return math.ceil(len(self.METHODS) / cols), cols

    @staticmethod
    def _hide_unused_axes(flat_axes, used):
        """Turn off the axes left empty by a grid wider than the method set."""

        for ax in flat_axes[used:]:
            ax.axis("off")

    def plot_confusion_matrices(self, figsize=None):
        """
        Draw the confusion matrix of every method.

        Returns
        -------
        tuple
            ``(fig, axes)``.
        """

        self._require_predictions()

        rows, cols = self._method_grid()
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize or (cols * 6, rows * 5)
        )
        flat_axes = axes.flatten()

        for ax, name in zip(flat_axes, self.METHODS):
            cm = confusion_matrix(self.y_test, self.labels[name])
            self._draw_confusion(
                ax, cm, self.class_names, f"{name} Confusion Matrix"
            )

        self._hide_unused_axes(flat_axes, len(self.METHODS))
        fig.tight_layout()
        self._save_figure(fig, "confusion_matrix_panel.png")

        return fig, axes

    def classification_metrics(self):
        """
        Score every method with scikit-learn's classification report.

        Returns
        -------
        dict
            'accuracy', 'macro_f1', 'weighted_f1' and 'macro_recall' as lists
            in METHODS order, plus 'f1_per_class' and 'acc_per_class' as
            (classes, methods) arrays.
        """

        self._require_predictions()

        classes_sorted = sorted(np.unique(self.y_test))
        class_names = self.class_names
        n_classes, n_methods = len(class_names), len(self.METHODS)

        accuracies, macro_f1s, weighted_f1s, macro_recalls = [], [], [], []
        f1_per_class = np.full((n_classes, n_methods), np.nan)
        acc_per_class = np.full((n_classes, n_methods), np.nan)

        for j, name in enumerate(self.METHODS):
            report = classification_report(
                self.y_test, self.labels[name],
                labels=classes_sorted, target_names=class_names,
                output_dict=True, zero_division=0,
            )

            accuracies.append(float(report["accuracy"]))
            macro_f1s.append(float(report["macro avg"]["f1-score"]))
            weighted_f1s.append(float(report["weighted avg"]["f1-score"]))
            macro_recalls.append(float(report["macro avg"]["recall"]))

            for i, class_name in enumerate(class_names):
                f1_per_class[i, j] = float(report[class_name]["f1-score"])
                # Per-class accuracy is the recall of that class.
                acc_per_class[i, j] = float(report[class_name]["recall"])

        return {
            "accuracy": accuracies,
            "macro_f1": macro_f1s,
            "weighted_f1": weighted_f1s,
            "macro_recall": macro_recalls,
            "f1_per_class": f1_per_class,
            "acc_per_class": acc_per_class,
        }

    def plot_classification_reports(self, figsize=(22, 16)):
        """
        Draw the classification report panel of every method.

        Four bar charts of aggregate scores plus two heatmaps resolved per
        class, since the aggregate hides which class each method fails on.

        Returns
        -------
        tuple
            ``(fig, axes, metrics)``.
        """

        metrics = self.classification_metrics()
        names = list(self.METHODS)
        positions = np.arange(len(names))

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        bar_panels = [
            (axes[0, 0], "accuracy", "Overall accuracy", "tab:blue"),
            (axes[0, 1], "macro_recall", "Macro recall", "tab:purple"),
            (axes[1, 0], "macro_f1", "Macro F1", "tab:green"),
            (axes[1, 1], "weighted_f1", "Weighted F1", "tab:orange"),
        ]
        for ax, key, label, color in bar_panels:
            values = metrics[key]
            bars = ax.bar(positions, values, color=color, alpha=0.88)
            ax.set_title(f"{label} per method (higher is better)")
            ax.set_ylabel(label)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(positions)
            ax.set_xticklabels(names, rotation=30, ha="right")
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, value,
                        f"{value:.2f}", ha="center", va="bottom", fontsize=8,
                    )

        heatmaps = [
            (axes[2, 0], "f1_per_class", "F1-score", "YlGnBu"),
            (axes[2, 1], "acc_per_class", "Per-class accuracy", "YlOrRd"),
        ]
        for ax, key, label, cmap in heatmaps:
            values = metrics[key]
            image = ax.imshow(
                values, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0
            )
            ax.set_title(f"{label} per class and method")
            ax.set_xlabel("Method")
            ax.set_ylabel("Class")
            ax.set_xticks(positions)
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(self.class_names)))
            ax.set_yticklabels(self.class_names)
            for i, j in itertools.product(
                    range(values.shape[0]), range(values.shape[1])):
                if np.isfinite(values[i, j]):
                    ax.text(
                        j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black",
                    )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
                label
            )

        fig.tight_layout(rect=(0, 0, 1, 0.98))
        self._save_figure(fig, "classification_report_panel.png")

        return fig, axes, metrics

    def plot_sample_predictions(self, index=0, figsize=None):
        """
        Draw every reconstruction of one test image with its prediction.

        Parameters
        ----------
        index : int
            Position in the test set to illustrate.

        Returns
        -------
        tuple
            ``(fig, axes)``.
        """

        self._require_predictions()

        if not 0 <= index < len(self.y_test):
            raise IndexError(
                f"index {index} is outside the test set of "
                f"{len(self.y_test)} images."
            )

        rows, cols = self._method_grid()
        title_font = {
            "family": "serif", "color": "black", "weight": "bold", "size": 10,
        }
        fig, axes = plt.subplots(
            rows, cols, figsize=figsize or (cols * 2.5, rows * 2.5)
        )
        flat_axes = axes.flatten()

        for ax, name in zip(flat_axes, self.METHODS):
            label = self.SHORT_LABELS.get(name, name)
            prediction = self.labels[name][index]
            confidence = self.confidences[name][index]

            ax.imshow(np.clip(self.sr_images[name][index], 0.0, 1.0))
            ax.set_title(
                f"{label} (Pred: {prediction}, Conf: {confidence:.2f})",
                fontdict=title_font,
            )
            ax.axis("off")

        self._hide_unused_axes(flat_axes, len(self.METHODS))
        fig.suptitle(f"True label: {self.y_test[index]}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self._save_figure(
            fig, "vgg16_predictions.png", dpi=300, also_eps=False
        )

        return fig, axes

    def confidence_metrics(self):
        """
        Split the mean confidence of every method by prediction outcome.

        Returns
        -------
        dict
            'mean_all', 'mean_correct', 'mean_wrong' and 'error_rate' as
            lists in METHODS order.
        """

        self._require_predictions()

        mean_all, mean_correct, mean_wrong, error_rate = [], [], [], []

        for name in self.METHODS:
            predictions = np.asarray(self.labels[name], dtype=int)
            confidences = np.asarray(self.confidences[name], dtype=float)
            correct = predictions == self.y_test

            mean_all.append(float(np.mean(confidences)))
            mean_correct.append(
                float(np.mean(confidences[correct])) if correct.any()
                else np.nan
            )
            mean_wrong.append(
                float(np.mean(confidences[~correct])) if (~correct).any()
                else np.nan
            )
            error_rate.append(1.0 - float(np.mean(correct)))

        return {
            "mean_all": mean_all,
            "mean_correct": mean_correct,
            "mean_wrong": mean_wrong,
            "error_rate": error_rate,
        }

    def plot_confidence(self, figsize=(12, 8)):
        """
        Draw the confidence panel of every method.

        Global mean confidence, then the same split into correct and wrong
        predictions, then the error rate. Reading the three together is what
        tells whether a method is confidently wrong.

        Returns
        -------
        tuple
            ``(fig, axes, metrics)``.
        """

        metrics = self.confidence_metrics()
        names = list(self.METHODS)
        positions = np.arange(len(names))
        title_font = {"family": "serif", "color": "black", "size": 14}

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        bars = axes[0].bar(
            positions, metrics["mean_all"], color="tab:blue", alpha=0.85
        )
        axes[0].set_ylabel("Mean confidence", fontsize=13)
        axes[0].set_title(
            "Global mean confidence per method", fontdict=title_font
        )
        for bar, value in zip(bars, metrics["mean_all"]):
            if np.isfinite(value):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=11,
                )

        width = 0.4
        axes[1].bar(
            positions - width / 2, metrics["mean_correct"], width=width,
            label="Correct predictions", color="tab:green", alpha=0.85,
        )
        axes[1].bar(
            positions + width / 2, metrics["mean_wrong"], width=width,
            label="Wrong predictions", color="tab:red", alpha=0.75,
        )
        axes[1].set_ylabel("Mean confidence", fontsize=13)
        axes[1].set_title(
            "Mean confidence by outcome", fontdict=title_font
        )
        axes[1].legend(loc="lower center", ncols=2, fontsize=9)
        for offset, key in (
                (-width / 2, "mean_correct"), (width / 2, "mean_wrong")):
            for position, value in zip(positions, metrics[key]):
                if np.isfinite(value):
                    axes[1].text(
                        position + offset, value, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=9,
                    )

        bars = axes[2].bar(
            positions, metrics["error_rate"], color="tab:red", alpha=0.8
        )
        axes[2].set_ylabel("Error rate", fontsize=13)
        axes[2].set_title(
            "Error rate per method (1 - accuracy)", fontdict=title_font
        )
        axes[2].set_xticks(positions)
        axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=11)
        for bar, value in zip(bars, metrics["error_rate"]):
            if np.isfinite(value):
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=11,
                )

        fig.tight_layout()
        self._save_figure(
            fig, "sr_confidence_panel.png", dpi=300, also_eps=False
        )

        return fig, axes, metrics
