import datetime
import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Conv2D, InputLayer
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from srlib.constants import (
    SRCNN_PATCH_SIZE,
    SRCNN_STRIDE,
    SRCNN_UPSCALE_INTERPOLATION,
    TIMESTAMP_FORMAT,
)
from srlib.dataset.loading import add_padding
from srlib.metrics import psnr, ssim
from srlib.model_registry import prepare_run_directory, save_run_metrics
from srlib.progress import stage
from srlib.deep_learning.callbacks import (
    EpochMemoryCallback,
    EpochTimeCallback,
    last_epoch_metrics,
    profile_evaluation,
)

class SRCNNModel:
    def __init__(self):
        self.model = None
        self._trained = False

    def setup_model(
            self, 
            input_shape=None, 
            learning_rate=1e-4, 
            from_pretrained=False, 
            pretrained_path=None):
        """Sets up the model: either loads pretrained or builds + compiles a new model."""
        
        if from_pretrained:
            if pretrained_path is None or not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"Pretrained model file not found at {pretrained_path}")
            
            self.model = load_model(pretrained_path, custom_objects={"psnr": psnr, "ssim": ssim})
            print(f"Loaded pretrained model from {pretrained_path}")
            self._trained = True
        else:
            if input_shape is None:
                raise ValueError("input_shape must be provided when not using a pretrained model.")
            
            self._build_model(input_shape)
            self._compile_model(learning_rate)

    def _build_model(self, input_shape):
        """Builds the SRCNN model using Sequential API."""
        
        self.model = Sequential([
            InputLayer(input_shape=input_shape), 
            Conv2D(96, (9, 9), activation="relu", padding="same"),
            Conv2D(32, (1, 1), activation="relu", padding="same"),
            Conv2D(3, (5, 5), activation="linear", padding="same")
        ])

    def _compile_model(self, learning_rate):
        """Compiles the model."""
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[psnr, ssim])
        self.model.summary()

    def fit(
            self,
            X_train,
            Y_train,
            X_val,
            Y_val,
            batch_size=16,
            epochs=50):
        """Trains the model over the extracted patch pairs and callbacks."""
        
        if self.model is None:
            raise ValueError("Model has not been set up.")
        
        devices = tf.config.list_physical_devices("GPU")
        if devices:
            print("Training on GPU:", devices[0].name)
        else:
            print("Training on CPU")
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
            EpochTimeCallback(),
            EpochMemoryCallback(track_gpu=True, gpu_device="GPU:0"),
        ]

        history = self.model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks
        )

        self._trained = True
        
        return history, callbacks[2], callbacks[3]

    def evaluate(self, X_test, Y_test):
        """Evaluates the model."""
        
        if not self._trained:
            raise RuntimeError("Model has not been trained.")
        
        results = self.model.evaluate(X_test, Y_test)
        print(f"Loss: {results[0]:.4f}, PSNR: {results[1]:.2f} dB, SSIM: {results[2]:.4f}")
        
        return results

    def evaluate_and_save(
            self, X_test, Y_test, history, time_cb, mem_cb, hr_h, hr_w,
            timestamp=None):
        """Evaluate the run, then persist the model and its metrics.

        The three steps travel together because they describe one training
        run: splitting them across cells is what lets a checkpoint be saved
        under one timestamp and its metrics under another.

        Cost is charged over training and over this evaluation only. The
        patch-wise reconstruction of a full frame belongs to the detection
        pipeline, so it is not attributed to the model.

        Parameters
        ----------
        X_test, Y_test : np.ndarray
            Test partition produced by ``load_srcnn_dataset``.
        history : keras.callbacks.History
            Returned by ``fit``.
        time_cb, mem_cb : EpochTimeCallback, EpochMemoryCallback
            Returned by ``fit`` alongside the history.
        hr_h, hr_w : int
            HR frame size, persisted so the pipeline can reassemble frames.
        timestamp : str, optional
            Run identifier. Defaults to the current time.

        Returns
        -------
        tuple
            ``(timestamp, run_dir, metrics)``.
        """

        timestamp = timestamp or datetime.datetime.now().strftime(
            TIMESTAMP_FORMAT
        )
        run_name = f"SRCNN_{timestamp}"

        with stage(f"SRCNN run {timestamp}") as step:
            step("evaluating on the test partition")
            results, eval_time_sec, eval_memory = profile_evaluation(
                lambda: self.evaluate(X_test, Y_test)
            )

            metrics = {
                "eval_loss": float(results[0]),
                "eval_psnr": float(results[1]),
                "eval_ssim": float(results[2]),
                **last_epoch_metrics(history.history),
                "epoch_time_sec": time_cb.mean_time_value(),
                "memory": mem_cb.as_dict(),
                "eval_time_sec": eval_time_sec,
                "eval_memory": eval_memory,
            }

            run_dir = prepare_run_directory("SRCNN", run_name)
            self.save(directory=run_dir, timestamp=timestamp)

            dimensions_path = os.path.join(run_dir, f"{run_name}_hrh_hrw.pkl")
            with open(dimensions_path, "wb") as f:
                pickle.dump((hr_h, hr_w), f)
            step(f"frame size -> {dimensions_path}")

            step(f"metrics    -> {save_run_metrics(run_dir, run_name, metrics)}")

        return timestamp, run_dir, metrics
    
    def super_resolve_image(self, lr_img, hr_h, hr_w, patch_size=SRCNN_PATCH_SIZE, stride=SRCNN_STRIDE, interpolation=SRCNN_UPSCALE_INTERPOLATION):
        """Super-resolve an in-memory LR RGB image array using padding and patch-wise inference.
        Args:
            lr_img: np.ndarray RGB image; dtype uint8 [0,255] or float32 [0,1] or [0,255].
            hr_h, hr_w: Target HR dimensions to which LR is first upscaled before SRCNN.
            patch_size, stride: Patch extraction parameters.
            interpolation: OpenCV interpolation used to upscale LR to (hr_w, hr_h).
                Defaults to the same shared constant the training loader uses,
                since SRCNN sees an already-upscaled image and a mismatch here
                would feed it a different input distribution than it learnt on.
        Returns:
            np.ndarray float32 RGB in [0,1] of shape (hr_h, hr_w, 3).
        """
        
        if not self._trained:
            raise RuntimeError("Model has not been trained.")
        if lr_img is None or not isinstance(lr_img, np.ndarray):
            raise ValueError("lr_img must be a numpy array (RGB).")
        
        def extract_patches_from_image(image, patch_size=33, stride=14):
            """Extracts patches from an image."""
            
            h, w, _ = image.shape
            patches = []
            positions = []

            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                    patches.append(patch)
                    positions.append((i, j))

            return np.array(patches), positions

        def reconstruct_from_patches(patches, positions, padded_shape, original_shape, patch_size=33):
            """Reconstructs image and crops to original size."""
            
            h_pad, w_pad = padded_shape[:2]
            h_orig, w_orig = original_shape
            
            reconstructed = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
            weight = np.zeros((h_pad, w_pad, 3), dtype=np.float32)

            for patch, (i, j) in zip(patches, positions):
                reconstructed[i:i+patch_size, j:j+patch_size, :] += patch
                weight[i:i+patch_size, j:j+patch_size, :] += 1.0

            # Avoid division by zero
            reconstructed = np.divide(
                reconstructed, 
                weight, 
                out=np.zeros_like(reconstructed), 
                where=weight!=0
            )
            
            # Crop back to the original size
            reconstructed = reconstructed[:h_orig, :w_orig, :]
            
            return np.clip(reconstructed, 0, 1)

        # Upscale LR to expected HR size
        img_lr_up = cv2.resize(lr_img, (hr_w, hr_h), interpolation=interpolation)

        # Pad with the shared helper so this grid matches the training one
        original_shape = img_lr_up.shape[:2]
        padded_img = add_padding(img_lr_up, patch_size, stride)

        # Extract patches
        patches, positions = extract_patches_from_image(padded_img, patch_size, stride)
        patches = np.array(patches)

        preds = self.model.predict(patches, batch_size=16, verbose=0)

        # Reconstruct
        return reconstruct_from_patches(preds, positions, padded_img.shape, original_shape, patch_size)

    def save(self, directory, timestamp):
        """Saves the model to a .h5 file with a timestamp."""
        
        if not self._trained:
            raise RuntimeError("Cannot save an untrained model.")
        if not directory:
            raise ValueError("Directory path must be provided.")
        
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"SRCNN_{timestamp}.h5")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")