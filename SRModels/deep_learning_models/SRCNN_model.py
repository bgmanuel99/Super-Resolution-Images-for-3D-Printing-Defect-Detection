import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Conv2D, InputLayer
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from SRModels.metrics import psnr, ssim
from SRModels.data_augmentation import AdvancedAugmentGenerator
from SRModels.deep_learning_models.callbacks import EpochTimeCallback, EpochMemoryCallback

class SRCNNModel:
    def __init__(self):
        self.model = None
        self._trained = False

    def setup_model(
            self, 
            input_shape=(33, 33, 3), 
            learning_rate=1e-4, 
            loss="mean_squared_error", 
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
            self._build_model(input_shape)
            self._compile_model(learning_rate, loss)

    def _build_model(self, input_shape):
        """Builds the SRCNN model using Sequential API."""
        
        self.model = Sequential([
            InputLayer(input_shape=input_shape), 
            Conv2D(64, (9, 9), activation="relu", padding="same"),
            Conv2D(32, (1, 1), activation="relu", padding="same"),
            Conv2D(3, (5, 5), activation="linear", padding="same")
        ])

    def _compile_model(self, learning_rate, loss):
        """Compiles the model."""
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[psnr, ssim])
        self.model.summary()

    def fit(
            self, 
            X_train, 
            Y_train, 
            X_val, 
            Y_val, 
            batch_size=16, 
            epochs=50, 
            use_augmentation=True, 
            use_mix=True, 
            augment_validation=False):
        """Trains the model with optional data augmentation and callbacks."""
        
        if self.model is None:
            raise ValueError("Model has not been set up.")
        
        devices = tf.config.list_physical_devices("GPU")
        if devices:
            print("Training on GPU:", devices[0].name)
        else:
            print("Training on CPU")

        # Callbacks
        callbacks = [
            EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1), 
            EpochTimeCallback(), 
            EpochMemoryCallback(track_cpu=True, track_gpu=True, gpu_device="GPU:0")
        ]
        
        if use_augmentation:
            train_gen = AdvancedAugmentGenerator(X_train, Y_train, batch_size=batch_size, shuffle=True, use_mix=use_mix)
            
            if augment_validation:
                # (Not recommended by default)
                val_gen = AdvancedAugmentGenerator(X_val, Y_val, batch_size=batch_size, shuffle=False, use_mix=use_mix)

                self.model.fit(
                    train_gen,
                    steps_per_epoch=len(train_gen),
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=len(val_gen),
                    callbacks=callbacks
                )
            else:
                self.model.fit(
                    train_gen,
                    steps_per_epoch=len(train_gen),
                    epochs=epochs,
                    validation_data=(X_val, Y_val),
                    callbacks=callbacks
                )
        else:
            self.model.fit(
                X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, Y_val),
                callbacks=callbacks
            )

        self._trained = True
        
        return callbacks[2], callbacks[3]  # Return time and memory callbacks

    def evaluate(self, X_test, Y_test):
        """Evaluates the model."""
        
        if not self._trained:
            raise RuntimeError("Model has not been trained.")
        
        results = self.model.evaluate(X_test, Y_test)
        print(f"Loss: {results[0]:.4f}, PSNR: {results[1]:.2f} dB, SSIM: {results[2]:.4f}")
        
        return results
    
    def super_resolve_image(self, lr_img, hr_h, hr_w, patch_size=33, stride=14, interpolation=cv2.INTER_CUBIC):
        """Super-resolve an in-memory LR RGB image array using padding and patch-wise inference.
        Args:
            lr_img: np.ndarray RGB image; dtype uint8 [0,255] or float32 [0,1] or [0,255].
            hr_h, hr_w: Target HR dimensions to which LR is first upscaled before SRCNN.
            patch_size, stride: Patch extraction parameters.
            interpolation: OpenCV interpolation used to upscale LR to (hr_w, hr_h).
        Returns:
            np.ndarray float32 RGB in [0,1] of shape (hr_h, hr_w, 3).
        """
        
        if not self._trained:
            raise RuntimeError("Model has not been trained.")
        if lr_img is None or not isinstance(lr_img, np.ndarray):
            raise ValueError("lr_img must be a numpy array (RGB).")
        
        def add_padding(image, patch_size, stride):
            """Add padding to ensure full coverage."""
            
            h, w, c = image.shape
            
            # Calcular cuánto padding se necesita
            pad_h = (patch_size - (h % stride)) % stride if h % stride != 0 else 0
            pad_w = (patch_size - (w % stride)) % stride if w % stride != 0 else 0
            
            # Agregar padding extra para asegurar cobertura completa
            pad_h = max(pad_h, patch_size - stride)
            pad_w = max(pad_w, patch_size - stride)
            
            # Padding reflejado (mirror) para mantener continuidad
            padded_img = np.pad(
                image, 
                ((0, pad_h), (0, pad_w), (0, 0)), 
                mode='reflect'
            )
            
            return padded_img, (h, w)
        
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

            # Evitar división por cero
            reconstructed = np.divide(
                reconstructed, 
                weight, 
                out=np.zeros_like(reconstructed), 
                where=weight!=0
            )
            
            # Recortar al tamaño original
            reconstructed = reconstructed[:h_orig, :w_orig, :]
            
            return np.clip(reconstructed, 0, 1)

        # Upscale LR to expected HR size
        img_lr_up = cv2.resize(lr_img, (hr_w, hr_h), interpolation=interpolation)

        # Agregar padding
        padded_img, original_shape = add_padding(img_lr_up, patch_size, stride)
        
        print(f"Original shape: {img_lr_up.shape}")
        print(f"Padded shape: {padded_img.shape}")

        # Extraer patches
        patches, positions = extract_patches_from_image(padded_img, patch_size, stride)
        patches = np.array(patches)
        
        print(f"Total patches: {len(patches)}")

        # Predict (measure only the model inference time & GPU memory)
        def _read_gpu_info(device="GPU:0"):
            try:
                return tf.config.experimental.get_memory_info(device)
            except Exception:
                return None

        gpu_begin = _read_gpu_info()
        t0 = time.perf_counter()

        preds = self.model.predict(patches, batch_size=16)

        elapsed = time.perf_counter() - t0
        gpu_end = _read_gpu_info()

        # Build metrics dict (MB for memory)
        def _mb(x):
            return None if x is None else float(x) / (1024.0 * 1024.0)


        cur_begin = gpu_begin.get("current") if isinstance(gpu_begin, dict) else None
        cur_end = gpu_end.get("current") if isinstance(gpu_end, dict) else None
        peak_begin = gpu_begin.get("peak") if isinstance(gpu_begin, dict) else None
        peak_end = gpu_end.get("peak") if isinstance(gpu_end, dict) else None

        # Approximate mean GPU memory usage as the average of begin and end 'current' values
        if cur_begin is not None and cur_end is not None:
            mean_current_bytes = (cur_begin + cur_end) / 2.0
            gpu_mean_current_mb = _mb(mean_current_bytes)
        else:
            gpu_mean_current_mb = _mb(cur_end) if cur_end is not None else None

        # Get peak GPU memory usage during inference
        gpu_peak_mb = None
        if peak_begin is not None and peak_end is not None:
            gpu_peak_mb = _mb(max(peak_begin, peak_end))
        elif peak_end is not None:
            gpu_peak_mb = _mb(peak_end)

        inference_metrics = {
            "time_sec": float(elapsed),
            "gpu_mean_current_mb": gpu_mean_current_mb,
            "gpu_peak_mb": gpu_peak_mb,
        }

        # Reconstruct
        sr_img = reconstruct_from_patches(preds, positions, padded_img.shape, original_shape, patch_size)

        return sr_img, inference_metrics

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