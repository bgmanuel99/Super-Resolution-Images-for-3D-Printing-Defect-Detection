import os
import sys
import time

import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Add, 
    Input, 
    Conv2D, 
    Lambda, 
    Activation
)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

from SRModels.metrics import psnr, ssim
from SRModels.deep_learning_models.callbacks import EpochTimeCallback, EpochMemoryCallback

class EDSR:
    def __init__(self):
        self.model = None
        self.scale_factor = None
        self.trained = False

    def setup_model(
            self, 
            scale_factor=2, 
            channels=3, 
            num_res_blocks=16, 
            num_filters=64, 
            res_scaling=0.1, 
            learning_rate=1e-4, 
            loss="mean_absolute_error", 
            from_pretrained=False, 
            pretrained_path=None):
        """Set up the EDSR model, either by loading a pretrained model or building a new one."""
        
        self.scale_factor = scale_factor
        
        if from_pretrained:
            if pretrained_path is None or not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"Pretrained model file not found at {pretrained_path}")
            
            self.model = load_model(pretrained_path, custom_objects={"psnr": psnr, "ssim": ssim})
            self.trained = True
            print(f"Loaded pretrained model from {pretrained_path}")
        else:
            self._build_model(scale_factor, channels, num_res_blocks, num_filters, res_scaling)
            self._compile_model(learning_rate, loss)
    
    def _residual_block(self, x, num_filters, res_scaling):
        """Build a residual block without batch normalization (key feature of EDSR)."""
        
        shortcut = x
        
        # First conv layer
        x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        
        # Second conv layer
        x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        
        # Scale the residual
        if res_scaling != 1.0:
            x = Lambda(lambda t: t * res_scaling)(x)
        
        # Add shortcut connection
        x = Add()([x, shortcut])
        
        return x

    def _upsampling_block(self, x, scale_factor, num_filters):
        """Create upsampling block using sub-pixel convolution."""
        
        if scale_factor == 2:
            x = Conv2D(num_filters * 4, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        elif scale_factor == 3:
            x = Conv2D(num_filters * 9, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = Lambda(lambda x: tf.nn.depth_to_space(x, 3))(x)
        elif scale_factor == 4:
            # Two 2x upsampling blocks
            x = Conv2D(num_filters * 4, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
            x = Conv2D(num_filters * 4, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        else:
            raise ValueError(f"Scale factor {scale_factor} not supported. Use 2, 3, or 4.")
        
        return x

    def _build_model(self, scale_factor, channels, num_res_blocks, num_filters, res_scaling):
        """Construct the EDSR model architecture using functional API."""
        
        inputs = Input(shape=(None, None, channels), name="input")
        
        # Initial convolution (head)
        x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        
        # Store for global residual connection
        head_output = x
        
        # Residual blocks (body)
        for _ in range(num_res_blocks):
            x = self._residual_block(x, num_filters, res_scaling)
        
        # Final convolution of the body
        x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        
        # Global residual connection
        x = Add()([x, head_output])
        
        # Upsampling blocks (tail)
        x = self._upsampling_block(x, scale_factor, num_filters)
        
        # Final convolution to produce RGB output
        x = Conv2D(channels, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        
        outputs = Lambda(lambda t: tf.clip_by_value(t, 0.0, 1.0), name="clip_0_1")(x)
        
        self.model = Model(inputs, outputs, name="EDSR")
        
    def _charbonnier_loss(self, y_true, y_pred, eps=1e-3):
        return tf.reduce_mean(tf.sqrt(tf.square(y_pred - y_true) + eps**2))

    def _compile_model(self, learning_rate, loss):
        """Compile the model with Adam optimizer and specified loss, including PSNR and SSIM metrics."""
        
        optimizer = Adam(
            learning_rate=learning_rate, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-8, 
            clipnorm=1.0
        )
        self.model.compile(optimizer=optimizer, loss=self._charbonnier_loss, metrics=[psnr, ssim])
        self.model.summary()

    def fit(
            self, 
            X_train, 
            Y_train, 
            X_val, 
            Y_val, 
            batch_size=16, 
            epochs=300):
        """Train the model using optional image data augmentation and standard callbacks."""
        
        if self.model is None:
            raise ValueError("Model is not built yet.")

        devices = tf.config.list_physical_devices("GPU")
        if devices:
            print("Training on GPU:", devices[0].name)
        else:
            print("Training on CPU")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
            EpochTimeCallback(),
            EpochMemoryCallback(track_gpu=True, gpu_device="GPU:0"),
        ]
        
        self.model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks
        )

        self.trained = True
        return callbacks[2], callbacks[3]  # Return time and memory callbacks

    def evaluate(self, X_test, Y_test):
        """Evaluate the model on test data and print loss, PSNR, and SSIM."""
        
        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        results = self.model.evaluate(X_test, Y_test)
        print(f"Loss: {results[0]:.4f}, PSNR: {results[1]:.2f} dB, SSIM: {results[2]:.4f}")
        
        return results
    
    def super_resolve_image(self, lr_img, patch_size_lr=48, stride=24):
        """Patch-based SR similar in flow to SRCNN, but accepts an in-memory LR numpy array.
        Steps: add padding, extract LR patches, batch-predict HR patches, reconstruct with
        overlap averaging, and crop to original HR size. No interpolation is used."""

        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        if self.scale_factor is None:
            raise ValueError("scale_factor is not set. Call setup_model first.")

        # --- Helpers to mirror SRCNN's structure (adapted for EDSR scaling) ---
        def add_padding(image, patch_size, stride):
            """Reflect-pad LR image to ensure full coverage by sliding window."""
            h, w, c = image.shape

            pad_h = (patch_size - (h % stride)) % stride if h % stride != 0 else 0
            pad_w = (patch_size - (w % stride)) % stride if w % stride != 0 else 0

            pad_h = max(pad_h, patch_size - stride)
            pad_w = max(pad_w, patch_size - stride)

            padded_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            return padded_img, (h, w)

        def extract_patches_from_image(image, patch_size=48, stride=24):
            """Extract LR patches and their top-left positions."""
            h, w, _ = image.shape
            patches = []
            positions = []
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patches.append(image[i:i+patch_size, j:j+patch_size, :])
                    positions.append((i, j))
            return np.asarray(patches, dtype=np.float32), positions

        def reconstruct_from_patches(hr_patches, positions, padded_lr_shape, original_lr_shape, patch_size_lr=48, scale=2):
            """Reconstruct HR image from predicted HR patches and crop to original upscale size."""
            h_lr_pad, w_lr_pad = padded_lr_shape[:2]
            h_lr_orig, w_lr_orig = original_lr_shape

            c = 3
            patch_size_hr = patch_size_lr * scale

            hr_h_pad = h_lr_pad * scale
            hr_w_pad = w_lr_pad * scale

            reconstructed = np.zeros((hr_h_pad, hr_w_pad, c), dtype=np.float32)
            weight = np.zeros_like(reconstructed, dtype=np.float32)

            for patch, (i, j) in zip(hr_patches, positions):
                hi = i * scale
                hj = j * scale
                reconstructed[hi:hi+patch_size_hr, hj:hj+patch_size_hr, :] += patch
                weight[hi:hi+patch_size_hr, hj:hj+patch_size_hr, :] += 1.0

            reconstructed = np.divide(
                reconstructed,
                weight,
                out=np.zeros_like(reconstructed),
                where=weight != 0
            )

            out_h = h_lr_orig * scale
            out_w = w_lr_orig * scale
            reconstructed = reconstructed[:out_h, :out_w, :]
            
            return np.clip(reconstructed, 0.0, 1.0)

        # --- Pad LR image ---
        lr_img_padded, original_lr_shape = add_padding(lr_img, patch_size_lr, stride)

        # --- Extract LR patches ---
        lr_patches, positions = extract_patches_from_image(lr_img_padded, patch_size_lr, stride)

        # --- Predict HR patches in batch (measure time and GPU memory around predict) ---
        def _read_gpu_info(device="GPU:0"):
            try:
                return tf.config.experimental.get_memory_info(device)
            except Exception:
                return None

        gpu_begin = _read_gpu_info()
        t0 = time.perf_counter()

        hr_patches = self.model.predict(lr_patches, batch_size=16, verbose=0)

        elapsed = time.perf_counter() - t0
        gpu_end = _read_gpu_info()

        def _mb(x):
            return None if x is None else float(x) / (1024.0 * 1024.0)

        cur_begin = gpu_begin.get("current") if isinstance(gpu_begin, dict) else None
        cur_end = gpu_end.get("current") if isinstance(gpu_end, dict) else None
        peak_begin = gpu_begin.get("peak") if isinstance(gpu_begin, dict) else None
        peak_end = gpu_end.get("peak") if isinstance(gpu_end, dict) else None

        if cur_begin is not None and cur_end is not None:
            mean_current_bytes = (cur_begin + cur_end) / 2.0
            gpu_mean_current_mb = _mb(mean_current_bytes)
        else:
            gpu_mean_current_mb = _mb(cur_end) if cur_end is not None else None

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

        # --- Reconstruct HR image and crop ---
        sr_img = reconstruct_from_patches(
            hr_patches,
            positions,
            lr_img_padded.shape,
            original_lr_shape,
            patch_size_lr=patch_size_lr,
            scale=self.scale_factor,
        )

        return sr_img, inference_metrics

    def save(self, directory, timestamp):
        """Save the trained model with a timestamp in the specified directory."""
        
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")
        if not directory:
            raise ValueError("Directory path must be provided.")

        os.makedirs(directory, exist_ok=True)
        
        path = os.path.join(directory, f"EDSR_x{self.scale_factor}_{timestamp}.h5")
        
        self.model.save(path)
        
        print(f"Model saved to {path}")