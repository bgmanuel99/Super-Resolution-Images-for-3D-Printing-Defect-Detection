import os

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

from srlib.dataset.loading import add_padding
from srlib.metrics import psnr, ssim
from srlib.deep_learning.callbacks import EpochMemoryCallback, EpochTimeCallback

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
        
        # Final convolution to produce RGB output. The output stays linear:
        # bounding it inside the graph would zero the gradient of every
        # pixel predicted outside the range, which is most of them early in
        # training and the saturated regions later on. The range is enforced
        # at reconstruction time instead, and Adam already caps the gradient
        # norm.
        outputs = Conv2D(
            channels, (3, 3), padding="same",
            kernel_initializer="he_normal", name="output",
        )(x)

        self.model = Model(inputs, outputs, name="EDSR")

    def _compile_model(self, learning_rate, loss):
        """Compile with Adam and the requested loss, tracking PSNR and SSIM.

        The EDSR paper trains with L1, which converges better than L2, so the
        caller's choice of loss is what gets compiled.
        """

        optimizer = Adam(
            learning_rate=learning_rate, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-8, 
            clipnorm=1.0
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[psnr, ssim])
        self.model.summary()

    def fit(
            self, 
            X_train, 
            Y_train, 
            X_val, 
            Y_val, 
            batch_size=16, 
            epochs=300):
        """Train the model over the extracted patch pairs and standard callbacks."""
        
        if self.model is None:
            raise ValueError("Model is not built yet.")

        devices = tf.config.list_physical_devices("GPU")
        if devices:
            print("Training on GPU:", devices[0].name)
        else:
            print("Training on CPU")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
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

        self.trained = True
        
        return history, callbacks[2], callbacks[3]

    def evaluate(self, X_test, Y_test):
        """Evaluate the model on test data and print loss, PSNR, and SSIM."""
        
        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        results = self.model.evaluate(X_test, Y_test)
        print(f"Loss: {results[0]:.4f}, PSNR: {results[1]:.2f} dB, SSIM: {results[2]:.4f}")
        
        return results
    
    def evaluate_and_save(
            self, X_test, Y_test, history, time_cb, mem_cb, timestamp=None):
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
            Test partition produced by ``load_edsr_dataset``.
        history : keras.callbacks.History
            Returned by ``fit``.
        time_cb, mem_cb : EpochTimeCallback, EpochMemoryCallback
            Returned by ``fit`` alongside the history.
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
        run_name = f"EDSR_{timestamp}"

        with stage(f"EDSR run {timestamp}") as step:
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

            run_dir = prepare_run_directory("EDSR", run_name)
            self.save(directory=run_dir, timestamp=timestamp)
            step(f"metrics    -> {save_run_metrics(run_dir, run_name, metrics)}")

        return timestamp, run_dir, metrics

    def super_resolve_image(self, lr_img, patch_size_lr=48, stride=24):
        """Patch-based SR similar in flow to SRCNN, but accepts an in-memory LR numpy array.
        Steps: add padding, extract LR patches, batch-predict HR patches, reconstruct with
        overlap averaging, and crop to original HR size. No interpolation is used."""

        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        if self.scale_factor is None:
            raise ValueError("scale_factor is not set. Call setup_model first.")

        # --- Helpers to mirror SRCNN's structure (adapted for EDSR scaling) ---
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

        # --- Pad LR image with the shared helper so this grid matches training ---
        original_lr_shape = lr_img.shape[:2]
        lr_img_padded = add_padding(lr_img, patch_size_lr, stride)

        # --- Extract LR patches ---
        lr_patches, positions = extract_patches_from_image(lr_img_padded, patch_size_lr, stride)

        # --- Predict HR patches in batch ---
        hr_patches = self.model.predict(lr_patches, batch_size=16, verbose=0)

        # --- Reconstruct HR image and crop ---
        return reconstruct_from_patches(
            hr_patches,
            positions,
            lr_img_padded.shape,
            original_lr_shape,
            patch_size_lr=patch_size_lr,
            scale=self.scale_factor,
        )

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