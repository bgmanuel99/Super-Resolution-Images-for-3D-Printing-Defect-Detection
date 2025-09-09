import os
import sys
import time

import numpy as np
import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from tensorflow_addons.layers import SpectralNormalization
from keras.layers import (
    Input, 
    Conv2D, 
    Add, 
    Lambda, 
    Concatenate, 
    LeakyReLU, 
    GlobalAveragePooling2D, 
    Dense, 
    Layer
)
from keras.backend import eval, mean, square, binary_crossentropy

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

from SRModels.deep_learning_models.callbacks import EpochTimeTracker, EpochMemoryTracker

class SelfAttention(Layer):
    """
    Self-Attention Layer for 2D feature maps.
    """
    
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        
        self.channels = channels

    def build(self, input_shape):
        self.f = Conv2D(self.channels // 8, 1, padding='same', name=self.name + "_f")
        self.g = Conv2D(self.channels // 8, 1, padding='same', name=self.name + "_g")
        self.h = Conv2D(self.channels // 2, 1, padding='same', name=self.name + "_h")
        self.v = Conv2D(self.channels, 1, padding='same', name=self.name + "_v")
        
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        f = self.f(x)  # [B, H, W, C//8]
        g = self.g(x)  # [B, H, W, C//8]
        h = self.h(x)  # [B, H, W, C//2]

        shape_f = tf.shape(f)
        shape_g = tf.shape(g)
        shape_h = tf.shape(h)

        f_flat = tf.reshape(f, [shape_f[0], -1, shape_f[-1]])  # [B, HW, C//8]
        g_flat = tf.reshape(g, [shape_g[0], -1, shape_g[-1]])  # [B, HW, C//8]
        h_flat = tf.reshape(h, [shape_h[0], -1, shape_h[-1]])  # [B, HW, C//2]

        s = tf.matmul(g_flat, f_flat, transpose_b=True)  # [B, HW, HW]
        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, h_flat)  # [B, HW, C//2]
        o = tf.reshape(o, tf.shape(h))  # [B, H, W, C//2]
        o = self.v(o)  # [B, H, W, C]

        x = Add()([x, o])
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ESRGAN:
    """
    Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) implementation.
    
    This class implements the ESRGAN architecture for image super-resolution,
    including the generator (RRDBNet), discriminator (VGG-style), and training logic.
    """
    
    def __init__(self):
        """
        Initialize ESRGAN model.
        
        Args:
            num_rrdb_blocks: Number of Residual-in-Residual Dense Blocks
        """
        
        # Initialize models
        self.generator = None
        self.discriminator = None
        self.vgg_model = None
        
        # Training parameters
        self.g_optimizer = None
        self.d_optimizer = None
        
        self.trained = False
        
    def setup_model(
            self, 
            scale_factor=2, 
            growth_channels=32, 
            num_rrdb_blocks=23, 
            input_shape=(None, None, 3),
            output_shape=(None, None, 3),
            from_trained=False, 
            generator_pretrained_path=None, 
            discriminator_pretrained_path=None):
        """
        Setup the ESRGAN models either from scratch or from pretrained weights.
        
        Args:
            scale_factor: Upscaling factor (2, 4, or 8)
            growth_channels: Number of growth channels in dense blocks
            lr_size: Low resolution image size
            hr_size: High resolution image size (calculated if None)
            channels: Number of image channels
            from_trained: If True, load pretrained models
            generator_pretrained_path: Path to pretrained generator model
            discriminator_pretrained_path: Path to pretrained discriminator model
        """
        
        # Persist scale_factor for inference utilities
        self.scale_factor = scale_factor

        if from_trained:
            # Check if paths exist
            if generator_pretrained_path is None or not os.path.exists(generator_pretrained_path):
                raise FileNotFoundError(f"Generator pretrained path does not exist: {generator_pretrained_path}")
            if discriminator_pretrained_path is None or not os.path.exists(discriminator_pretrained_path):
                raise FileNotFoundError(f"Discriminator pretrained path does not exist: {discriminator_pretrained_path}")
            
            # Load pretrained models
            self.generator = load_model(generator_pretrained_path, custom_objects={
                "SelfAttention": SelfAttention,
                "SpectralNormalization": SpectralNormalization,
            })
            self.discriminator = load_model(discriminator_pretrained_path, custom_objects={
                "SpectralNormalization": SpectralNormalization,
            })
            self.vgg_model = self._build_vgg_model(output_shape)
            
            self.trained = True
            
            print(f"- Generator loaded from: {generator_pretrained_path}")
            print(f"- Discriminator loaded from: {discriminator_pretrained_path}")
            print("- VGG model built for perceptual loss")
        else:
            self.generator = self._build_generator(input_shape, scale_factor, growth_channels, num_rrdb_blocks)
            self.discriminator = self._build_discriminator(output_shape)
            self.vgg_model = self._build_vgg_model(output_shape)

            self._compile_models()
        
    def _compile_models(self):
        """
        Compile the models with optimizers.
        
        Args:
            g_lr: Generator learning rate
            d_lr: Discriminator learning rate
            
            beta_1: Beta1 parameter for Adam optimizer
            beta_2: Beta2 parameter for Adam optimizer
        """
        
        self.g_optimizer = Adam(
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True
            ), 
            beta_1=0.9, 
            beta_2=0.999
        )
        self.d_optimizer = Adam(
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-5,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True
            ), 
            beta_1=0.9, 
            beta_2=0.999
        )
        
        print("=" * 50)
        print("GENERATOR SUMMARY")
        print("=" * 50)
        self.generator.summary()
        
        print("\n" + "=" * 50)
        print("DISCRIMINATOR SUMMARY")
        print("=" * 50)
        self.discriminator.summary()
        
        print("\n" + "=" * 50)
        print("VGG FEATURE EXTRACTOR SUMMARY")
        print("=" * 50)
        self.vgg_model.summary()
        
    def _dense_block(self, x, growth_rate, name="dense_block"):
        """
        Create a dense block.
        
        Args:
            x: Input tensor
            growth_rate: Number of filters to add per layer
            name: Block name
            
        Returns:
            Output tensor
        """
        
        # Store input for skip connection
        input_tensor = x
        input_channels = x.shape[-1]
        
        # First conv layer
        x1 = Conv2D(growth_rate, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
        x1_concat = Concatenate(name=f"{name}_concat1")([x, x1])
        
        # Second conv layer
        x2 = Conv2D(growth_rate, 3, padding="same", activation="relu", name=f"{name}_conv2")(x1_concat)
        x2_concat = Concatenate(name=f"{name}_concat2")([x, x1, x2])
        
        # Third conv layer
        x3 = Conv2D(growth_rate, 3, padding="same", activation="relu", name=f"{name}_conv3")(x2_concat)
        x3_concat = Concatenate(name=f"{name}_concat3")([x, x1, x2, x3])
        
        # Fourth conv layer
        x4 = Conv2D(growth_rate, 3, padding="same", activation="relu", name=f"{name}_conv4")(x3_concat)
        x4_concat = Concatenate(name=f"{name}_concat4")([x, x1, x2, x3, x4])
        
        # Fifth conv layer (output layer)
        x5 = Conv2D(input_channels, 3, padding="same", name=f"{name}_conv5")(x4_concat)
        
        # Residual scaling
        x5 = Lambda(lambda t: t * 0.2, name=f"{name}_scale")(x5)
        
        # Skip connection
        output = Add(name=f"{name}_add")([input_tensor, x5])
        
        return output
    
    def _rrdb_block(self, x, growth_channels, name="rddb"):
        """
        Create a Residual-in-Residual Dense Block (RRDB).
        
        Args:
            x: Input tensor
            growth_channels: Number of growth channels
            name: Block name
            
        Returns:
            Output tensor
        """
        
        input_tensor = x
        
        # Three dense blocks
        x = self._dense_block(x, growth_channels, f"{name}_dense1")
        x = self._dense_block(x, growth_channels, f"{name}_dense2")
        x = self._dense_block(x, growth_channels, f"{name}_dense3")
        
        # Residual scaling
        x = Lambda(lambda t: t * 0.2, name=f"{name}_scale")(x)
        
        # Skip connection
        output = Add(name=f"{name}_add")([input_tensor, x])
        
        return output
    
    def _upsample_block(self, x, filters, name="upsample"):
        """
        Create an upsampling block using sub-pixel convolution.
        
        Args:
            x: Input tensor
            filters: Number of filters
            name: Block name
            
        Returns:
            Upsampled tensor
        """
        
        x = Conv2D(filters * 4, 3, padding="same", name=f"{name}_conv")(x)
        x = Lambda(lambda t: tf.nn.depth_to_space(t, 2), name=f"{name}_pixelshuffle")(x)
        x = LeakyReLU(alpha=0.2, name=f"{name}_leaky")(x)
        
        return x
    
    def _build_generator(self, input_shape, scale_factor, growth_channels, num_rrdb_blocks):
        """
        Build the generator network (RRDBNet).
        
        Returns:
            Generator model
        """
        
        inputs = Input(shape=input_shape, name="lr_input")
        
        # Initial convolution
        x = Conv2D(64, 3, padding="same", name="initial_conv")(inputs)
        trunk_output = x
        
        # RRDB blocks
        for i in range(num_rrdb_blocks):
            x = self._rrdb_block(x, growth_channels, f"rrdb_{i}")
        
        # Trunk convolution
        x = Conv2D(64, 3, padding="same", name="trunk_conv")(x)
        
        # Trunk connection
        x = Add(name="trunk_add")([trunk_output, x])
        
        # Self-Attention after RRDB trunk
        x = SelfAttention(64, name="self_attention_trunk")(x)
        
        # Upsampling blocks
        num_upsample = int(np.log2(scale_factor))
        for i in range(num_upsample):
            x = self._upsample_block(x, 64, f"upsample_{i}")
            
            # Self-Attention after first upsampling
            if i == 0:
                x = SelfAttention(64, name=f"self_attention_upsample_{i}")(x)
        
        # Final convolution layers
        x = Conv2D(64, 3, padding="same", activation="relu", name="final_conv1")(x)
        outputs = Conv2D(inputs.shape[-1], 3, padding="same", activation="tanh", name="final_conv2")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="Generator")
        
        return model
    
    def _build_discriminator(self, output_shape):
        """
        Build the discriminator network (VGG-style).
        
        Returns:
            Discriminator model
        """
        
        inputs = Input(shape=output_shape, name="hr_input")
        
        # Initial convolution
        x = SpectralNormalization(Conv2D(64, 3, padding="same", name="disc_conv1"))(inputs)
        x = LeakyReLU(alpha=0.2, name="disc_leaky1")(x)
        
        # Convolutional blocks (reduced depth and channels to lower parameter count)
        filters = [64, 64, 128, 128, 256]
        strides = [2, 1, 2, 1, 2]

        for i, (f, s) in enumerate(zip(filters, strides)):
            x = SpectralNormalization(Conv2D(f, 3, strides=s, padding="same", name=f"disc_conv{i+2}"))(x)
            x = LeakyReLU(alpha=0.2, name=f"disc_leaky{i+2}")(x)
        
        # Global average pooling and dense layers
        x = GlobalAveragePooling2D(name="disc_gap")(x)
        x = SpectralNormalization(Dense(256, name="disc_dense1"))(x)
        x = LeakyReLU(alpha=0.2, name="disc_leaky_dense1")(x)
        outputs = SpectralNormalization(Dense(1, activation="sigmoid", name="disc_output"))(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="Discriminator")
        
        return model
    
    def _build_vgg_model(self, output_shape):
        """
        Build VGG model for perceptual loss.
        
        Returns:
            VGG model for feature extraction
        """
        
        vgg = VGG19(
            include_top=False, 
            weights="imagenet", 
            input_shape=output_shape
        )
        
        # Extract features from conv5_4 layer
        vgg.trainable = False
        outputs = vgg.get_layer("block5_conv4").output
        
        model = Model(inputs=vgg.input, outputs=outputs, name="VGG_Feature_Extractor")
        
        return model
    
    def _preprocess_vgg_input(self, x):
        """Preprocess input for VGG model."""
        
        # Convert from [-1, 1] to [0, 255]
        x = (x + 1) * 127.5
        
        # Apply VGG preprocessing
        return preprocess_input(x)
    
    def _perceptual_loss(self, hr_real, hr_fake):
        """
        Calculate perceptual loss using VGG features.
        
        Args:
            hr_real: Real high-resolution images
            hr_fake: Generated high-resolution images
            
        Returns:
            Perceptual loss
        """
        
        # Preprocess inputs for VGG
        hr_real_vgg = self._preprocess_vgg_input(hr_real)
        hr_fake_vgg = self._preprocess_vgg_input(hr_fake)
        
        # Extract features
        real_features = self.vgg_model(hr_real_vgg)
        fake_features = self.vgg_model(hr_fake_vgg)
        
        # Calculate MSE loss
        return mean(square(real_features - fake_features))
    
    def _pixel_loss(self, hr_real, hr_fake):
        """
        Calculate pixel-wise L1 loss.
        
        Args:
            hr_real: Real high-resolution images
            hr_fake: Generated high-resolution images
            
        Returns:
            Pixel loss
        """
        
        return mean(abs(hr_real - hr_fake))
    
    def _adversarial_loss(self, y_true, y_pred):
        """
        Calculate adversarial loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Adversarial loss
        """
        
        return mean(binary_crossentropy(y_true, y_pred))
    
    def _spectral_loss(self, hr_real, hr_fake):
        """Spectral (Fourier) loss for texture preservation."""
        
        # Compute FFT2 for each image in the batch
        hr_real_fft = tf.signal.fft2d(tf.cast(hr_real, tf.complex64))
        hr_fake_fft = tf.signal.fft2d(tf.cast(hr_fake, tf.complex64))
        
        # Use magnitude (abs) for comparison
        real_mag = tf.abs(hr_real_fft)
        fake_mag = tf.abs(hr_fake_fft)
        
        # L1 loss between magnitude spectra
        return tf.reduce_mean(tf.abs(real_mag - fake_mag))
    
    def _train_step(self, lr_images, hr_images):
        """
        Perform one training step.
        
        Args:
            lr_images: Low-resolution images
            hr_images: High-resolution images
            
        Returns:
            Dictionary containing losses
        """
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            # Generate fake images
            hr_fake = self.generator(lr_images, training=True)
            
            # Discriminator predictions
            d_real = self.discriminator(hr_images, training=True)
            d_fake = self.discriminator(hr_fake, training=True)
            
            # Discriminator losses
            d_loss_real = self._adversarial_loss(tf.ones_like(d_real), d_real)
            d_loss_fake = self._adversarial_loss(tf.zeros_like(d_fake), d_fake)
            d_loss = d_loss_real + d_loss_fake
        
        # Update discriminator
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as g_tape:
            # Generate fake images
            hr_fake = self.generator(lr_images, training=True)
            
            # Discriminator prediction for fake images
            d_fake = self.discriminator(hr_fake, training=True)
            
            # Generator losses
            g_adversarial_loss = self._adversarial_loss(tf.ones_like(d_fake), d_fake)
            g_perceptual_loss = self._perceptual_loss(hr_images, hr_fake)
            g_pixel_loss = self._pixel_loss(hr_images, hr_fake)
            g_spectral_loss = self._spectral_loss(hr_images, hr_fake)
            
            # Combined generator loss
            g_loss = (
                g_adversarial_loss 
                + 1.0 * g_perceptual_loss 
                + 100.0 * g_pixel_loss 
                + 1.0 * g_spectral_loss)
        
        # Update generator
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        return {
            "g_loss": g_loss, 
            "d_loss": d_loss, 
        }
    
    def fit(
        self,
        X_train=None,
        Y_train=None,
        train_dataset=None,
        X_val=None,
        Y_val=None,
        val_dataset=None,
        epochs=100,
        batch_size=16,
        steps_per_epoch=None,
        val_steps=None,
        normalize=True,
        save_dir=None):
        """
        Train the ESRGAN model and optionally save a 5x5 SR preview grid each epoch.

        Formas de entrada:
        - Proporcionar (X_train, Y_train) y opcionalmente (X_val, Y_val)
        - O proporcionar directamente un train_dataset (tf.data.Dataset) ya preparado

        Parámetros:
        X_train, Y_train: ndarrays en rango [0,1]
        train_dataset: tf.data.Dataset que produce (lr, hr) en [0,1] o [-1,1]
        steps_per_epoch: obligatorio si la fuente es infinita (repeat)
        normalize: si True convierte batches de [0,1] a [-1,1]
        save_dir: si se proporciona, guarda al final de cada época una cuadrícula 5x5
          con salidas del generador para monitorizar el progreso
        """
        # Validaciones básicas
        if train_dataset is None and (X_train is None or Y_train is None):
            raise ValueError("Debe aportar (X_train,Y_train) o un train_dataset")

        # Info dispositivo
        devices = tf.config.list_physical_devices('GPU')
        if devices:
            print("Training on GPU:", [d.name for d in devices])
        else:
            print("Training on CPU")

        # Construcción del dataset de entrenamiento
        if train_dataset is None:
            # Dataset desde arrays
            train_dataset = (
                tf.data.Dataset
                .from_tensor_slices((X_train, Y_train))
                .shuffle(len(X_train))
                .batch(batch_size)
                .repeat()
            )
            if steps_per_epoch is None:
                steps_per_epoch = int(np.ceil(len(X_train)/batch_size))
        else:
            # Dataset externo: debe proveer batching o lo añadimos si no.
            # Para mantener consistencia, forzamos repeat() aquí.
            # No alteramos estructura si ya está batched (supuesto del usuario).
            train_dataset = train_dataset.repeat()
            if steps_per_epoch is None:
                raise ValueError("Debe indicar steps_per_epoch cuando aporta un dataset externo")

        # Normalización a [-1,1] si procede
        if normalize:
            train_dataset = train_dataset.map(lambda x,y: (x*2.0 - 1.0, y*2.0 - 1.0), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Dataset de validación
        val_data_struct = None
        if val_dataset is not None:
            val_data_struct = val_dataset
        elif X_val is not None and Y_val is not None:
            val_data_struct = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size)
            if val_steps is None:
                val_steps = int(np.ceil(len(X_val)/batch_size))
        
        if val_data_struct is not None and normalize:
            val_data_struct = val_data_struct.map(lambda x,y: (x*2.0 - 1.0, y*2.0 - 1.0), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        # Preparación de guardado de previews
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Cache para un batch fijo de previsualización (consistente entre épocas)
        preview_lr_cache = None
        preview_cache_is_normalized = False  # True si proviene del dataset ya en [-1,1]

        def _prepare_preview_batch():
            nonlocal preview_lr_cache, preview_cache_is_normalized
            if preview_lr_cache is not None:
                return preview_lr_cache, preview_cache_is_normalized

            n_max = 25
            batch = None
            is_norm = False
            
            if X_val is not None and len(X_val) > 0:
                batch = X_val[:min(n_max, len(X_val))]
                is_norm = False
            elif X_train is not None and len(X_train) > 0:
                batch = X_train[:min(n_max, len(X_train))]
                is_norm = False
            else:
                src = val_data_struct if val_data_struct is not None else train_dataset
                for lr_b, _ in src.take(1):
                    lr_np = lr_b.numpy()
                    batch = lr_np[:min(n_max, lr_np.shape[0])]
                    is_norm = True if normalize else False
                if batch is None:
                    raise RuntimeError("No se pudo obtener un batch de previsualización para guardar imágenes.")

            preview_lr_cache = batch.astype(np.float32)
            preview_cache_is_normalized = is_norm
            return preview_lr_cache, preview_cache_is_normalized

        def _to_uint8(img):
            img = np.clip(img, 0.0, 1.0)
            return (img * 255.0).round().astype(np.uint8)

        def _save_sr_grid(epoch_idx):
            if save_dir is None:
                return
            
            lr_preview, is_norm = _prepare_preview_batch()
            
            lr_in = lr_preview if is_norm else (lr_preview * 2.0 - 1.0)
            
            sr = self.generator(lr_in, training=False).numpy()
            
            sr = (sr + 1.0) / 2.0

            # Construir grid 5x5
            n = min(25, sr.shape[0])
            rows, cols = 5, 5
            h, w, ch = sr.shape[1], sr.shape[2], sr.shape[3]
            grid = np.zeros((rows * h, cols * w, ch), dtype=np.uint8)
            for idx in range(n):
                r = idx // cols
                cidx = idx % cols
                tile = _to_uint8(sr[idx])
                grid[r*h:(r+1)*h, cidx*w:(cidx+1)*w] = tile

            # Guardar PNG
            png = tf.image.encode_png(grid)
            out_path = os.path.join(save_dir, f"epoch_{epoch_idx:03d}_sr_grid.png")
            tf.io.write_file(out_path, png)

        # Trackers
        time_tracker = EpochTimeTracker()
        memory_tracker = EpochMemoryTracker(track_gpu=True, gpu_device="GPU:0")

        # Bucle de entrenamiento
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            if time_tracker is not None:
                time_tracker.begin_epoch()
            if memory_tracker is not None:
                memory_tracker.begin_epoch()
            
            epoch_losses = {
                "g_loss": [],
                "val_g_loss": [],
                "d_loss": [],
                "psnr": [],
                "val_psnr": [],
                "ssim": [], 
                "val_ssim": [],
                "g_lr": [],
                "d_lr": []
            }

            # Iterate over training batches
            for step, (lr_batch, hr_batch) in enumerate(train_dataset.take(steps_per_epoch)):
                losses = self._train_step(lr_batch, hr_batch)
                for key, value in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(float(value.numpy()))

                # Perceptual metrics every 10 steps
                hr_fake = self.generator(lr_batch, training=False)
                hr_real_eval = (hr_batch + 1.0) / 2.0
                hr_gen_eval = (hr_fake + 1.0) / 2.0
                psnr_score = tf.reduce_mean(tf.image.psnr(hr_real_eval, hr_gen_eval, max_val=1.0))
                ssim_score = tf.reduce_mean(tf.image.ssim(hr_real_eval, hr_gen_eval, max_val=1.0))
                epoch_losses["psnr"].append(float(psnr_score.numpy()))
                epoch_losses["ssim"].append(float(ssim_score.numpy()))
                epoch_losses["g_lr"].append(float(self.g_optimizer._decayed_lr(tf.float32).numpy()))
                epoch_losses["d_lr"].append(float(self.d_optimizer._decayed_lr(tf.float32).numpy()))

                if (step+1) % 10 == 0 or (step+1) == steps_per_epoch:
                    print(
                        f"  Step {step+1}/{steps_per_epoch} G_loss={epoch_losses['g_loss'][-1]:.4f} "
                        f"D_loss={epoch_losses['d_loss'][-1]:.4f} PSNR={epoch_losses['psnr'][-1]:.2f} "
                        f"SSIM={epoch_losses['ssim'][-1]:.4f}")

            # Resumen epoch
            avg_losses = {k: np.mean(v) for k,v in epoch_losses.items()}
            print(
                f"- Epoch Summary - G_loss: {avg_losses['g_loss']:.4f}, D_loss: {avg_losses['d_loss']:.4f}, "
                f"PSNR: {avg_losses['psnr']:.2f}, SSIM: {avg_losses['ssim']:.4f}")

            # Validación si existe
            if val_data_struct is not None:
                val_psnr, val_ssim, val_g_losses = [], [], []
                for i, (lr_v, hr_v) in enumerate(val_data_struct.take(val_steps)):
                    # Forward pass
                    hr_fake_v = self.generator(lr_v, training=False)

                    # Compute generator validation loss (same formula as training, no grads)
                    d_fake_v = self.discriminator(hr_fake_v, training=False)
                    g_adv_v = self._adversarial_loss(tf.ones_like(d_fake_v), d_fake_v)
                    g_perc_v = self._perceptual_loss(hr_v, hr_fake_v)
                    g_pix_v  = self._pixel_loss(hr_v, hr_fake_v)
                    g_spec_v = self._spectral_loss(hr_v, hr_fake_v)
                    g_loss_v = g_adv_v + 1.0 * g_perc_v + 100.0 * g_pix_v + 1.0 * g_spec_v
                    val_g_losses.append(float(g_loss_v.numpy()))

                    # PSNR / SSIM in [0,1]
                    hr_real_eval = (hr_v + 1.0) / 2.0
                    hr_gen_eval  = (hr_fake_v + 1.0) / 2.0
                    val_psnr.append(float(tf.reduce_mean(tf.image.psnr(hr_real_eval, hr_gen_eval, 1.0)).numpy()))
                    val_ssim.append(float(tf.reduce_mean(tf.image.ssim(hr_real_eval, hr_gen_eval, 1.0)).numpy()))

                val_psnr_mean = float(np.mean(val_psnr)) if len(val_psnr) > 0 else float('nan')
                val_ssim_mean = float(np.mean(val_ssim)) if len(val_ssim) > 0 else float('nan')
                val_g_loss_mean = float(np.mean(val_g_losses)) if len(val_g_losses) > 0 else float('nan')

                # Registrar en epoch_losses
                epoch_losses["val_psnr"] = val_psnr_mean
                epoch_losses["val_ssim"] = val_ssim_mean
                epoch_losses["val_g_loss"] = val_g_loss_mean

                print(
                    f"  Validation -> PSNR: {val_psnr_mean:.2f}, SSIM: {val_ssim_mean:.4f}, G_loss: {val_g_loss_mean:.4f}")

            # Guardar previsualización (grid 5x5) al final de cada época
            _save_sr_grid(epoch + 1)

            self.trained = True

            # End-of-epoch tracking
            if memory_tracker is not None:
                memory_tracker.end_epoch()
            if time_tracker is not None:
                time_tracker.end_epoch()

        return epoch_losses, time_tracker, memory_tracker
    
    def evaluate(self, test_dataset):
        """
        Evaluate the trained model with a test dataset.

        Computes avg_g_loss (using the same generator loss formula as training)
        and reports PSNR/SSIM averages. Pixel and perceptual losses are not
        returned as separate metrics in evaluation.

        Args:
            test_dataset: Test dataset (tf.data.Dataset)

        Returns:
            dict: {"avg_psnr", "avg_ssim", "avg_g_loss"}
        """

        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        print("Evaluating model on test dataset...")

        # Initialize metrics
        total_psnr = 0.0
        total_ssim = 0.0
        total_g_loss = 0.0
        num_batches = 0

        for lr_batch, hr_batch in test_dataset:
            # Generate high-resolution images
            hr_generated = self.generator(lr_batch, training=False)

            # Compute generator loss components (match training formula)
            d_fake = self.discriminator(hr_generated, training=False)
            g_adversarial_loss = self._adversarial_loss(tf.ones_like(d_fake), d_fake)
            g_perceptual_loss = self._perceptual_loss(hr_batch, hr_generated)
            g_pixel_loss = self._pixel_loss(hr_batch, hr_generated)
            g_spectral_loss = self._spectral_loss(hr_batch, hr_generated)

            g_loss = (
                g_adversarial_loss
                + 1.0 * g_perceptual_loss
                + 100.0 * g_pixel_loss
                + 1.0 * g_spectral_loss
            )
            total_g_loss += eval(g_loss)

            # Convert to [0, 1] range for PSNR and SSIM
            hr_real_eval = (hr_batch + 1.0) / 2.0
            hr_gen_eval = (hr_generated + 1.0) / 2.0

            # Calculate PSNR and SSIM
            psnr_score = tf.image.psnr(hr_real_eval, hr_gen_eval, max_val=1.0)
            ssim_score = tf.image.ssim(hr_real_eval, hr_gen_eval, max_val=1.0)

            total_psnr += eval(mean(psnr_score))
            total_ssim += eval(mean(ssim_score))

            num_batches += 1

        # Calculate averages
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        avg_g_loss = total_g_loss / num_batches

        metrics = {
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_g_loss": avg_g_loss,
        }

        print("Evaluation Results:")
        print(f"  Average PSNR: {avg_psnr:.4f}")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        print(f"  Average G Loss: {avg_g_loss:.4f}")

        return metrics
    
    def super_resolve_image(self, lr_img, patch_size_lr=48, stride=24, batch_size=16):
        """Patch-wise super-resolution using the ESRGAN generator.
        Follows SRCNN/EDSR flow: reflect padding, patch extraction, batch predict, overlap-averaged reconstruction.
        Accounts for ESRGAN's [-1,1] tanh output by normalizing inputs to [-1,1] and denormalizing outputs to [0,1].

        Args:
            lr_img_path: Path to LR image file.
            patch_size_lr: LR patch size used for sliding window.
            stride: Stride for LR patch extraction.
            batch_size: Batch size for generator prediction.

        Returns:
            np.ndarray float32 RGB image in [0,1] with shape (H*scale, W*scale, 3).
        """

        if not self.trained:
            raise RuntimeError("Model has not been trained or loaded.")
        if self.generator is None:
            raise RuntimeError("Generator is not initialized.")
        if not hasattr(self, 'scale_factor') or self.scale_factor is None:
            raise ValueError("scale_factor is not set. Ensure setup_model was called.")

        scale = self.scale_factor

        # --- Helpers mirroring EDSR/SRCNN structure ---
        def add_padding(image, patch_size, stride):
            h, w, c = image.shape
            pad_h = (patch_size - (h % stride)) % stride if h % stride != 0 else 0
            pad_w = (patch_size - (w % stride)) % stride if w % stride != 0 else 0
            pad_h = max(pad_h, patch_size - stride)
            pad_w = max(pad_w, patch_size - stride)
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            return padded, (h, w)

        def extract_lr_patches(img, patch_size, stride):
            h, w, _ = img.shape
            patches, positions = [], []
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patches.append(img[i:i+patch_size, j:j+patch_size, :])
                    positions.append((i, j))
            if len(patches) == 0:
                return np.empty((0, patch_size, patch_size, 3), dtype=np.float32), positions
            return np.asarray(patches, dtype=np.float32), positions

        def reconstruct_from_hr_patches(hr_patches, positions, padded_lr_shape, original_lr_shape, patch_size_lr, scale):
            h_lr_pad, w_lr_pad = padded_lr_shape[:2]
            h_lr_orig, w_lr_orig = original_lr_shape
            c = 3
            patch_size_hr = patch_size_lr * scale
            H_hr_pad = h_lr_pad * scale
            W_hr_pad = w_lr_pad * scale

            recon = np.zeros((H_hr_pad, W_hr_pad, c), dtype=np.float32)
            weight = np.zeros_like(recon, dtype=np.float32)

            for patch, (i, j) in zip(hr_patches, positions):
                hi, hj = i * scale, j * scale
                recon[hi:hi+patch_size_hr, hj:hj+patch_size_hr] += patch
                weight[hi:hi+patch_size_hr, hj:hj+patch_size_hr] += 1.0

            recon = np.divide(recon, weight, out=np.zeros_like(recon), where=weight != 0)
            out_h, out_w = h_lr_orig * scale, w_lr_orig * scale
            return np.clip(recon[:out_h, :out_w, :], 0.0, 1.0)

        # Pad LR image
        lr_padded, lr_orig_shape = add_padding(lr_img, patch_size_lr, stride)

        # Extract LR patches and normalize to [-1,1]
        lr_patches, positions = extract_lr_patches(lr_padded, patch_size_lr, stride)

        lr_patches_norm = (lr_patches * 2.0) - 1.0

        # --- Predict HR patches in batch (measure time and GPU memory around predict) ---
        def _read_gpu_info(device="GPU:0"):
            try:
                return tf.config.experimental.get_memory_info(device)
            except Exception:
                return None

        gpu_begin = _read_gpu_info()
        t0 = time.perf_counter()

        hr_patches = self.generator.predict(lr_patches_norm, batch_size=batch_size, verbose=0)

        elapsed = time.perf_counter() - t0
        gpu_end = _read_gpu_info()

        hr_patches = (hr_patches + 1.0) / 2.0  # to [0,1]

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

        # Reconstruct HR image and crop to target size
        sr_img = reconstruct_from_hr_patches(
            hr_patches, positions, lr_padded.shape, lr_orig_shape, patch_size_lr, scale
        )

        return sr_img, inference_metrics
    
    def save(self, directory, timestamp):
        """Save the trained model with a timestamp in the specified directory."""
        
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")
        
        os.makedirs(directory, exist_ok=True)
        
        generator_path = os.path.join(directory, f"ESRGAN_generator_x{self.scale_factor}_{timestamp}.h5")
        discriminator_path = os.path.join(directory, f"ESRGAN_discriminator_x{self.scale_factor}_{timestamp}.h5")
        
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
        
        print(f"Generator model saved to {generator_path}")
        print(f"Discriminator model saved to {discriminator_path}")