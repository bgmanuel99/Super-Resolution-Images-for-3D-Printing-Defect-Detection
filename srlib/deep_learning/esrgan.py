import datetime
import os

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
    MaxPooling2D, 
    Dense, 
    Layer
)
from keras.backend import eval, mean, square, binary_crossentropy

from srlib.progress import format_duration, stage
from srlib.constants import (
    ESRGAN_GROWTH_CHANNELS,
    ESRGAN_LOSS_WEIGHTS,
    ESRGAN_PATCH_SIZE,
    ESRGAN_RRDB_BLOCKS,
    ESRGAN_STRIDE,
    ESRGAN_SCALE_FACTOR,
    ESRGAN_DISCRIMINATOR_LR,
    ESRGAN_GENERATOR_LR,
    ESRGAN_LR_DECAY_HALVINGS,
    ESRGAN_LR_DECAY_RATE,
    ESRGAN_PREVIEW_EVERY,
    ESRGAN_PREVIEW_SUBDIR,
    TIMESTAMP_FORMAT,
)
from srlib.dataset.loading import add_padding
from srlib.model_registry import (
    collect_staged_previews,
    prepare_run_directory,
    save_epoch_log,
    save_model_summary,
    save_run_metrics,
    stage_preview_directory,
)
from srlib.deep_learning.callbacks import (
    EpochMemoryTracker,
    EpochTimeTracker,
    last_epoch_metrics,
    profile_evaluation,
)

class SelfAttention(Layer):
    """
    Self-Attention Layer for 2D feature maps.

    The attention map is quadratic in the number of spatial positions, so
    the key and value branches are pooled by two before it is formed, as
    the SAGAN formulation does. Without that pooling the map is
    ``[B, HW, HW]``: at the upsampled resolution of the generator that is
    a gigabyte per batch, held twice over by the gradient tape.
    """
    
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        
        self.channels = channels

    def build(self, input_shape):
        self.f = Conv2D(self.channels // 8, 1, padding='same', name=self.name + "_f")
        self.g = Conv2D(self.channels // 8, 1, padding='same', name=self.name + "_g")
        self.h = Conv2D(self.channels // 2, 1, padding='same', name=self.name + "_h")
        self.v = Conv2D(self.channels, 1, padding='same', name=self.name + "_v")
        self.pool = MaxPooling2D(
            pool_size=2, strides=2, padding='same', name=self.name + "_pool"
        )
        
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        f = self.f(x)  # key   [B, H, W, C//8]
        g = self.g(x)  # query [B, H, W, C//8]
        h = self.h(x)  # value [B, H, W, C//2]

        # Only the key and the value are pooled. The query stays at full
        # resolution, so every output position is still attended to; what
        # shrinks is the set of positions it attends over.
        f = self.pool(f)  # [B, H/2, W/2, C//8]
        h = self.pool(h)  # [B, H/2, W/2, C//2]

        shape_x = tf.shape(x)
        batch = shape_x[0]

        f_flat = tf.reshape(f, [batch, -1, self.channels // 8])  # [B, HW/4, C//8]
        g_flat = tf.reshape(g, [batch, -1, self.channels // 8])  # [B, HW,   C//8]
        h_flat = tf.reshape(h, [batch, -1, self.channels // 2])  # [B, HW/4, C//2]

        s = tf.matmul(g_flat, f_flat, transpose_b=True)  # [B, HW, HW/4]
        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, h_flat)  # [B, HW, C//2]

        # Restored against the input, not against the pooled value, whose
        # spatial size is now half of it.
        o = tf.reshape(
            o, [batch, shape_x[1], shape_x[2], self.channels // 2]
        )
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

        The networks are left unbuilt; setup_model creates or loads them.
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
            scale_factor=ESRGAN_SCALE_FACTOR, 
            growth_channels=ESRGAN_GROWTH_CHANNELS, 
            num_rrdb_blocks=ESRGAN_RRDB_BLOCKS, 
            input_shape=(None, None, 3),
            output_shape=(None, None, 3),
            from_trained=False, 
            generator_pretrained_path=None, 
            discriminator_pretrained_path=None):
        """
        Setup the ESRGAN models either from scratch or from pretrained weights.
        
        Args:
            scale_factor: Upscaling factor (2, 4, or 8)
            growth_channels: Growth channels of the dense blocks. Defaults to
                the value declared in constants, which keeps the generator at
                the capacity of EDSR.
            num_rrdb_blocks: Number of Residual-in-Residual Dense Blocks,
                declared alongside growth_channels for the same reason.
            input_shape: Shape of the low-resolution input
            output_shape: Shape of the high-resolution output
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
        Compile the generator and the discriminator with their optimizers.

        Both start at a constant Adam rate, the discriminator an order of
        magnitude lower than the generator so that it does not overpower it
        early in training. The decay is installed later by
        ``_schedule_learning_rates``, which is the first point at which the
        length of the run is known.
        """
        
        self.g_optimizer = Adam(
            learning_rate=ESRGAN_GENERATOR_LR, beta_1=0.9, beta_2=0.999
        )
        self.d_optimizer = Adam(
            learning_rate=ESRGAN_DISCRIMINATOR_LR, beta_1=0.9, beta_2=0.999
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
        """
        Spectral (Fourier) L1 loss for texture preservation.

        ``tf.signal.fft2d`` transforms the two innermost dimensions, so the
        NHWC tensors are transposed to NCHW for the transform to run over
        (height, width) instead of (width, channel).

        The magnitude is divided by ``sqrt(H * W)``, the unitary convention,
        because ``fft2d`` is unnormalised and its output would otherwise
        grow with the patch size and dominate the other loss terms.

        Returns:
            Scalar L1 distance between the two magnitude spectra.
        """

        real_nchw = tf.transpose(hr_real, [0, 3, 1, 2])
        fake_nchw = tf.transpose(hr_fake, [0, 3, 1, 2])

        real_mag = tf.abs(tf.signal.fft2d(tf.cast(real_nchw, tf.complex64)))
        fake_mag = tf.abs(tf.signal.fft2d(tf.cast(fake_nchw, tf.complex64)))

        shape = tf.shape(hr_real)
        norm = tf.sqrt(tf.cast(shape[1] * shape[2], tf.float32))

        return tf.reduce_mean(tf.abs(real_mag - fake_mag)) / norm

    def _generator_loss(self, hr_real, hr_fake, d_fake):
        """
        Combine the four generator terms with their configured weights.

        Shared by training, validation and evaluation so that the three
        report the same quantity.

        Args:
            hr_real: Reference high-resolution images
            hr_fake: Generated high-resolution images
            d_fake: Discriminator output for the generated images

        Returns:
            tuple: (total_loss, components) with components keyed as in
                ESRGAN_LOSS_WEIGHTS.
        """

        components = {
            "adversarial": self._adversarial_loss(tf.ones_like(d_fake), d_fake),
            "perceptual": self._perceptual_loss(hr_real, hr_fake),
            "pixel": self._pixel_loss(hr_real, hr_fake),
            "spectral": self._spectral_loss(hr_real, hr_fake),
        }
        total = tf.add_n([
            ESRGAN_LOSS_WEIGHTS[name] * value
            for name, value in components.items()
        ])

        return total, components

    
    def _schedule_learning_rates(self, epochs, steps_per_epoch):
        """
        Install the decay schedule now that the length of the run is known.

        The interval between two halvings is derived from the total number
        of optimiser steps, so both rates fall by the same factor over any
        run regardless of how many epochs or patches it covers. A fixed
        interval cannot do that, and getting it wrong in the frozen
        direction is silent: the losses simply stop moving.

        Args:
            epochs: Number of epochs the run will cover
            steps_per_epoch: Optimiser steps per epoch
        """

        total_steps = int(epochs) * int(steps_per_epoch)
        decay_steps = max(
            1, total_steps // max(1, int(ESRGAN_LR_DECAY_HALVINGS))
        )

        for optimizer, initial in (
                (self.g_optimizer, ESRGAN_GENERATOR_LR),
                (self.d_optimizer, ESRGAN_DISCRIMINATOR_LR)):
            if optimizer is None:
                continue
            optimizer.learning_rate = (
                tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial,
                    decay_steps=decay_steps,
                    decay_rate=ESRGAN_LR_DECAY_RATE,
                    staircase=True,
                )
            )

        print(
            f"- LR schedule: x{ESRGAN_LR_DECAY_RATE} every {decay_steps} "
            f"steps, {ESRGAN_LR_DECAY_HALVINGS} times over {total_steps} "
            f"steps"
        )

    @tf.function(reduce_retracing=True)
    def _train_step(self, lr_images, hr_images):
        """
        Perform one training step over both networks.

        The two updates are taken from a single forward pass held by one
        persistent tape. Alternating them needs the generator evaluated
        once per tape, and the generator is the expensive half of the pair;
        the gradient of each loss is still read against its own variables
        only, so neither update leaks into the other network.

        PSNR and SSIM are returned from the same ``hr_fake``. The generator
        holds no layer whose behaviour depends on the training flag, so a
        second pass with ``training=False`` would return an identical
        tensor at the cost of a third traversal of the network.

        Args:
            lr_images: Low-resolution images in [-1, 1]
            hr_images: High-resolution images in [-1, 1]

        Returns:
            Dictionary with both losses and both perceptual metrics.
        """

        with tf.GradientTape(persistent=True) as tape:
            hr_fake = self.generator(lr_images, training=True)

            d_real = self.discriminator(hr_images, training=True)
            d_fake = self.discriminator(hr_fake, training=True)

            d_loss = (
                self._adversarial_loss(tf.ones_like(d_real), d_real)
                + self._adversarial_loss(tf.zeros_like(d_fake), d_fake)
            )
            g_loss, _ = self._generator_loss(hr_images, hr_fake, d_fake)

        d_grads = tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        del tape

        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        hr_real_eval = (hr_images + 1.0) / 2.0
        hr_gen_eval = (hr_fake + 1.0) / 2.0

        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "psnr": tf.reduce_mean(
                tf.image.psnr(hr_real_eval, hr_gen_eval, max_val=1.0)
            ),
            "ssim": tf.reduce_mean(
                tf.image.ssim(hr_real_eval, hr_gen_eval, max_val=1.0)
            ),
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
        save_dir=None,
        preview_every=ESRGAN_PREVIEW_EVERY):
        """
        Train the ESRGAN model, saving 5x5 SR preview grids as it goes.

        Input forms:
        - Provide (X_train, Y_train) and optionally (X_val, Y_val)
        - Or provide an already prepared train_dataset (tf.data.Dataset)

        Parameters:
        X_train, Y_train: ndarrays in the [0,1] range
        train_dataset: tf.data.Dataset yielding (lr, hr) in [0,1] or [-1,1]
        steps_per_epoch: required when the source is infinite (repeat)
        normalize: when True, converts batches from [0,1] to [-1,1]
        save_dir: where the preview grids are written. Defaults to a staging
          directory of this training session, which 'evaluate_and_save'
          moves into the run directory. Pass an empty string to disable the
          previews entirely.
        preview_every: epochs between two grids. The first epoch is always
          rendered, so the run starts from a visible baseline.

        Returns:
        (history, time_tracker, memory_tracker), where history maps each
        metric to one value per epoch: the mean over that epoch for the
        training metrics and the validation mean for the val_ prefixed ones.
        The val_ lists stay empty when no validation source is given.
        """
        # Basic validation
        if train_dataset is None and (X_train is None or Y_train is None):
            raise ValueError("Provide (X_train, Y_train) or a train_dataset")

        # Device info
        devices = tf.config.list_physical_devices('GPU')
        if devices:
            print("Training on GPU:", [d.name for d in devices])
        else:
            print("Training on CPU")

        # Building the training dataset
        if train_dataset is None:
            # Dataset from arrays
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
            # External dataset: it must provide batching, otherwise we add it.
            # repeat() is forced here for consistency.
            # The structure is left untouched if it is already batched
            # (assumed to be the caller's responsibility).
            train_dataset = train_dataset.repeat()
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required when an external dataset is provided")

        # Normalisation to [-1,1] where applicable
        if normalize:
            train_dataset = train_dataset.map(lambda x,y: (x*2.0 - 1.0, y*2.0 - 1.0), num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Validation dataset
        val_data_struct = None
        if val_dataset is not None:
            val_data_struct = val_dataset
        elif X_val is not None and Y_val is not None:
            val_data_struct = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size)
            if val_steps is None:
                val_steps = int(np.ceil(len(X_val)/batch_size))
        
        if val_data_struct is not None and normalize:
            val_data_struct = val_data_struct.map(lambda x,y: (x*2.0 - 1.0, y*2.0 - 1.0), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        # The run directory does not exist yet, so the grids are staged
        # under this training session and moved once the model is saved.
        self._preview_session = datetime.datetime.now().strftime(
            TIMESTAMP_FORMAT
        )
        if save_dir is None:
            save_dir = stage_preview_directory(self._preview_session)
        elif save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Cache of a fixed preview batch, so the same patches are rendered at
        # every epoch and two grids differ only by what the generator learnt.
        preview_cache = None

        def _prepare_preview_batch():
            nonlocal preview_cache
            if preview_cache is not None:
                return preview_cache

            n_max = 8
            lr_batch, hr_batch = None, None
            is_norm = False

            if X_val is not None and len(X_val) > 0:
                lr_batch, hr_batch = X_val, Y_val
            elif X_train is not None and len(X_train) > 0:
                lr_batch, hr_batch = X_train, Y_train

            if lr_batch is not None:
                # The patches of one image are contiguous in these arrays, so
                # the leading slice of them is a set of neighbouring crops of
                # a single frame. Spreading the picks over the whole array is
                # what makes the grid show unrelated content.
                idx = np.unique(
                    np.linspace(
                        0, len(lr_batch) - 1, min(n_max, len(lr_batch))
                    ).round().astype(int)
                )
                lr_batch, hr_batch = lr_batch[idx], hr_batch[idx]
            else:
                src = (
                    val_data_struct if val_data_struct is not None
                    else train_dataset
                )
                for lr_b, hr_b in src.take(1):
                    lr_batch = lr_b.numpy()[:n_max]
                    hr_batch = hr_b.numpy()[:n_max]
                    is_norm = bool(normalize)
                if lr_batch is None:
                    raise RuntimeError("Could not obtain a preview batch to save images.")

            preview_cache = (
                lr_batch.astype(np.float32),
                hr_batch.astype(np.float32),
                is_norm,
            )
            return preview_cache

        def _to_uint8(img):
            img = np.clip(img, 0.0, 1.0)
            return (img * 255.0).round().astype(np.uint8)

        def _save_preview_grid(epoch_idx):
            """
            Write one row per preview patch as LR | SR | HR.

            The generator output on its own says nothing: a 48 px crop looks
            plausible from the first epoch, because the network is fed the
            low-resolution image and not a noise vector. What the grid has
            to show is the distance still left to the reference, so the
            input and the target are rendered beside it.
            """

            if not save_dir:
                return

            lr_preview, hr_preview, is_norm = _prepare_preview_batch()

            lr_in = lr_preview if is_norm else (lr_preview * 2.0 - 1.0)
            sr = (self.generator(lr_in, training=False).numpy() + 1.0) / 2.0

            # Only a batch taken from a normalised dataset needs mapping back.
            lr_view = (lr_preview + 1.0) / 2.0 if is_norm else lr_preview
            hr_view = (hr_preview + 1.0) / 2.0 if is_norm else hr_preview

            # The LR patch is repeated, not resampled, up to HR size: the
            # three columns then share a scale and whatever blur the left
            # column shows is the generator's input rather than an artefact
            # introduced by the figure.
            ratio = hr_view.shape[1] // lr_view.shape[1]
            lr_view = np.repeat(
                np.repeat(lr_view, ratio, axis=1), ratio, axis=2
            )

            # A 48 px tile is too small to judge, so every tile is repeated
            # by an integer factor. Nearest-neighbour keeps the zoom honest.
            zoom, gutter = 3, 4
            h = hr_view.shape[1] * zoom
            w = hr_view.shape[2] * zoom
            rows = len(sr)

            grid = np.full(
                (rows * h + (rows - 1) * gutter, 3 * w + 2 * gutter, 3),
                255, dtype=np.uint8,
            )
            for r in range(rows):
                top = r * (h + gutter)
                for col, tile in enumerate((lr_view[r], sr[r], hr_view[r])):
                    tile = _to_uint8(tile)
                    tile = np.repeat(
                        np.repeat(tile, zoom, axis=0), zoom, axis=1
                    )
                    left = col * (w + gutter)
                    grid[top:top + h, left:left + w] = tile

            png = tf.image.encode_png(grid)
            out_path = os.path.join(
                save_dir, f"epoch_{epoch_idx:03d}_lr_sr_hr.png"
            )
            tf.io.write_file(out_path, png)

        # Trackers
        time_tracker = EpochTimeTracker()
        memory_tracker = EpochMemoryTracker(track_gpu=True, gpu_device="GPU:0")

        # One entry per epoch, so that history[key][-1] is the mean over the
        # last epoch and means the same as it does in a keras History.
        train_keys = ("g_loss", "d_loss", "psnr", "ssim", "g_lr", "d_lr")
        val_keys = ("val_g_loss", "val_psnr", "val_ssim")
        history = {key: [] for key in train_keys + val_keys}

        # The decay interval needs the total number of steps, which is only
        # settled once epochs and steps_per_epoch are both resolved.
        self._schedule_learning_rates(epochs, steps_per_epoch)

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            if time_tracker is not None:
                time_tracker.begin_epoch()
            if memory_tracker is not None:
                memory_tracker.begin_epoch()

            # Per-step accumulator, reset on every epoch.
            epoch_losses = {key: [] for key in train_keys}

            # Iterate over training batches
            for step, (lr_batch, hr_batch) in enumerate(train_dataset.take(steps_per_epoch)):
                # The step returns the perceptual metrics alongside the two
                # losses, computed on the forward pass it already needed.
                losses = self._train_step(lr_batch, hr_batch)
                for key, value in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(float(value.numpy()))

                epoch_losses["g_lr"].append(float(self.g_optimizer._decayed_lr(tf.float32).numpy()))
                epoch_losses["d_lr"].append(float(self.d_optimizer._decayed_lr(tf.float32).numpy()))

                if memory_tracker is not None:
                    memory_tracker.sample_step()

                if (step+1) % 10 == 0 or (step+1) == steps_per_epoch:
                    print(
                        f"  Step {step+1}/{steps_per_epoch} G_loss={epoch_losses['g_loss'][-1]:.4f} "
                        f"D_loss={epoch_losses['d_loss'][-1]:.4f} PSNR={epoch_losses['psnr'][-1]:.2f} "
                        f"SSIM={epoch_losses['ssim'][-1]:.4f}")

            # Epoch summary
            for key, values in epoch_losses.items():
                history[key].append(
                    float(np.mean(values)) if values else float("nan")
                )
            print(
                f"- Epoch Summary - G_loss: {history['g_loss'][-1]:.4f}, "
                f"D_loss: {history['d_loss'][-1]:.4f}, "
                f"PSNR: {history['psnr'][-1]:.2f}, "
                f"SSIM: {history['ssim'][-1]:.4f}")

            # Validation, when available
            if val_data_struct is not None:
                val_psnr, val_ssim, val_g_losses = [], [], []
                for lr_v, hr_v in val_data_struct.take(val_steps):
                    # Forward pass
                    hr_fake_v = self.generator(lr_v, training=False)

                    # Generator validation loss, without gradients
                    d_fake_v = self.discriminator(hr_fake_v, training=False)
                    g_loss_v, _ = self._generator_loss(hr_v, hr_fake_v, d_fake_v)
                    val_g_losses.append(float(g_loss_v.numpy()))

                    # PSNR / SSIM in [0,1]
                    hr_real_eval = (hr_v + 1.0) / 2.0
                    hr_gen_eval  = (hr_fake_v + 1.0) / 2.0
                    val_psnr.append(float(tf.reduce_mean(tf.image.psnr(hr_real_eval, hr_gen_eval, 1.0)).numpy()))
                    val_ssim.append(float(tf.reduce_mean(tf.image.ssim(hr_real_eval, hr_gen_eval, 1.0)).numpy()))

                val_psnr_mean = float(np.mean(val_psnr)) if len(val_psnr) > 0 else float('nan')
                val_ssim_mean = float(np.mean(val_ssim)) if len(val_ssim) > 0 else float('nan')
                val_g_loss_mean = float(np.mean(val_g_losses)) if len(val_g_losses) > 0 else float('nan')

                history["val_psnr"].append(val_psnr_mean)
                history["val_ssim"].append(val_ssim_mean)
                history["val_g_loss"].append(val_g_loss_mean)

                print(
                    f"  Validation -> PSNR: {val_psnr_mean:.2f}, SSIM: {val_ssim_mean:.4f}, G_loss: {val_g_loss_mean:.4f}")

            # First epoch as the baseline, then one grid every few epochs:
            # what the previews are for is the trend, not every step.
            if epoch == 0 or (epoch + 1) % preview_every == 0:
                _save_preview_grid(epoch + 1)

            self.trained = True

            # End-of-epoch tracking
            if memory_tracker is not None:
                memory_tracker.end_epoch()
            if time_tracker is not None:
                time_tracker.end_epoch()

                # The custom loop has no Keras progress bar, so the epoch
                # duration is the only cue of how long the run will take.
                elapsed = time_tracker.epoch_times_sec[-1]
                remaining = elapsed * (epochs - epoch - 1)
                print(
                    f"  Epoch time: {format_duration(elapsed)}"
                    f"   (~{format_duration(remaining)} left)\n",
                    flush=True,
                )

        return history, time_tracker, memory_tracker
    
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

            d_fake = self.discriminator(hr_generated, training=False)
            g_loss, _ = self._generator_loss(hr_batch, hr_generated, d_fake)
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
    
    def evaluate_and_save(
            self, test_dataset, history, time_cb, mem_cb, timestamp=None):
        """Evaluate the run, then persist both networks and the metrics.

        The three steps travel together because they describe one training
        run: splitting them across cells is what lets a checkpoint be saved
        under one timestamp and its metrics under another.

        Cost is charged over training and over this evaluation only. The
        patch-wise reconstruction of a full frame belongs to the detection
        pipeline, so it is not attributed to the model.

        Parameters
        ----------
        test_dataset : tf.data.Dataset
            Batched test pairs, normalised the way training saw them.
        history : dict
            Per-epoch curves returned by ``fit``.
        time_cb, mem_cb : EpochTimeTracker, EpochMemoryTracker
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
        run_name = f"ESRGAN_{timestamp}"

        with stage(f"ESRGAN run {timestamp}") as step:
            step("evaluating on the test partition")
            results, eval_time_sec, eval_memory = profile_evaluation(
                lambda: self.evaluate(test_dataset)
            )

            # The generator loss stands in for the loss of the other two
            # models, so the reported key set stays identical across them.
            metrics = {
                "eval_loss": float(results["avg_g_loss"]),
                "eval_psnr": float(results["avg_psnr"]),
                "eval_ssim": float(results["avg_ssim"]),
                **last_epoch_metrics(
                    {**history, "loss": history.get("g_loss"),
                     "val_loss": history.get("val_g_loss")}
                ),
                "epoch_time_sec": time_cb.mean_time_value(),
                "memory": mem_cb.as_dict(),
                "eval_time_sec": eval_time_sec,
                "eval_memory": eval_memory,
            }

            run_dir = prepare_run_directory("ESRGAN", run_name)
            self.save(directory=run_dir, timestamp=timestamp)
            step(f"metrics    -> {save_run_metrics(run_dir, run_name, metrics)}")
            step("summary    -> " + save_model_summary(
                run_dir, run_name,
                {"GENERATOR": self.generator,
                 "DISCRIMINATOR": self.discriminator,
                 "VGG FEATURE EXTRACTOR": self.vgg_model},
            ))

            # The custom loop keeps the epoch durations outside the history,
            # so they are folded back in to match what the Keras models log.
            step("epochs     -> " + save_epoch_log(
                run_dir, run_name,
                {**history, "epoch_time_sec": time_cb.epoch_times_sec},
            ))

            # Previews of sessions that were never saved travel with this
            # run rather than being lost, tagged with the session that
            # produced them.
            moved = collect_staged_previews(run_dir)
            if moved:
                total = sum(moved.values())
                step(f"previews   {total} grid(s) from {len(moved)} session(s) "
                     f"-> {os.path.join(run_dir, ESRGAN_PREVIEW_SUBDIR)}")
                for session, count in moved.items():
                    own = " (this run)" if session == self._preview_session else ""
                    step(f"             {session}: {count} grid(s){own}")

        return timestamp, run_dir, metrics

    def super_resolve_image(self, lr_img, patch_size_lr=ESRGAN_PATCH_SIZE, stride=ESRGAN_STRIDE, batch_size=16):
        """Patch-wise super-resolution using the ESRGAN generator.
        Follows SRCNN/EDSR flow: reflect padding, patch extraction, batch predict, overlap-averaged reconstruction.
        Accounts for ESRGAN's [-1,1] tanh output by normalizing inputs to [-1,1] and denormalizing outputs to [0,1].

        Args:
            lr_img: LR RGB image array.
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

        # Pad LR image with the shared helper so this grid matches training
        lr_orig_shape = lr_img.shape[:2]
        lr_padded = add_padding(lr_img, patch_size_lr, stride)

        # Extract LR patches and normalize to [-1,1]
        lr_patches, positions = extract_lr_patches(lr_padded, patch_size_lr, stride)

        lr_patches_norm = (lr_patches * 2.0) - 1.0

        # --- Predict HR patches in batch ---
        hr_patches = self.generator.predict(
            lr_patches_norm, batch_size=batch_size, verbose=0
        )

        hr_patches = (hr_patches + 1.0) / 2.0  # to [0,1]

        # Reconstruct HR image and crop to target size
        return reconstruct_from_hr_patches(
            hr_patches, positions, lr_padded.shape, lr_orig_shape, patch_size_lr, scale
        )
    
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