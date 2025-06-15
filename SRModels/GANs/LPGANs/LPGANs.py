import tensorflow as tf
from keras import layers, models, optimizers, losses

# Laplacian Pyramid GAN (LAPGAN) with 3 levels (32x32 -> 64x64 -> 128x128):contentReference[oaicite:0]{index=0}
class LaplacianPyramidGAN:
    def __init__(self, latent_dim=100):
        """Initialize the LAPGAN with 3 levels (32x32, 64x64, 128x128)."""
        self.latent_dim = latent_dim
        self.img_channels = 3
        # Build generator and discriminator for each level
        self.gen32 = self.build_generator_32()
        self.disc32 = self.build_discriminator_32()
        self.gen64 = self.build_generator_64()
        self.disc64 = self.build_discriminator_64()
        self.gen128 = self.build_generator_128()
        self.disc128 = self.build_discriminator_128()
        # Loss function for adversarial training
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        # Optimizers for each generator/discriminator
        self.gen32_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc32_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.gen64_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc64_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.gen128_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc128_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        # Setup checkpoints for all models and optimizers:contentReference[oaicite:1]{index=1}
        self.ckpt = tf.train.Checkpoint(gen32=self.gen32, disc32=self.disc32,
                                        gen64=self.gen64, disc64=self.disc64,
                                        gen128=self.gen128, disc128=self.disc128,
                                        gen32_optimizer=self.gen32_optimizer,
                                        disc32_optimizer=self.disc32_optimizer,
                                        gen64_optimizer=self.gen64_optimizer,
                                        disc64_optimizer=self.disc64_optimizer,
                                        gen128_optimizer=self.gen128_optimizer,
                                        disc128_optimizer=self.disc128_optimizer)
        self.ckpt_manager = None

    def build_generator_32(self):
        """Builds a generator model for 32x32 image generation."""
        model = models.Sequential(name="Generator_32")
        n_nodes = 4 * 4 * 256
        model.add(layers.Dense(n_nodes, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((4, 4, 256)))
        # Upsample to 8x8
        model.add(layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # Upsample to 16x16
        model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # Upsample to 32x32
        model.add(layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # Final output 32x32x3 with tanh activation (range [-1,1]):contentReference[oaicite:2]{index=2}
        model.add(layers.Conv2DTranspose(self.img_channels, (4,4), strides=(1,1), padding='same', use_bias=False, activation='tanh'))
        return model

    def build_discriminator_32(self):
        """Builds a discriminator model for 32x32 images."""
        model = models.Sequential(name="Discriminator_32")
        model.add(layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(32,32,self.img_channels)))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(256, (3,3), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Flatten())
        model.add(layers.Dense(1))  # Output is a single logit for real/fake
        return model

    def build_generator_64(self):
        """Builds a conditional generator model for 64x64 image generation."""
        # Two inputs: upsampled 32x32 base image and a noise vector
        base_input = layers.Input(shape=(64, 64, self.img_channels))
        noise_input = layers.Input(shape=(self.latent_dim,))
        # Process noise vector to spatial feature map
        x = layers.Dense(4*4*256, use_bias=False)(noise_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((4, 4, 256))(x)
        x = layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False)(x)  # 8x8
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False)(x)  # 16x16
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False)(x)   # 32x32
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', use_bias=False)(x)   # 64x64
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        # Process base image to features
        y = layers.Conv2D(32, (3,3), strides=(1,1), padding='same', use_bias=False)(base_input)
        y = layers.LeakyReLU()(y)
        # Combine noise and base features
        merged = layers.Concatenate()([x, y])
        # Final conv layers to produce 64x64x3 image
        merged = layers.Conv2D(64, (3,3), padding='same')(merged)
        merged = layers.LeakyReLU()(merged)
        merged = layers.Conv2D(self.img_channels, (3,3), padding='same', activation='tanh')(merged)
        return models.Model(inputs=[base_input, noise_input], outputs=merged, name="Generator_64")

    def build_discriminator_64(self):
        """Builds a discriminator model for 64x64 images."""
        model = models.Sequential(name="Discriminator_64")
        model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=(64,64,self.img_channels)))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, (4,4), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(256, (4,4), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def build_generator_128(self):
        """Builds a conditional generator model for 128x128 image generation."""
        base_input = layers.Input(shape=(128, 128, self.img_channels))
        noise_input = layers.Input(shape=(self.latent_dim,))
        # Noise path
        x = layers.Dense(4*4*512, use_bias=False)(noise_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((4, 4, 512))(x)
        x = layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False)(x)  # 8x8
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False)(x)  # 16x16
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False)(x)   # 32x32
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', use_bias=False)(x)   # 64x64
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2DTranspose(16, (4,4), strides=(2,2), padding='same', use_bias=False)(x)   # 128x128
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        # Base image path
        y = layers.Conv2D(32, (3,3), strides=(1,1), padding='same', use_bias=False)(base_input)
        y = layers.LeakyReLU()(y)
        # Combine noise and base features
        merged = layers.Concatenate()([x, y])
        merged = layers.Conv2D(64, (3,3), padding='same')(merged)
        merged = layers.LeakyReLU()(merged)
        merged = layers.Conv2D(self.img_channels, (3,3), padding='same', activation='tanh')(merged)
        return models.Model(inputs=[base_input, noise_input], outputs=merged, name="Generator_128")

    def build_discriminator_128(self):
        """Builds a discriminator model for 128x128 images."""
        model = models.Sequential(name="Discriminator_128")
        model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=(128,128,self.img_channels)))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, (4,4), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(256, (4,4), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def train(self, epochs=10000, batch_size=64, checkpoint_dir='./lapgan_ckpts'):
        """
        Train the LAPGAN level by level on CIFAR-10:contentReference[oaicite:3]{index=3}.
        Checkpoints are saved to the specified directory.
        """
        # Load CIFAR-10 dataset:contentReference[oaicite:4]{index=4}
        (trainX, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        # Convert to float and scale to [-1,1]:contentReference[oaicite:5]{index=5}
        trainX = trainX.astype('float32')
        trainX = (trainX - 127.5) / 127.5
        # Prepare dataset
        dataset = tf.data.Dataset.from_tensor_slices(trainX).shuffle(buffer_size=10000).batch(batch_size)
        # Setup checkpoint manager
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=3)
        # Training loops (sequentially by pyramid level)
        for level, (gen, disc, g_opt, d_opt, size) in enumerate([
            (self.gen32, self.disc32, self.gen32_optimizer, self.disc32_optimizer, 32),
            (self.gen64, self.disc64, self.gen64_optimizer, self.disc64_optimizer, 64),
            (self.gen128, self.disc128, self.gen128_optimizer, self.disc128_optimizer, 128)
        ], start=1):
            print(f"Training level {level} for size {size}x{size}")
            for epoch in range(epochs):
                for real_images in dataset:
                    # Prepare real images at this level
                    if size == 32:
                        real = real_images  # already 32x32
                    else:
                        real = tf.image.resize(real_images, (size, size))
                    # Generate noise and fake images
                    noise = tf.random.normal([real.shape[0], self.latent_dim])
                    if size == 32:
                        fake = gen(noise, training=True)
                    else:
                        # Upsample lower-resolution real images as conditional base:contentReference[oaicite:6]{index=6}
                        base = tf.image.resize(real_images, (size//2, size//2))
                        base = tf.image.resize(base, (size, size))
                        fake = gen([base, noise], training=True)
                    # Train discriminator with real images
                    with tf.GradientTape() as disc_tape:
                        real_logits = disc(real, training=True)
                        fake_logits = disc(fake, training=True)
                        d_loss_real = self.cross_entropy(tf.ones_like(real_logits), real_logits)
                        d_loss_fake = self.cross_entropy(tf.zeros_like(fake_logits), fake_logits)
                        d_loss = d_loss_real + d_loss_fake
                    d_gradients = disc_tape.gradient(d_loss, disc.trainable_variables)
                    d_opt.apply_gradients(zip(d_gradients, disc.trainable_variables))
                    # Train generator to fool discriminator
                    with tf.GradientTape() as gen_tape:
                        if size == 32:
                            fake = gen(noise, training=True)
                        else:
                            fake = gen([base, noise], training=True)
                        fake_logits_for_g = disc(fake, training=False)
                        g_loss = self.cross_entropy(tf.ones_like(fake_logits_for_g), fake_logits_for_g)
                    g_gradients = gen_tape.gradient(g_loss, gen.trainable_variables)
                    g_opt.apply_gradients(zip(g_gradients, gen.trainable_variables))
                # Save checkpoint at each epoch (demonstration):contentReference[oaicite:7]{index=7}
                ckpt_save_path = self.ckpt_manager.save()
                print(f"Level {level} Epoch {epoch+1}: checkpoint saved at {ckpt_save_path}")

    def restore(self, checkpoint_dir):
        """Restore model weights from the latest checkpoint in the directory."""
        self.ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print(f"Restored from {checkpoint_dir}")

    def generate(self, z1, z2=None, z3=None):
        """
        Generate images from the trained LAPGAN given noise vectors.
        z1: noise for 32x32 generator
        z2: optional noise for 64x64 generator (default random)
        z3: optional noise for 128x128 generator (default random)
        """
        # Generate 32x32 image
        fake32 = self.gen32(z1, training=False)
        result = [fake32]
        # Generate 64x64
        if z2 is None:
            z2 = tf.random.normal([z1.shape[0], self.latent_dim])
        base64 = tf.image.resize(fake32, (64, 64))
        fake64 = self.gen64([base64, z2], training=False)
        result.append(fake64)
        # Generate 128x128
        if z3 is None:
            z3 = tf.random.normal([z1.shape[0], self.latent_dim])
        base128 = tf.image.resize(fake64, (128, 128))
        fake128 = self.gen128([base128, z3], training=False)
        result.append(fake128)
        return result
