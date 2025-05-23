import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential
from keras.datasets import mnist
from keras.losses import BinaryCrossentropy
from keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, ReLU, Flatten, Input

class DCGAN:
    def __init__(self, checkpoint_dir='dcgan_checkpoints', noise_dim=100):
        self.noise_dim = noise_dim
        self.checkpoint_dir = checkpoint_dir
        self.cross_entropy = BinaryCrossentropy(from_logits=True)

        # Crear modelos
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Optimizers
        self.generator_optimizer = Adam(1e-4)
        self.discriminator_optimizer = Adam(1e-4)

        # Checkpoint Manager
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator,
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=3)

        # Cargar datos
        (train_images, _), _ = mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        self.train_images = (train_images - 127.5) / 127.5  # Normalización [-1, 1]
        self.batch_size = 256
        self.buffer_size = 60000
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_images).shuffle(self.buffer_size).batch(self.batch_size)

        # Seed para generación constante
        self.seed = tf.random.normal([25, self.noise_dim])

    # DCGAN Generator Architecture
    def _build_generator(self):
        model = Sequential([
            Input(shape=(self.noise_dim,)),
            
            # Project and reshape
            Dense(4 * 4 * 1024, use_bias=False),
            Reshape((4, 4, 1024)),
            BatchNormalization(),
            ReLU(),
            
            # Upsample to 8x8
            Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            
            # Upsample to 16x16
            Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            
            # Upsample to 32x32
            Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            
            # Final layer to 64x64
            Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh')
        ])
        
        return model

    # DCGAN Discriminator Architecture
    def _build_discriminator(self):
        model = Sequential([
            Input(shape=(64, 64, 3)),
            
            # 64x64 -> 32x32
            Conv2D(64, kernel_size=5, strides=2, padding='same', use_bias=False),
            LeakyReLU(negative_slope=0.2),
            
            # 32x32 -> 16x16
            Conv2D(128, kernel_size=5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.2),
            
            # 16x16 -> 8x8
            Conv2D(256, kernel_size=5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.2),
            
            # 8x8 -> 4x4
            Conv2D(512, kernel_size=5, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(negative_slope=0.2),
            
            # Final classification
            Flatten(),
            Dense(1, use_bias=False)
        ])
        
        return model

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        
        return real_loss + fake_loss

    @tf.function
    def _train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss
    
    def train(self, epochs=50):
        for epoch in range(epochs):
            for image_batch in self.train_dataset:
                gen_loss, disc_loss = self._train_step(image_batch)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
                self.generate_and_save_images(epoch)
                self.manager.save()

    def restore_latest_checkpoint(self):
        latest = self.manager.latest_checkpoint
        
        if latest:
            self.checkpoint.restore(latest).expect_partial()
            print(f"Modelo restaurado desde {latest}")
            self._verify_restored_model()
        else:
            print("No se encontró ningún checkpoint para restaurar.")

    def _verify_restored_model(self):
        try:
            self.generate_and_plot_images()
            print("Verificación del modelo restaurado completada correctamente.")
        except Exception as e:
            print("Error al verificar el modelo restaurado:", e)

    def generate_and_save_images(self, epoch):
        predictions = self.generator(self.seed, training=False)
        
        _ = plt.figure(figsize=(4, 4))
        
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            
        plt.savefig(f'image_at_epoch_{epoch:04d}.png')
        plt.close()