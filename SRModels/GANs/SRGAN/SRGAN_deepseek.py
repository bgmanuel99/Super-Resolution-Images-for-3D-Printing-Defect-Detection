"""
SRGAN Implementation in TensorFlow/Keras
Archivo único ejecutable (train_srgan.py)

Usage:
python train_srgan.py --train_lr_dir [DIR] --train_hr_dir [DIR] --val_lr_dir [DIR] --val_hr_dir [DIR]
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import time

# Configuración por defecto
DEFAULT_LR_SIZE = 64
DEFAULT_HR_SIZE = 256
DEFAULT_SCALE_FACTOR = 4
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 100
DEFAULT_PRETRAIN_EPOCHS = 10

class SRGAN:
    def __init__(self, lr_size=DEFAULT_LR_SIZE, hr_size=DEFAULT_HR_SIZE):
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.scale_factor = hr_size // lr_size
        
        # Build models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.vgg = self.build_vgg()
        
        # Optimizers
        self.generator_optimizer = keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = keras.optimizers.Adam(1e-4)
        
        # Checkpoint
        self.checkpoint_dir = "./checkpoints_srgan"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        
        # Results directory
        self.results_dir = "./results_srgan"
        os.makedirs(self.results_dir, exist_ok=True)

    def residual_block(self, x):
        """Bloque residual con 2 convoluciones y conexión skip."""
        filters = x.shape[-1]
        initial = x
        
        x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        
        x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        return layers.Add()([initial, x])

    def upsample_block(self, x, filters):
        """Bloque de upsampling con sub-pixel convolution."""
        x = layers.Conv2D(filters * 4, kernel_size=3, padding="same")(x)
        x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        return layers.PReLU(shared_axes=[1, 2])(x)

    def build_generator(self):
        """Construye el generador SRGAN."""
        inputs = keras.Input((self.lr_size, self.lr_size, 3))
        
        # Capa inicial
        x = layers.Conv2D(64, kernel_size=9, padding="same")(inputs)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        initial = x
        
        # Bloques residuales (16 según el paper original)
        for _ in range(16):
            x = self.residual_block(x)
        
        # Capas finales
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([initial, x])  # Conexión skip global
        
        # Upsampling x2 (2 bloques para factor 4)
        x = self.upsample_block(x, 64)
        x = self.upsample_block(x, 64)
        
        # Capa de salida
        x = layers.Conv2D(3, kernel_size=9, padding="same", activation="tanh")(x)
        
        return keras.Model(inputs, x, name="generator")

    def build_discriminator(self):
        """Construye el discriminador SRGAN."""
        inputs = keras.Input((self.hr_size, self.hr_size, 3))
        
        x = layers.Conv2D(64, kernel_size=3, padding="same")(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Bloques convolucionales con reducción de dimensión
        filters = 64
        for i in range(4):
            filters = min(filters * 2, 512)
            x = layers.Conv2D(filters, kernel_size=3, strides=2 if i < 3 else 1, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Capas densas finales
        x = layers.Flatten()(x)
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dense(1, activation="sigmoid")(x)
        
        return keras.Model(inputs, x, name="discriminator")

    def build_vgg(self):
        """Construye el modelo VGG19 para la pérdida de contenido."""
        vgg = keras.applications.VGG19(
            include_top=False, 
            weights="imagenet", 
            input_shape=(self.hr_size, self.hr_size, 3)
        )
        vgg.trainable = False
        
        # Usamos salidas de las capas de bloque 5
        outputs = [vgg.layers[20].output]
        return keras.Model(vgg.input, outputs)

    def content_loss(self, hr, sr):
        """Pérdida de contenido basada en características VGG."""
        sr = keras.applications.vgg19.preprocess_input(sr)
        hr = keras.applications.vgg19.preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return keras.losses.mean_squared_error(hr_features, sr_features)

    def generator_loss(self, fake_output):
        """Pérdida del generador (adversarial)."""
        cross_entropy = keras.losses.BinaryCrossentropy()
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """Pérdida del discriminador."""
        cross_entropy = keras.losses.BinaryCrossentropy()
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def load_and_preprocess_image(self, path, resize_to=None):
        """Carga y preprocesa una imagen."""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        if resize_to:
            img = tf.image.resize(img, resize_to, method="bicubic")
        
        # Normalizar a [-1, 1]
        img = (img - 0.5) / 0.5
        return img

    def create_dataset(self, lr_dir, hr_dir, batch_size=8):
        """Crea un dataset de pares LR-HR."""
        lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(('.jpg', '.png'))])
        hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('.jpg', '.png'))])
        
        # Verificar que tenemos el mismo número de imágenes
        assert len(lr_paths) == len(hr_paths), "Número de imágenes LR y HR no coincide"
        
        dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
        dataset = dataset.map(
            lambda lr, hr: (
                self.load_and_preprocess_image(lr, (self.lr_size, self.lr_size)),
                self.load_and_preprocess_image(hr, (self.hr_size, self.hr_size))
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset

    @tf.function
    def pretrain_step(self, lr_images, hr_images):
        """Paso de pre-entrenamiento para el generador."""
        with tf.GradientTape() as tape:
            sr_images = self.generator(lr_images, training=True)
            loss = self.content_loss(hr_images, sr_images)
        
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return loss

    def pretrain_generator(self, dataset, epochs):
        """Pre-entrena el generador con pérdida de contenido."""
        print("Iniciando pre-entrenamiento del generador...")
        
        for epoch in range(epochs):
            start = time.time()
            
            total_loss = 0
            for batch, (lr_images, hr_images) in enumerate(dataset):
                loss = self.pretrain_step(lr_images, hr_images)
                total_loss += loss
                
                if batch % 100 == 0:
                    print(f"Batch {batch}, Pérdida: {loss:.4f}")
            
            # Guardar checkpoint cada época
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            
            print(f"Época {epoch + 1}, Pérdida: {total_loss / (batch + 1):.4f}, Tiempo: {time.time() - start:.2f}s")
        
        print("Pre-entrenamiento completado.")

    @tf.function
    def train_step(self, lr_images, hr_images):
        """Paso de entrenamiento para el SRGAN."""
        # Generar imágenes super-resolucionadas
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            sr_images = self.generator(lr_images, training=True)
            
            # Discriminar imágenes reales y generadas
            real_output = self.discriminator(hr_images, training=True)
            fake_output = self.discriminator(sr_images, training=True)
            
            # Calcular pérdidas
            gen_content_loss = self.content_loss(hr_images, sr_images)
            gen_adversarial_loss = 1e-3 * self.generator_loss(fake_output)
            gen_total_loss = gen_content_loss + gen_adversarial_loss
            
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        # Aplicar gradientes al generador
        gen_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        # Aplicar gradientes al discriminador
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_content_loss, gen_adversarial_loss, disc_loss

    def evaluate_model(self, dataset):
        """Evalúa el modelo en el conjunto de validación."""
        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        
        for lr_images, hr_images in dataset.take(10):  # Evaluar en 10 batches
            sr_images = self.generator(lr_images, training=False)
            
            # Convertir de [-1, 1] a [0, 1]
            hr_images = (hr_images + 1) / 2
            sr_images = (sr_images + 1) / 2
            
            total_psnr += tf.reduce_mean(tf.image.psnr(hr_images, sr_images, max_val=1.0))
            total_ssim += tf.reduce_mean(tf.image.ssim(hr_images, sr_images, max_val=1.0))
            num_batches += 1
        
        return {
            "psnr": total_psnr / num_batches,
            "ssim": total_ssim / num_batches
        }

    def generate_samples(self, dataset, epoch, num_samples=3):
        """Genera muestras de super-resolución y las guarda."""
        for i, (lr_images, hr_images) in enumerate(dataset.take(num_samples)):
            sr_images = self.generator(lr_images, training=False)
            
            # Convertir de [-1, 1] a [0, 255]
            lr_images = (lr_images + 1) * 127.5
            hr_images = (hr_images + 1) * 127.5
            sr_images = (sr_images + 1) * 127.5
            
            # Guardar imágenes
            for j in range(lr_images.shape[0]):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(tf.cast(lr_images[j], tf.uint8))
                axes[0].set_title("Baja Resolución")
                axes[0].axis("off")
                
                axes[1].imshow(tf.cast(sr_images[j], tf.uint8))
                axes[1].set_title("Super-Resolución")
                axes[1].axis("off")
                
                axes[2].imshow(tf.cast(hr_images[j], tf.uint8))
                axes[2].set_title("Alta Resolución (Real)")
                axes[2].axis("off")
                
                plt.savefig(f"{self.results_dir}/epoch_{epoch}_sample_{i}_{j}.png")
                plt.close()

    def train(self, train_dataset, val_dataset, epochs, pretrain_epochs):
        """Entrena el modelo completo."""
        # Cargar último checkpoint si existe
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Checkpoint cargado: {latest_checkpoint}")
        
        # Pre-entrenamiento del generador
        self.pretrain_generator(train_dataset, pretrain_epochs)
        
        print("Iniciando entrenamiento completo del SRGAN...")
        
        for epoch in range(epochs):
            start = time.time()
            
            total_content_loss = 0
            total_adversarial_loss = 0
            total_disc_loss = 0
            
            for batch, (lr_images, hr_images) in enumerate(train_dataset):
                content_loss, adversarial_loss, disc_loss = self.train_step(lr_images, hr_images)
                
                total_content_loss += content_loss
                total_adversarial_loss += adversarial_loss
                total_disc_loss += disc_loss
                
                if batch % 100 == 0:
                    print(f"Batch {batch}, Contenido: {content_loss:.4f}, Adversarial: {adversarial_loss:.4f}, Disc: {disc_loss:.4f}")
            
            # Guardar checkpoint y mostrar progreso
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            
            # Calcular métricas en el conjunto de validación
            val_metrics = self.evaluate_model(val_dataset)
            
            print(f"Época {epoch + 1}")
            print(f"  Pérdidas - Contenido: {total_content_loss/(batch+1):.4f}, Adversarial: {total_adversarial_loss/(batch+1):.4f}, Disc: {total_disc_loss/(batch+1):.4f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f}, Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Tiempo: {time.time() - start:.2f}s")
            
            # Generar ejemplos de muestra
            if (epoch + 1) % 5 == 0:
                self.generate_samples(val_dataset, epoch + 1)
        
        print("Entrenamiento completado.")

    def super_resolve_image(self, image_path, output_path=None):
        """Aplica super-resolución a una imagen personalizada."""
        # Cargar y preprocesar imagen
        lr_image = self.load_and_preprocess_image(image_path, (self.lr_size, self.lr_size))
        lr_image = tf.expand_dims(lr_image, axis=0)  # Añadir dimensión de batch
        
        # Generar super-resolución
        sr_image = self.generator(lr_image, training=False)
        
        # Convertir de [-1, 1] a [0, 255]
        lr_image = (lr_image + 1) * 127.5
        sr_image = (sr_image + 1) * 127.5
        
        # Mostrar resultados
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(tf.cast(lr_image[0], tf.uint8))
        plt.title("Baja Resolución")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(tf.cast(sr_image[0], tf.uint8))
        plt.title("Super-Resolución")
        plt.axis("off")
        
        if output_path:
            sr_pil = Image.fromarray(tf.cast(sr_image[0], tf.uint8).numpy())
            sr_pil.save(output_path)
            print(f"Imagen guardada en {output_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento de SRGAN')
    parser.add_argument('--train_lr_dir', type=str, required=True, help='Directorio con imágenes de baja resolución para entrenamiento')
    parser.add_argument('--train_hr_dir', type=str, required=True, help='Directorio con imágenes de alta resolución para entrenamiento')
    parser.add_argument('--val_lr_dir', type=str, required=True, help='Directorio con imágenes de baja resolución para validación')
    parser.add_argument('--val_hr_dir', type=str, required=True, help='Directorio con imágenes de alta resolución para validación')
    parser.add_argument('--lr_size', type=int, default=DEFAULT_LR_SIZE, help='Tamaño de imágenes de baja resolución')
    parser.add_argument('--hr_size', type=int, default=DEFAULT_HR_SIZE, help='Tamaño de imágenes de alta resolución')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Número de épocas de entrenamiento')
    parser.add_argument('--pretrain_epochs', type=int, default=DEFAULT_PRETRAIN_EPOCHS, help='Número de épocas de pre-entrenamiento')
    
    args = parser.parse_args()
    
    # Crear modelo SRGAN
    srgan = SRGAN(lr_size=args.lr_size, hr_size=args.hr_size)
    
    # Crear datasets
    train_dataset = srgan.create_dataset(args.train_lr_dir, args.train_hr_dir, args.batch_size)
    val_dataset = srgan.create_dataset(args.val_lr_dir, args.val_hr_dir, args.batch_size)
    
    # Entrenar modelo
    srgan.train(train_dataset, val_dataset, args.epochs, args.pretrain_epochs)
    
    # Evaluación final
    final_metrics = srgan.evaluate_model(val_dataset)
    print("\nMétricas finales en validación:")
    print(f"PSNR: {final_metrics['psnr']:.2f} dB")
    print(f"SSIM: {final_metrics['ssim']:.4f}")

if __name__ == "__main__":
    main()