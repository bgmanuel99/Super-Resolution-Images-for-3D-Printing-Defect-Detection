import os
import sys
import datetime

from keras import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
)

class FineTunedVGG16:
    def __init__(self):
        self.model = None
        self.trained = False

    def setup_model(
            self,
            input_shape=(128, 128, 3),
            num_classes=2,
            train_last_n_layers=4,
            base_trainable=False,
            dropout_rate=0.2,
            l2_reg=0.0,
            learning_rate=1e-3,
            loss="sparse_categorical_crossentropy",
            from_pretrained=False,
            pretrained_path=None):
        """
            Set up the VGG16 classifier, either by loading a pretrained model 
            or building a new one.
        """
        
        if from_pretrained:
            if pretrained_path is None or not os.path.isfile(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained model file not found at {pretrained_path}"
                )
            self.model = load_model(pretrained_path)
            self.trained = True
            print(f"Loaded pretrained model from {pretrained_path}")
        else:
            self.build_vgg16(
                input_shape=input_shape,
                num_classes=num_classes,
                train_last_n_layers=train_last_n_layers,
                base_trainable=base_trainable,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
            )
            self.compile(learning_rate=learning_rate, loss=loss)

    def build_vgg16(
            self,
            input_shape=(128, 128, 3),
            num_classes=2,
            train_last_n_layers=4,
            base_trainable=False,
            dropout_rate=0.2,
            l2_reg=0.0):
        """Build a VGG16-based model with ImageNet weights"""
        
        assert input_shape[-1] == 3, "Input must have 3 channels (RGB)."

        base = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

        # Freeze all layers by default
        base.trainable = False

        # Optionally unfreeze last N layers
        if base_trainable and train_last_n_layers > 0:
            for layer in base.layers[-train_last_n_layers:]:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = True

        inputs = Input(shape=input_shape)
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D(name="gap")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        kernel_reg = l2(l2_reg) if l2_reg > 0 else None
        x = Dense(
            256, activation="relu", kernel_regularizer=kernel_reg
        )(x)
        x = Dropout(dropout_rate)(x) if dropout_rate > 0 else x
        outputs = Dense(
            num_classes, activation="softmax", name="predictions"
        )(x)
        self.model = Model(inputs, outputs, name="vgg16_finetune")

    def compile(
            self, 
            learning_rate=1e-3, 
            loss="sparse_categorical_crossentropy"):
        if self.model is None:
            raise ValueError("Model is not built yet.")
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer, loss=loss, metrics=["accuracy"]
        )
        self.model.summary()

    def fit(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=32,
            epochs=50,
            use_augmentation=True):
        if self.model is None:
            raise ValueError("Model is not built yet.")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,min_lr=1e-7, verbose=1)
        ]

        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
            )

            # Assume X_train and y_train are your training data and labels
            train_generator = datagen.flow(X_train, y_train, batch_size=32)

            history = self.model.fit(
                train_generator,
                steps_per_epoch=len(train_generator), 
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )
            
        self.trained = True
        
        return history

    def evaluate(self, X_test, y_test):
        if not self.trained:
            raise RuntimeError("Model has not been trained.")
        
        results = self.model.evaluate(X_test, y_test)
        print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        
        return results

    def classify_defects_method(
        self,
        image,
        patch_size=None,
        stride=None,
        batch_size=32,
    ):
        """Classify an image by extracting patches and aggregating predictions.

        Uses majority voting across patch predictions. The predicted class is
        the one with the most votes (argmax over per-patch probabilities).
        Confidence is the mean probability of the winning class across all
        patches. Ties are broken by the class with higher mean probability.

        Args:
            image: np.ndarray HxWxC (RGB). dtype uint8/[0,255] or float.
            patch_size: int size for square patches. Defaults to model input
                size if None.
            stride: int step between patch starts. Defaults to patch_size//2
                if None.
            batch_size: inference batch size for predict.

        Returns:
            (predicted_class: int, confidence: float)
        """

        if self.model is None:
            raise ValueError("Model is not built yet.")
        if image is None:
            raise ValueError("image must be provided")

        import numpy as np

        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB array")

        # Determine patch/input size
        _, in_h, in_w, in_c = self.model.input_shape
        if patch_size is None:
            if in_h is None or in_w is None:
                raise ValueError(
                    "Model input size is dynamic; please set patch_size."
                )
            patch_size = int(in_h)
        if stride is None:
            stride = max(1, patch_size // 2)

        def add_padding(image_arr, psize, st):
            h, w, _ = image_arr.shape
            pad_h = (psize - (h % st)) % st if (h % st) != 0 else 0
            pad_w = (psize - (w % st)) % st if (w % st) != 0 else 0
            pad_h = max(pad_h, psize - st)
            pad_w = max(pad_w, psize - st)
            if pad_h == 0 and pad_w == 0:
                return image_arr, (h, w)
            padded = np.pad(
                image_arr,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )
            return padded, (h, w)

        def extract_patches(image_arr, psize, st):
            h, w, _ = image_arr.shape
            patches = []
            for i in range(0, h - psize + 1, st):
                for j in range(0, w - psize + 1, st):
                    patches.append(
                        image_arr[i:i + psize, j:j + psize, :]
                    )
            return np.asarray(patches, dtype=np.float32)

        padded, _ = add_padding(img, patch_size, stride)
        patches = extract_patches(padded, patch_size, stride)

        probs = self.model.predict(
            patches, batch_size=batch_size, verbose=0
        )
        # Ensure 2D [N, num_classes]
        probs = np.asarray(probs)
        if probs.ndim != 2:
            probs = probs.reshape((probs.shape[0], -1))

        # Majority voting across patches
        num_classes = int(probs.shape[1])
        patch_preds = np.argmax(probs, axis=1)
        votes = np.bincount(patch_preds, minlength=num_classes)

        # Handle ties by choosing the class with highest mean probability
        top_vote = votes.max()
        top_classes = np.where(votes == top_vote)[0]
        if len(top_classes) == 1:
            winning_class = int(top_classes[0])
        else:
            mean_probs = probs.mean(axis=0)
            winning_class = int(
                top_classes[np.argmax(mean_probs[top_classes])]
            )

        confidence = float(probs[:, winning_class].mean())

        return winning_class, confidence

    def save(self, directory, timestamp):
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")
        
        os.makedirs(directory, exist_ok=True)
        
        path = os.path.join(directory, f"VGG16_{timestamp}.h5")
        
        self.model.save(path)
        
        print(f"Model saved to {path}")