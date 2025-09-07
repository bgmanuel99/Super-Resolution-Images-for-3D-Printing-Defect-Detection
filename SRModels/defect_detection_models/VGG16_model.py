import os
import sys
import datetime

from keras import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

from SRModels.data_augmentation import AdvancedAugmentGenerator

class FineTunedVGG16:
    def __init__(self):
        self.model = None
        self.trained = False

    def setup_model(self,
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
        """Set up the VGG16 classifier, either by loading a pretrained model or building a new one."""
        if from_pretrained:
            if pretrained_path is None or not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"Pretrained model file not found at {pretrained_path}")
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

    def build_vgg16(self,
                    input_shape=(128, 128, 3),
                    num_classes=2,
                    train_last_n_layers=4,
                    base_trainable=False,
                    dropout_rate=0.2,
                    l2_reg=0.0):
        """
        Build a VGG16-based model with ImageNet weights and a custom classification head.
        """
        assert input_shape[-1] == 3, "Input must have 3 channels (RGB)."

        # Load VGG16 base with ImageNet weights and no top
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

        # Build head
        inputs = Input(shape=input_shape)
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D(name="gap")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        kernel_reg = l2(l2_reg) if l2_reg > 0 else None
        x = Dense(256, activation="relu", kernel_regularizer=kernel_reg)(x)
        x = Dropout(dropout_rate)(x) if dropout_rate > 0 else x

        outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
        self.model = Model(inputs, outputs, name="vgg16_finetune")

    def compile(self, learning_rate=1e-3, loss="sparse_categorical_crossentropy"):
        if self.model is None:
            raise ValueError("Model is not built yet.")
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        self.model.summary()

    def fit(self,
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=32,
            epochs=50,
            use_augmentation=True,
            use_mix=True,
            augment_validation=False):
        if self.model is None:
            raise ValueError("Model is not built yet.")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1)
        ]

        if use_augmentation:
            train_gen = AdvancedAugmentGenerator(X_train, y_train, batch_size=batch_size, shuffle=True, use_mix=use_mix)

            if augment_validation:
                val_gen = AdvancedAugmentGenerator(X_val, y_val, batch_size=batch_size, shuffle=False, use_mix=use_mix)
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
                    validation_data=(X_val, y_val),
                    callbacks=callbacks
                )
        else:
            self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )

        self.trained = True

    def evaluate(self, X_test, y_test):
        if not self.trained:
            raise RuntimeError("Model has not been trained.")
        results = self.model.evaluate(X_test, y_test)
        print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        return results

    def save(self, directory="models/VGG16"):
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")
        
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"VGG16_{timestamp}.h5")
        self.model.save(path)
        print(f"Model saved to {path}")