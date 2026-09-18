import datetime
import os

import numpy as np

from keras import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Input, Layer
)

# Keras 2.10 only exposes the serialization registry under the tensorflow
# namespace; 'keras.utils.register_keras_serializable' does not exist here.
from tensorflow.keras.utils import register_keras_serializable

from srlib.progress import stage
from srlib.constants import TIMESTAMP_FORMAT
from srlib.dataset.loading import add_padding

# 'model_registry' owns the on-disk layout of a run, and it is also what
# the pipeline uses to find these checkpoints again.
from srlib.model_registry import (
    prepare_run_directory,
    save_epoch_log,
    save_model_summary,
    save_run_metrics,
)

# Registering the layer lets 'load_model' rebuild it from a saved '.h5'
# without the caller passing 'custom_objects'.
@register_keras_serializable(package="srlib")
class VGG16Preprocessing(Layer):
    """
    Apply the canonical ImageNet preprocessing expected by VGG16.

    The pretrained weights were fitted on BGR inputs with the channel means
    subtracted, while the loaders deliver RGB in ``[0, 1]``. Converting
    inside the graph is what keeps training and inference in agreement,
    since the transform then travels with the saved model.

    Notes
    -----
    The layer expects inputs already scaled to ``[0, 1]``. It has no
    weights and does not change the tensor shape.
    """

    def call(self, inputs):
        return preprocess_input(inputs * 255.0)

    def compute_output_shape(self, input_shape):
        return input_shape

class FineTunedVGG16:
    def __init__(self):
        self.model = None
        self.trained = False
        # Remembered from setup for the fine-tuning phase.
        self.train_last_n_layers = 0
        self.loss = None

    def setup_model(
            self,
            input_shape=(128, 128, 3),
            num_classes=2,
            train_last_n_layers=4,
            dropout_rate=0.2,
            l2_reg=0.0,
            learning_rate=1e-3,
            loss="sparse_categorical_crossentropy",
            from_pretrained=False,
            pretrained_path=None):
        """
        Set up the VGG16 classifier, either by loading a pretrained model
        or building a new one.

        A freshly built model starts with the whole backbone frozen, which
        is what the first training phase needs. The number of layers to
        open later is stored rather than applied now.

        Parameters
        ----------
        train_last_n_layers : int
            How many of the backbone's final layers the fine-tuning phase
            will unfreeze. Not applied during setup.
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
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
            )
            self.compile(learning_rate=learning_rate, loss=loss)

        self.train_last_n_layers = int(train_last_n_layers)
        self.loss = loss

    def build_vgg16(
            self,
            input_shape=(128, 128, 3),
            num_classes=2,
            dropout_rate=0.2,
            l2_reg=0.0):
        """
        Build a VGG16-based model with ImageNet weights.

        The backbone is left fully frozen: only the new classification head
        is trainable. Call ``unfreeze_top_layers`` to move on to the
        fine-tuning phase.
        """

        assert input_shape[-1] == 3, "Input must have 3 channels (RGB)."

        base = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

        base.trainable = False

        inputs = Input(shape=input_shape)
        # Maps the input onto the distribution the ImageNet weights expect.
        x = VGG16Preprocessing(name="vgg16_preprocess")(inputs)
        x = base(x)
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
        self.loss = loss
        self.model.summary()

    def get_base_model(self):
        """
        Return the nested VGG16 backbone.

        It is a model used as a layer, so its ``trainable`` flag gates every
        layer inside it regardless of their individual flags.
        """

        if self.model is None:
            raise ValueError("Model is not built yet.")

        nested = [
            layer for layer in self.model.layers if isinstance(layer, Model)
        ]
        if len(nested) != 1:
            raise RuntimeError(
                "Expected exactly one nested backbone model, found "
                f"{len(nested)}."
            )

        return nested[0]

    def count_parameters(self):
        """
        Return the ``(trainable, frozen)`` parameter counts of the model.

        Returns
        -------
        tuple of int
            Number of trainable and of frozen parameters.
        """

        if self.model is None:
            raise ValueError("Model is not built yet.")

        trainable = int(
            sum(np.prod(w.shape) for w in self.model.trainable_weights)
        )
        frozen = int(
            sum(np.prod(w.shape) for w in self.model.non_trainable_weights)
        )

        return trainable, frozen

    def unfreeze_top_layers(
            self,
            train_last_n_layers=None,
            learning_rate=1e-5,
            loss=None):
        """
        Open the last N backbone layers for fine-tuning and recompile.

        Order matters: the container's ``trainable`` flag overrides those of
        the layers inside it, so the backbone is opened first and the layers
        that must stay fixed are frozen afterwards. The recompile is not
        optional either, since ``fit`` would otherwise reuse the train
        function traced from the previous variable set and keep ignoring the
        newly opened weights.

        Parameters
        ----------
        train_last_n_layers : int, optional
            Layers to unfreeze. Defaults to the value given to
            ``setup_model``.
        learning_rate : float
            Fine-tuning learning rate. Must stay well below the one used for
            the head, since the pretrained filters only need small
            corrections.
        loss : str, optional
            Loss to recompile with. Defaults to the one already in use.

        Returns
        -------
        tuple of int
            Parameter counts ``(trainable, frozen)`` after unfreezing.
        """

        base = self.get_base_model()

        if train_last_n_layers is None:
            train_last_n_layers = self.train_last_n_layers
        train_last_n_layers = int(train_last_n_layers)

        if train_last_n_layers <= 0:
            raise ValueError(
                "train_last_n_layers must be positive to fine-tune, got "
                f"{train_last_n_layers}."
            )
        if train_last_n_layers > len(base.layers):
            raise ValueError(
                f"Cannot unfreeze {train_last_n_layers} layers: the backbone "
                f"only has {len(base.layers)}."
            )

        base.trainable = True
        for layer in base.layers[:-train_last_n_layers]:
            layer.trainable = False

        self.compile(
            learning_rate=learning_rate,
            loss=loss if loss is not None else self.loss,
        )

        return self.count_parameters()

    def fit(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=32,
            epochs=50,
            use_augmentation=True,
            patience=10):
        """
        Train the model as currently compiled, for a single phase.

        Parameters
        ----------
        patience : int
            Epochs without validation improvement before stopping early.
            Fine-tuning improves more slowly than head training and needs a
            larger value to avoid being cut off prematurely.
        """

        if self.model is None:
            raise ValueError("Model is not built yet.")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                # Half the early-stopping patience, so the rate can drop a
                # couple of times before the run is cut.
                patience=max(1, patience // 2),
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
            )

            train_generator = datagen.flow(
                X_train, y_train, batch_size=batch_size
            )

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

    def fit_two_phases(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=32,
            head_epochs=150,
            finetune_epochs=150,
            head_patience=10,
            finetune_patience=15,
            finetune_learning_rate=1e-5,
            train_last_n_layers=None,
            use_augmentation=True):
        """
        Train the classifier in the two phases fine-tuning requires.

        Phase 1 trains only the new classification head, with the backbone
        frozen. Phase 2 opens the last N backbone layers and continues at a
        much lower learning rate.

        The split is what makes the pretrained weights survive: the large
        initial gradients of a head starting from random values would
        overwrite the ImageNet features if the convolutions were open.

        Parameters
        ----------
        finetune_learning_rate : float
            Learning rate for phase 2, deliberately far below the head's.
        train_last_n_layers : int, optional
            Layers to unfreeze in phase 2. Defaults to the value given to
            ``setup_model``.

        Returns
        -------
        tuple
            The ``(head_history, finetune_history)`` pair.
        """

        if self.model is None:
            raise ValueError("Model is not built yet.")

        trainable, frozen = self.count_parameters()
        print(
            f"\nPhase 1/2 - training the head only "
            f"({trainable:,} trainable, {frozen:,} frozen)"
        )
        head_history = self.fit(
            X_train, y_train, X_val, y_val,
            batch_size=batch_size,
            epochs=head_epochs,
            use_augmentation=use_augmentation,
            patience=head_patience,
        )

        trainable, frozen = self.unfreeze_top_layers(
            train_last_n_layers=train_last_n_layers,
            learning_rate=finetune_learning_rate,
        )
        print(
            f"\nPhase 2/2 - fine-tuning the last "
            f"{train_last_n_layers or self.train_last_n_layers} backbone "
            f"layers at lr={finetune_learning_rate:g} "
            f"({trainable:,} trainable, {frozen:,} frozen)"
        )
        finetune_history = self.fit(
            X_train, y_train, X_val, y_val,
            batch_size=batch_size,
            epochs=finetune_epochs,
            use_augmentation=use_augmentation,
            patience=finetune_patience,
        )

        return head_history, finetune_history

    def evaluate(self, X_test, y_test):
        if not self.trained:
            raise RuntimeError("Model has not been trained.")
        
        results = self.model.evaluate(X_test, y_test)
        print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        
        return results

    def evaluate_and_save(
            self, X_test, y_test, head_history, finetune_history,
            timestamp=None):
        """Evaluate the run, then persist the model and its metrics.

        The two phases are concatenated into one curve per metric, so the
        training plot shows a single run with the unfreeze marked on it.

        Parameters
        ----------
        X_test, y_test : np.ndarray
            Test partition of the classifier.
        head_history, finetune_history : keras.callbacks.History
            The pair returned by ``fit_two_phases``.
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
        run_name = f"VGG16_{timestamp}"

        def joined(key):
            return (
                list(head_history.history.get(key, []))
                + list(finetune_history.history.get(key, []))
            )

        with stage(f"VGG16 run {timestamp}") as step:
            step("evaluating on the test partition")
            results = self.evaluate(X_test, y_test)
            trainable, frozen = self.count_parameters()

            metrics = {
                "eval_loss": float(results[0]),
                "eval_accuracy": float(results[1]),
                "final_train_loss": joined("loss"),
                "final_val_loss": joined("val_loss"),
                "final_train_accuracy": joined("accuracy"),
                "final_val_accuracy": joined("val_accuracy"),
                "head_epochs_run": len(head_history.history.get("loss", [])),
                "trainable_params": trainable,
                "frozen_params": frozen,
            }

            run_dir = prepare_run_directory("VGG16", run_name)
            self.save(directory=run_dir, timestamp=timestamp)
            step(f"metrics    -> {save_run_metrics(run_dir, run_name, metrics)}")
            step(f"summary    -> {save_model_summary(run_dir, run_name, self.model)}")

            # The two phases are kept apart here, unlike in the metrics,
            # because the epoch numbering restarts at the unfreeze.
            step("epochs     -> " + save_epoch_log(
                run_dir, run_name,
                {"Phase 1/2 - head only": head_history.history,
                 "Phase 2/2 - backbone fine-tuning": finetune_history.history},
            ))

        return timestamp, run_dir, metrics

    def classify_defects_method(
        self,
        image,
        patch_size=None,
        stride=None,
        batch_size=32,
    ):
        """Classify an image by extracting patches and aggregating predictions.

        Uses majority voting across patch predictions: every patch votes for
        its own argmax and the winning class is the most voted one. Ties are
        broken by the class with higher mean probability. Confidence is the
        mean probability of the winning class across all patches.

        Args:
            image: np.ndarray HxWxC, RGB, float in [0, 1]. The ImageNet
                preprocessing lives inside the model, so an image on the
                [0, 255] scale is rejected instead of being scaled twice and
                yielding meaningless predictions. Mild overshoot from
                interpolation is clipped rather than rejected.
            patch_size: int size for square patches. Defaults to model input
                size if None.
            stride: int step between patch starts. Defaults to patch_size//2
                if None.
            batch_size: inference batch size for predict.

        Returns:
            (winning_class: int, confidence: float)
        """

        if self.model is None:
            raise ValueError("Model is not built yet.")
        if image is None:
            raise ValueError("image must be provided")

        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB array")

        img = img.astype(np.float32)

        # The in-graph preprocessing assumes the [0, 1] scale, so another
        # scale would produce quietly wrong output. Mild overshoot is fine
        # and gets clipped: kernels with negative lobes and the iterative
        # back-projection routinely return values just outside the range.
        low, high = float(img.min()), float(img.max())
        if low < -1.0 or high > 2.0:
            raise ValueError(
                "image must be RGB float in [0, 1], got range "
                f"[{low:.3f}, {high:.3f}]. Divide by 255 if the image is on "
                "the [0, 255] scale."
            )
        if low < 0.0 or high > 1.0:
            img = np.clip(img, 0.0, 1.0)

        _, in_h, in_w, _ = self.model.input_shape
        if patch_size is None:
            if in_h is None or in_w is None:
                raise ValueError(
                    "Model input size is dynamic; please set patch_size."
                )
            patch_size = int(in_h)
        if stride is None:
            stride = max(1, patch_size // 2)

        def extract_patches(image_arr, psize, st):
            h, w, _ = image_arr.shape
            patches = []
            for i in range(0, h - psize + 1, st):
                for j in range(0, w - psize + 1, st):
                    patches.append(
                        image_arr[i:i + psize, j:j + psize, :]
                    )
            return np.asarray(patches, dtype=np.float32)

        # Shared with the training loader: the majority vote is only valid if
        # the patches seen here come from the same grid the model was fit on.
        padded = add_padding(img, patch_size, stride)
        patches = extract_patches(padded, patch_size, stride)

        probs = self.model.predict(
            patches, batch_size=batch_size, verbose=0
        )
        probs = np.asarray(probs)
        if probs.ndim != 2:
            probs = probs.reshape((probs.shape[0], -1))

        num_classes = int(probs.shape[1])
        patch_preds = np.argmax(probs, axis=1)
        votes = np.bincount(patch_preds, minlength=num_classes)

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
        """Save the model under the run name the registry resolves."""

        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")
        
        os.makedirs(directory, exist_ok=True)
        
        path = os.path.join(directory, f"VGG16_{timestamp}.h5")
        
        self.model.save(path)
        
        print(f"Model saved to {path}")