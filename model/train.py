#!/usr/bin/env python3
"""Train, evaluate, and export the browser doodle classifier.

This is the single training entry point for the project. It:

1. creates deterministic, non-overlapping train/validation/test splits;
2. streams balanced samples from memory-mapped Quick Draw bitmap files;
3. trains with restrained geometric and stroke-width augmentation;
4. uses early stopping, learning-rate reduction, and best-weight restoration;
5. reports top-1/top-3 accuracy, confusion matrices, and per-class recall; and
6. exports the TensorFlow.js model and its ordered ``class_names.json`` contract.

Run from any directory with:

    python model/train.py

Required packages: tensorflow, tensorflowjs, numpy, and matplotlib. On Apple
Silicon, install the TensorFlow distribution appropriate for your environment.
"""

import csv
import json
import math
import os
import random
import shutil
import sys
import zlib
from pathlib import Path

if not (3, 10) <= sys.version_info[:2] <= (3, 11):
    raise RuntimeError(
        "This training environment requires Python 3.10 or 3.11. "
        f"Found Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Create a Python 3.11 virtual environment and install "
        "model/requirements.txt."
    )

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras import Model, layers


###############################################################################
# CONFIGURATION
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "quickdraw_data" / "numpy_bitmap"
MODEL_OUTPUT_DIRECTORY = PROJECT_ROOT / "public" / "tfjs_model"
TRAINING_OUTPUT_DIRECTORY = PROJECT_ROOT / "model" / "training_artifacts"
EVALUATION_DIRECTORY = TRAINING_OUTPUT_DIRECTORY / "evaluation"

SEED = 2026
IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 512
MAX_EPOCHS = 50

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
# The remaining 0.15 is held out for the final test set.

MAX_AVAILABLE_SAMPLES_PER_CLASS = 70_000
TRAIN_SAMPLES_PER_CLASS_PER_EPOCH = 12_000
VALIDATION_SAMPLES_PER_CLASS = 5_000
TEST_SAMPLES_PER_CLASS = 5_000

INITIAL_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 6
LR_REDUCTION_PATIENCE = 2
LR_REDUCTION_FACTOR = 0.5
MINIMUM_LEARNING_RATE = 1e-6

# The order is the semantic contract for the classifier's output neurons.
CATEGORIES = [
    "circle",
    "square",
    "triangle",
    "star",
    "line",
    "cup",
    "clock",
    "chair",
    "book",
    "laptop",
    "cell phone",
    "key",
    "umbrella",
    "car",
    "cat",
    "dog",
    "bird",
    "fish",
    "tree",
    "flower",
    "sun",
    "cloud",
    "eye",
    "hand",
    "face",
    "smiley face",
    "scissors",
    "pencil",
    "hammer",
    "guitar",
    "bicycle",
    "airplane",
    "sailboat",
    "apple",
    "banana",
    "pizza",
]


###############################################################################
# REPRODUCIBILITY
###############################################################################
def configure_reproducibility():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # TF_DETERMINISTIC_OPS still covers TensorFlow versions without this API.
        pass


def category_seed(category):
    """Return a stable seed that does not depend on Python's randomized hash."""
    return SEED + zlib.crc32(category.encode("utf-8"))


###############################################################################
# DATA SPLITS AND MEMORY-EFFICIENT INPUT PIPELINE
###############################################################################
class BitmapStore:
    """Open class files once and create deterministic, disjoint split indices."""

    def __init__(self, categories, data_directory, max_samples_per_class):
        self.categories = categories
        self.arrays = []
        self.split_indices = []

        for category in categories:
            path = data_directory / f"{category}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing Quick Draw data: {path}")

            array = np.load(path, mmap_mode="r")
            sample_count = min(len(array), max_samples_per_class)
            if sample_count < 3:
                raise ValueError(f"Not enough samples to split class '{category}'.")

            indices = np.arange(sample_count, dtype=np.int64)
            rng = np.random.default_rng(category_seed(category))
            rng.shuffle(indices)

            train_end = int(sample_count * TRAIN_FRACTION)
            validation_end = train_end + int(
                sample_count * VALIDATION_FRACTION,
            )
            splits = {
                "train": indices[:train_end],
                "validation": indices[train_end:validation_end],
                "test": indices[validation_end:],
            }

            if sum(len(split) for split in splits.values()) != sample_count:
                raise RuntimeError(f"Split accounting failed for '{category}'.")
            if any(len(split) == 0 for split in splits.values()):
                raise ValueError(f"A split is empty for class '{category}'.")

            self.arrays.append(array)
            self.split_indices.append(splits)

        self._print_summary()

    def _print_summary(self):
        first = self.split_indices[0]
        print(
            "Loaded "
            f"{len(self.categories)} classes with deterministic "
            f"{len(first['train'])}/{len(first['validation'])}/{len(first['test'])} "
            "train/validation/test samples per class."
        )


class BalancedBitmapSequence(tf.keras.utils.Sequence):
    """Yield balanced batches without loading the complete dataset into memory."""

    def __init__(
        self,
        store,
        split,
        batch_size,
        samples_per_class=None,
        shuffle=False,
    ):
        super().__init__()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.store = store
        self.split = split
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.shuffle = shuffle
        self.epoch = 0
        self.labels = np.empty(0, dtype=np.int32)
        self.sample_indices = np.empty(0, dtype=np.int64)
        self._rebuild_references()

    def _select_epoch_indices(self, class_index):
        available = self.store.split_indices[class_index][self.split]
        requested = self.samples_per_class
        if requested is None or requested >= len(available):
            return available.copy()

        if not self.shuffle:
            return available[:requested].copy()

        # A deterministic rolling window gives broad coverage across epochs and
        # avoids repeatedly drawing an unrelated random subset.
        offset = (self.epoch * requested) % len(available)
        end = offset + requested
        if end <= len(available):
            return available[offset:end].copy()
        return np.concatenate((available[offset:], available[: end - len(available)]))

    def _rebuild_references(self):
        label_parts = []
        index_parts = []

        for class_index in range(len(self.store.categories)):
            selected = self._select_epoch_indices(class_index)
            label_parts.append(
                np.full(len(selected), class_index, dtype=np.int32),
            )
            index_parts.append(selected)

        labels = np.concatenate(label_parts)
        sample_indices = np.concatenate(index_parts)

        if self.shuffle:
            rng = np.random.default_rng(SEED + 10_000 + self.epoch)
            permutation = rng.permutation(len(labels))
            labels = labels[permutation]
            sample_indices = sample_indices[permutation]

        self.labels = labels
        self.sample_indices = sample_indices

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, len(self.labels))
        batch_labels = self.labels[start:end]
        batch_sample_indices = self.sample_indices[start:end]
        batch_images = np.empty(
            (len(batch_labels), IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            dtype=np.float32,
        )

        for class_index in np.unique(batch_labels):
            positions = np.flatnonzero(batch_labels == class_index)
            source_indices = batch_sample_indices[positions]
            bitmaps = self.store.arrays[class_index][source_indices]
            batch_images[positions, :, :, 0] = bitmaps.reshape(
                -1,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ).astype(np.float32) / 255.0

        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            self.epoch += 1
            self._rebuild_references()


###############################################################################
# AUGMENTATION AND MODEL
###############################################################################
class StrokeWidthAugmentation(layers.Layer):
    """Randomly dilate or erode a small share of training doodles by one pixel."""

    def __init__(self, dilation_probability=0.12, erosion_probability=0.08, **kwargs):
        super().__init__(**kwargs)
        self.dilation_probability = dilation_probability
        self.erosion_probability = erosion_probability

    def call(self, inputs, training=None):
        if not training:
            return inputs

        dilated = tf.nn.max_pool2d(inputs, ksize=3, strides=1, padding="SAME")
        eroded = -tf.nn.max_pool2d(-inputs, ksize=3, strides=1, padding="SAME")
        choices = tf.random.uniform(
            [tf.shape(inputs)[0], 1, 1, 1],
            seed=SEED + 4,
        )
        augmented = tf.where(
            choices < self.dilation_probability,
            dilated,
            inputs,
        )
        augmented = tf.where(
            choices
            > 1.0 - self.erosion_probability,
            eroded,
            augmented,
        )
        return tf.clip_by_value(augmented, 0.0, 1.0)

    def get_config(self):
        return {
            **super().get_config(),
            "dilation_probability": self.dilation_probability,
            "erosion_probability": self.erosion_probability,
        }


def build_augmentation_pipeline():
    return tf.keras.Sequential(
        [
            layers.RandomTranslation(
                height_factor=0.06,
                width_factor=0.06,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="bilinear",
                seed=SEED + 1,
            ),
            # Keras rotation factors are fractions of a full turn: 8 / 360
            # produces the intended restrained range of approximately +/-8°.
            layers.RandomRotation(
                factor=8 / 360,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="bilinear",
                seed=SEED + 2,
            ),
            layers.RandomZoom(
                height_factor=(-0.08, 0.08),
                width_factor=(-0.08, 0.08),
                fill_mode="constant",
                fill_value=0.0,
                interpolation="bilinear",
                seed=SEED + 3,
            ),
            StrokeWidthAugmentation(name="stroke_width_augmentation"),
        ],
        name="training_augmentation",
    )


def convolution_block(inputs, filters, name, pool=True):
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv_1",
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn_1")(x)
    x = layers.Activation("relu", name=f"{name}_relu_1")(x)
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv_2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn_2")(x)
    x = layers.Activation("relu", name=f"{name}_relu_2")(x)
    if pool:
        x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return x


def build_classifier(num_classes):
    """A compact three-block CNN with more capacity than the original model."""
    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
        name="bitmap_input",
    )
    x = convolution_block(inputs, 48, "block_1")
    x = convolution_block(x, 96, "block_2")
    x = convolution_block(x, 192, "block_3", pool=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(128, activation="relu", name="embedding")(x)
    x = layers.Dropout(0.35, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, name="class_logits")(x)
    return Model(inputs, outputs, name="doodle_classifier")


def build_training_model(classifier):
    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
        name="training_bitmap_input",
    )
    augmented = build_augmentation_pipeline()(inputs)
    outputs = classifier(augmented)
    return Model(inputs, outputs, name="augmented_doodle_classifier")


###############################################################################
# EVALUATION REPORTS
###############################################################################
def save_matrix_csv(path, matrix):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual / predicted", *CATEGORIES])
        for category, row in zip(CATEGORIES, matrix):
            writer.writerow([category, *row])


def collect_predictions(classifier, sequence):
    labels = []
    logits = []
    for batch_index in range(len(sequence)):
        batch_images, batch_labels = sequence[batch_index]
        batch_logits = classifier.predict_on_batch(batch_images)
        labels.append(batch_labels)
        logits.append(np.asarray(batch_logits))
    return np.concatenate(labels), np.concatenate(logits)


def create_evaluation_reports(classifier, test_sequence, test_metrics):
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    true_labels, logits = collect_predictions(classifier, test_sequence)
    predicted_labels = np.argmax(logits, axis=1)

    confusion = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=np.int64)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(len(CATEGORIES), dtype=np.float64),
        where=support > 0,
    )
    normalized_confusion = np.divide(
        confusion,
        support[:, None],
        out=np.zeros_like(confusion, dtype=np.float64),
        where=support[:, None] > 0,
    )

    save_matrix_csv(EVALUATION_DIRECTORY / "confusion_matrix.csv", confusion)
    save_matrix_csv(
        EVALUATION_DIRECTORY / "confusion_matrix_normalized.csv",
        normalized_confusion,
    )

    with (EVALUATION_DIRECTORY / "per_class_recall.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class", "support", "recall"])
        for category, class_support, class_recall in zip(
            CATEGORIES,
            support,
            recall,
        ):
            writer.writerow([category, int(class_support), float(class_recall)])

    serializable_metrics = {
        key: float(value) for key, value in test_metrics.items()
    }
    serializable_metrics["macro_recall"] = float(np.mean(recall))
    with (EVALUATION_DIRECTORY / "test_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(serializable_metrics, file, indent=2)
        file.write("\n")

    plot_confusion_matrices(confusion, normalized_confusion)
    plot_per_class_recall(recall)

    print("\nTest metrics")
    for key, value in serializable_metrics.items():
        print(f"  {key}: {value:.4f}")


def plot_confusion_matrices(confusion, normalized_confusion):
    figure, axes = plt.subplots(1, 2, figsize=(24, 10), constrained_layout=True)
    for axis, matrix, title, maximum in (
        (axes[0], confusion, "Confusion matrix (counts)", None),
        (axes[1], normalized_confusion, "Confusion matrix (row-normalized)", 1.0),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=90, fontsize=7)
        axis.set_yticks(range(len(CATEGORIES)), CATEGORIES, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.savefig(EVALUATION_DIRECTORY / "confusion_matrices.png", dpi=180)
    plt.close(figure)


def plot_per_class_recall(recall):
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.bar(CATEGORIES, recall, color="#6046ff")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall")
    axis.set_title("Per-class recall")
    axis.tick_params(axis="x", rotation=75, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(EVALUATION_DIRECTORY / "per_class_recall.png", dpi=180)
    plt.close(figure)


###############################################################################
# EXPORT
###############################################################################
def export_model(classifier):
    staging_directory = TRAINING_OUTPUT_DIRECTORY / "tfjs_export"
    if staging_directory.exists():
        shutil.rmtree(staging_directory)
    staging_directory.mkdir(parents=True)

    # Convert completely before replacing the currently deployable model. A
    # converter failure therefore leaves the previous browser model intact.
    tfjs.converters.save_keras_model(classifier, str(staging_directory))
    contract = {
        "schemaVersion": 1,
        "classNames": CATEGORIES,
        "input": {
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "channels": CHANNELS,
            "range": [0, 1],
            "foreground": 1,
            "background": 0,
            "source": "full-canvas",
        },
    }
    with (staging_directory / "class_names.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(contract, file, indent=2)
        file.write("\n")

    MODEL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for pattern in ("model.json", "group*-shard*of*.bin", "class_names.json"):
        for path in MODEL_OUTPUT_DIRECTORY.glob(pattern):
            path.unlink()
    for artifact in staging_directory.iterdir():
        shutil.copy2(artifact, MODEL_OUTPUT_DIRECTORY / artifact.name)

    print(f"Exported TensorFlow.js model to {MODEL_OUTPUT_DIRECTORY}")


###############################################################################
# TRAINING ENTRY POINT
###############################################################################
def main():
    configure_reproducibility()
    TRAINING_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    store = BitmapStore(
        CATEGORIES,
        DATA_DIRECTORY,
        MAX_AVAILABLE_SAMPLES_PER_CLASS,
    )
    train_sequence = BalancedBitmapSequence(
        store,
        split="train",
        batch_size=BATCH_SIZE,
        samples_per_class=TRAIN_SAMPLES_PER_CLASS_PER_EPOCH,
        shuffle=True,
    )
    validation_sequence = BalancedBitmapSequence(
        store,
        split="validation",
        batch_size=BATCH_SIZE,
        samples_per_class=VALIDATION_SAMPLES_PER_CLASS,
    )
    test_sequence = BalancedBitmapSequence(
        store,
        split="test",
        batch_size=BATCH_SIZE,
        samples_per_class=TEST_SAMPLES_PER_CLASS,
    )

    print(
        f"Per epoch: {len(train_sequence.labels):,} train, "
        f"{len(validation_sequence.labels):,} validation, and "
        f"{len(test_sequence.labels):,} held-out test samples."
    )

    classifier = build_classifier(len(CATEGORIES))
    training_model = build_training_model(classifier)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=INITIAL_LEARNING_RATE,
    )
    training_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=3,
                name="top_3_accuracy",
            ),
        ],
    )
    classifier.summary()

    best_weights_path = TRAINING_OUTPUT_DIRECTORY / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_weights_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=LR_REDUCTION_FACTOR,
            patience=LR_REDUCTION_PATIENCE,
            min_lr=MINIMUM_LEARNING_RATE,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            TRAINING_OUTPUT_DIRECTORY / "training_history.csv",
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = training_model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint is the source of truth even if callback ordering changes.
    training_model.load_weights(best_weights_path)
    with (TRAINING_OUTPUT_DIRECTORY / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            file,
            indent=2,
        )
        file.write("\n")

    test_metrics = training_model.evaluate(
        test_sequence,
        return_dict=True,
        verbose=1,
    )
    create_evaluation_reports(classifier, test_sequence, test_metrics)
    export_model(classifier)


if __name__ == "__main__":
    main()
