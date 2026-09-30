"""
CNN model with transfer learning and two output heads.

Architecture:
    input (0-255 RGB)
      -> backbone-specific normalisation
      -> pretrained CNN backbone (ImageNet weights, no top)
      -> GlobalAveragePooling2D -> Dropout
      -> Dense(256, ReLU, L2) -> Dropout
      -> head "stage": Dense(5, softmax)  - DR stage 0-4
      -> head "dr":    Dense(1, sigmoid)  - DR present / absent

Two heads let the network learn "is there disease?" and "how severe?"
together, which is exactly what the coursework brief asks for.
"""
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

import config

# name -> (keras application class, preprocess function, last conv layer for Grad-CAM)
BACKBONES = {
    "efficientnetb0": (keras.applications.EfficientNetB0,
                       keras.applications.efficientnet.preprocess_input, "top_activation"),
    "efficientnetb3": (keras.applications.EfficientNetB3,
                       keras.applications.efficientnet.preprocess_input, "top_activation"),
    "resnet50":       (keras.applications.ResNet50,
                       keras.applications.resnet50.preprocess_input, "conv5_block3_out"),
    "mobilenetv2":    (keras.applications.MobileNetV2,
                       keras.applications.mobilenet_v2.preprocess_input, "out_relu"),
}


def build_model(backbone: str = config.BACKBONE, img_size: int = config.IMG_SIZE,
                dropout: float = config.DROPOUT, weights=config.PRETRAINED_WEIGHTS
                ) -> Tuple[keras.Model, keras.Model]:
    """
    Build the two-head transfer-learning model.

    Returns (model, base) - `base` is the backbone so its layers can be
    frozen / unfrozen between the two training stages.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(BACKBONES)}")
    app_cls, preprocess_fn, _ = BACKBONES[backbone]

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = keras.layers.Lambda(preprocess_fn, name="normalise")(inputs)
    # input_tensor keeps the backbone layers in the same graph, which makes
    # Grad-CAM (reading the last conv layer) straightforward.
    base = app_cls(include_top=False, weights=weights, input_tensor=x)

    reg = keras.regularizers.l2(config.L2_WEIGHT_DECAY)
    h = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    h = keras.layers.Dropout(dropout, name="dropout_1")(h)
    h = keras.layers.Dense(256, activation="relu", kernel_regularizer=reg, name="dense_1")(h)
    h = keras.layers.Dropout(dropout, name="dropout_2")(h)
    stage_out = keras.layers.Dense(config.NUM_CLASSES, activation="softmax", name="stage")(h)
    dr_out = keras.layers.Dense(1, activation="sigmoid", name="dr")(h)

    model = keras.Model(inputs, {"stage": stage_out, "dr": dr_out}, name=f"dr_{backbone}")
    return model, base


def freeze_backbone(base: keras.Model) -> None:
    """Stage 1: freeze all backbone weights so only the new head is trained."""
    base.trainable = False


def unfreeze_top_layers(base: keras.Model, n_layers: int = config.UNFREEZE_LAYERS) -> None:
    """
    Stage 2: unfreeze the last `n_layers` of the backbone for fine-tuning.

    BatchNormalization layers stay frozen: with small batches their running
    statistics would be overwritten by noisy estimates and wreck the
    pretrained features.
    """
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def count_trainable(model: keras.Model) -> int:
    """Number of trainable parameters (reported for each training stage)."""
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))


# ---------------------------------------------------------------------------
# Loss functions for class imbalance
# ---------------------------------------------------------------------------
def weighted_categorical_crossentropy(class_weights):
    """Cross-entropy where each sample is scaled by the weight of its true class."""
    w = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return ce * tf.reduce_sum(y_true * w, axis=-1)
    return loss


def categorical_focal_loss(class_weights=None, gamma: float = config.FOCAL_GAMMA):
    """
    Focal loss (Lin et al., 2017): (1 - p_t)^gamma down-weights easy examples
    so training concentrates on hard, often minority-class, images.
    """
    w = tf.constant(class_weights if class_weights is not None
                    else [1.0] * config.NUM_CLASSES, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        alpha = tf.reduce_sum(y_true * w, axis=-1)
        return -alpha * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
    return loss


def compile_model(model: keras.Model, lr: float, balance: str, class_weights=None) -> None:
    """Compile with a stage loss that matches the chosen balancing method."""
    if balance == "weights":
        stage_loss = weighted_categorical_crossentropy(class_weights)
    elif balance == "focal":
        stage_loss = categorical_focal_loss(class_weights)
    else:   # "none" or "oversample" (balance already handled by the data)
        stage_loss = keras.losses.CategoricalCrossentropy()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={"stage": stage_loss, "dr": keras.losses.BinaryCrossentropy()},
        loss_weights={"stage": 1.0, "dr": config.BINARY_LOSS_WEIGHT},
        metrics={"stage": [keras.metrics.CategoricalAccuracy(name="accuracy")],
                 "dr": [keras.metrics.BinaryAccuracy(name="accuracy"),
                        keras.metrics.AUC(name="auc")]},
    )


def load_trained_model(run_dir, weights_file: str = "best.weights.h5") -> Tuple[keras.Model, dict]:
    """
    Rebuild a model from its saved run_config.json and load its weights.

    Saving weights + config (instead of a full model file) avoids problems
    deserialising custom losses and Lambda layers across Keras versions.
    """
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model, _ = build_model(cfg["backbone"], cfg["img_size"], cfg["dropout"], weights=None)
    model.load_weights(run_dir / weights_file)
    return model, cfg
