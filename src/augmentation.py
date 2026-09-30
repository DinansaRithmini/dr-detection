"""
Data augmentation for the TRAINING set only.

Validation and test images are never augmented: they must represent real,
unseen data, otherwise the reported metrics would be optimistic (leakage).

Chosen transformations and why:
    RandomFlip (h + v)   - left/right eyes are mirror images; fundus cameras
                           can capture either orientation.
    RandomRotation       - a retina has no natural "up", so any angle is valid.
    RandomZoom (+-10%)   - mimics different fields of view / crop tightness.
    RandomBrightness     - mimics exposure differences between cameras.
    RandomContrast       - mimics differences in camera sensors and pupil dilation.

Deliberately avoided:
    Hue / strong colour shifts - lesion colour carries diagnostic meaning
                                 (red haemorrhages vs yellow exudates).
    Heavy shear / elastic warps - would distort the round anatomy unrealistically.
"""
from pathlib import Path

import numpy as np
from tensorflow import keras

import config


def build_augmenter(seed: int = config.SEED) -> keras.Sequential:
    """Return a Keras model that randomly augments a batch of 0-255 images."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            keras.layers.RandomRotation(factor=0.5, fill_mode="constant", fill_value=0.0, seed=seed),
            keras.layers.RandomZoom(height_factor=(-0.1, 0.1), fill_mode="constant", fill_value=0.0, seed=seed),
            keras.layers.RandomBrightness(factor=0.1, value_range=(0, 255), seed=seed),
            keras.layers.RandomContrast(factor=0.1, seed=seed),
        ],
        name="augmentation",
    )


def save_augmentation_figure(img: np.ndarray, n: int = 8,
                             out_path: Path = config.FIGURES_DIR / "augmentation_examples.png") -> Path:
    """Save the original image next to `n` random augmented versions of it."""
    import matplotlib.pyplot as plt

    augmenter = build_augmenter()
    batch = np.repeat(img[None].astype("float32"), n, axis=0)
    augmented = np.clip(augmenter(batch, training=True).numpy(), 0, 255).astype("uint8")

    cols = 3
    rows = int(np.ceil((n + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.ravel()
    axes[0].imshow(img)
    axes[0].set_title("Original")
    for i in range(n):
        axes[i + 1].imshow(augmented[i])
        axes[i + 1].set_title(f"Augmented {i + 1}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
