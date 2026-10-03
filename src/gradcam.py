"""
Grad-CAM explainability (Selvaraju et al., 2017).

Grad-CAM shows which regions of the retina most influenced a prediction:
the gradients of the chosen class score with respect to the last
convolutional feature maps are averaged into one weight per channel,
and the weighted sum of the maps gives a coarse heatmap.

Usage (save a Grad-CAM grid of test images for the report):
    python -m src.gradcam --exp-name effb0_full_weights --n 8
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import config
from src.model import BACKBONES


def build_gradcam_model(model: keras.Model, backbone: str) -> keras.Model:
    """Model that returns (last conv feature maps, stage probabilities)."""
    last_conv = BACKBONES[backbone][2]
    return keras.Model(model.inputs,
                       [model.get_layer(last_conv).output, model.get_layer("stage").output])


def gradcam_heatmap(grad_model: keras.Model, img: np.ndarray, class_idx: int = None):
    """
    Compute a Grad-CAM heatmap for one preprocessed image.

    Args:
        img: (H, W, 3) array in 0-255, already preprocessed.
        class_idx: stage to explain; None = the predicted stage.
    Returns:
        (heatmap in [0, 1] of the feature-map size, stage probabilities)
    """
    x = tf.convert_to_tensor(img[None].astype("float32"))
    with tf.GradientTape() as tape:
        feature_maps, probs = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(probs[0]))
        class_score = probs[:, class_idx]
    grads = tape.gradient(class_score, feature_maps)
    channel_weights = tf.reduce_mean(grads, axis=(0, 1, 2))        # importance of each channel
    cam = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)
    cam = tf.nn.relu(cam)                                           # keep positive evidence only
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), probs[0].numpy()


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resize the heatmap to the image, colour it (jet) and blend it on top."""
    h = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colour = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * img + alpha * colour, 0, 255))


def save_gradcam_grid(model, backbone: str, df, image_dir: Path, out_path: Path, n: int = 8) -> Path:
    """Save image / heatmap pairs for `n` images (half correct, half wrong if possible)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grad_model = build_gradcam_model(model, backbone)
    rows = []
    for _, r in df.iterrows():
        img = cv2.cvtColor(cv2.imread(str(Path(image_dir) / f"{r['id_code']}.png")), cv2.COLOR_BGR2RGB)
        heat, probs = gradcam_heatmap(grad_model, img)
        rows.append((img, heat, int(r["stage"]), int(probs.argmax()), float(probs.max())))
    # rows: (image, heatmap, true stage, predicted stage, confidence)
    correct = [i for i, x in enumerate(rows) if x[2] == x[3]]
    wrong = [i for i, x in enumerate(rows) if x[2] != x[3]]
    picked = correct[: n // 2] + wrong[: n - n // 2]
    # If one group (correct/wrong) ran short, fill the remaining slots with any other images
    picked += [i for i in range(len(rows)) if i not in picked][: n - len(picked)]
    chosen = [rows[i] for i in picked]

    fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)), squeeze=False)
    for ax_row, (img, heat, t, p, conf) in zip(axes, chosen):
        ax_row[0].imshow(img)
        ax_row[0].set_title(f"True: {config.CLASS_NAMES[t]}", fontsize=9)
        ax_row[1].imshow(overlay_heatmap(img, heat))
        ax_row[1].set_title(f"Pred: {config.CLASS_NAMES[p]} ({conf:.0%})", fontsize=9)
        for a in ax_row:
            a.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    from src.dataset import prepare_splits
    from src.model import load_trained_model
    from src.preprocessing import cache_dataset

    p = argparse.ArgumentParser(description="Grad-CAM figure for a trained run")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    run_dir = config.MODELS_DIR / args.exp_name
    model, cfg = load_trained_model(run_dir)
    _, _, test_df = prepare_splits()
    sample = pd.concat([g.sample(min(len(g), 6), random_state=config.SEED)
                        for _, g in test_df.groupby("stage")])
    image_dir = cache_dataset(sample["id_code"], cfg["preprocess"], cfg["img_size"])
    out = save_gradcam_grid(model, cfg["backbone"], sample.reset_index(drop=True), image_dir,
                            run_dir / "gradcam_examples.png", args.n)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
