"""
Model evaluation and report figures.

Produces, for a given split:
    - classification report (precision / recall / F1 per stage)
    - confusion matrices (counts and row-normalised)
    - Quadratic Weighted Kappa (official APTOS metric - respects stage order)
    - one-vs-rest ROC curves + AUC per stage
    - binary DR / No-DR metrics: sensitivity, specificity, AUC, confusion matrix
    - a grid of misclassified images for error analysis

Stand-alone usage (re-evaluate a trained run, optionally with TTA and a speed test):
    python -m src.evaluate --exp-name effb0_full_weights --tta --benchmark
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)

import config


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_history(hist: pd.DataFrame, out_path: Path, switch_epoch: int = None) -> None:
    """Loss and stage-accuracy curves (train vs validation) across both stages."""
    panels = [("loss", "Total loss"), ("stage_accuracy", "Stage accuracy"), ("dr_auc", "DR AUC")]
    panels = [p for p in panels if p[0] in hist.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels):
        ax.plot(hist["epoch"], hist[key], marker="o", ms=3, label="Train")
        if f"val_{key}" in hist.columns:
            ax.plot(hist["epoch"], hist[f"val_{key}"], marker="o", ms=3, label="Validation")
        if switch_epoch:
            ax.axvline(switch_epoch + 0.5, color="grey", ls="--", lw=1, label="Start fine-tuning")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(model, df: pd.DataFrame, image_dir: Path, batch_size: int = config.BATCH_SIZE,
            tta: bool = False):
    """
    Return (stage_probs (N, 5), dr_probs (N,)).

    With `tta` (test-time augmentation) predictions are averaged over the
    original image and its horizontal, vertical and double flips.
    """
    from src.dataset import make_dataset
    import tensorflow as tf

    ds = make_dataset(df, image_dir, training=False, batch_size=batch_size)
    images = ds.map(lambda x, y: x)
    flips = [lambda x: x]
    if tta:
        flips += [lambda x: tf.image.flip_left_right(x), lambda x: tf.image.flip_up_down(x),
                  lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x))]
    stage_sum, dr_sum = 0, 0
    for f in flips:
        out = model.predict(images.map(f), verbose=0)
        stage_sum = stage_sum + out["stage"]
        dr_sum = dr_sum + out["dr"].ravel()
    return stage_sum / len(flips), dr_sum / len(flips)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, labels, out_path: Path, title: str, normalise: bool) -> None:
    """Heatmap of a confusion matrix (rows = true class, columns = predicted)."""
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalise else cm
    plt.figure(figsize=(1.3 * len(labels) + 2, 1.1 * len(labels) + 1.5))
    sns.heatmap(data, annot=True, fmt=".2f" if normalise else "d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.ylabel("True stage")
    plt.xlabel("Predicted stage")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_true: np.ndarray, stage_probs: np.ndarray, dr_true: np.ndarray,
             dr_probs: np.ndarray, out_path: Path) -> dict:
    """One-vs-rest ROC per stage, plus the binary DR head. Returns AUCs."""
    aucs = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for c, name in enumerate(config.CLASS_NAMES):
        y_c = (y_true == c).astype(int)
        if y_c.min() == y_c.max():          # class missing from this split
            continue
        fpr, tpr, _ = roc_curve(y_c, stage_probs[:, c])
        aucs[name] = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {aucs[name]:.3f})")
    axes[0].set_title("Stage classification (one-vs-rest)")
    if dr_true.min() != dr_true.max():
        fpr, tpr, _ = roc_curve(dr_true, dr_probs)
        aucs["DR vs No DR"] = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"DR head (AUC = {aucs['DR vs No DR']:.3f})")
    axes[1].set_title("Binary DR detection")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return aucs


def plot_misclassified(df: pd.DataFrame, y_true, y_pred, probs, image_dir: Path,
                       out_path: Path, n: int = 8) -> None:
    """Show the `n` most confident mistakes - the most informative errors."""
    import cv2

    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        img = cv2.cvtColor(cv2.imread(str(Path(image_dir) / f"{df.iloc[i]['id_code']}.png")), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"True: {config.CLASS_NAMES[y_true[i]]}\nPred: {config.CLASS_NAMES[y_pred[i]]} "
                     f"({probs[i, y_pred[i]]:.0%})", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full evaluation of one split
# ---------------------------------------------------------------------------
def evaluate_split(model, df: pd.DataFrame, image_dir: Path, run_dir: Path, split: str,
                   batch_size: int = config.BATCH_SIZE, tta: bool = False) -> dict:
    """Compute every metric and figure for `split`, save them in run_dir/<split>/."""
    out = Path(run_dir) / (split + ("_tta" if tta else ""))
    out.mkdir(parents=True, exist_ok=True)

    stage_probs, dr_probs = predict(model, df, image_dir, batch_size, tta)
    y_true = df["stage"].values
    y_pred = stage_probs.argmax(axis=1)
    dr_true = df["dr"].values
    dr_pred = (dr_probs >= 0.5).astype(int)
    dr_from_stage = (y_pred > 0).astype(int)     # binary result derived from the stage head

    labels = list(range(config.NUM_CLASSES))
    report = classification_report(y_true, y_pred, labels=labels, target_names=config.CLASS_NAMES,
                                   digits=3, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=labels, target_names=config.CLASS_NAMES,
                              digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion(cm, config.CLASS_NAMES, out / "confusion_matrix.png", f"Confusion matrix ({split})", False)
    plot_confusion(cm, config.CLASS_NAMES, out / "confusion_matrix_normalised.png",
                   f"Normalised confusion matrix ({split})", True)

    def sens_spec(true, pred):
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
        return tp / max(tp + fn, 1), tn / max(tn + fp, 1)

    sens, spec = sens_spec(dr_true, dr_pred)
    sens_s, spec_s = sens_spec(dr_true, dr_from_stage)
    plot_confusion(confusion_matrix(dr_true, dr_pred, labels=[0, 1]), config.BINARY_NAMES,
                   out / "confusion_matrix_binary.png", f"DR vs No DR ({split})", False)

    aucs = plot_roc(y_true, stage_probs, dr_true, dr_probs, out / "roc_curves.png")
    plot_misclassified(df, y_true, y_pred, stage_probs, image_dir, out / "misclassified.png")

    pd.DataFrame({"id_code": df["id_code"], "true_stage": y_true, "pred_stage": y_pred,
                  **{f"p_{c}": stage_probs[:, i] for i, c in enumerate(config.CLASS_NAMES)},
                  "true_dr": dr_true, "p_dr": dr_probs}).to_csv(out / "predictions.csv", index=False)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "stage_macro_auc": float(np.mean([v for k, v in aucs.items() if k in config.CLASS_NAMES]))
                           if aucs else float("nan"),
        "dr_accuracy": accuracy_score(dr_true, dr_pred),
        "dr_sensitivity": sens,
        "dr_specificity": spec,
        "dr_auc": roc_auc_score(dr_true, dr_probs) if dr_true.min() != dr_true.max() else float("nan"),
        "dr_from_stage_sensitivity": sens_s,
        "dr_from_stage_specificity": spec_s,
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def benchmark(model, img_size: int, n: int = 50) -> dict:
    """Average single-image inference time (for the deployment-feasibility discussion)."""
    x = np.random.randint(0, 255, (1, img_size, img_size, 3)).astype("float32")
    model.predict(x, verbose=0)                   # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(x, verbose=0)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"ms_per_image": round(ms, 1), "params": int(model.count_params()),
            "size_mb_float32": round(model.count_params() * 4 / 1e6, 1)}


def main():
    from src.dataset import prepare_splits
    from src.model import load_trained_model
    from src.preprocessing import cache_dataset

    p = argparse.ArgumentParser(description="Evaluate a trained run")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--tta", action="store_true", help="test-time augmentation (flips)")
    p.add_argument("--benchmark", action="store_true", help="measure inference speed")
    args = p.parse_args()

    run_dir = config.MODELS_DIR / args.exp_name
    model, cfg = load_trained_model(run_dir)
    _, val_df, test_df = prepare_splits()
    df = test_df if args.split == "test" else val_df
    image_dir = cache_dataset(df["id_code"], cfg["preprocess"], cfg["img_size"])

    metrics = evaluate_split(model, df, image_dir, run_dir, args.split, cfg["batch_size"], args.tta)
    print(json.dumps(metrics, indent=2))
    if args.benchmark:
        b = benchmark(model, cfg["img_size"])
        (run_dir / "benchmark.json").write_text(json.dumps(b, indent=2))
        print(json.dumps(b, indent=2))


if __name__ == "__main__":
    main()
