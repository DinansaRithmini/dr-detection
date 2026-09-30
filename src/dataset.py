"""
Dataset handling: labels, duplicate removal, stratified splits, class balancing
and tf.data input pipelines.

Main entry point:
    prepare_splits()  -> (train_df, val_df, test_df), cached in data/splits/
    make_dataset(df)  -> tf.data.Dataset yielding (image, {"stage", "dr"})
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import config
from src.augmentation import build_augmenter


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def load_labels(csv_path: Path = config.LABELS_CSV) -> pd.DataFrame:
    """Load APTOS train.csv and add the binary DR label (0 = No DR, 1 = DR)."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"diagnosis": "stage"})
    df["dr"] = (df["stage"] > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Duplicate detection (prevents train/test leakage)
# ---------------------------------------------------------------------------
def compute_hashes(df: pd.DataFrame, image_dir: Path = config.RAW_IMAGE_DIR) -> np.ndarray:
    """
    Perceptual hash (pHash) of every image, as an (N, 64) boolean array.

    pHash is based on low-frequency image content, so re-saved, resized or
    slightly re-exposed copies of the same photograph get almost equal hashes.
    Results are cached because hashing 3,600 images takes a few minutes.
    """
    import imagehash
    from PIL import Image

    cache = config.SPLITS_DIR / "phash.npy"
    if cache.exists():
        hashes = np.load(cache)
        if len(hashes) == len(df):
            return hashes
    hashes = []
    for image_id in tqdm(df["id_code"], desc="Hashing images"):
        with Image.open(image_dir / f"{image_id}.png") as im:
            hashes.append(imagehash.phash(im).hash.flatten())
    hashes = np.array(hashes, dtype=bool)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, hashes)
    return hashes


def find_duplicate_groups(hashes: np.ndarray,
                          threshold: int = config.DUPLICATE_HASH_THRESHOLD) -> np.ndarray:
    """
    Group images whose hashes differ by <= `threshold` bits.

    Pairwise Hamming distances are computed with two matrix products, then a
    union-find joins duplicates transitively. Returns a group id per image.
    """
    h = hashes.astype(np.float32)
    # Hamming distance = number of positions where the bits differ
    dist = h @ (1 - h).T + (1 - h) @ h.T

    parent = np.arange(len(h))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows, cols = np.where(np.triu(dist <= threshold, k=1))
    for i, j in zip(rows, cols):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(h))])


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Keep one image per duplicate group.

    Groups whose copies carry DIFFERENT stage labels are dropped entirely,
    because we cannot know which grade is correct. Returns the cleaned frame
    and a small report dictionary for the write-up.
    """
    df = df.copy()
    df["group"] = find_duplicate_groups(compute_hashes(df))
    sizes = df.groupby("group")["id_code"].transform("size")
    n_labels = df.groupby("group")["stage"].transform("nunique")

    conflicting = df[(sizes > 1) & (n_labels > 1)]
    consistent = df[n_labels == 1].drop_duplicates("group", keep="first")

    report = {
        "original_images": len(df),
        "duplicate_groups": int((df.groupby("group").size() > 1).sum()),
        "images_in_duplicate_groups": int((sizes > 1).sum()),
        "conflicting_label_images_dropped": len(conflicting),
        "images_after_cleaning": len(consistent),
    }
    return consistent.drop(columns="group").reset_index(drop=True), report


def save_duplicate_examples(df: pd.DataFrame, n: int = 6,
                            out_path: Path = config.FIGURES_DIR / "duplicate_examples.png") -> Path:
    """
    Show `n` flagged duplicate groups side by side (with their labels) so you
    can confirm visually that they really are copies of the same photograph.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    df = df.copy()
    df["group"] = find_duplicate_groups(compute_hashes(df))
    groups = [g for _, g in df.groupby("group") if len(g) > 1][:n]
    if not groups:
        print("No duplicates found at the current threshold.")
        return out_path
    fig, axes = plt.subplots(len(groups), 2, figsize=(6, 3 * len(groups)), squeeze=False)
    for ax_row, g in zip(axes, groups):
        for ax, (_, r) in zip(ax_row, g.head(2).iterrows()):
            ax.imshow(Image.open(config.RAW_IMAGE_DIR / f"{r['id_code']}.png"))
            ax.set_title(f"{r['id_code']}  stage {r['stage']}", fontsize=8)
        for ax in ax_row:
            ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def prepare_splits(force: bool = False, dedupe: bool = True,
                   seed: int = config.SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create (or load cached) stratified train / validation / test splits.

    Splitting happens AFTER duplicate removal and BEFORE any augmentation or
    balancing, so no information from val/test leaks into training.
    """
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [config.SPLITS_DIR / f"{s}.csv" for s in ("train", "val", "test")]
    if not force and all(p.exists() for p in paths):
        return tuple(pd.read_csv(p) for p in paths)

    df = load_labels()
    if dedupe:
        df, report = remove_duplicates(df)
        pd.Series(report).to_csv(config.SPLITS_DIR / "duplicate_report.csv", header=["value"])
        print("Duplicate report:", report)

    train_df, temp_df = train_test_split(
        df, test_size=config.VAL_FRAC + config.TEST_FRAC,
        stratify=df["stage"], random_state=seed)
    rel_test = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["stage"], random_state=seed)

    for split, p in zip((train_df, val_df, test_df), paths):
        split.reset_index(drop=True).to_csv(p, index=False)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_summary(train_df, val_df, test_df) -> pd.DataFrame:
    """Per-class image counts for each split (a ready-made report table)."""
    table = pd.DataFrame({
        name: d["stage"].value_counts().sort_index()
        for name, d in (("train", train_df), ("val", val_df), ("test", test_df))
    }).fillna(0).astype(int)
    table.index = [config.CLASS_NAMES[i] for i in table.index]
    table.loc["Total"] = table.sum()
    return table


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------
def get_class_weights(train_df: pd.DataFrame) -> np.ndarray:
    """'Balanced' weights: w_c = N / (K * n_c), so rare classes count for more."""
    classes = np.arange(config.NUM_CLASSES)
    return compute_class_weight("balanced", classes=classes, y=train_df["stage"].values).astype("float32")


def oversample(train_df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """
    Repeat minority-class rows until every class matches the largest class.

    Because augmentation is random, each repeated copy is seen differently
    by the network, which reduces (but does not remove) overfitting risk.
    """
    max_n = train_df["stage"].value_counts().max()
    parts = [g.sample(max_n, replace=True, random_state=seed) if len(g) < max_n else g
             for _, g in train_df.groupby("stage")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------
def make_dataset(df: pd.DataFrame, image_dir: Path, training: bool,
                 batch_size: int = config.BATCH_SIZE,
                 seed: int = config.SEED) -> tf.data.Dataset:
    """
    Build a tf.data pipeline from cached preprocessed images.

    Yields (image, {"stage": one-hot (5,), "dr": (1,)}) batches.
    Images stay in the 0-255 range; the model normalises them itself.
    Augmentation is applied only when `training` is True.
    """
    paths = [str(Path(image_dir) / f"{i}.png") for i in df["id_code"]]
    stage = tf.one_hot(df["stage"].values, config.NUM_CLASSES)
    dr = df["dr"].values.astype("float32")[:, None]

    ds = tf.data.Dataset.from_tensor_slices((paths, {"stage": stage, "dr": dr}))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, labels):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        return tf.cast(img, tf.float32), labels

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def save_class_distribution_figure(dfs: dict, out_path: Path, title: str) -> Path:
    """Bar chart of class counts for one or more labelled DataFrames."""
    import matplotlib.pyplot as plt

    counts = pd.DataFrame({k: v["stage"].value_counts().sort_index() for k, v in dfs.items()}).fillna(0)
    counts.index = [config.CLASS_NAMES[i] for i in counts.index]
    ax = counts.plot(kind="bar", figsize=(8, 4.5), rot=0)
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    ax.set_ylabel("Number of images")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
