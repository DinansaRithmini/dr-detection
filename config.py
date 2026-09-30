"""
Central configuration for the Diabetic Retinopathy (DR) stage detection project.

Every script imports its settings from here, so an experiment can be reproduced
by knowing only this file and the command-line flags used for training.
"""
import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_IMAGE_DIR = DATA_DIR / "train_images"      # APTOS 2019 training images (.png)
LABELS_CSV = DATA_DIR / "train.csv"            # columns: id_code, diagnosis
PROCESSED_DIR = DATA_DIR / "processed"         # cached preprocessed images
SPLITS_DIR = DATA_DIR / "splits"               # train/val/test CSVs
MODELS_DIR = ROOT_DIR / "models"               # one sub-folder per experiment
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EXPERIMENT_LOG = OUTPUTS_DIR / "experiments.csv"

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
# Binary task: stage 0 = no DR, stages 1-4 = DR present
BINARY_NAMES = ["No DR", "DR"]

# ---------------------------------------------------------------------------
# Data split (stratified) - fractions of the de-duplicated dataset
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
# Perceptual-hash distance (0-64 bits) at or below which two images count as duplicates.
# Keep this small: all fundus photos look alike at low resolution. Check the pairs it
# flags with dataset.save_duplicate_examples() and lower it if they are not true copies.
DUPLICATE_HASH_THRESHOLD = 2

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE = 224            # use 300 for EfficientNetB3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BEN_GRAHAM_SIGMA = 10     # Gaussian sigma used in Ben Graham's local colour normalisation
DENOISE_KERNEL = 3        # median-blur kernel size for noise removal

# ---------------------------------------------------------------------------
# Training defaults (all can be overridden from the train.py command line)
# ---------------------------------------------------------------------------
SEED = 42
BACKBONE = "efficientnetb0"     # efficientnetb0 | efficientnetb3 | resnet50 | mobilenetv2
# Set the environment variable DR_WEIGHTS=none to train from scratch (offline testing only)
PRETRAINED_WEIGHTS = None if os.environ.get("DR_WEIGHTS", "imagenet").lower() == "none" else "imagenet"
BATCH_SIZE = 32
DROPOUT = 0.3
L2_WEIGHT_DECAY = 1e-4

EPOCHS_HEAD = 10          # stage 1: frozen backbone, train new head only
LR_HEAD = 1e-3
EPOCHS_FINETUNE = 30      # stage 2: unfreeze top layers of backbone
LR_FINETUNE = 1e-5
UNFREEZE_LAYERS = 40      # how many backbone layers to unfreeze in stage 2

EARLY_STOP_PATIENCE = 6
LR_PATIENCE = 3
LR_FACTOR = 0.3

BALANCE_METHOD = "weights"   # none | weights | oversample | focal
FOCAL_GAMMA = 2.0
BINARY_LOSS_WEIGHT = 0.5     # weight of the DR/No-DR head in the combined loss


def set_seed(seed: int = SEED) -> None:
    """Fix every source of randomness so experiments are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    """Create every output folder used by the project if it does not exist."""
    for d in (PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
