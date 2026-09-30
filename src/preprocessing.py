"""
Image preprocessing for retinal fundus photographs.

Pipeline (each step can be switched off for the ablation study):
    1. crop_black_border  - remove the empty black area around the retina
    2. resize             - fixed square input size for the CNN
    3. denoise            - median blur to suppress sensor / JPEG noise
    4. apply_clahe        - local contrast enhancement of vessels and lesions
    5. ben_graham         - local colour normalisation + edge enhancement
    6. circular_mask      - re-blacken the corners introduced by step 5

The three named variants used in the ablation study are:
    "raw"   -> crop + resize only
    "clahe" -> crop + resize + denoise + CLAHE
    "full"  -> the complete pipeline above
"""
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
from tqdm import tqdm

import config

# Which steps each ablation variant switches on
VARIANTS: Dict[str, Dict[str, bool]] = {
    "raw":   {"denoise": False, "clahe": False, "ben_graham": False},
    "clahe": {"denoise": True,  "clahe": True,  "ben_graham": False},
    "full":  {"denoise": True,  "clahe": True,  "ben_graham": True},
}


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image from disk and return it as an RGB uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_black_border(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop away the black background around the circular retina.

    Pixels darker than `threshold` in the grey image are treated as background.
    Removing them means the CNN spends its capacity on the retina itself, and
    every image ends up with the retina filling a similar area of the frame.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = grey > threshold
    if not mask.any():                      # completely dark image - leave it
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def resize(img: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Resize to a square `size` x `size` image (INTER_AREA is best for shrinking)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, ksize: int = config.DENOISE_KERNEL) -> np.ndarray:
    """Median blur removes salt-and-pepper noise while keeping edges sharp."""
    return cv2.medianBlur(img, ksize)


def apply_clahe(img: np.ndarray,
                clip_limit: float = config.CLAHE_CLIP_LIMIT,
                tile_grid=config.CLAHE_TILE_GRID) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation on the lightness channel.

    Working in LAB colour space means only brightness contrast is changed, so
    lesion colours (red haemorrhages, yellow exudates) are not distorted.
    The clip limit stops CLAHE from amplifying noise in flat regions.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_chan = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB)


def ben_graham(img: np.ndarray, sigma: float = config.BEN_GRAHAM_SIGMA) -> np.ndarray:
    """
    Ben Graham's preprocessing (winner of the 2015 Kaggle DR competition).

    Subtracting a heavily blurred copy removes slow lighting changes across
    the retina (uneven illumination between cameras) and leaves fine detail
    such as vessels, microaneurysms and exudates.  It acts as edge enhancement.
        out = 4 * img - 4 * GaussianBlur(img) + 128
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def circular_mask(img: np.ndarray, scale: float = 0.95) -> np.ndarray:
    """
    Black out everything outside a centred circle.

    Ben Graham's step turns the black corners grey; masking them again
    stops the network from learning anything from the image border.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * scale), 1, thickness=-1)
    return img * mask[:, :, None]


def preprocess_image(img: np.ndarray, variant: str = "full",
                     size: int = config.IMG_SIZE) -> np.ndarray:
    """
    Run the preprocessing pipeline on one RGB image.

    Args:
        img: RGB uint8 image of any size.
        variant: "raw", "clahe" or "full" (see VARIANTS).
        size: output side length in pixels.
    Returns:
        RGB uint8 image of shape (size, size, 3), values 0-255.
        Model-specific normalisation happens inside the model (src/model.py),
        so cached images stay viewable and backbone-independent.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}")
    steps = VARIANTS[variant]

    img = crop_black_border(img)
    img = resize(img, size)
    if steps["denoise"]:
        img = denoise(img)
    if steps["clahe"]:
        img = apply_clahe(img)
    if steps["ben_graham"]:
        img = ben_graham(img)
        img = circular_mask(img)
    return img


def pipeline_stages(img: np.ndarray, size: int = config.IMG_SIZE) -> Dict[str, np.ndarray]:
    """Return every intermediate image of the full pipeline (for report figures)."""
    stages = {"Original": resize(img, size)}
    x = crop_black_border(img)
    stages["Cropped"] = resize(x, size)
    x = denoise(stages["Cropped"])
    stages["Denoised"] = x
    x = apply_clahe(x)
    stages["CLAHE"] = x
    x = circular_mask(ben_graham(x))
    stages["Ben Graham"] = x
    return stages


def cache_dataset(image_ids, variant: str = "full", size: int = config.IMG_SIZE,
                  src_dir: Path = config.RAW_IMAGE_DIR) -> Path:
    """
    Preprocess every image once and save it to data/processed/<variant>_<size>/.

    Training then reads the small cached PNGs instead of re-running OpenCV
    every epoch, which makes experiments much faster and fully repeatable.
    Images already in the cache are skipped.
    """
    out_dir = config.PROCESSED_DIR / f"{variant}_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in image_ids if not (out_dir / f"{i}.png").exists()]
    for image_id in tqdm(todo, desc=f"Preprocessing ({variant}, {size}px)"):
        img = load_image(src_dir / f"{image_id}.png")
        out = preprocess_image(img, variant, size)
        cv2.imwrite(str(out_dir / f"{image_id}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out_dir


def save_pipeline_figure(image_paths, out_path: Path = config.FIGURES_DIR / "preprocessing_pipeline.png") -> Path:
    """Save a grid (rows = images, columns = pipeline stages) for the report."""
    import matplotlib.pyplot as plt

    rows = [pipeline_stages(load_image(p)) for p in image_paths]
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    for r, stages in enumerate(rows):
        for c, (name, im) in enumerate(stages.items()):
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(name, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
