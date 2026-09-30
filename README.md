# Diabetic Retinopathy Stage Detection

A CNN with transfer learning that classifies retinal fundus photographs from the
[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
dataset. It predicts:

- **DR stage:** 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative
- **DR presence:** DR vs No DR, via a second output head

Video demonstration: _add your hosted video URL here_

## Pipeline

```
raw image -> crop black border -> resize -> median denoise -> CLAHE (LAB L-channel)
          -> Ben Graham local colour normalisation -> circular mask      (src/preprocessing.py)
          -> random flips / rotation / zoom / brightness / contrast      (src/augmentation.py, train only)
          -> pretrained backbone -> GAP -> Dense -> [stage head, DR head] (src/model.py)
```

Data handling (`src/dataset.py`) removes perceptual-hash duplicates before a stratified
70/15/15 split, so no copy of a test image is ever seen during training.

## Project structure

| Path | Purpose |
|---|---|
| `config.py` | every setting (paths, image size, learning rates, seed) |
| `src/preprocessing.py` | image preprocessing pipeline and caching |
| `src/augmentation.py` | training-time augmentation |
| `src/dataset.py` | labels, duplicate removal, splits, class balancing, tf.data |
| `src/model.py` | backbones, two-head model, freezing, weighted and focal losses |
| `src/train.py` | two-stage training with callbacks, logs every run |
| `src/evaluate.py` | metrics, confusion matrices, ROC, error analysis, TTA, speed test |
| `src/gradcam.py` | Grad-CAM explanations |
| `app/app.py` | Gradio prototype |
| `notebooks/01_eda.ipynb` | dataset exploration and report figures |
| `notebooks/02_compare_experiments.ipynb` | experiment comparison tables and charts |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements-lock.txt
```

Download the data into `data/`. This needs a Kaggle API token in `~/.kaggle/kaggle.json`,
and you must accept the competition rules on Kaggle first.

```bash
cd data
kaggle competitions download -c aptos2019-blindness-detection
unzip -q aptos2019-blindness-detection.zip && rm aptos2019-blindness-detection.zip
cd ..
```

Only `train.csv` has labels, so the project makes its own validation and test splits from it.

## Usage

Run everything from the project root.

```bash
# 0. Smoke test (a few images, 1 epoch per stage) - checks everything runs
python -m src.train --quick

# 1. Main model
python -m src.train --backbone efficientnetb0 --exp-name effb0_full_weights

# 2. Backbone comparison
python -m src.train --backbone resnet50    --exp-name resnet50_full_weights
python -m src.train --backbone mobilenetv2 --exp-name mobilenetv2_full_weights

# 3. Preprocessing ablation
python -m src.train --preprocess raw   --exp-name effb0_raw_weights
python -m src.train --preprocess clahe --exp-name effb0_clahe_weights

# 4. Class-balancing comparison
python -m src.train --balance oversample --exp-name effb0_full_oversample
python -m src.train --balance focal      --exp-name effb0_full_focal

# 5. Hyperparameter tuning (examples)
python -m src.train --lr-finetune 3e-5 --exp-name effb0_lr3e-5
python -m src.train --dropout 0.5      --exp-name effb0_dropout0.5
python -m src.train --unfreeze 80      --exp-name effb0_unfreeze80

# 6. Final model extras: test-time augmentation, speed test, Grad-CAM
python -m src.evaluate --exp-name effb0_full_weights --tta --benchmark
python -m src.gradcam  --exp-name effb0_full_weights --n 8

# 7. Demo app
python app/app.py --exp-name effb0_full_weights
```

Each run saves `run_config.json`, `history.csv`, `training_curves.png`, the best
weights, and `val/` + `test/` folders under `models/<exp-name>/`. The test folder
contains the classification report, confusion matrices, ROC curves, misclassified
images and predictions. A summary row is added to `outputs/experiments.csv`.

## Results

_Fill in from `outputs/experiments.csv` once training is finished._

| Model | Test accuracy | Macro F1 | QWK | DR sensitivity | DR specificity |
|---|---|---|---|---|---|
| EfficientNetB0 | | | | | |

## Disclaimer

A university coursework prototype. It is not a medical device and must not be used for diagnosis.
