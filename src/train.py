"""
Train a DR stage classifier with two-stage transfer learning.

Usage (from the project root):
    python -m src.train --exp-name effb0_full_weights
    python -m src.train --backbone resnet50 --exp-name resnet50_full_weights
    python -m src.train --preprocess raw --exp-name ablation_raw
    python -m src.train --balance focal --exp-name effb0_focal
    python -m src.train --quick                     # 1-epoch smoke test

Each run writes to models/<exp-name>/:
    best.weights.h5     best weights (lowest validation loss)
    run_config.json     every setting used, so the run can be rebuilt
    history.csv         per-epoch metrics for both stages
and appends one summary row to outputs/experiments.csv.
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow import keras

import config
from src import dataset, evaluate, model as model_lib
from src.preprocessing import cache_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train DR stage detection model")
    p.add_argument("--exp-name", default=None, help="folder name under models/")
    p.add_argument("--backbone", default=config.BACKBONE, choices=list(model_lib.BACKBONES))
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--preprocess", default="full", choices=["raw", "clahe", "full"])
    p.add_argument("--balance", default=config.BALANCE_METHOD,
                   choices=["none", "weights", "oversample", "focal"])
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--epochs-head", type=int, default=config.EPOCHS_HEAD)
    p.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE)
    p.add_argument("--lr-head", type=float, default=config.LR_HEAD)
    p.add_argument("--lr-finetune", type=float, default=config.LR_FINETUNE)
    p.add_argument("--unfreeze", type=int, default=config.UNFREEZE_LAYERS)
    p.add_argument("--no-dedupe", action="store_true", help="skip duplicate removal")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--quick", action="store_true",
                   help="tiny subset and 1 epoch per stage - checks the pipeline runs")
    return p.parse_args()


def make_callbacks(run_dir, stage_name: str, best_so_far=None):
    """
    EarlyStopping      - stop when val_loss stops improving, keep best weights
    ReduceLROnPlateau  - lower the learning rate when progress stalls
    ModelCheckpoint    - save the best weights across BOTH stages
    CSVLogger          - per-epoch metrics for the report
    """
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=config.LR_FACTOR,
                                          patience=config.LR_PATIENCE, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.weights.h5"), monitor="val_loss",
                                        save_best_only=True, save_weights_only=True,
                                        initial_value_threshold=best_so_far, verbose=1),
        keras.callbacks.CSVLogger(str(run_dir / f"history_{stage_name}.csv")),
    ]


def main():
    args = parse_args()
    config.set_seed(args.seed)
    config.ensure_dirs()

    exp_name = args.exp_name or f"{args.backbone}_{args.preprocess}_{args.balance}_{datetime.now():%Y%m%d_%H%M}"
    if args.quick:
        exp_name = "quick_test"
        args.epochs_head, args.epochs_finetune = 1, 1
    run_dir = config.MODELS_DIR / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Experiment: {exp_name} ===")

    # ---- 1. Data --------------------------------------------------------
    train_df, val_df, test_df = dataset.prepare_splits(dedupe=not args.no_dedupe, seed=args.seed)
    if args.quick:
        train_df = train_df.groupby("stage").head(12)
        val_df = val_df.groupby("stage").head(4)
        test_df = test_df.groupby("stage").head(4)
    print(dataset.split_summary(train_df, val_df, test_df))

    all_ids = pd.concat([train_df, val_df, test_df])["id_code"]
    image_dir = cache_dataset(all_ids, variant=args.preprocess, size=args.img_size)

    class_weights = dataset.get_class_weights(train_df)
    print("Class weights:", np.round(class_weights, 3))
    fit_df = dataset.oversample(train_df, args.seed) if args.balance == "oversample" else train_df

    train_ds = dataset.make_dataset(fit_df, image_dir, training=True, batch_size=args.batch_size, seed=args.seed)
    val_ds = dataset.make_dataset(val_df, image_dir, training=False, batch_size=args.batch_size)

    # ---- 2. Model -------------------------------------------------------
    model, base = model_lib.build_model(args.backbone, args.img_size, args.dropout,
                                        weights=config.PRETRAINED_WEIGHTS)

    run_config = {**vars(args), "exp_name": exp_name, "class_weights": class_weights.tolist(),
                  "pretrained_weights": config.PRETRAINED_WEIGHTS,
                  "started": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    t0 = time.time()

    # ---- 3. Stage 1: train the new head, backbone frozen -----------------
    model_lib.freeze_backbone(base)
    model_lib.compile_model(model, args.lr_head, args.balance, class_weights)
    print(f"\nStage 1 - trainable params: {model_lib.count_trainable(model):,}")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                   callbacks=make_callbacks(run_dir, "stage1"), verbose=2)
    best_val = float(np.min(h1.history["val_loss"]))

    # ---- 4. Stage 2: fine-tune the top backbone layers -------------------
    model_lib.unfreeze_top_layers(base, args.unfreeze)
    model_lib.compile_model(model, args.lr_finetune, args.balance, class_weights)  # recompile after unfreezing
    print(f"\nStage 2 - trainable params: {model_lib.count_trainable(model):,}")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                   callbacks=make_callbacks(run_dir, "stage2", best_val), verbose=2)
    train_minutes = (time.time() - t0) / 60

    # ---- 5. Save history and curves --------------------------------------
    hist = pd.concat([pd.DataFrame(h1.history).assign(stage=1),
                      pd.DataFrame(h2.history).assign(stage=2)], ignore_index=True)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(run_dir / "history.csv", index=False)
    evaluate.plot_history(hist, run_dir / "training_curves.png", switch_epoch=len(h1.history["loss"]))

    # ---- 6. Evaluate the best weights on validation and test --------------
    model.load_weights(run_dir / "best.weights.h5")
    val_metrics = evaluate.evaluate_split(model, val_df, image_dir, run_dir, "val", args.batch_size)
    test_metrics = evaluate.evaluate_split(model, test_df, image_dir, run_dir, "test", args.batch_size)

    run_config["train_minutes"] = round(train_minutes, 2)
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    row = {"exp_name": exp_name, "backbone": args.backbone, "preprocess": args.preprocess,
           "balance": args.balance, "img_size": args.img_size, "dropout": args.dropout,
           "lr_finetune": args.lr_finetune, "unfreeze": args.unfreeze,
           "epochs_run": len(hist), "train_minutes": round(train_minutes, 1),
           "params": model.count_params(),
           **{f"val_{k}": v for k, v in val_metrics.items()},
           **{f"test_{k}": v for k, v in test_metrics.items()}}
    log = config.EXPERIMENT_LOG
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"\nDone in {train_minutes:.1f} min. Results in {run_dir}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
