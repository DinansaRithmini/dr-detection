"""
Gradio prototype: upload a retinal fundus photo and get
    - the preprocessed image the model actually sees
    - the probability of each DR stage
    - a DR present / absent result
    - a Grad-CAM heatmap showing where the model looked

Run from the project root:
    python app/app.py --exp-name effb0_full_weights
    python app/app.py                      # picks the run with the best test QWK
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # allow "import config"

import gradio as gr
import numpy as np
import pandas as pd

import config
from src.gradcam import build_gradcam_model, gradcam_heatmap, overlay_heatmap
from src.model import load_trained_model
from src.preprocessing import preprocess_image

DISCLAIMER = ("Research prototype for a university coursework project. "
              "It is not a medical device and must not be used for diagnosis. "
              "Always consult a qualified eye-care professional.")


def pick_experiment(name: str = None) -> str:
    """Use the given run, otherwise the run with the highest test QWK in experiments.csv."""
    if name:
        return name
    if config.EXPERIMENT_LOG.exists():
        log = pd.read_csv(config.EXPERIMENT_LOG)
        log = log[log["exp_name"] != "quick_test"]
        if len(log):
            return log.sort_values("test_qwk", ascending=False).iloc[0]["exp_name"]
    raise SystemExit("No trained model found. Train one first: python -m src.train")


def build_interface(exp_name: str) -> gr.Blocks:
    model, cfg = load_trained_model(config.MODELS_DIR / exp_name)
    grad_model = build_gradcam_model(model, cfg["backbone"])

    def analyse(image):
        if image is None:
            raise gr.Error("Upload a retinal fundus image to analyse.")
        processed = preprocess_image(np.asarray(image.convert("RGB")), cfg["preprocess"], cfg["img_size"])
        out = model.predict(processed[None].astype("float32"), verbose=0)
        stage_probs = out["stage"][0]
        p_dr = float(out["dr"][0, 0])

        heat, _ = gradcam_heatmap(grad_model, processed)
        stage = int(stage_probs.argmax())
        verdict = (f"### {'DR detected' if p_dr >= 0.5 else 'No DR detected'}\n"
                   f"Probability of DR: **{p_dr:.1%}**  \n"
                   f"Most likely stage: **{stage} - {config.CLASS_NAMES[stage]}** "
                   f"({stage_probs[stage]:.1%})")
        return (processed, overlay_heatmap(processed, heat),
                {f"{i} - {n}": float(p) for i, (n, p) in enumerate(zip(config.CLASS_NAMES, stage_probs))},
                verdict)

    examples_dir = config.RAW_IMAGE_DIR
    examples = []
    split = config.SPLITS_DIR / "test.csv"
    if split.exists() and examples_dir.exists():
        test_df = pd.read_csv(split)
        examples = [[str(examples_dir / f"{r.id_code}.png")]
                    for r in test_df.groupby("stage").head(1).itertuples()]

    with gr.Blocks(title="Diabetic Retinopathy Stage Detection") as demo:
        gr.Markdown("# Diabetic retinopathy stage detection\n"
                    f"Model: `{cfg['backbone']}`, preprocessing: `{cfg['preprocess']}`, "
                    f"input {cfg['img_size']}px")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Fundus photograph")
                btn = gr.Button("Analyse image", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=inp, label="Test images (one per stage)")
            with gr.Column():
                verdict = gr.Markdown()
                probs = gr.Label(num_top_classes=5, label="Stage probabilities")
        with gr.Row():
            pre = gr.Image(label="What the model sees (preprocessed)")
            cam = gr.Image(label="Grad-CAM: regions that drove the prediction")
        gr.Markdown(f"*{DISCLAIMER}*")
        btn.click(analyse, inputs=inp, outputs=[pre, cam, probs, verdict])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default=None)
    p.add_argument("--share", action="store_true", help="public link (needed on Colab/Kaggle)")
    a = p.parse_args()
    build_interface(pick_experiment(a.exp_name)).launch(share=a.share)
