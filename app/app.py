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
from gradio.themes.utils import colors

import config
from src.gradcam import build_gradcam_model, gradcam_heatmap, overlay_heatmap
from src.model import load_trained_model
from src.preprocessing import preprocess_image

DISCLAIMER = (
    "Research & clinical prototype for coursework demonstration. "
    "Not a certified medical device; strictly prohibited for primary clinical diagnosis. "
    "Always consult a board-certified ophthalmologist or eye-care professional."
)

THEME = gr.themes.Default(
    primary_hue=colors.blue,
    secondary_hue=colors.slate,
    neutral_hue=colors.slate,
    font=(
        gr.themes.GoogleFont("Google Sans"),
        gr.themes.GoogleFont("Plus Jakarta Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ),
).set(
    body_background_fill="#f8fafc",
    body_background_fill_dark="#f8fafc",
    body_text_color="#0f172a",
    body_text_color_dark="#0f172a",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_width="0px",
    block_border_width_dark="0px",
    block_shadow="none",
    block_shadow_dark="none",
    panel_background_fill="transparent",
    panel_background_fill_dark="transparent",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: light !important;
    --primary: #1e60ff;
    --primary-light: #eff6ff;
    --primary-dark: #0052e0;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border-light: #edf2f7;
    --bg-page: #f8fafc;
    --card-radius: 20px;
}

* {
    font-family: 'Google Sans', 'Plus Jakarta Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    box-sizing: border-box;
}

body, .gradio-container, .dark {
    background: var(--bg-page) !important;
    color: var(--text-main) !important;
    color-scheme: light !important;
}

.gradio-container {
    max-width: 1220px !important;
    margin: 0 auto !important;
    padding: 16px 20px 48px !important;
}

/* Eliminate ALL nested Gradio block borders, paddings and shadows */
.gradio-container .block,
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .form,
.gradio-container fieldset,
.gradio-container .gr-group,
.gradio-container .gap-4,
.gradio-container .compact {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 16px !important;
}

/* Top Navbar */
.med-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0 20px;
    margin-bottom: 8px;
    flex-wrap: wrap;
    gap: 16px;
}

.med-navbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.med-logo-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1e60ff 0%, #0052e0 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 16px rgba(30, 96, 255, 0.28);
    flex-shrink: 0;
}

.med-brand-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.med-brand-tag {
    font-size: 0.72rem;
    font-weight: 700;
    background: #eff6ff;
    color: #1e60ff;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid #dbeafe;
}

.med-brand-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 1px;
}

.med-navbar-right {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}

.med-nav-pills {
    display: flex;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 4px;
    gap: 4px;
}

.med-pill-item {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    padding: 5px 14px;
    border-radius: 999px;
}

.med-pill-item.active {
    background: #eff6ff;
    color: #1e60ff;
}

.med-pill-meta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-main);
}

.med-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
}

/* Page Headline */
.med-page-heading {
    margin-bottom: 20px;
}

.med-main-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text-main);
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}

.med-main-desc {
    font-size: 0.92rem;
    color: var(--text-muted);
    margin: 0;
}

/* Single Layer Clean White Cards */
.med-card {
    background: #ffffff !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--card-radius) !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04), 0 2px 6px -1px rgba(15, 23, 42, 0.02) !important;
    display: flex;
    flex-direction: column;
    height: 100%;
}

.med-card-heading {
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}

.med-card-heading h3 {
    margin: 0 0 3px 0 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
}

.med-card-heading p {
    margin: 0 !important;
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
}

/* Image Upload & Display */
.gradio-container .image-container,
.gradio-container [data-testid="image"] {
    background: #f8fafc !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

.gradio-container .image-container:hover {
    border-color: #94a3b8 !important;
}

.preview-card .image-container,
.preview-card [data-testid="image"] {
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
}

/* Primary Button */
button.primary {
    background: linear-gradient(135deg, #1e60ff 0%, #0052e0 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.96rem !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 13px 28px !important;
    box-shadow: 0 8px 20px rgba(30, 96, 255, 0.25) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-top: 14px !important;
    width: 100% !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #1652df 0%, #0042b8 100%) !important;
    box-shadow: 0 10px 24px rgba(30, 96, 255, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Examples gallery */
.med-examples-wrap {
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid #f1f5f9;
}

.med-examples-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.gr-examples {
    border: none !important;
    padding: 0 !important;
}

.gr-examples table, .gr-examples div[role="button"] {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    overflow: hidden !important;
}

/* Diagnostic Assessment States */
.med-assessment-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 38px 20px;
    text-align: center;
    background: #f8fafc;
    border: 1.5px dashed #cbd5e1;
    border-radius: 16px;
    min-height: 290px;
}

.med-placeholder-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: #eff6ff;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
}

.med-assessment-placeholder h4 {
    margin: 0 0 6px;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-main);
}

.med-assessment-placeholder p {
    margin: 0 0 16px;
    font-size: 0.84rem;
    color: var(--text-muted);
    max-width: 380px;
    line-height: 1.45;
}

.med-placeholder-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
}

.med-placeholder-pill {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
}

.med-placeholder-check {
    color: #10b981;
    margin-right: 3px;
}

/* Result Verdict Card (Medicom Style) */
.med-result-box {
    display: flex;
    flex-direction: column;
    gap: 16px;
    animation: fadeIn 0.25s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

.med-verdict-banner {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px 18px;
}

.med-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.status-chip-healthy {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}

.status-chip-warning {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.med-verdict-text {
    margin: 0;
    font-size: 0.83rem;
    color: #475569;
    line-height: 1.45;
}

/* 3-Column Metrics (Medicom numbers) */
.med-kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}

.med-kpi-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 12px 14px;
}

.med-kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

.med-kpi-num {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text-main);
    line-height: 1.15;
}

.med-kpi-primary {
    font-size: 1.2rem;
    font-weight: 800;
    color: #1e60ff;
    line-height: 1.15;
}

.med-kpi-sub {
    font-size: 0.76rem;
    font-weight: 600;
    color: #64748b;
    margin-top: 2px;
}

/* Stage Probability Progress Bars */
.med-stage-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 16px;
}

.med-stage-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 12px;
}

.med-stage-rows {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.med-stage-row {
    display: grid;
    grid-template-columns: 130px 1fr 45px;
    align-items: center;
    gap: 10px;
}

.stage-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
}

.is-top-stage .stage-title {
    color: #1e60ff;
    font-weight: 800;
}

.stage-track {
    height: 9px;
    background: #f1f5f9;
    border-radius: 999px;
    overflow: hidden;
}

.stage-fill {
    height: 100%;
    border-radius: 999px;
    background: #cbd5e1;
    transition: width 0.4s ease;
}

.is-top-stage .stage-fill {
    background: linear-gradient(90deg, #1e60ff 0%, #3b82f6 100%) !important;
}

.stage-percentage {
    font-size: 0.78rem;
    font-weight: 700;
    color: #334155;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.is-top-stage .stage-percentage {
    color: #1e60ff;
    font-weight: 800;
}

/* Footer */
.med-page-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-top: 28px;
    padding: 14px 20px;
    text-align: center;
    flex-wrap: wrap;
}

.med-pill-footer {
    background: #eff6ff;
    color: #1e60ff;
    border: 1px solid #dbeafe;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
}

.med-footer-note {
    margin: 0;
    font-size: 0.8rem;
    color: #64748b;
    max-width: 750px;
}

footer { display: none !important; }
"""

# Injected into <head>: reloads the page once with ?__theme=light so Gradio never
# renders in dark mode (the custom CSS above is designed for a light background only).
FORCE_LIGHT_HEAD = """
<script>
(function () {
    var url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "light") {
        url.searchParams.set("__theme", "light");
        window.location.replace(url.toString());
    }
})();
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""

EMPTY_ASSESSMENT_HTML = """
<div class="med-assessment-placeholder">
    <div class="med-placeholder-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#1e60ff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="m10 15 5-3-5-3v6Z"></path>
        </svg>
    </div>
    <h4>Awaiting Examination</h4>
    <p>Upload a retinal fundus photograph on the left and click <strong>Analyse Retinal Scan</strong> to generate automated grading, risk metrics, and Grad-CAM explainability maps.</p>
    <div class="med-placeholder-badges">
        <span class="med-placeholder-pill"><span class="med-placeholder-check">✓</span> Binary DR Screening</span>
        <span class="med-placeholder-pill"><span class="med-placeholder-check">✓</span> 5-Stage ICDR Grading</span>
        <span class="med-placeholder-pill"><span class="med-placeholder-check">✓</span> Grad-CAM Heatmaps</span>
    </div>
</div>
"""


def format_assessment_html(p_dr: float, stage_probs: np.ndarray) -> str:
    """Build the result card HTML from the two model heads (binary DR probability + 5 stage probabilities)."""
    stage = int(stage_probs.argmax())
    stage_name = config.CLASS_NAMES[stage]
    # The verdict (healthy / DR) follows the binary head; the stage shown is the stage head's argmax.
    # The two heads can disagree on borderline images.
    is_dr = p_dr >= 0.5

    if not is_dr:
        chip_html = '<span class="med-status-chip status-chip-healthy"><span class="med-status-dot"></span> No DR Detected (Healthy)</span>'
        desc_text = "Retinal vasculature and macula appear within normal limits. Routine annual screening recommended."
        risk_text = "Low Risk"
    else:
        chip_html = f'<span class="med-status-chip status-chip-warning"><span class="med-status-dot" style="background:#ef4444;box-shadow:0 0 0 3px rgba(239,68,68,0.2);"></span> DR Detected &middot; Stage {stage} ({stage_name})</span>'
        desc_text = f"Pathological signs consistent with {stage_name} Diabetic Retinopathy detected. Specialist ophthalmic referral advised."
        # Moderate stage (2) and above is treated as high risk
        risk_text = "High Risk" if stage >= 2 else "Moderate Risk"

    # One progress-bar row per stage; the top-scoring stage is highlighted via CSS class
    rows_html = ""
    for i, name in enumerate(config.CLASS_NAMES):
        prob = float(stage_probs[i])
        pct = prob * 100
        is_top = (i == stage)
        cls = "is-top-stage" if is_top else ""
        rows_html += f"""
        <div class="med-stage-row {cls}">
            <div class="stage-title">Stage {i} &middot; {name}</div>
            <div class="stage-track">
                <div class="stage-fill" style="width: {pct:.1f}%"></div>
            </div>
            <div class="stage-percentage">{pct:.1f}%</div>
        </div>
        """

    return f"""
    <div class="med-result-box">
        <div class="med-verdict-banner">
            {chip_html}
            <p class="med-verdict-text">{desc_text}</p>
        </div>
        
        <div class="med-kpi-grid">
            <div class="med-kpi-card">
                <div class="med-kpi-label">DR Risk Score</div>
                <div class="med-kpi-num">{p_dr * 100:.1f}%</div>
                <div class="med-kpi-sub">{risk_text}</div>
            </div>
            <div class="med-kpi-card">
                <div class="med-kpi-label">Primary Finding</div>
                <div class="med-kpi-primary">Stage {stage}</div>
                <div class="med-kpi-sub">{stage_name}</div>
            </div>
            <div class="med-kpi-card">
                <div class="med-kpi-label">Confidence</div>
                <div class="med-kpi-num">{stage_probs[stage] * 100:.1f}%</div>
                <div class="med-kpi-sub">Multihead Score</div>
            </div>
        </div>

        <div class="med-stage-section">
            <div class="med-stage-header">
                <span>Stage Probability Distribution</span>
                <span style="font-weight:600;color:#64748b;font-size:0.75rem;">ICDR Standard</span>
            </div>
            <div class="med-stage-rows">
                {rows_html}
            </div>
        </div>
    </div>
    """


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
        """Gradio callback: preprocess -> predict -> Grad-CAM -> render results."""
        if image is None:
            raise gr.Error("Please upload a retinal fundus image to analyse.")
        # Use the same preprocessing variant and size the model was trained with
        processed = preprocess_image(np.asarray(image.convert("RGB")), cfg["preprocess"], cfg["img_size"])
        out = model.predict(processed[None].astype("float32"), verbose=0)
        stage_probs = out["stage"][0]
        p_dr = float(out["dr"][0, 0])

        heat, _ = gradcam_heatmap(grad_model, processed)
        assessment_html = format_assessment_html(p_dr, stage_probs)
        return processed, overlay_heatmap(processed, heat), assessment_html

    # Clickable examples: one raw test-set image per DR stage (skipped if data isn't available locally)
    examples_dir = config.RAW_IMAGE_DIR
    examples = []
    split = config.SPLITS_DIR / "test.csv"
    if split.exists() and examples_dir.exists():
        test_df = pd.read_csv(split)
        examples = [[str(examples_dir / f"{r.id_code}.png")]
                    for r in test_df.groupby("stage").head(1).itertuples()]

    with gr.Blocks(title="Diabetic Retinopathy Detection") as demo:
        # Top Navbar (Direct on canvas, no wrapper boxes)
        gr.HTML(
            f"""
            <nav class="med-navbar">
                <div class="med-navbar-brand">
                    <div class="med-logo-icon">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="3"></circle>
                            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path>
                        </svg>
                    </div>
                    <div>
                        <div class="med-brand-name">Medicom <span class="med-brand-tag">RetinaAI</span></div>
                        <div class="med-brand-sub">Diabetic Retinopathy Automated Screening</div>
                    </div>
                </div>
                <div class="med-navbar-right">
                    <div class="med-nav-pills">
                        <span class="med-pill-item active">Screening</span>
                        <span class="med-pill-item">Explainability</span>
                    </div>
                    <div class="med-pill-meta">
                        <span class="med-status-dot"></span>
                        <span>{cfg['backbone'].upper()} &middot; {cfg['preprocess'].upper()} {cfg['img_size']}px</span>
                    </div>
                </div>
            </nav>
            <div class="med-page-heading">
                <h1 class="med-main-title">Retinal Health Screening</h1>
                <p class="med-main-desc">Deep learning-powered ICDR 5-stage grading and Grad-CAM lesion localization</p>
            </div>
            """
        )

        # Main 2-Column Dashboard (Examination & Diagnostic Results)
        with gr.Row():
            with gr.Column(scale=5, elem_classes=["med-card"]):
                gr.HTML(
                    """
                    <div class="med-card-heading">
                        <h3>Fundus Examination</h3>
                        <p>Upload a macula-centered or optic-disc retinal photograph</p>
                    </div>
                    """
                )
                inp = gr.Image(type="pil", show_label=False)
                btn = gr.Button("Analyse Retinal Scan", variant="primary", size="lg")
                if examples:
                    gr.HTML('<div class="med-examples-wrap"><div class="med-examples-title">Sample Test Cases (Click to load)</div></div>')
                    gr.Examples(examples, inputs=inp, label=None)

            with gr.Column(scale=5, elem_classes=["med-card"]):
                gr.HTML(
                    """
                    <div class="med-card-heading">
                        <h3>Diagnostic Assessment</h3>
                        <p>Multi-head risk score & automated ICDR stage classification</p>
                    </div>
                    """
                )
                assessment_view = gr.HTML(EMPTY_ASSESSMENT_HTML)

        # Neural Network Explainability Row
        with gr.Row():
            with gr.Column(scale=5, elem_classes=["med-card", "preview-card"]):
                gr.HTML(
                    """
                    <div class="med-card-heading">
                        <h3>Preprocessed Network Input</h3>
                        <p>Standardized CLAHE contrast-enhanced view (224×224px)</p>
                    </div>
                    """
                )
                pre = gr.Image(show_label=False, interactive=False)

            with gr.Column(scale=5, elem_classes=["med-card", "preview-card"]):
                gr.HTML(
                    """
                    <div class="med-card-heading">
                        <h3>Grad-CAM Attention Heatmap</h3>
                        <p>Class activation map highlighting pathological biomarkers</p>
                    </div>
                    """
                )
                cam = gr.Image(show_label=False, interactive=False)

        # Footer Notice
        gr.HTML(
            f"""
            <div class="med-page-footer">
                <span class="med-pill-footer">Clinical Notice</span>
                <p class="med-footer-note">{DISCLAIMER}</p>
            </div>
            """
        )

        btn.click(analyse, inputs=inp, outputs=[pre, cam, assessment_view])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default=None)
    p.add_argument("--share", action="store_true", help="public link (needed on Colab/Kaggle)")
    p.add_argument("--port", type=int, default=None, help="port to run the Gradio app on")
    a = p.parse_args()
    build_interface(pick_experiment(a.exp_name)).launch(
        server_port=a.port, share=a.share, theme=THEME, css=CSS, head=FORCE_LIGHT_HEAD
    )
