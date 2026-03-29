import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from model import EmotionPredictor


# ── Initialize ───────────────────────────────────────────────────────
predictor = EmotionPredictor()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade")

# Muted, harmonious palette
EMOTION_META = {
    "angry":    {"label": "Angry",    "css": "#e06c75", "bgr": (117, 108, 224)},
    "disgust":  {"label": "Disgust",  "css": "#7ec89f", "bgr": (159, 200, 126)},
    "fear":     {"label": "Fear",     "css": "#c49cde", "bgr": (222, 156, 196)},
    "happy":    {"label": "Happy",    "css": "#e5c17c", "bgr": (124, 193, 229)},
    "sad":      {"label": "Sad",      "css": "#7caed6", "bgr": (214, 174, 124)},
    "surprise": {"label": "Surprise", "css": "#d6956e", "bgr": (110, 149, 214)},
    "neutral":  {"label": "Neutral",  "css": "#abb2bf", "bgr": (191, 178, 171)},
}


def predict_emotion(image):
    """Predict emotion from an uploaded image or webcam frame."""
    if image is None:
        return None, _empty_html()

    frame = image.copy()
    frame_bgr = (
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3
        else frame
    )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    detected = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(detected) == 0:
        return frame, _no_face_html()

    faces = [max(detected, key=lambda r: r[2] * r[3])]
    output = frame_bgr.copy()

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(frame_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

        img_pil = PILImage.fromarray(face_rgb)
        tensor = predictor.transform(img_pil).unsqueeze(0).to(predictor.device)
        with torch.inference_mode():
            logits = predictor.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        top_idx = int(probs.argmax())
        emotion = predictor.classes[top_idx]
        confidence = float(probs[top_idx])
        meta = EMOTION_META.get(emotion, {"label": emotion, "css": "#abb2bf", "bgr": (191, 178, 171)})
        color_bgr = meta["bgr"]

        # Clean bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), color_bgr, 2, cv2.LINE_AA)

        # Minimal label pill
        label = f"{emotion.capitalize()}  {confidence*100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc, th = 0.5, 1
        (tw, tht), _ = cv2.getTextSize(label, font, sc, th)
        lx = x
        ly = max(y - 8, tht + 6)

        pill_overlay = output.copy()
        cv2.rectangle(pill_overlay, (lx, ly - tht - 5), (lx + tw + 10, ly + 3), (15, 15, 20), -1)
        cv2.addWeighted(pill_overlay, 0.75, output, 0.25, 0, output)
        cv2.putText(output, label, (lx + 5, ly - 1), font, sc, (240, 240, 240), th, cv2.LINE_AA)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    conf_data = {predictor.classes[i]: float(probs[i]) for i in range(len(predictor.classes))}
    return output_rgb, _result_html(emotion, confidence, conf_data)


# ── HTML Builders ────────────────────────────────────────────────────

def _empty_html():
    return """
    <div class="result-empty">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="3"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <path d="m21 15-5-5L5 21"/>
        </svg>
        <span>Upload an image to analyze</span>
    </div>"""


def _no_face_html():
    return """
    <div class="result-empty" style="color:#d6956e;">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <path d="M8 15s1.5-2 4-2 4 2 4 2"/>
            <line x1="9" y1="9" x2="9.01" y2="9"/>
            <line x1="15" y1="9" x2="15.01" y2="9"/>
        </svg>
        <span style="font-weight:500;">No face detected</span>
        <span class="result-hint">Try a well-lit, front-facing photo</span>
    </div>"""


def _result_html(emotion, confidence, conf_data):
    meta = EMOTION_META.get(emotion, {"label": emotion, "css": "#abb2bf"})
    color = meta["css"]

    bars = ""
    for i, (emo, prob) in enumerate(sorted(conf_data.items(), key=lambda x: -x[1])):
        m = EMOTION_META.get(emo, {"label": emo, "css": "#64748b"})
        pct = prob * 100
        is_top = emo == emotion
        weight = "600" if is_top else "400"
        text_color = m["css"] if is_top else "#9ca3b0"
        bar_opacity = "1" if is_top else "0.45"
        delay = i * 50

        bars += f"""
        <div class="conf-row" style="animation-delay:{delay}ms;">
            <span class="conf-label" style="color:{text_color};font-weight:{weight};">{m['label']}</span>
            <div class="conf-track">
                <div class="conf-fill" style="width:{pct:.1f}%;background:{m['css']};opacity:{bar_opacity};
                     animation-delay:{delay + 80}ms;"></div>
            </div>
            <span class="conf-pct" style="color:{text_color};font-weight:{weight};">{pct:.1f}%</span>
        </div>"""

    return f"""
    <div class="result-card">
        <div class="result-hero">
            <div class="result-dot" style="background:{color};"></div>
            <div class="result-info">
                <span class="result-emotion" style="color:{color};">{meta['label']}</span>
                <span class="result-conf">Confidence {confidence*100:.1f}%</span>
            </div>
            <span class="result-pct" style="color:{color};">{confidence*100:.0f}%</span>
        </div>
        <div class="conf-section">
            <span class="conf-heading">Breakdown</span>
            {bars}
        </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════
#  CSS — Emil Kowalski design engineering principles
# ═══════════════════════════════════════════════════════════════════════

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
    --surface: #111318;
    --surface-raised: #191c24;
    --surface-overlay: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --border-hover: rgba(255,255,255,0.14);
    --text-primary: #eaeaf0;
    --text-secondary: #b0b8c8;
    --text-tertiary: #6b7488;
    --accent: #8b8bf5;
    --accent-glow: rgba(139,139,245,0.15);
}

/* ── Global ── */
body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}
body {
    background: var(--surface) !important;
}
.gradio-container {
    max-width: 940px !important;
    margin: auto !important;
    width: 100% !important;
}

/* ═══════════════════════════════════════════════════════════════════
   HEADER — polished with gradient accent bar and better hierarchy
   ═══════════════════════════════════════════════════════════════════ */
.app-header {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 2rem 1.5rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #e8713a 0%, #d4904a 25%, #b8a84a 50%, #6dbfa5 75%, #3faa9e 100%);
    margin-bottom: 1.25rem;
    animation: headerIn 500ms var(--ease-out) both;
}
.app-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(255,255,255,0.12) 0%, transparent 60%);
    pointer-events: none;
}
@keyframes headerIn {
    from { opacity: 0; transform: scale(0.97) translateY(6px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.app-header h1 {
    position: relative;
    color: #fff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.15);
    line-height: 1.2;
}
.app-header h3 {
    position: relative;
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    margin: 0 0 1rem !important;
    letter-spacing: 0.01em;
}
.header-icons {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    margin-top: 0.25rem;
}
.header-skill-icons {
    height: 32px;
    border-radius: 8px;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.15));
    opacity: 0.9;
    transition: opacity 200ms ease, transform 160ms var(--ease-out);
}
.gradio-icon {
    width: 32px; height: 32px;
    flex-shrink: 0;
    border-radius: 8px;
    overflow: hidden;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.15));
    opacity: 0.9;
    transition: opacity 200ms ease, transform 160ms var(--ease-out);
}
.gradio-icon svg {
    width: 100%; height: 100%;
    display: block;
}
@media (hover: hover) and (pointer: fine) {
    .header-skill-icons:hover,
    .gradio-icon:hover {
        opacity: 1;
        transform: translateY(-1px);
    }
}

/* ── Cards ── */
.card {
    background: var(--surface-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 200ms ease !important;
}
@media (hover: hover) and (pointer: fine) {
    .card:hover {
        border-color: var(--border-hover) !important;
    }
}

.card-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin-bottom: 0.6rem;
}
.card-label svg {
    width: 14px; height: 14px;
    stroke-width: 1.5;
    color: var(--accent);
}

/* ── Detect Button ── */
.detect-btn {
    background: var(--accent) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px !important;
    letter-spacing: 0.005em !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px var(--accent-glow) !important;
    transition: transform 160ms var(--ease-out), background 200ms ease, box-shadow 200ms ease !important;
    position: relative !important;
    z-index: 10 !important;
}
@media (hover: hover) and (pointer: fine) {
    .detect-btn:hover {
        background: #7c7cf0 !important;
        box-shadow: 0 4px 16px rgba(139,139,245,0.3) !important;
    }
}
.detect-btn:active {
    transform: scale(0.97) !important;
}

.hint {
    text-align: center;
    font-size: 0.68rem;
    color: var(--text-tertiary);
    margin-top: 0.35rem;
}

/* ── Result panel ── */
.result-panel {
    background: transparent !important;
    border: none !important;
}
.result-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    padding: 2rem 1rem;
    color: var(--text-tertiary);
    font-size: 0.78rem;
}
.result-empty svg { opacity: 0.5; }
.result-hint { font-size: 0.68rem; color: var(--text-tertiary); }

.result-card {
    padding: 0.85rem 0.9rem;
    font-family: 'Inter', system-ui, sans-serif;
    animation: resultIn 250ms var(--ease-out) both;
}
@keyframes resultIn {
    from { opacity: 0; transform: scale(0.96) translateY(4px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
.result-hero {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.65rem 0.75rem;
    background: var(--surface-overlay);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.75rem;
    animation: resultIn 250ms var(--ease-out) both;
    animation-delay: 60ms;
}
.result-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.result-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.result-emotion {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.result-conf {
    font-size: 0.68rem;
    color: var(--text-secondary);
    font-weight: 400;
}
.result-pct {
    font-size: 1.2rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}

.conf-section {
    padding: 0.6rem 0.65rem;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
}
.conf-heading {
    display: block;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-tertiary);
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-bottom: 0.3rem;
    animation: rowIn 220ms var(--ease-out) both;
}
@keyframes rowIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.conf-label {
    width: 3.8rem;
    font-size: 0.7rem;
    flex-shrink: 0;
}
.conf-track {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
    animation: barIn 350ms var(--ease-in-out) both;
}
@keyframes barIn {
    from { transform: scaleX(0); transform-origin: left; }
    to   { transform: scaleX(1); transform-origin: left; }
}
.conf-pct {
    width: 2.5rem;
    text-align: right;
    font-size: 0.68rem;
    font-variant-numeric: tabular-nums;
}

/* ═══════════════════════════════════════════════════════════════════
   ABOUT SECTION — Fixed text contrast + table visibility
   ═══════════════════════════════════════════════════════════════════ */
.about-wrap { margin-top: 0.75rem; }
.about-wrap .label-wrap {
    background: var(--surface-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    transition: border-color 200ms ease !important;
}
/* ── About prose text — high contrast ── */
.about-wrap .prose {
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
    line-height: 1.65 !important;
}
.about-wrap .prose strong,
.about-wrap .prose b {
    color: #fff !important;
    font-weight: 600 !important;
}
.about-wrap .prose p {
    color: var(--text-secondary) !important;
    margin-bottom: 0.6rem !important;
}
/* ── About table — clearly visible ── */
.about-wrap .prose table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    margin-top: 0.5rem !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}
.about-wrap .prose th,
.about-wrap .prose td {
    padding: 0.55rem 0.75rem !important;
    text-align: left !important;
    border-bottom: 1px solid var(--border) !important;
    font-size: 0.78rem !important;
}
.about-wrap .prose th {
    background: rgba(139,139,245,0.08) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}
.about-wrap .prose td {
    background: var(--surface-raised) !important;
    color: var(--text-primary) !important;
}
.about-wrap .prose td strong {
    color: var(--accent) !important;
}
.about-wrap .prose tr:last-child td,
.about-wrap .prose tr:last-child th {
    border-bottom: none !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 0.8rem 0 0.4rem;
    font-size: 0.68rem;
    color: var(--text-tertiary);
}
.app-footer a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
    transition: color 200ms ease;
}
@media (hover: hover) and (pointer: fine) {
    .app-footer a:hover { color: #a5a5ff; }
}

/* ── Misc ── */
footer { display: none !important; }
.image-container, .upload-container { border-radius: 8px !important; }

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
    .app-logo, .result-card, .result-hero, .conf-row, .conf-fill {
        animation: none !important;
        opacity: 1 !important;
        transform: none !important;
    }
    .detect-btn:active { transform: none !important; }
    *, *::before, *::after { transition-duration: 0.01ms !important; }
}
"""


# ── Theme ────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#111318",
    body_background_fill_dark="#111318",
    block_background_fill="#191c24",
    block_background_fill_dark="#191c24",
    block_border_color="rgba(255,255,255,0.08)",
    block_border_color_dark="rgba(255,255,255,0.08)",
    block_label_text_color="#6b7488",
    block_label_text_color_dark="#6b7488",
    block_title_text_color="#eaeaf0",
    block_title_text_color_dark="#eaeaf0",
    input_background_fill="#14161c",
    input_background_fill_dark="#14161c",
    input_border_color="rgba(255,255,255,0.08)",
    input_border_color_dark="rgba(255,255,255,0.08)",
    button_primary_background_fill="#8b8bf5",
    button_primary_background_fill_dark="#8b8bf5",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    border_color_primary="rgba(139,139,245,0.2)",
    border_color_primary_dark="rgba(139,139,245,0.2)",
)


# ── UI ───────────────────────────────────────────────────────────────

with gr.Blocks(title="Smilo — Emotion Detection") as demo:

    # ──── Gradient Header ────
    gr.HTML("""
        <div class="app-header">
            <h1>Smilo 😃</h1>
            <h3>Real-Time Emotion Detection powered by PyTorch</h3>
            <div class="header-icons">
                <img src="https://skillicons.dev/icons?i=pytorch,opencv" alt="PyTorch & OpenCV" class="header-skill-icons" />
                <div class="gradio-icon" title="Gradio">
                    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="gf1" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#FF9900"/>
                                <stop offset="100%" stop-color="#FF4D00"/>
                            </linearGradient>
                            <linearGradient id="gf2" x1="100%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stop-color="#FFD700"/>
                                <stop offset="100%" stop-color="#FF6600"/>
                            </linearGradient>
                        </defs>
                        <!-- Top Diamond -->
                        <path d="M50 15 L85 35 L50 55 L15 35 Z" fill="none" stroke="url(#gf1)" stroke-width="14" stroke-linejoin="round" stroke-linecap="round"/>
                        <!-- Bottom Diamond -->
                        <path d="M50 45 L85 65 L50 85 L15 65 Z" fill="none" stroke="url(#gf2)" stroke-width="14" stroke-linejoin="round" stroke-linecap="round"/>
                    </svg>
                </div>
            </div>
        </div>
    """)

    with gr.Row(equal_height=True):

        # ──── Input Card ────
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML("""
                <div class="card-label">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                    </svg>
                    Input
                </div>
            """)
            image_input = gr.Image(
                label=None, show_label=False,
                type="numpy",
                sources=["upload", "webcam"],
                elem_id="image-input",
                height=320,
            )
            submit_btn = gr.Button("Detect Emotion", elem_classes=["detect-btn"], size="lg")
            gr.HTML('<p class="hint">Upload a photo, use webcam, or click Detect</p>')

        # ──── Output Card ────
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML("""
                <div class="card-label">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                    </svg>
                    Analysis
                </div>
            """)
            image_output = gr.Image(
                label=None, show_label=False,
                type="numpy", interactive=False,
                height=320,
            )
            result_html = gr.HTML(value=_empty_html(), elem_classes=["result-panel"])

    # ──── About ────
    with gr.Accordion("About Smilo", open=False, elem_classes=["about-wrap"]):
        gr.Markdown("""
**Smilo** uses a custom-trained CNN to classify 7 facial emotions in real time.

| Component | Details |
|-----------|---------|
| **Detection** | Haar Cascade (OpenCV) |
| **Classifier** | 3-layer CNN · PyTorch |
| **Emotions** | Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise |
| **Input** | 128×128 px, auto-resized |
        """)

    gr.HTML('<div class="app-footer">Built with <a href="https://gradio.app">Gradio</a> · <a href="https://pytorch.org">PyTorch</a></div>')

    # ──── Events ────
    # Button click — always works (upload + webcam snapshot)
    submit_btn.click(
        fn=predict_emotion,
        inputs=[image_input],
        outputs=[image_output, result_html],
    )
    # Auto-detect when image changes (upload or webcam stop)
    image_input.change(
        fn=predict_emotion,
        inputs=[image_input],
        outputs=[image_output, result_html],
    )
    # Live webcam stream
    image_input.stream(
        fn=predict_emotion,
        inputs=[image_input],
        outputs=[image_output, result_html],
        stream_every=0.2,
    )

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)
