# ============================================================
# streamlit_app.py — Fruit Quality AI
# ============================================================

import os
import io
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ── Auto-download model weights if missing (cloud deploy) ─
WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "app", "model", "model_weights.weights.h5")

if not os.path.exists(WEIGHTS_PATH):
    import gdown
    MODEL_URL = None
    # Try Streamlit secrets first (cloud)
    try:
        MODEL_URL = st.secrets.get("MODEL_WEIGHTS_URL", None)
    except Exception:
        pass
    # Fall back to environment variable (local)
    if not MODEL_URL:
        MODEL_URL = os.environ.get("MODEL_WEIGHTS_URL", "")

    if MODEL_URL:
        print("⬇️ Downloading model weights...")
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        gdown.download(MODEL_URL, WEIGHTS_PATH, quiet=False)
        print("✓ Model weights ready")
    else:
        st.error(
            "❌ Model weights not found and MODEL_WEIGHTS_URL "
            "is not set. Please add it to Streamlit secrets.")
        st.stop()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f8fa; }
[data-testid="stHeader"]           { background: transparent; }
#MainMenu, footer                  { visibility: hidden; }

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 12px;
    padding: 8px;
    border: 1px solid #e0e0e0;
}
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 12px 16px;
    border: 1px solid #e8e8e8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stExpander"] {
    background: white;
    border-radius: 10px;
    border: 1px solid #e8e8e8;
}
.section-header {
    font-size: 14px;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.grade-badge {
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 16px;
}
.llm-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #e8e8e8;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.topbar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 18px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ── Load predictor and LLM (cached) ──────────────────────
@st.cache_resource
def load_predictor():
    from app.utils.predictor import predict
    return predict

@st.cache_resource
def load_llm():
    from app.utils.llm import get_llm_report
    return get_llm_report

predict_fn = load_predictor()
get_llm_fn = load_llm()

# ── Constants ─────────────────────────────────────────────
GRADE_COLOR = {
    "A": "#27ae60", "B": "#f39c12",
    "C": "#e67e22", "F": "#e74c3c"
}
GRADE_EMOJI = {"A": "🟢", "B": "🟡", "C": "🟠", "F": "🔴"}
GRADE_BG    = {
    "A": "linear-gradient(135deg,#eafaf1,#d5f5e3)",
    "B": "linear-gradient(135deg,#fefde7,#fef9c3)",
    "C": "linear-gradient(135deg,#fef5e7,#fdebd0)",
    "F": "linear-gradient(135deg,#fdedec,#fadbd8)"
}

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div style="display:flex;align-items:center;gap:12px">
        <span style="font-size:32px">🍎</span>
        <div>
            <div style="font-size:22px;font-weight:700;
                        letter-spacing:-0.3px">
                Fruit Quality AI
            </div>
            <div style="font-size:13px;opacity:0.7;margin-top:2px">
                EfficientNetB0 + CBAM · 99.57% accuracy ·
                Apple · Banana · Orange
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

# ── LEFT — Upload ─────────────────────────────────────────
with left:
    st.markdown(
        '<div class="section-header">📤 Upload Fruit Image</div>',
        unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        raw_bytes = uploaded.read()
        pil_image = Image.open(io.BytesIO(raw_bytes))

        display_img = pil_image.copy()
        if display_img.width > 400:
            ratio = 400 / display_img.width
            display_img = display_img.resize(
                (400, int(display_img.height * ratio)),
                Image.LANCZOS)

        st.image(display_img,
                 caption=f"📁 {uploaded.name} "
                          f"({pil_image.width}×{pil_image.height}px)",
                 use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            analyze_btn = st.button(
                "🔍  Analyze Fruit",
                type="primary",
                use_container_width=True)
        with col_b:
            use_llm = st.toggle("AI Report", value=True)

        st.markdown("""
<div style="background:white;border-radius:10px;padding:14px 16px;
            border:1px solid #e8e8e8;margin-top:12px;font-size:13px;
            color:#666;">
<strong>💡 Tips for best results</strong><br>
- Plain background (white / light)<br>
- Single fruit in frame<br>
- Good natural lighting<br>
- Fruit fills 60%+ of image
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div style="background:white;border-radius:14px;padding:48px 24px;
            text-align:center;border:2px dashed #d0d0d0;color:#aaa;">
    <div style="font-size:48px;margin-bottom:12px">🍊</div>
    <div style="font-size:16px;font-weight:600;
                color:#888;margin-bottom:6px">
        Drop your fruit image here
    </div>
    <div style="font-size:13px">
        Supports JPG, JPEG, PNG, WEBP · Max 10MB
    </div>
</div>
""", unsafe_allow_html=True)
        analyze_btn = False
        use_llm     = True

# ── RIGHT — Results ───────────────────────────────────────
with right:
    st.markdown(
        '<div class="section-header">📊 Analysis Results</div>',
        unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
<div style="background:white;border-radius:14px;padding:48px 24px;
            text-align:center;border:1px solid #e8e8e8;color:#bbb;">
    <div style="font-size:40px;margin-bottom:12px">📊</div>
    <div style="font-size:15px;font-weight:500;color:#ccc">
        Results will appear here
    </div>
    <div style="font-size:13px;margin-top:6px">
        Upload an image and click Analyze
    </div>
</div>
""", unsafe_allow_html=True)

    elif uploaded and analyze_btn:

        with st.spinner("🔬 Analyzing fruit..."):
            result = predict_fn(pil_image)

        if result["status"] == "uncertain":
            st.warning(f"⚠️ {result['message']}")
            st.markdown("**Top predictions:**")
            for p in result["top3"]:
                st.progress(
                    p["prob"],
                    text=f"{p['class'].replace('_',' ').title()}"
                         f"  ·  {p['prob']:.1%}")

        elif result["status"] == "error":
            st.error(f"❌ {result['message']}")

        else:
            grade = result["grade"]
            color = GRADE_COLOR[grade]
            emoji = GRADE_EMOJI[grade]
            bg    = GRADE_BG[grade]

            if result.get("warning"):
                st.warning(f"⚠️ {result['warning']}")

            st.markdown(
                f"""<div class="grade-badge"
                    style="background:{bg};
                           border:2px solid {color}30;">
                    <div style="font-size:13px;font-weight:600;
                                color:{color};opacity:0.8;
                                text-transform:uppercase;
                                letter-spacing:0.08em;
                                margin-bottom:4px">
                        {result['fruit'].capitalize()} · Quality Assessment
                    </div>
                    <div style="font-size:34px;font-weight:800;
                                color:{color};line-height:1.1">
                        {emoji} Grade {grade}
                    </div>
                    <div style="font-size:16px;color:{color};
                                opacity:0.85;margin-top:4px;
                                font-weight:500">
                        {result['condition'].capitalize()} ·
                        {result['grade_label']}
                    </div>
                </div>""",
                unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence",
                       f"{result['confidence']:.1%}")
            c2.metric("Freshness Score",
                       f"{result['freshness_score']:.4f}")
            c3.metric("Shelf Life", result["shelf_life"])

            c4, c5 = st.columns(2)
            c4.metric("Risk Level",     result["risk_level"])
            c5.metric("Recommendation", result["recommendation"])

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-header">Top Predictions</div>',
                unsafe_allow_html=True)
            for p in result["top3"]:
                lbl = p["class"].replace("_", " ").title()
                st.progress(p["prob"],
                            text=f"{lbl}  ·  {p['prob']:.2%}")

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("🔬 Grad-CAM — Model Attention Map"):
                st.image(
                    result["gradcam"],
                    caption="Red = regions the model focused on most.",
                    use_column_width=True)

            if use_llm:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-header">'
                    '🤖 AI Quality Report</div>',
                    unsafe_allow_html=True)

                with st.spinner("Generating AI report..."):
                    llm = get_llm_fn(result)

                if llm.get("status") == "success":
                    st.markdown(
                        f"<div style='font-size:12px;color:#aaa;"
                        f"margin-bottom:12px;'>"
                        f"Generated by {llm['model']}</div>",
                        unsafe_allow_html=True)

                    sections = [
                        ("📋 Quality Summary",
                         "quality_summary",  "#3498db"),
                        ("🌡️ Storage Advice",
                         "storage_advice",   "#27ae60"),
                        ("⏰ Consumption Window",
                         "consumption_window","#e67e22"),
                        ("🥗 Nutritional Impact",
                         "nutritional_impact","#8e44ad"),
                    ]

                    for row_items in [sections[:2], sections[2:]]:
                        cols = st.columns(2)
                        for col, (title, key, clr) in zip(
                                cols, row_items):
                            with col:
                                content = llm.get(key, "")
                                st.markdown(
                                    f"""<div class="llm-card"
                                    style="border-left:4px solid {clr};">
                                    <div style="font-size:13px;
                                        font-weight:700;
                                        color:{clr};
                                        margin-bottom:8px">
                                        {title}
                                    </div>
                                    <div style="font-size:13.5px;
                                        color:#444;line-height:1.6">
                                        {content}
                                    </div>
                                    </div>""",
                                    unsafe_allow_html=True)

                elif llm.get("status") == "unavailable":
                    st.warning(
                        "⚠️ AI Report unavailable. "
                        "Check API keys in Streamlit secrets.")

    elif uploaded and not analyze_btn:
        st.markdown("""
<div style="background:white;border-radius:14px;padding:36px 24px;
            text-align:center;border:1px solid #e8e8e8;color:#bbb;">
    <div style="font-size:36px;margin-bottom:10px">👈</div>
    <div style="font-size:14px;color:#aaa">
        Click <strong style="color:#888">Analyze Fruit</strong>
        to run the AI model
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:16px;color:#bbb;font-size:12px;
            border-top:1px solid #e8e8e8;">
    Fruit Quality AI &nbsp;·&nbsp; EfficientNetB0 + CBAM &nbsp;·&nbsp;
    99.57% accuracy &nbsp;·&nbsp; 18,984 training images &nbsp;·&nbsp;
    Built with TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)