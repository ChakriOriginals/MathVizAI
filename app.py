"""
MathVizAI – Streamlit Frontend
A clean, dark-themed UI for generating math animations.
"""

import io
import json
import time
from pathlib import Path

import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MathVizAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0e14;
    color: #e8eaf6;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #48cae4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #8b9ecf;
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.step-card {
    background: #161822;
    border: 1px solid #2a2d42;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}

.step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6c63ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}

.step-title {
    font-size: 1rem;
    font-weight: 600;
    color: #c8d0f0;
}

.status-success { color: #4caf50; }
.status-failed  { color: #f44336; }
.status-running { color: #ffc107; }

.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #48cae4);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.88; }

.code-box {
    background: #0a0b10;
    border: 1px solid #2a2d42;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    overflow-x: auto;
    white-space: pre-wrap;
    color: #a8b4e0;
}

hr { border-color: #2a2d42; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">MathVizAI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Transform math concepts into beautiful animations — powered by AI + Manim</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Input Panel ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📝 Input")

    input_mode = st.radio(
        "Input type",
        ["Topic / text", "Upload PDF"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_text = ""
    uploaded_file = None

    if input_mode == "Topic / text":
        raw_text = st.text_area(
            "Enter a math topic or short excerpt",
            placeholder="e.g.  Central Limit Theorem\n\nor paste a paragraph from a paper...",
            height=200,
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a PDF (max 10 pages)",
            type=["pdf"],
            help="Only the first 10 pages are processed.",
        )

    difficulty = st.selectbox(
        "Difficulty level",
        ["undergraduate", "high_school"],
        index=0,
    )

    generate_clicked = st.button("🎬  Generate Animation", use_container_width=True)


# ── Status / Output Panel ─────────────────────────────────────────────────────
with col_right:
    st.markdown("### 📊 Pipeline Status")
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    st.markdown("### 🎥 Output")
    video_placeholder = st.empty()
    download_placeholder = st.empty()


# ── Generation Logic ──────────────────────────────────────────────────────────
if generate_clicked:
    has_input = raw_text.strip() or uploaded_file is not None
    if not has_input:
        st.error("Please enter a topic/text or upload a PDF before generating.")
        st.stop()

    # Reset UI
    status_placeholder.empty()
    video_placeholder.empty()
    download_placeholder.empty()
    progress_bar.progress(0)

    steps = [
        ("Parser Agent", "Extracting mathematical content..."),
        ("Concept Agent", "Identifying core concepts..."),
        ("Pedagogy Planner", "Designing learning sequence..."),
        ("Scene Generator", "Building animation instructions..."),
        ("Code Generator", "Writing Manim code..."),
        ("Renderer", "Rendering video..."),
    ]

    with status_placeholder.container():
        for i, (name, desc) in enumerate(steps):
            st.markdown(
                f'<div class="step-card">'
                f'<div class="step-label">Step {i+1} of {len(steps)}</div>'
                f'<div class="step-title">⏳ {name} — {desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    progress_bar.progress(5)

    # ── Call API ──────────────────────────────────────────────────────────────
    with st.spinner("Running pipeline (this may take 1–3 minutes)..."):
        try:
            if uploaded_file is not None:
                response = requests.post(
                    f"{API_BASE}/generate-video-from-pdf",
                    files={"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")},
                    data={"difficulty_level": difficulty},
                    timeout=1200,
                )
            else:
                response = requests.post(
                    f"{API_BASE}/generate-video",
                    json={"topic_or_text": raw_text, "difficulty_level": difficulty},
                    timeout=1200,
                )
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot reach the backend at http://localhost:8000.  "
                "Make sure the FastAPI server is running:\n\n"
                "`uvicorn backend.api.main:app --reload`"
            )
            st.stop()
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The pipeline took too long — try a simpler topic.")
            st.stop()

    progress_bar.progress(95)

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        st.stop()

    data = response.json()
    progress_bar.progress(100)

    # ── Display Results ───────────────────────────────────────────────────────
    if data.get("status") == "success":
        status_placeholder.success("✅ Pipeline completed successfully!")

        job_id = data["job_id"]
        video_path = data.get("video_path", "")

        if video_path and Path(video_path).exists():
            with video_placeholder:
                video_bytes = Path(video_path).read_bytes()
                st.video(video_bytes)

            with download_placeholder:
                st.download_button(
                    label="⬇️  Download MP4",
                    data=Path(video_path).read_bytes(),
                    file_name=f"mathvizai_{job_id[:8]}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
        else:
            st.info(
                f"Video rendered successfully! Download via API:\n\n"
                f"`GET {API_BASE}/download/{job_id}`"
            )

        # ── Debug / Trace Expander ────────────────────────────────────────────
        trace = data.get("pipeline_trace")
        if trace:
            with st.expander("🔍 Pipeline trace (debug)", expanded=False):
                st.json(trace)

    else:
        error_msg = data.get("error", "Unknown error.")
        status_placeholder.error(f"❌ Pipeline failed:\n\n{error_msg}")

        trace = data.get("pipeline_trace")
        if trace:
            with st.expander("🔍 Debug trace", expanded=True):
                st.json(trace)


# ── Sidebar: Examples ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Example topics")
    examples = [
        "Central Limit Theorem",
        "Eigenvectors and eigenvalues",
        "Fourier Transform intuition",
        "Gradient descent optimization",
        "Bayes' Theorem",
        "The derivative as a limit",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            # Streamlit can't programmatically fill a text_area from a button
            # but we surface the example text so the user can copy it
            st.info(f"Copy this into the input box:\n\n**{ex}**")

    st.markdown("---")
    st.markdown("### ⚙️ API")
    st.markdown(
        f"Backend: [{API_BASE}]({API_BASE}/docs)\n\nSwagger docs available at `/docs`"
    )
