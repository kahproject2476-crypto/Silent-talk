# -*- coding: utf-8 -*-
"""
SilentTalk — Two-Way ISL Communication
Professional redesigned UI.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.gesture.predictor import GesturePredictor
from src.speech.stt import SpeechToText
from src.speech.tts import TextToSpeech
from src.translation.indic_translator import TuluTranslator
from src.utils.helpers import append_conversation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SilentTalk",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Reset & base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #0A0F1E;
        color: #E2E8F0;
    }
    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

    /* ── Navbar ── */
    .st-navbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 1rem 2rem;
        background: rgba(15, 23, 42, 0.95);
        border-bottom: 1px solid rgba(59,130,246,0.2);
        backdrop-filter: blur(12px);
        position: sticky; top: 0; z-index: 999;
        margin: 0 -2rem 2rem -2rem;
    }
    .st-brand {
        display: flex; align-items: center; gap: 0.6rem;
    }
    .st-brand-icon { font-size: 1.6rem; }
    .st-brand-name {
        font-size: 1.4rem; font-weight: 700;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .st-brand-tag {
        font-size: 0.7rem; color: #64748B; margin-top: -4px; letter-spacing: 0.05em;
    }
    .st-team {
        font-size: 0.75rem; color: #64748B; text-align: right; line-height: 1.6;
    }
    .st-team strong { color: #94A3B8; }

    /* ── Mode tabs ── */
    .mode-tabs {
        display: flex; gap: 0.5rem; margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(59,130,246,0.15);
        padding-bottom: 0;
    }
    .mode-tab {
        padding: 0.6rem 1.4rem;
        border-radius: 8px 8px 0 0;
        font-size: 0.875rem; font-weight: 500;
        cursor: pointer; border: none;
        transition: all 0.2s;
        background: transparent; color: #64748B;
        border-bottom: 2px solid transparent;
    }
    .mode-tab:hover { color: #CBD5E1; background: rgba(59,130,246,0.05); }
    .mode-tab.active {
        color: #3B82F6;
        border-bottom: 2px solid #3B82F6;
        background: rgba(59,130,246,0.08);
    }

    /* ── Cards / panels ── */
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(59,130,246,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(8px);
    }
    .panel-title {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #475569; margin-bottom: 1rem;
    }

    /* ── Camera frame ── */
    .camera-placeholder {
        width: 100%; aspect-ratio: 4/3;
        background: #0D1526;
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        flex-direction: column; gap: 0.75rem;
        color: #334155;
    }
    .camera-placeholder svg { width: 48px; height: 48px; opacity: 0.4; }
    .camera-placeholder p { font-size: 0.875rem; margin: 0; }

    /* ── Gesture result ── */
    .gesture-badge {
        display: flex; align-items: center; gap: 1rem;
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 12px; padding: 1rem 1.25rem;
        margin-top: 0.75rem;
    }
    .gesture-label {
        font-size: 1.6rem; font-weight: 700; color: #F8FAFC;
        letter-spacing: 0.02em; flex: 1;
    }
    .gesture-conf { font-size: 0.8rem; color: #64748B; text-align: right; }
    .conf-bar-bg {
        width: 80px; height: 5px;
        background: rgba(255,255,255,0.1); border-radius: 9999px; margin-top: 4px;
    }
    .conf-bar-fill {
        height: 5px; border-radius: 9999px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        transition: width 0.3s ease;
    }
    .gesture-idle {
        font-size: 0.875rem; color: #475569;
        border: 1px dashed rgba(59,130,246,0.2);
        border-radius: 12px; padding: 1rem 1.25rem;
        text-align: center; margin-top: 0.75rem;
    }

    /* ── Sentence builder ── */
    .sentence-display {
        background: #070C18;
        border: 1px solid rgba(59,130,246,0.15);
        border-radius: 10px;
        padding: 0.875rem 1rem;
        font-size: 1.1rem; font-weight: 500;
        color: #F1F5F9;
        min-height: 54px;
        letter-spacing: 0.04em;
        word-break: break-all;
        margin-bottom: 0.75rem;
    }
    .sentence-placeholder { color: #334155; }

    /* ── Translation result ── */
    .tulu-box {
        background: linear-gradient(135deg, rgba(139,92,246,0.12), rgba(59,130,246,0.08));
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 12px; padding: 1.25rem;
        margin-top: 1rem;
    }
    .tulu-label { font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #8B5CF6; margin-bottom: 0.4rem; }
    .tulu-text { font-size: 1.2rem; color: #E2E8F0; line-height: 1.6; }

    /* ── STT result ── */
    .transcript-box {
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.2);
        border-radius: 12px; padding: 1.25rem;
        margin-top: 1rem;
    }
    .transcript-label { font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #10B981; margin-bottom: 0.4rem; }
    .transcript-text { font-size: 1rem; color: #E2E8F0; line-height: 1.7; }

    /* ── Status pill ── */
    .status-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.72rem; font-weight: 500;
        padding: 0.25rem 0.75rem; border-radius: 9999px;
    }
    .status-live {
        background: rgba(16,185,129,0.15); color: #10B981;
        border: 1px solid rgba(16,185,129,0.3);
    }
    .status-offline {
        background: rgba(239,68,68,0.12); color: #EF4444;
        border: 1px solid rgba(239,68,68,0.25);
    }
    .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .dot.pulse { animation: pulse 1.5s infinite; }
    @keyframes pulse {
        0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 8px !important; font-weight: 500 !important;
        font-size: 0.875rem !important; transition: all 0.2s !important;
        border: 1px solid rgba(59,130,246,0.3) !important;
        background: rgba(59,130,246,0.1) !important;
        color: #93C5FD !important;
    }
    .stButton > button:hover {
        background: rgba(59,130,246,0.2) !important;
        border-color: rgba(59,130,246,0.5) !important;
        color: #DBEAFE !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6, #6366F1) !important;
        border: none !important; color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9 !important; transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(239,68,68,0.1) !important;
        border: 1px solid rgba(239,68,68,0.3) !important;
        color: #FCA5A5 !important;
    }

    /* ── Inputs ── */
    .stTextArea textarea, .stTextInput input {
        background: #070C18 !important; color: #E2E8F0 !important;
        border: 1px solid rgba(59,130,246,0.2) !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(59,130,246,0.5) !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.1) !important;
    }
    label { color: #64748B !important; font-size: 0.8rem !important; }

    /* ── About page ── */
    .about-hero {
        text-align: center; padding: 3rem 1rem 2rem;
    }
    .about-hero h1 {
        font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .about-hero p { color: #64748B; font-size: 1.05rem; }
    .tech-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
        margin: 1.5rem 0;
    }
    .tech-card {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(59,130,246,0.15);
        border-radius: 12px; padding: 1.25rem;
        text-align: center;
    }
    .tech-card .icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .tech-card h4 { font-size: 0.875rem; font-weight: 600;
        color: #CBD5E1; margin: 0 0 0.25rem; }
    .tech-card p { font-size: 0.75rem; color: #475569; margin: 0; }
    .arch-flow {
        display: flex; align-items: center; gap: 0.5rem;
        flex-wrap: wrap; justify-content: center;
        padding: 1.5rem; background: rgba(7,12,24,0.8);
        border-radius: 12px; border: 1px solid rgba(59,130,246,0.1);
    }
    .arch-step {
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 8px; padding: 0.5rem 0.875rem;
        font-size: 0.78rem; color: #93C5FD; font-weight: 500;
    }
    .arch-arrow { color: #334155; font-size: 1rem; }

    /* ── Tabs override ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important; gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #475569 !important; border-radius: 8px !important;
        font-size: 0.875rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,0.1) !important;
        color: #3B82F6 !important;
    }
    .stTabs [data-baseweb="tab-border"] { background: rgba(59,130,246,0.2) !important; }

    /* ── Checkbox ── */
    .stCheckbox label { color: #94A3B8 !important; font-size: 0.875rem !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_predictor() -> GesturePredictor:
    return GesturePredictor()

@st.cache_resource
def get_translator() -> TuluTranslator:
    return TuluTranslator()

@st.cache_resource
def get_tts() -> TextToSpeech:
    return TextToSpeech()

@st.cache_resource
def get_stt() -> SpeechToText:
    return SpeechToText()


# ── Session state ─────────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "mode": "sign",          # "sign" | "speech" | "about"
        "recognized_text": "",
        "tulu_text": "",
        "speech_text": "",
        "last_gesture": "",
        "last_confidence": 0.0,
        "camera_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Navbar ────────────────────────────────────────────────────────────────────
def render_navbar() -> None:
    predictor = get_predictor()
    status_class = "status-live" if predictor.model_loaded else "status-offline"
    status_dot   = "dot pulse"  if predictor.model_loaded else "dot"
    status_text  = "Model Ready" if predictor.model_loaded else "No Model"

    st.markdown(f"""
    <div class="st-navbar">
        <div class="st-brand">
            <span class="st-brand-icon">🤟</span>
            <div>
                <div class="st-brand-name">SilentTalk</div>
                <div class="st-brand-tag">ISL · TULU · AI</div>
            </div>
        </div>
        <span class="status-pill {status_class}">
            <span class="{status_dot}"></span>{status_text}
        </span>
        <div class="st-team">
            <strong>Akshitha &nbsp;·&nbsp; Hardhika &nbsp;·&nbsp; Kavya</strong><br>
            Guide: Mr. Kiran Ankalakoti &nbsp;·&nbsp; CEC Mangaluru 2025–26
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Mode selector ─────────────────────────────────────────────────────────────
def render_mode_selector() -> str:
    col1, col2, col3, _ = st.columns([1.4, 1.4, 1, 6])
    with col1:
        if st.button("🤟  Sign → Speech", use_container_width=True,
                     type="primary" if st.session_state.mode == "sign" else "secondary"):
            st.session_state.mode = "sign"
            st.rerun()
    with col2:
        if st.button("🎙️  Speech → Text", use_container_width=True,
                     type="primary" if st.session_state.mode == "speech" else "secondary"):
            st.session_state.mode = "speech"
            st.rerun()
    with col3:
        if st.button("ℹ️  About", use_container_width=True,
                     type="primary" if st.session_state.mode == "about" else "secondary"):
            st.session_state.mode = "about"
            st.rerun()
    return st.session_state.mode


# ── Sign → Speech mode ────────────────────────────────────────────────────────
def sign_to_speech_mode() -> None:
    cam_col, out_col = st.columns([1.4, 1], gap="large")

    # ── Left: camera panel ────────────────────────────────────────────────────
    with cam_col:
        st.markdown('<div class="panel-title">Live Camera Feed</div>', unsafe_allow_html=True)

        frame_placeholder = st.empty()
        gesture_placeholder = st.empty()

        # Camera toggle
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("▶  Start Camera", use_container_width=True, type="primary",
                                  disabled=st.session_state.camera_running)
        with btn_col2:
            stop_btn = st.button("■  Stop Camera", use_container_width=True, type="secondary",
                                 disabled=not st.session_state.camera_running)

        if start_btn:
            st.session_state.camera_running = True
            st.rerun()
        if stop_btn:
            st.session_state.camera_running = False
            get_predictor().reset_history()
            st.rerun()

        # Camera idle placeholder
        if not st.session_state.camera_running:
            frame_placeholder.markdown("""
            <div class="camera-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
                    <path d="M23 7l-7 5 7 5V7z"/>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                </svg>
                <p>Camera is off — click Start Camera</p>
            </div>""", unsafe_allow_html=True)
            gesture_placeholder.markdown(
                '<div class="gesture-idle">🖐 Show a hand gesture to the camera</div>',
                unsafe_allow_html=True)

    # ── Right: output panel ───────────────────────────────────────────────────
    with out_col:
        st.markdown('<div class="panel-title">Recognition & Translation</div>', unsafe_allow_html=True)

        # Sentence display
        sentence = st.session_state.recognized_text
        if sentence:
            st.markdown(f'<div class="sentence-display">{sentence}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="sentence-display sentence-placeholder">Your sentence will appear here...</div>',
                unsafe_allow_html=True)

        # Action buttons
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("＋ Add Word", use_container_width=True):
                if st.session_state.last_gesture:
                    st.session_state.recognized_text += " " + st.session_state.last_gesture \
                        if st.session_state.recognized_text else st.session_state.last_gesture
                    st.rerun()
        with b2:
            if st.button("⌫ Backspace", use_container_width=True):
                words = st.session_state.recognized_text.strip().split()
                st.session_state.recognized_text = " ".join(words[:-1])
                st.rerun()
        with b3:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.recognized_text = ""
                st.session_state.tulu_text = ""
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        translate_btn = st.button("🔄  Translate to Tulu & Speak",
                                  type="primary", use_container_width=True,
                                  disabled=not sentence.strip())

        # Tulu translation result
        if st.session_state.tulu_text:
            st.markdown(f"""
            <div class="tulu-box">
                <div class="tulu-label">Tulu Translation</div>
                <div class="tulu-text">{st.session_state.tulu_text}</div>
            </div>""", unsafe_allow_html=True)

    # ── Translation trigger (outside columns so audio renders full-width) ─────
    if translate_btn and sentence.strip():
        with st.spinner("Translating to Tulu..."):
            tulu = get_translator().translate(sentence)
            st.session_state.tulu_text = tulu
        with st.spinner("Generating speech..."):
            audio_path = get_tts().synthesize(tulu)
        append_conversation({"mode": "sign_to_speech", "english": sentence, "tulu": tulu})
        if audio_path:
            st.audio(str(audio_path), format="audio/mp3")
        st.rerun()

    # ── Live camera loop ──────────────────────────────────────────────────────
    if st.session_state.camera_running:
        predictor = get_predictor()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        if not cap.isOpened():
            st.error("Cannot open webcam. Check camera permissions.")
            st.session_state.camera_running = False
            cap.release()
            return

        ret, frame = cap.read()
        cap.release()

        if ret:
            result = predictor.process_frame(frame)
            annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            if result["gesture"]:
                st.session_state.last_gesture = result["gesture"]
                st.session_state.last_confidence = result["confidence"]
                conf_pct = int(result["confidence"] * 100)
                gesture_placeholder.markdown(f"""
                <div class="gesture-badge">
                    <div class="gesture-label">{result['gesture']}</div>
                    <div class="gesture-conf">
                        {conf_pct}% confidence
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{conf_pct}%"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                gesture_placeholder.markdown(
                    '<div class="gesture-idle">🖐 Show a hand gesture to the camera</div>',
                    unsafe_allow_html=True)
        else:
            gesture_placeholder.markdown(
                '<div class="gesture-idle" style="color:#EF4444">Could not read frame</div>',
                unsafe_allow_html=True)

        time.sleep(0.04)
        st.rerun()


# ── Speech → Text mode ────────────────────────────────────────────────────────
def speech_to_text_mode() -> None:
    st.markdown('<div class="panel-title">Two-Way Communication — Hearing Users → Deaf Users</div>',
                unsafe_allow_html=True)

    tab_live, tab_upload = st.tabs(["🎙️  Live Microphone", "📁  Upload Audio File"])

    with tab_live:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av, wave
            import numpy as np

            if "audio_frames" not in st.session_state:
                st.session_state.audio_frames = []

            def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
                st.session_state.audio_frames.append(
                    (frame.to_ndarray(), frame.sample_rate, frame.layout.name))
                return frame

            st.markdown("""
            <div class="glass-card" style="margin-bottom:1rem">
                <div style="font-size:0.875rem;color:#94A3B8;line-height:1.7">
                    Click <strong style="color:#3B82F6">Start</strong> to open your microphone,
                    speak clearly, then click <strong style="color:#EF4444">Stop</strong>.
                    Hit <em>Transcribe</em> to convert your speech to text via Whisper.
                </div>
            </div>""", unsafe_allow_html=True)

            ctx = webrtc_streamer(
                key="stt-mic",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                audio_frame_callback=audio_frame_callback,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🎤  Transcribe Recording", type="primary",
                             use_container_width=True, key="transcribe_live",
                             disabled=not st.session_state.audio_frames):
                    with st.spinner("Transcribing with Whisper..."):
                        arrays = [f[0] for f in st.session_state.audio_frames]
                        sr = st.session_state.audio_frames[0][1]
                        import numpy as _np
                        combined = _np.concatenate(arrays, axis=-1)
                        if combined.ndim > 1:
                            combined = combined.mean(axis=0)
                        tmp_wav = config.AUDIO_DIR / "live_recording.wav"
                        config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                        with wave.open(str(tmp_wav), "wb") as wf:
                            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                            wf.writeframes((combined * 32767).astype(_np.int16).tobytes())
                        st.session_state.speech_text = get_stt().transcribe(tmp_wav)
                        st.session_state.audio_frames = []
                    append_conversation({"mode": "stt_live",
                                         "transcript": st.session_state.speech_text})
                    st.rerun()
            with c2:
                if st.button("✕  Clear Buffer", use_container_width=True, key="clear_audio"):
                    st.session_state.audio_frames = []
                    st.rerun()

        except ImportError:
            st.warning("Install streamlit-webrtc: `pip install streamlit-webrtc av`")

    with tab_upload:
        st.markdown("<br>", unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload audio (WAV · MP3 · M4A · WebM)",
                                      type=["wav", "mp3", "m4a", "webm"], key="audio_upload")
        if audio_file:
            st.audio(audio_file)
            if st.button("🎤  Transcribe with Whisper", type="primary",
                         use_container_width=True, key="transcribe_upload"):
                with st.spinner("Transcribing... (first run downloads Whisper model)"):
                    with tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(audio_file.name).suffix) as tmp:
                        tmp.write(audio_file.read())
                    st.session_state.speech_text = get_stt().transcribe(Path(tmp.name))
                append_conversation({"mode": "stt_upload",
                                     "transcript": st.session_state.speech_text})
                st.rerun()

    # Shared transcript
    if st.session_state.speech_text:
        st.markdown(f"""
        <div class="transcript-box">
            <div class="transcript-label">Transcript</div>
            <div class="transcript-text">{st.session_state.speech_text}</div>
        </div>""", unsafe_allow_html=True)

        if st.button("✕  Clear Transcript", key="clear_transcript"):
            st.session_state.speech_text = ""
            st.rerun()


# ── About mode ────────────────────────────────────────────────────────────────
def about_mode() -> None:
    st.markdown("""
    <div class="about-hero">
        <h1>SilentTalk</h1>
        <p>Two-Way Communication Using Indian Sign Language Recognition</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="panel-title">System Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="arch-flow">
            <span class="arch-step">📷 Webcam</span>
            <span class="arch-arrow">→</span>
            <span class="arch-step">MediaPipe</span>
            <span class="arch-arrow">→</span>
            <span class="arch-step">Random Forest</span>
            <span class="arch-arrow">→</span>
            <span class="arch-step">English Text</span>
        </div>
        <div style="text-align:center;margin:0.5rem 0;color:#334155;font-size:1.1rem">↓</div>
        <div class="arch-flow">
            <span class="arch-step">🔊 gTTS</span>
            <span class="arch-arrow">←</span>
            <span class="arch-step">IndicTrans2</span>
            <span class="arch-arrow">←</span>
            <span class="arch-step">Tulu Text</span>
        </div>
        <div style="text-align:center;margin:0.75rem 0;color:#334155">─────────────</div>
        <div class="arch-flow">
            <span class="arch-step">🎙️ Microphone</span>
            <span class="arch-arrow">→</span>
            <span class="arch-step">Whisper STT</span>
            <span class="arch-arrow">→</span>
            <span class="arch-step">Text Output</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Model Status</div>', unsafe_allow_html=True)
        predictor = get_predictor()
        if predictor.model_loaded:
            import json
            labels = json.loads((config.LABELS_PATH).read_text(encoding="utf-8"))
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.75rem">
                    <span style="color:#94A3B8;font-size:0.875rem">Random Forest Classifier</span>
                    <span class="status-pill status-live"><span class="dot pulse"></span>Loaded</span>
                </div>
                <div style="font-size:0.8rem;color:#64748B;margin-bottom:0.5rem">
                    {len(labels)} trained gesture classes
                </div>
                <div style="font-size:0.75rem;color:#334155;line-height:1.8;word-break:break-word">
                    {" · ".join(labels)}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <span class="status-pill status-offline"><span class="dot"></span>No model loaded</span>
                <p style="font-size:0.8rem;color:#64748B;margin-top:0.75rem">
                    Run: <code>python scripts/train_model.py</code>
                </p>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="panel-title">Technology Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="tech-grid">
            <div class="tech-card">
                <div class="icon">👁️</div>
                <h4>MediaPipe</h4>
                <p>21-point hand landmark extraction</p>
            </div>
            <div class="tech-card">
                <div class="icon">🌲</div>
                <h4>Random Forest</h4>
                <p>45-class ISL gesture classifier</p>
            </div>
            <div class="tech-card">
                <div class="icon">🧠</div>
                <h4>CNN-LSTM</h4>
                <p>Dynamic gesture sequences</p>
            </div>
            <div class="tech-card">
                <div class="icon">🌐</div>
                <h4>IndicTrans2</h4>
                <p>English → Tulu translation</p>
            </div>
            <div class="tech-card">
                <div class="icon">🔊</div>
                <h4>gTTS</h4>
                <p>Tulu speech synthesis</p>
            </div>
            <div class="tech-card">
                <div class="icon">🎙️</div>
                <h4>Whisper</h4>
                <p>Speech-to-text for hearing users</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel-title">Project Goals</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex;flex-direction:column;gap:0.75rem">
                <div style="display:flex;gap:0.75rem;align-items:flex-start">
                    <span style="color:#3B82F6;font-size:1rem;margin-top:1px">①</span>
                    <span style="font-size:0.875rem;color:#94A3B8">
                        Recognize ISL gestures in real-time using computer vision
                    </span>
                </div>
                <div style="display:flex;gap:0.75rem;align-items:flex-start">
                    <span style="color:#8B5CF6;font-size:1rem;margin-top:1px">②</span>
                    <span style="font-size:0.875rem;color:#94A3B8">
                        Translate recognized text to Tulu language with speech output
                    </span>
                </div>
                <div style="display:flex;gap:0.75rem;align-items:flex-start">
                    <span style="color:#EC4899;font-size:1rem;margin-top:1px">③</span>
                    <span style="font-size:0.875rem;color:#94A3B8">
                        Enable two-way communication between deaf and hearing users
                    </span>
                </div>
                <div style="display:flex;gap:0.75rem;align-items:flex-start">
                    <span style="color:#10B981;font-size:1rem;margin-top:1px">④</span>
                    <span style="font-size:0.875rem;color:#94A3B8">
                        Expand vocabulary with CNN-LSTM for dynamic gesture sequences
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    init_session_state()
    inject_css()
    render_navbar()
    render_mode_selector()

    st.markdown("---", unsafe_allow_html=False)

    mode = st.session_state.mode
    if mode == "sign":
        sign_to_speech_mode()
    elif mode == "speech":
        speech_to_text_mode()
    else:
        about_mode()


if __name__ == "__main__":
    main()
