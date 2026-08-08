"""
SilentTalk - Two-Way Communication Using Indian Sign Language Recognition

Main Streamlit application.
"""

from __future__ import annotations

import sys
import tempfile
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

st.set_page_config(
    page_title="SilentTalk - ISL Communication",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .sub-header { color: #64748b; margin-bottom: 1.5rem; }
    .result-box {
        background: #f0f9ff; border-left: 4px solid #0284c7;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def init_session_state() -> None:
    defaults = {
        "recognized_text": "",
        "tulu_text": "",
        "speech_text": "",
        "last_gesture": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> str:
    st.sidebar.title("SilentTalk")
    st.sidebar.markdown("**Two-Way ISL Communication**")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Select Mode",
        [
            "Sign → Text → Tulu → Speech",
            "Speech → Text (Whisper)",
            "About Project",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Team**")
    st.sidebar.markdown("Akshitha | Hardhika | Kavya")
    st.sidebar.markdown("Guide: Mr. Kiran Ankalakoti")
    st.sidebar.markdown("CEC, Mangaluru | AY 2025-26")

    predictor = get_predictor()
    if not predictor.model_loaded:
        st.sidebar.warning(
            "No trained gesture model found. Run data collection & training first."
        )
        st.sidebar.code("python scripts/collect_gestures.py\npython scripts/train_model.py")

    return mode


def sign_to_speech_mode() -> None:
    st.markdown('<p class="main-header">Sign Language → Tulu Speech</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Show ISL gestures to the webcam. Recognized text is translated to Tulu and spoken.</p>',
        unsafe_allow_html=True,
    )

    col_cam, col_output = st.columns([1.2, 1])

    with col_cam:
        st.subheader("Live Webcam Feed")
        run_camera = st.checkbox("Start Camera", value=False)
        frame_placeholder = st.empty()
        gesture_placeholder = st.empty()

    with col_output:
        st.subheader("Recognition Output")
        english_text = st.text_area(
            "Recognized ISL Text (English)",
            value=st.session_state.recognized_text,
            height=100,
            key="english_area",
        )

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            add_letter = st.button("Add to Sentence")
        with col_btn2:
            clear_text = st.button("Clear Text")
        with col_btn3:
            backspace = st.button("Backspace")

        if add_letter and st.session_state.last_gesture:
            st.session_state.recognized_text += st.session_state.last_gesture
        if clear_text:
            st.session_state.recognized_text = ""
        if backspace:
            st.session_state.recognized_text = st.session_state.recognized_text[:-1]

        english_text = st.session_state.recognized_text
        st.text_area("Current Sentence", value=english_text, height=80, disabled=True)

        translate_btn = st.button("Translate to Tulu & Speak", type="primary")

    if translate_btn and english_text.strip():
        with st.spinner("Translating to Tulu..."):
            translator = get_translator()
            tulu = translator.translate(english_text)
            st.session_state.tulu_text = tulu

        with st.spinner("Generating speech..."):
            tts = get_tts()
            audio_path = tts.synthesize(tulu)

        append_conversation(
            {
                "mode": "sign_to_speech",
                "english": english_text,
                "tulu": tulu,
            }
        )

        if audio_path:
            st.audio(str(audio_path), format="audio/mp3")

    if st.session_state.tulu_text:
        st.markdown(
            f'<div class="result-box"><strong>Tulu Translation:</strong> {st.session_state.tulu_text}</div>',
            unsafe_allow_html=True,
        )

    # ── Camera state management ──────────────────────────────────────────────
    # Track whether the camera is actively running across Streamlit reruns.
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    # Sync checkbox → state
    if run_camera and not st.session_state.camera_running:
        st.session_state.camera_running = True
        st.rerun()
    if not run_camera and st.session_state.camera_running:
        st.session_state.camera_running = False
        st.rerun()

    if st.session_state.camera_running:
        predictor = get_predictor()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        if not cap.isOpened():
            st.error("Could not open webcam. Check camera permissions.")
            st.session_state.camera_running = False
            cap.release()
            return

        # Render the Stop button BEFORE the loop so Streamlit sees it on reruns
        with col_cam:
            stop_btn = st.button("Stop Camera", type="secondary")

        if stop_btn:
            st.session_state.camera_running = False
            cap.release()
            predictor.reset_history()
            st.rerun()

        # Single-frame capture per Streamlit rerun — avoids blocking the event loop
        ret, frame = cap.read()
        cap.release()

        if ret:
            result = predictor.process_frame(frame)
            annotated = result["annotated_frame"]
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            if result["gesture"]:
                st.session_state.last_gesture = result["gesture"]
                gesture_placeholder.success(
                    f"Gesture: **{result['gesture']}** ({result['confidence']:.0%} confidence)"
                )
            else:
                gesture_placeholder.info("Show a hand gesture to the camera")
        else:
            gesture_placeholder.warning("Could not read frame from camera.")

        # Trigger next rerun to get the next frame
        import time
        time.sleep(0.04)  # ~25 fps
        st.rerun()

    else:
        st.info("Enable **Start Camera** to begin gesture recognition.")


def speech_to_text_mode() -> None:
    st.markdown('<p class="main-header">Speech → Text (Two-Way Communication)</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Record live audio or upload a file. OpenAI Whisper converts speech to text.</p>',
        unsafe_allow_html=True,
    )

    tab_live, tab_upload = st.tabs(["🎙️ Live Microphone", "📁 Upload Audio File"])

    # ── Live mic via streamlit-webrtc ────────────────────────────────────────
    with tab_live:
        st.markdown("Click **Start** to open the microphone, speak, then click **Stop**.")

        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
            import av
            import io
            import wave

            # Accumulate audio frames in session state
            if "audio_frames" not in st.session_state:
                st.session_state.audio_frames = []

            # Callback to collect raw audio frames
            def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
                pcm = frame.to_ndarray()  # shape: (channels, samples)
                st.session_state.audio_frames.append(
                    (pcm, frame.sample_rate, frame.layout.name)
                )
                return frame

            ctx = webrtc_streamer(
                key="stt-mic",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                audio_frame_callback=audio_frame_callback,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
            )

            # Transcribe once the stream stops
            if not ctx.state.playing and st.session_state.audio_frames:
                if st.button("Transcribe recorded audio", type="primary", key="transcribe_live"):
                    with st.spinner("Saving audio and transcribing..."):
                        # Write accumulated PCM frames to a WAV file
                        pcm_all, sample_rate, layout = st.session_state.audio_frames[0]
                        import numpy as np

                        arrays = [f[0] for f in st.session_state.audio_frames]
                        pcm_combined = np.concatenate(arrays, axis=-1)

                        # Flatten to mono if multi-channel
                        if pcm_combined.ndim > 1:
                            pcm_combined = pcm_combined.mean(axis=0)

                        tmp_wav = config.AUDIO_DIR / "live_recording.wav"
                        config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                        with wave.open(str(tmp_wav), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(sample_rate)
                            pcm_int16 = (pcm_combined * 32767).astype(np.int16)
                            wf.writeframes(pcm_int16.tobytes())

                        stt = get_stt()
                        text = stt.transcribe(tmp_wav)
                        st.session_state.speech_text = text
                        st.session_state.audio_frames = []

                    append_conversation({"mode": "speech_to_text_live", "transcript": text})
                    st.rerun()

            if st.button("Clear recording buffer", key="clear_audio"):
                st.session_state.audio_frames = []
                st.rerun()

        except ImportError:
            st.warning(
                "streamlit-webrtc is not installed. Install it with:\n\n"
                "```\npip install streamlit-webrtc av\n```\n\n"
                "Then restart the app."
            )

    # ── File upload ──────────────────────────────────────────────────────────
    with tab_upload:
        audio_file = st.file_uploader(
            "Upload audio (WAV, MP3, M4A, WebM)",
            type=["wav", "mp3", "m4a", "webm"],
            key="audio_upload",
        )

        if audio_file is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(audio_file.name).suffix
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = Path(tmp.name)

            st.audio(audio_file)

            if st.button("Transcribe with Whisper", type="primary", key="transcribe_upload"):
                with st.spinner("Transcribing... (first run downloads the Whisper model)"):
                    stt = get_stt()
                    text = stt.transcribe(tmp_path)
                    st.session_state.speech_text = text

                append_conversation({"mode": "speech_to_text", "transcript": text})

    # ── Shared transcript output ─────────────────────────────────────────────
    if st.session_state.speech_text:
        st.markdown("---")
        st.markdown(
            f'<div class="result-box"><strong>Transcript:</strong> {st.session_state.speech_text}</div>',
            unsafe_allow_html=True,
        )


def about_mode() -> None:
    st.markdown('<p class="main-header">About SilentTalk</p>', unsafe_allow_html=True)
    st.markdown(
        """
        **SilentTalk** is an AI-powered two-way communication system for Indian Sign Language (ISL).

        ### Objectives
        1. Recognize ISL gestures using computer vision and deep learning (CNN/LSTM)
        2. Convert gestures to **Tulu** text and speech via IndicTrans2 and TTS
        3. Enable two-way communication using **OpenAI Whisper** STT

        ### Architecture
        ```
        Webcam → MediaPipe Landmarks → CNN/LSTM Classifier → English Text
              → IndicTrans2 → Tulu Text → gTTS → Speech Output

        Microphone → Whisper STT → Text (for hearing users → deaf users)
        ```

        ### Tech Stack
        - Python, OpenCV, MediaPipe, TensorFlow, scikit-learn
        - IndicTrans2, OpenAI Whisper, gTTS
        - Streamlit UI

        ### Phase-II Workflow
        1. Collect gesture data: `python scripts/collect_gestures.py`
        2. Train model: `python scripts/train_model.py`
        3. Run app: `streamlit run app.py`
        """
    )


def main() -> None:
    init_session_state()
    mode = render_sidebar()

    if mode == "Sign → Text → Tulu → Speech":
        sign_to_speech_mode()
    elif mode == "Speech → Text (Whisper)":
        speech_to_text_mode()
    else:
        about_mode()


if __name__ == "__main__":
    main()
