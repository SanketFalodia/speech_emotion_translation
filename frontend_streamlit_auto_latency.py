"""
Streamlit Frontend - Emotional Voice Translation System
Modern, User-Friendly Interface with Live Audio Recording
Single Workflow: Record → Translate → Analyze → Listen
AUTO PROCESSING + LATENCY TRACKING
"""

import streamlit as st
import requests
import io
import os
import tempfile
from pathlib import Path
import time
import numpy as np
from scipy.io.wavfile import write
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
from datetime import datetime


# Configuration

API_BASE_URL = "http://localhost:8000"
SAMPLE_RATE = 16000
RECORDING_DURATION = 10


st.set_page_config(
    page_title="🎤 Voice Translation",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)


 
# Modern CSS Styling (Blue/Teal Theme)
 


st.markdown("""
<style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00a8e8;
        --accent-color: #00d9ff;
        --success-color: #00c853;
        --warning-color: #ff9800;
        --error-color: #ff5252;
        --bg-light: #f5f7fa;
        --bg-white: #ffffff;
        --text-primary: #1a1a2e;
        --text-secondary: #65757d;
        --border-color: #e0e0e0;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main Container */
    .main {
        background-color: var(--bg-light);
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #0066cc 0%, #00a8e8 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.2);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
    }
    
    /* Card Styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: box-shadow 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.15);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #00a8e8 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f1ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0066cc;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Emotion Badge */
    .emotion-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }
    
    .emotion-joy {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #1a1a2e;
    }
    
    .emotion-sadness {
        background: linear-gradient(135deg, #4169E1, #1E90FF);
        color: white;
    }
    
    .emotion-anger {
        background: linear-gradient(135deg, #DC143C, #FF4500);
        color: white;
    }
    
    .emotion-fear {
        background: linear-gradient(135deg, #8B008B, #9932CC);
        color: white;
    }
    
    .emotion-surprise {
        background: linear-gradient(135deg, #FF6347, #FF1493);
        color: white;
    }
    
    .emotion-disgust {
        background: linear-gradient(135deg, #90EE90, #32CD32);
        color: #1a1a2e;
    }
    
    .emotion-neutral {
        background: linear-gradient(135deg, #A9A9A9, #808080);
        color: white;
    }
    
    /* Status Indicators */
    .status-success {
        background-color: #e8f5e9;
        border-left: 4px solid #00c853;
        padding: 1rem;
        border-radius: 6px;
        color: #1b5e20;
    }
    
    .status-error {
        background-color: #ffebee;
        border-left: 4px solid #ff5252;
        padding: 1rem;
        border-radius: 6px;
        color: #b71c1c;
    }
    
    .status-info {
        background-color: #e3f2fd;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 6px;
        color: #0d47a1;
    }
    
    .status-processing {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 6px;
        color: #e65100;
    }
    
    /* Recording Indicator */
    .recording-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #ff5252;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Progress Bar */
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Text Input Styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 2px solid var(--border-color);
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #0066cc;
    }
    
    /* Audio Player */
    .audio-player {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    /* Latency Display */
    .latency-badge {
        background: linear-gradient(135deg, #FF6347, #FF4500);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.9rem;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border-color);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.8rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State


if 'api_status' not in st.session_state:
    st.session_state.api_status = False
    
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
    
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = None
    
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = None

if 'processing_latency' not in st.session_state:
    st.session_state.processing_latency = None

if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False


# Helper Functions


def check_api_health():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def record_audio_live(duration=RECORDING_DURATION, sr=SAMPLE_RATE):
    """Record audio from microphone"""
    try:
        st.info(f"🎙️ Recording for {duration} seconds... Please speak now!")
        
        # Record
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
        
        # Progress indicator
        progress_bar = st.progress(0)
        for i in range(duration):
            time.sleep(1)
            progress_bar.progress((i + 1) / duration)
        
        sd.wait()
        st.success("✅ Recording complete!")
        
        return audio, sr
    except Exception as e:
        st.error(f"❌ Recording failed: {str(e)}")
        return None, None


def save_temp_audio(audio, sr):
    """Save audio to temporary file"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, sr, audio)
        return temp_file.name
    except Exception as e:
        st.error(f"❌ Failed to save audio: {str(e)}")
        return None


def emotion_badge_html(emotion: str, score: float = 1.0) -> str:
    """Create HTML badge for emotion"""
    emotion_class = f"emotion-{emotion.lower()}"
    score_text = f"({score*100:.1f}%)" if score < 1.0 else ""
    return f'<span class="emotion-badge {emotion_class}">{emotion.upper()} {score_text}</span>'


def translate_speech_workflow(audio_file, source_lang, target_lang):
    """Complete speech translation workflow with latency tracking"""
    
    workflow_steps = []
    start_time = time.time()  # ⏱️ START TRACKING LATENCY
    
    # Step 1: Speech to Text
    step_start = time.time()
    with st.spinner("🔄 Converting speech to text..."):
        try:
            files = {"file": open(audio_file, 'rb')}
            response = requests.post(
                f"{API_BASE_URL}/speech-to-text",
                files=files,
                params={"language": source_lang},
                timeout=30
            )
            
            step_duration = time.time() - step_start
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    recognized_text = data.get("text", "")
                    workflow_steps.append(("📝 Speech Recognition", recognized_text, "✅", f"{step_duration:.2f}s"))
                else:
                    st.error(f"❌ Speech recognition failed: {data.get('error')}")
                    return None
            else:
                st.error(f"❌ API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"❌ Speech recognition error: {str(e)}")
            return None
    
    # Step 2: Translate
    step_start = time.time()
    with st.spinner("🌍 Translating text..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/translate",
                params={
                    "text": recognized_text,
                    "source_language": source_lang,
                    "target_language": target_lang
                },
                timeout=30
            )
            
            step_duration = time.time() - step_start
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    translated_text = data.get("translated_text", "")
                    detected_emotion = data.get("detected_emotion", "neutral")
                    emotion_score = data.get("emotion_score", 0.0)
                    
                    workflow_steps.append(("🌐 Translation", translated_text, "✅", f"{step_duration:.2f}s"))
                    workflow_steps.append(("😊 Emotion Detection", f"{detected_emotion} ({emotion_score*100:.1f}%)", "✅", "0.00s"))
                    
                    st.session_state.emotion_data = {
                        "emotion": detected_emotion,
                        "score": emotion_score
                    }
                else:
                    st.error(f"❌ Translation failed: {data.get('error')}")
                    return None
            else:
                st.error(f"❌ API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"❌ Translation error: {str(e)}")
            return None
    
    # Step 3: Generate Emotional Speech
    step_start = time.time()
    with st.spinner("🎤 Generating emotional speech..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/tts",
                json={
                    "text": translated_text,
                    "language": target_lang,
                    "emotion": detected_emotion
                },
                timeout=30
            )
            
            step_duration = time.time() - step_start
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    audio_file = data.get("audio_file")
                    download_url = data.get("download_url")
                    
                    workflow_steps.append(("🔊 Speech Synthesis", "Generated with emotion", "✅", f"{step_duration:.2f}s"))
                    
                    # Download audio
                    try:
                        audio_response = requests.get(
                            f"{API_BASE_URL}{download_url}",
                            timeout=30
                        )
                        
                        # Calculate total latency
                        total_latency = time.time() - start_time  # ⏱️ STOP TRACKING
                        st.session_state.processing_latency = total_latency
                        
                        st.session_state.translation_result = {
                            "original_text": recognized_text,
                            "translated_text": translated_text,
                            "emotion": detected_emotion,
                            "emotion_score": emotion_score,
                            "audio": audio_response.content,
                            "steps": workflow_steps,
                            "total_latency": total_latency  # ⏱️ ADD TO RESULT
                        }
                        return st.session_state.translation_result
                    except Exception as e:
                        st.warning(f"⚠️ Could not download audio: {str(e)}")
                        return None
                else:
                    st.error(f"❌ Speech synthesis failed: {data.get('error')}")
                    return None
            else:
                st.error(f"❌ API Error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"❌ Speech synthesis error: {str(e)}")
            return None


# Main UI

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🎤 Voice Translation Engine</h1>
</div>
""", unsafe_allow_html=True)


# API Status Check
col1, col2, col3 = st.columns([2, 1, 1])


with col1:
    st.session_state.api_status = check_api_health()
    if st.session_state.api_status:
        st.markdown('<div class="status-success">✅ Backend API Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="status-error">
        ❌ Backend Not Connected<br>
        Run: <code>python -m uvicorn app:app --reload</code>
        </div>
        ''', unsafe_allow_html=True)


with col2:
    st.metric("API Status", "🟢 Online" if st.session_state.api_status else "🔴 Offline")


st.divider()


# Main Content
col_config, col_workflow = st.columns([1, 2])

# Sidebar Configuration

with col_config:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚙️ Settings</div>', unsafe_allow_html=True)
    
    # Language Selection
    st.subheader("🌐 Languages")
    source_lang = st.selectbox(
        "Source Language",
        ["en", "hi", "es"],
        format_func=lambda x: {
            "en": "English",
            "hi": "Hindi",
            "es": "Spanish"
        }.get(x, x),
        key="source_lang"
    )
    
    target_lang = st.selectbox(
        "Target Language",
        ["en", "hi", "es"],
        format_func=lambda x: {
            "en": "English",
            "hi": "Hindi",
            "es": "Spanish"
        }.get(x, x),
        key="target_lang"
    )
    
    # Recording Duration
    st.subheader("⏱️ Recording")
    duration = st.slider(
        "Duration (seconds)",
        min_value=5,
        max_value=60,
        value=RECORDING_DURATION,
        step=1
    )
    
    st.markdown('</div>', unsafe_allow_html=True)


# Main Workflow

with col_workflow:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎙️ Live Audio Recording</div>', unsafe_allow_html=True)
    
    # Record Button
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔴 Record Audio", use_container_width=True, key="record_btn"):
            if not st.session_state.api_status:
                st.error("❌ Backend API is not running!")
            else:
                audio, sr = record_audio_live(duration=duration)
                if audio is not None:
                    # Save audio
                    audio_file = save_temp_audio(audio, sr)
                    st.session_state.recorded_audio = audio_file
                    st.success("✅ Audio recorded successfully!")
    
    with col_btn2:
        if st.button("📂 Upload Audio", use_container_width=True, key="upload_btn"):
            st.info("Upload feature - file picker coming soon")
    
    with col_btn3:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
            st.session_state.recorded_audio = None
            st.session_state.translation_result = None
            st.session_state.emotion_data = None
            st.session_state.processing_latency = None
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


st.divider()


# AUTO-PROCESS AUDIO WHEN AVAILABLE
 
if st.session_state.recorded_audio and not st.session_state.translation_result and not st.session_state.is_processing:
    if st.session_state.api_status:
        st.session_state.is_processing = True
        
        # Show processing status
        st.markdown('<div class="status-processing">⏳ Processing your audio... Please wait</div>', unsafe_allow_html=True)
        
        # Start processing
        result = translate_speech_workflow(
            st.session_state.recorded_audio,
            source_lang,
            target_lang
        )
        
        st.session_state.is_processing = False
        
        if result:
            st.rerun()  # Refresh to show results

# Display Results

if st.session_state.translation_result:
    result = st.session_state.translation_result
    
    # ⏱️ LATENCY DISPLAY AT TOP
    st.markdown(f"""
    <div class="latency-badge">
        ⏱️ Total Processing Time: {result.get('total_latency', 0):.2f} seconds
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Workflow Steps with Individual Timings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Processing Steps</div>', unsafe_allow_html=True)
    
    for i, step_data in enumerate(result.get("steps", []), 1):
        if len(step_data) == 4:
            step_name, step_value, step_status, step_time = step_data
        else:
            step_name, step_value, step_status = step_data
            step_time = "N/A"
        
        col_step1, col_step2, col_step3, col_step4 = st.columns([0.5, 2.5, 0.8, 0.8])
        
        with col_step1:
            st.write(f"**{i}.**")
        with col_step2:
            st.write(f"**{step_name}**")
            st.caption(step_value)
        with col_step3:
            st.write(step_status)
        with col_step4:
            st.caption(f"⏱️ {step_time}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Results Display
    col_text1, col_text2 = st.columns(2)
    
    with col_text1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Original Text</div>', unsafe_allow_html=True)
        st.text_area(
            "Original",
            value=result["original_text"],
            height=100,
            disabled=True,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_text2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🌐 Translated Text</div>', unsafe_allow_html=True)
        st.text_area(
            "Translated",
            value=result["translated_text"],
            height=100,
            disabled=True,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Emotion & Metrics
    col_emotion, col_audio = st.columns([1, 1])
    
    with col_emotion:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">😊 Detected Emotion</div>', unsafe_allow_html=True)
        
        emotion = result["emotion"]
        score = result["emotion_score"]
        
        st.markdown(emotion_badge_html(emotion, score), unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Emotion", emotion.capitalize())
        with col_m2:
            st.metric("Confidence", f"{score*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_audio:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔊 Emotional Speech</div>', unsafe_allow_html=True)
        
        # Audio Player
        st.audio(result["audio"], format="audio/mpeg")
        
        # Download Button
        st.download_button(
            label="⬇️ Download Audio",
            data=result["audio"],
            file_name=f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime="audio/mpeg",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)


else:
    if not st.session_state.recorded_audio:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #65757d;">
            <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">🎙️ Ready to Start?</h3>
            <p>Click "Record Audio" button to start recording, or upload an audio file.</p>
            <p style="margin-top: 1rem; font-size: 0.9rem;">
                Your speech will be transcribed, translated, and converted back to speech with emotions! 🎉
            </p>
        </div>
        """, unsafe_allow_html=True)
