"""
Streamlit Frontend - Emotional Voice Translation System
Updated: Auto-Duration Recording (Smart Silence Detection)
"""

import streamlit as st
import requests
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile

# Configuration
API_BASE_URL = "http://localhost:8000"
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01  # Amplitude threshold for silence
SILENCE_DURATION = 2.5    # Seconds of silence to trigger stop
MAX_DURATION = 60         # Maximum safety duration in seconds

# Language Mapping
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-CN",
    "Russian": "ru",
    "Arabic": "ar"
}

st.set_page_config(
    page_title=" Voice Translation Pro",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00a8e8;
        --bg-light: #f5f7fa;
        --text-secondary: #65757d;
        --border-color: #e0e0e0;
    }
    * { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: var(--bg-light); }
    
    .header-container {
        background: linear-gradient(135deg, #0066cc 0%, #00a8e8 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.2);
    }
    .header-title { font-size: 2.5rem; font-weight: 700; margin: 0; color: white; }
    
    .card {
        background: white; padding: 2rem; border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); margin-bottom: 1.5rem;
    }
    .card-title { font-size: 1.3rem; font-weight: 600; color: #1a1a2e; margin-bottom: 1rem; }
    
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #00a8e8 100%);
        color: white; border: none; padding: 0.8rem 2rem;
        border-radius: 8px; font-weight: 600; width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,102,204,0.3); }
    
    .latency-badge {
        background: linear-gradient(135deg, #FF6347, #FF4500);
        color: white; padding: 0.8rem 1.5rem; border-radius: 10px;
        font-weight: 600; text-align: center; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'api_status' not in st.session_state: st.session_state.api_status = False
if 'recorded_audio' not in st.session_state: st.session_state.recorded_audio = None
if 'translation_result' not in st.session_state: st.session_state.translation_result = None
if 'voice_analysis' not in st.session_state: st.session_state.voice_analysis = None

# Helper Functions
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except: return False

def record_audio_smart():
    """Records audio until silence is detected"""
    try:
        st.toast("🎙️ Listening... Speak now!", icon="🔴")
        placeholder = st.empty()
        placeholder.markdown('<div style="color:#d63031; font-weight:bold;">🔴 Recording... (Stop speaking to finish)</div>', unsafe_allow_html=True)
        
        recorded_frames = []
        silent_chunks = 0
        chunk_duration = 0.5  # seconds
        chunk_size = int(SAMPLE_RATE * chunk_duration)
        max_chunks = int(MAX_DURATION / chunk_duration)
        silence_limit_chunks = int(SILENCE_DURATION / chunk_duration)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            for _ in range(max_chunks):
                chunk, overflow = stream.read(chunk_size)
                recorded_frames.append(chunk)
                
                # Check volume (RMS)
                rms = np.sqrt(np.mean(chunk**2))
                
                if rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0  # Reset if speech detected
                
                # Stop if silence persists
                if silent_chunks > silence_limit_chunks:
                    placeholder.success("✅ Silence detected. Processing...")
                    break
        
        # Convert list of chunks to single array
        audio_data = np.concatenate(recorded_frames, axis=0)
        
        # Trim the silence from the end
        if silent_chunks > 0:
            trim_samples = int(silent_chunks * chunk_size)
            audio_data = audio_data[:-trim_samples]
            
        # Convert to int16 for saving
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        return audio_int16, SAMPLE_RATE
    except Exception as e:
        st.error(f"Recording Error: {e}")
        return None, None

def save_temp_audio(audio, sr):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, sr, audio)
    return temp.name

def process_workflow(audio_file, src_lang_code, tgt_lang_code, voice_engine):
    start_time = time.time()
    steps = []
    
    # 1. Voice Analysis
    with st.spinner("🔍 Analyzing voice characteristics..."):
        try:
            files = {"file": open(audio_file, 'rb')}
            resp = requests.post(f"{API_BASE_URL}/analyze-voice", files=files)
            if resp.status_code == 200 and isinstance(resp.json(), dict):
                analysis = resp.json().get("analysis", {})
                st.session_state.voice_analysis = analysis
                steps.append(("📊 Voice Analysis", f"{analysis.get('voice_type','female').title()}", "✅"))
            else:
                st.session_state.voice_analysis = {"voice_type": "female"} 
        except Exception: 
            st.session_state.voice_analysis = {"voice_type": "female"}

    # 2. Speech to Text
    with st.spinner("📝 Transcribing..."):
        try:
            files = {"file": open(audio_file, 'rb')}
            resp = requests.post(f"{API_BASE_URL}/speech-to-text", files=files, params={"language": src_lang_code})
            
            if resp.status_code != 200:
                st.error("Transcription failed")
                return None
            
            data = resp.json()
            if not isinstance(data, dict): return None
                
            original_text = data.get("text", "")
            if not original_text:
                st.warning("No speech detected.")
                return None
            steps.append(("📝 Transcription", original_text[:30]+"...", "✅"))
        except Exception as e:
            st.error(f"Error: {e}")
            return None

    # 3. Translate
    with st.spinner("🌍 Translating..."):
        try:
            resp = requests.post(f"{API_BASE_URL}/translate", 
                               params={"text": original_text, "source_language": src_lang_code, "target_language": tgt_lang_code})
            
            if resp.status_code != 200:
                st.error("Translation failed")
                return None

            data = resp.json()
            if isinstance(data, list) and data: data = data[0]
            if not isinstance(data, dict): return None

            translated_text = data.get("translated_text", "")
            emotion = data.get("detected_emotion", "neutral")
            steps.append(("🌍 Translation", translated_text[:30]+"...", "✅"))
            steps.append(("😊 Emotion", emotion.upper(), "✅"))
        except Exception: return None

    # 4. TTS
    with st.spinner("🔊 Generating Audio..."):
        gender = st.session_state.voice_analysis.get("voice_type", "female")
        tts_payload = {
            "text": translated_text,
            "language": tgt_lang_code,
            "emotion": emotion,
            "gender": gender,
            "engine": "edge" if voice_engine == "Neural (Edge TTS)" else "gtts"
        }
        
        try:
            resp = requests.post(f"{API_BASE_URL}/tts", json=tts_payload)
            if resp.status_code == 200:
                audio_data = resp.json()
                dl_url = f"{API_BASE_URL}{audio_data.get('download_url')}"
                audio_content = requests.get(dl_url).content
                return {
                    "original": original_text,
                    "translated": translated_text,
                    "emotion": emotion,
                    "audio": audio_content,
                    "steps": steps,
                    "latency": time.time() - start_time
                }
        except Exception: st.error("TTS Failed")
    return None

# Main UI
st.markdown('<div class="header-container"><h1 class="header-title">🎤 Voice Translation Engine </h1></div>', unsafe_allow_html=True)

# Status
api_live = check_api_health()
st.session_state.api_status = api_live
if api_live:
    st.markdown('<div style="color: green; font-weight: bold;">✅ Backend Connected</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color: red; font-weight: bold;">❌ Backend Offline</div>', unsafe_allow_html=True)

col_conf, col_main = st.columns([1, 2])

with col_conf:
    st.markdown('<div class="card"><div class="card-title">⚙️ Settings</div>', unsafe_allow_html=True)
    
    src_lang_name = st.selectbox("Speak Language", list(LANGUAGES.keys()), index=1)
    tgt_lang_name = st.selectbox("Translate To", list(LANGUAGES.keys()), index=0)
    
    src_lang_code = LANGUAGES[src_lang_name]
    tgt_lang_code = LANGUAGES[tgt_lang_name]
    
    voice_engine = st.radio("Voice Model", ["Neural (Edge TTS)", "Standard (gTTS)"])
    
    # REMOVED DURATION SLIDER HERE
    st.info("ℹ️ Recording automatically stops when you stop speaking.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_main:
    st.markdown('<div class="card"><div class="card-title">🎙️ Action Center</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔴 Record Live", use_container_width=True):
            if not api_live: st.error("Start backend first!")
            else:
                # Calls the new SMART recorder
                audio, sr = record_audio_smart()
                if audio is not None:
                    fpath = save_temp_audio(audio, sr)
                    st.session_state.recorded_audio = fpath
                    
    with c2:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.recorded_audio = None
            st.session_state.translation_result = None
            st.rerun()

    if st.session_state.recorded_audio and not st.session_state.translation_result:
        res = process_workflow(st.session_state.recorded_audio, src_lang_code, tgt_lang_code, voice_engine)
        if res:
            st.session_state.translation_result = res
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.translation_result:
    res = st.session_state.translation_result
    st.markdown(f'<div class="latency-badge">⏱️ Total Time: {res["latency"]:.2f}s</div>', unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        st.metric("Detected Emotion", res['emotion'].capitalize())
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
        v = st.session_state.voice_analysis if st.session_state.voice_analysis else {}
        st.metric("Detected Voice", f"{v.get('voice_type','--').title()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    c_text, c_audio = st.columns(2)
    with c_text:
        st.markdown('<div class="card"><b>Original:</b><br>' + res['original'] + '<hr><b>Translated:</b><br>' + res['translated'] + '</div>', unsafe_allow_html=True)
    with c_audio:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.audio(res['audio'], format='audio/mp3')
        st.download_button("⬇️ Download MP3", res['audio'], "translation.mp3", "audio/mpeg", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)