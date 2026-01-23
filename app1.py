"""
FastAPI Backend for Emotional Voice Cloning & TTS System
Fixed: Strict dictionary returns to prevent frontend parsing errors.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import uuid
import asyncio
import traceback
import json

# Audio processing
import soundfile as sf
import librosa
import numpy as np
from scipy.io.wavfile import write

# Speech & NLP
from gtts import gTTS
from deep_translator import GoogleTranslator
from transformers import pipeline

# New: Edge TTS for natural emotion
import edge_tts

# Data models
from pydantic import BaseModel

# ============================================================================
# Configuration & Setup
# ============================================================================

app = FastAPI(
    title="Emotional Voice Cloning API (Enhanced)",
    description="FastAPI backend for multilingual speech processing with Edge TTS & Cloning",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Initialize Emotion Classifier
print("🔄 Loading models...")
try:
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )
    print("✅ Emotion classifier loaded")
except Exception as e:
    print(f"⚠️ Could not load emotion classifier: {e}")
    emotion_classifier = None

# ============================================================================
# Edge TTS & Emotion Configuration
# ============================================================================

# Emotion-optimized voice mapping
EMOTION_VOICES = {
    'en': {
        'joy': {'male': 'en-US-GuyNeural', 'female': 'en-US-JennyNeural'},
        'sadness': {'male': 'en-US-EricNeural', 'female': 'en-US-SaraNeural'},
        'anger': {'male': 'en-US-GuyNeural', 'female': 'en-US-JennyNeural'},
        'fear': {'male': 'en-US-EricNeural', 'female': 'en-US-AriaNeural'},
        'neutral': {'male': 'en-US-GuyNeural', 'female': 'en-US-AriaNeural'}
    },
    # Fallbacks for other languages
    'es': {'default': {'male': 'es-ES-AlvaroNeural', 'female': 'es-ES-ElviraNeural'}},
    'fr': {'default': {'male': 'fr-FR-HenriNeural', 'female': 'fr-FR-DeniseNeural'}},
    'de': {'default': {'male': 'de-DE-ConradNeural', 'female': 'de-DE-KatjaNeural'}},
    'hi': {'default': {'male': 'hi-IN-MadhurNeural', 'female': 'hi-IN-SwaraNeural'}},
    'it': {'default': {'male': 'it-IT-DiegoNeural', 'female': 'it-IT-ElsaNeural'}},
    'ja': {'default': {'male': 'ja-JP-KeitaNeural', 'female': 'ja-JP-NanamiNeural'}},
    'zh': {'default': {'male': 'zh-CN-YunxiNeural', 'female': 'zh-CN-XiaoxiaoNeural'}},
    'ko': {'default': {'male': 'ko-KR-InJoonNeural', 'female': 'ko-KR-SunHiNeural'}},
}

# ============================================================================
# Data Models
# ============================================================================

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    emotion: str = "neutral"
    gender: str = "female"
    engine: str = "edge" 
    reference_audio: str | None = None

# ============================================================================
# Utility Functions
# ============================================================================

def cleanup_file(file_path: str):
    async def delayed_cleanup():
        await asyncio.sleep(3600)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
    return delayed_cleanup()

def analyze_pitch_and_gender(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # YIN algorithm for pitch
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), 
                        fmax=librosa.note_to_hz('C7'), sr=sr)
        
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) > 0:
            avg_pitch = float(np.median(valid_f0))
        else:
            avg_pitch = 180.0
        
        gender = "male" if avg_pitch < 165 else "female"
        
        return {
            "pitch_hz": round(avg_pitch, 2),
            "voice_type": gender,
            "duration": float(len(y) / sr)
        }
    except Exception as e:
        print(f"Analysis Error: {e}")
        return {"pitch_hz": 0.0, "voice_type": "female", "error": str(e)}

async def generate_edge_tts(text, lang, emotion, gender):
    try:
        emotion = emotion.lower()
        # Select Voice
        lang_config = EMOTION_VOICES.get(lang, EMOTION_VOICES.get(lang.split('-')[0], {}))
        
        if not lang_config:
             lang_config = EMOTION_VOICES['en']

        if emotion in lang_config:
            voice = lang_config[emotion].get(gender, lang_config[emotion].get('female'))
        elif 'default' in lang_config:
            voice = lang_config['default'].get(gender, lang_config['default'].get('female'))
        else:
            voice = "en-US-AriaNeural"

        print(f"🎤 Edge TTS: Generating '{emotion}' speech using {voice}")

        output_file = OUTPUT_DIR / f"edge_{uuid.uuid4()}.mp3"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_file))
        return str(output_file)

    except Exception as e:
        print(f"Edge TTS Failed: {e}")
        raise e

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "ok", "engine": "Edge TTS + OpenVoice Hooks"}

@app.post("/analyze-voice")
async def analyze_voice_endpoint(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        result = analyze_pitch_and_gender(str(file_path))
        os.remove(file_path)
        return {"success": True, "analysis": result}
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...), language: str = "en"):
    try:
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        import speech_recognition as sr_lib
        recognizer = sr_lib.Recognizer()
        with sr_lib.AudioFile(str(file_path)) as source:
            audio = recognizer.record(source)
            
        try:
            text = recognizer.recognize_google(audio, language=language)
        except sr_lib.UnknownValueError:
            text = ""
        except sr_lib.RequestError:
            text = "API Unavailable"

        os.remove(file_path)
        return {"success": True, "text": text}
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

@app.post("/translate")
async def translate_text(
    text: str,
    source_language: str = "auto",
    target_language: str = "en"
):
    """
    Translates text and detects emotion.
    ALWAYS returns a JSON Dictionary.
    """
    try:
        # Handle empty text input gracefully
        if not text or not text.strip():
            return {"success": False, "error": "No text provided for translation"}

        translator = GoogleTranslator(source=source_language, target=target_language)
        translated = translator.translate(text)
        
        # Detect emotion
        emotion = "neutral"
        score = 0.0
        if emotion_classifier:
            try:
                res = emotion_classifier(translated[:512]) 
                # Ensure we handle the list/dict structure of pipeline output safely
                if isinstance(res, list) and len(res) > 0:
                    first_res = res[0]
                    if isinstance(first_res, list) and len(first_res) > 0:
                        first_res = first_res[0] # Nested list case [[{}]]
                    
                    if isinstance(first_res, dict):
                        emotion = first_res.get('label', 'neutral')
                        score = float(first_res.get('score', 0.0))
            except Exception as e:
                print(f"Emotion detection warning: {e}")
                # Continue without crashing, default to neutral
        
        return {
            "success": True, 
            "translated_text": translated,
            "detected_emotion": emotion,
            "emotion_score": score
        }
    except Exception as e:
        print(f"Translation Error: {e}")
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

@app.post("/tts")
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    try:
        output_file = None
        
        if request.engine == "edge":
            output_file = await generate_edge_tts(
                request.text, 
                request.language, 
                request.emotion, 
                request.gender
            )
        else: 
            tts = gTTS(text=request.text, lang=request.language, slow=False)
            output_file = OUTPUT_DIR / f"gtts_{uuid.uuid4()}.mp3"
            tts.save(str(output_file))

        if not output_file:
            raise Exception("Failed to generate audio")

        filename = os.path.basename(str(output_file))
        background_tasks.add_task(cleanup_file, str(output_file))

        return {
            "success": True,
            "audio_file": filename,
            "download_url": f"/download/{filename}",
            "engine_used": request.engine
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(str(file_path), media_type="audio/mpeg", filename=filename)
    return HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)