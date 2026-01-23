"""
FastAPI Backend for Emotional Voice Cloning & TTS System
Handles audio processing, speech recognition, translation, and synthesis
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import uuid
import asyncio
from typing import Optional
import traceback

# Audio processing
import soundfile as sf
import librosa
import numpy as np
from scipy.io.wavfile import write

# Speech & NLP
import sounddevice as sd
from gtts import gTTS
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import pipeline

# Data models
from pydantic import BaseModel
import json

# ============================================================================
# Configuration & Setup
# ============================================================================

app = FastAPI(
    title="Emotional Voice Cloning API",
    description="FastAPI backend for multilingual speech processing with emotion",
    version="1.0.0"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# Initialize models
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
# Data Models
# ============================================================================

class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en"
    emotion: str = "neutral"
    use_voice_cloning: bool = False

class SpeechTranslationRequest(BaseModel):
    target_language: str
    emotion: str = "neutral"
    use_voice_cloning: bool = False

class EmotionalTTSRequest(BaseModel):
    text: str
    language: str = "en"
    emotion: str = "neutral"

class VoiceAnalysisResponse(BaseModel):
    pitch_hz: float
    voice_type: str
    characteristics: dict

# ============================================================================
# Utility Functions
# ============================================================================

def cleanup_file(file_path: str):
    """Cleanup temporary files after delay"""
    async def delayed_cleanup():
        await asyncio.sleep(3600)  # 1 hour
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
    return delayed_cleanup()

def analyze_voice(audio_path: str) -> dict:
    """Analyze voice characteristics"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # Use YIN algorithm for better pitch detection
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), 
                        fmax=librosa.note_to_hz('C7'), sr=sr)
        
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) > 0:
            avg_pitch = float(np.median(valid_f0))
        else:
            # Fallback
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(float(pitch))
            avg_pitch = float(np.median(pitch_values)) if pitch_values else 180.0
        
        avg_pitch = float(np.clip(avg_pitch, 80, 400))
        
        # Determine voice type
        voice_type = "male" if avg_pitch < 165 else "female"
        
        return {
            "pitch_hz": avg_pitch,
            "voice_type": voice_type,
            "energy": float(np.mean(np.abs(y))),
            "duration_seconds": float(len(y) / sr)
        }
    except Exception as e:
        return {"error": str(e)}

def apply_emotional_effects(y: np.ndarray, sr: int, emotion: str) -> np.ndarray:
    """Apply emotional effects to audio"""
    
    emotion_params = {
        'joy': {'pitch': 2.5, 'speed': 1.18, 'volume': 1.3},
        'sadness': {'pitch': -3.0, 'speed': 0.82, 'volume': 0.65},
        'anger': {'pitch': 1.8, 'speed': 1.28, 'volume': 1.6},
        'fear': {'pitch': 3.5, 'speed': 1.35, 'volume': 1.3},
        'surprise': {'pitch': 4.0, 'speed': 1.22, 'volume': 1.5},
        'disgust': {'pitch': -1.5, 'speed': 0.92, 'volume': 0.9},
        'neutral': {'pitch': 0.0, 'speed': 1.0, 'volume': 1.0}
    }
    
    params = emotion_params.get(emotion.lower(), emotion_params['neutral'])
    
    # Apply pitch shift
    if abs(params['pitch']) > 0.1:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=params['pitch'])
    
    # Apply time stretch (speed)
    if abs(params['speed'] - 1.0) > 0.05:
        y = librosa.effects.time_stretch(y, rate=params['speed'])
    
    # Apply volume
    y = y * params['volume']
    
    # Prevent clipping
    max_val = np.abs(y).max()
    if max_val > 1.0:
        y = y / max_val * 0.95
    
    return y

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Emotional Voice Cloning API"}

# ============================================================================
# Voice Analysis Endpoints
# ============================================================================

@app.post("/analyze-voice")
async def analyze_voice_endpoint(file: UploadFile = File(...)):
    """Analyze voice characteristics from uploaded audio"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze
        characteristics = analyze_voice(str(file_path))
        
        # Cleanup
        os.remove(file_path)
        
        return {
            "success": True,
            "characteristics": characteristics
        }
    except Exception as e:
        return {"success": False, "error": str(e)}, 400

# ============================================================================
# TTS Endpoints
# ============================================================================

@app.post("/tts")
async def text_to_speech(request: TextToSpeechRequest, background_tasks: BackgroundTasks):
    """Convert text to speech with optional emotion"""
    try:
        # Generate base speech
        tts = gTTS(text=request.text, lang=request.language, slow=False)
        
        # Save temporarily
        temp_file = TEMP_DIR / f"{uuid.uuid4()}.mp3"
        tts.save(str(temp_file))
        
        # If emotion requested, apply effects
        if request.emotion != "neutral":
            y, sr = librosa.load(str(temp_file), sr=None)
            y = apply_emotional_effects(y, sr, request.emotion)
            
            output_file = OUTPUT_DIR / f"tts_{uuid.uuid4()}.wav"
            sf.write(str(output_file), y, sr)
            os.remove(temp_file)
        else:
            output_file = OUTPUT_DIR / f"tts_{uuid.uuid4()}.mp3"
            shutil.move(str(temp_file), str(output_file))
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, str(output_file))
        
        return {
            "success": True,
            "audio_file": output_file.name,
            "download_url": f"/download/{output_file.name}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}, 400

# ============================================================================
# Speech-to-Text Endpoints
# ============================================================================

@app.post("/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = "en"
):
    """Convert speech to text using Google Speech Recognition"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load audio
        y, sr = librosa.load(str(file_path), sr=16000)
        
        # In production, you'd use a proper speech recognition API
        # For now, return placeholder
        # Using: SpeechRecognition library with Google API
        
        import speech_recognition as sr_lib
        recognizer = sr_lib.Recognizer()
        
        with sr_lib.AudioFile(str(file_path)) as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio, language=language)
        except sr_lib.UnknownValueError:
            text = ""
        
        # Cleanup
        os.remove(file_path)
        
        return {
            "success": True,
            "text": text,
            "language": language
        }
    except Exception as e:
        return {"success": False, "error": str(e)}, 400

# ============================================================================
# Translation Endpoints
# ============================================================================

@app.post("/translate")
async def translate_text(
    text: str,
    source_language: str = "auto",
    target_language: str = "en"
):
    """Translate text between languages"""
    try:
        translator = GoogleTranslator(source=source_language, target=target_language)
        translated = translator.translate(text)
        
        # Detect emotion
        emotion = "neutral"
        emotion_score = 0.0
        
        if emotion_classifier:
            try:
                result = emotion_classifier(translated)
                emotion = result[0][0]['label']
                emotion_score = float(result[0][0]['score'])
            except:
                pass
        
        return {
            "success": True,
            "original_text": text,
            "translated_text": translated,
            "source_language": source_language,
            "target_language": target_language,
            "detected_emotion": emotion,
            "emotion_score": emotion_score
        }
    except Exception as e:
        return {"success": False, "error": str(e)}, 400

# ============================================================================
# Emotion Detection
# ============================================================================

@app.post("/detect-emotion")
async def detect_emotion(text: str):
    """Detect emotion in text"""
    try:
        if not emotion_classifier:
            return {"success": False, "error": "Emotion classifier not loaded"}
        
        result = emotion_classifier(text)
        emotion = result[0][0]['label']
        score = float(result[0][0]['score'])
        
        return {
            "success": True,
            "text": text,
            "emotion": emotion,
            "score": score
        }
    except Exception as e:
        return {"success": False, "error": str(e)}, 400

# ============================================================================
# File Download
# ============================================================================

@app.get("/download/{filename}")
async def download_file(filename: str, background_tasks: BackgroundTasks):
    """Download generated audio file"""
    try:
        file_path = OUTPUT_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Schedule cleanup after download
        background_tasks.add_task(cleanup_file, str(file_path))
        
        return FileResponse(
            str(file_path),
            media_type="audio/mpeg",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# Batch Processing
# ============================================================================

@app.post("/batch-tts")
async def batch_tts(texts: list[str], language: str = "en", emotion: str = "neutral"):
    """Process multiple texts"""
    try:
        results = []
        
        for text in texts:
            tts = gTTS(text=text, lang=language, slow=False)
            output_file = OUTPUT_DIR / f"batch_{uuid.uuid4()}.mp3"
            tts.save(str(output_file))
            
            results.append({
                "text": text,
                "audio_file": output_file.name,
                "download_url": f"/download/{output_file.name}"
            })
        
        return {
            "success": True,
            "total": len(results),
            "results": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}, 400

# ============================================================================
# Root
# ============================================================================

@app.get("/")
async def root():
    """API documentation"""
    return {
        "name": "Emotional Voice Cloning API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "voice_analysis": "POST /analyze-voice",
            "text_to_speech": "POST /tts",
            "speech_to_text": "POST /speech-to-text",
            "translate": "POST /translate",
            "detect_emotion": "POST /detect-emotion",
            "download": "GET /download/{filename}",
            "batch_tts": "POST /batch-tts"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)