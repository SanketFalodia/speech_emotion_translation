'''
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
import requests
from langdetect import detect
from gtts import gTTS
import os
import platform
from transformers import pipeline
from deep_translator import GoogleTranslator
import json

# Initialize emotion classifier
print("Loading emotion classifier...")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Record audio using sounddevice
def record_audio(duration=5, sample_rate=16000):
    print(f"🎙️ Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("✅ Recording complete!")
    return audio, sample_rate

# Save audio to file
def save_audio(audio, sample_rate, filename="temp_audio.wav"):
    write(filename, sample_rate, audio)
    return filename

# Recognize speech using Google Speech API
def recognize_speech_from_file(audio_file, language="en-US"):
    try:
        # Read the audio file
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        # Use Google Speech-to-Text API (free tier)
        url = "http://www.google.com/speech-api/v2/recognize"
        params = {
            'output': 'json',
            'lang': language,
            'key': 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw'  # Free demo key
        }
        headers = {'Content-Type': 'audio/l16; rate=16000;'}
        
        response = requests.post(url, params=params, headers=headers, data=audio_data)
        
        # Parse response
        for line in response.content.decode('utf-8').strip().split('\n'):
            if line:
                try:
                    result = json.loads(line)
                    if 'result' in result and result['result']:
                        transcript = result['result'][0]['alternative'][0]['transcript']
                        print(f"✅ Recognized Text: {transcript}")
                        return transcript
                except:
                    continue
        
        print("❌ Could not understand the audio.")
        return None
        
    except Exception as e:
        print(f"❌ Speech recognition failed: {e}")
        return None

# Detect language from text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Translate using deep_translator
def translate_text(text, dest_lang):
    try:
        translated = GoogleTranslator(source='auto', target=dest_lang).translate(text)
        return translated
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return None

# Convert text to speech
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        print(f"🔊 Playing audio...")
        if platform.system() == "Windows":
            os.system("start output.mp3")
        elif platform.system() == "Darwin":
            os.system("afplay output.mp3")
        else:
            os.system("mpg123 output.mp3")
    except Exception as e:
        print(f"❌ Text-to-speech failed: {e}")

# Detect emotion from text
def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        label = result[0][0]['label']
        score = result[0][0]['score']
        return f"{label.capitalize()} ({score * 100:.1f}%)"
    except Exception as e:
        return f"Error detecting emotion: {e}"

# Main workflow
def main():
    print("=" * 50)
    print("🎙️  Multilingual Speech Emotion Recognition")
    print("=" * 50)
    
    print("\n📝 Supported languages:")
    print("   en - English")
    print("   hi - Hindi")
    print("   bn - Bengali")
    print("   es - Spanish")
    print("   fr - French")
    print("   de - German")
    print()
    
    input_lang = input("Enter the language you'll speak (e.g., 'en', 'hi', 'bn', 'es'): ").strip()
    
    # Map language codes to Google Speech Recognition format
    lang_map = {
        'en': 'en-US',
        'hi': 'hi-IN',
        'bn': 'bn-IN',
        'es': 'es-ES',
        'fr': 'fr-FR',
        'de': 'de-DE'
    }
    
    recognition_lang = lang_map.get(input_lang, 'en-US')
    
    # Record audio
    duration = int(input("Enter recording duration in seconds (default 5): ") or "5")
    audio, sample_rate = record_audio(duration=duration)
    
    # Save audio to file
    audio_file = save_audio(audio, sample_rate)
    
    # Recognize speech
    text = recognize_speech_from_file(audio_file, language=recognition_lang)
    
    # Clean up temporary file
    if os.path.exists(audio_file):
        os.remove(audio_file)

    if text:
        src_lang = detect_language(text)
        print(f"🌐 Detected Language: {src_lang}")

        emotion = detect_emotion(text)
        print(f"😊 Emotion: {emotion}")

        dest_lang = input("\nEnter target language code for translation (e.g., 'en', 'fr', 'hi', 'bn'): ").strip()
        translated = translate_text(text, dest_lang)
        if translated:
            print(f"📝 Translated Text: {translated}")
            text_to_speech(translated, lang=dest_lang)
        else:
            print("❌ Translation failed. Please try again.")
    else:
        print("❌ No speech detected. Please try again.")

# Entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        '''
import sounddevice as sd
from scipy.io.wavfile import write
import os
import platform
import json
import requests
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import pipeline
from emotion_tts import AdvancedEmotionalTTS

# Initialize emotion classifier
print("Loading emotion classifier...")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# -------------------- AUDIO FUNCTIONS --------------------

def record_audio(duration=5, sample_rate=16000):
    print(f"🎙️ Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording complete!")
    return audio, sample_rate

def save_audio(audio, sample_rate, filename="temp_audio.wav"):
    write(filename, sample_rate, audio)
    return filename

# -------------------- SPEECH RECOGNITION --------------------

def recognize_speech_from_file(audio_file, language="en-US"):
    try:
        with open(audio_file, 'rb') as f:
            audio_data = f.read()

        url = "http://www.google.com/speech-api/v2/recognize"
        params = {'output': 'json', 'lang': language, 'key': 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw'}
        headers = {'Content-Type': 'audio/l16; rate=16000;'}

        response = requests.post(url, params=params, headers=headers, data=audio_data)

        for line in response.content.decode('utf-8').strip().split('\n'):
            if line:
                try:
                    result = json.loads(line)
                    if 'result' in result and result['result']:
                        transcript = result['result'][0]['alternative'][0]['transcript']
                        print(f"✅ Recognized Text: {transcript}")
                        return transcript
                except:
                    continue

        print("❌ Could not understand the audio.")
        return None

    except Exception as e:
        print(f"❌ Speech recognition failed: {e}")
        return None

# -------------------- LANGUAGE DETECTION & TRANSLATION --------------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_text(text, dest_lang):
    try:
        translated = GoogleTranslator(source='auto', target=dest_lang).translate(text)
        return translated
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return None

# -------------------- EMOTION DETECTION --------------------

def detect_emotion(text):
    """
    Returns:
        label (str): emotion name
        score (float): confidence (0–1)
    """
    result = emotion_classifier(text)
    label = result[0][0]['label'].lower()
    score = result[0][0]['score']
    return label, score

# -------------------- MAIN WORKFLOW --------------------

def main():
    print("=" * 50)
    print("🎙️  Multilingual Speech Emotion Recognition")
    print("=" * 50)

    print("\n📝 Supported languages:")
    print("   en - English")
    print("   hi - Hindi")
    print("   bn - Bengali")
    print("   es - Spanish")
    print("   fr - French")
    print("   de - German\n")

    input_lang = input("Enter the language you'll speak (e.g., 'en', 'hi', 'bn', 'es'): ").strip()

    lang_map = {
        'en': 'en-US',
        'hi': 'hi-IN',
        'bn': 'bn-IN',
        'es': 'es-ES',
        'fr': 'fr-FR',
        'de': 'de-DE'
    }
    recognition_lang = lang_map.get(input_lang, 'en-US')

    duration = int(input("Enter recording duration in seconds (default 5): ") or "5")
    audio, sample_rate = record_audio(duration=duration)
    audio_file = save_audio(audio, sample_rate)

    text = recognize_speech_from_file(audio_file, language=recognition_lang)
    if os.path.exists(audio_file):
        os.remove(audio_file)

    if text:
        src_lang = detect_language(text)
        print(f"🌐 Detected Language: {src_lang}")

        emotion_label, emotion_score = detect_emotion(text)
        print(f"😊 Emotion: {emotion_label} ({emotion_score*100:.1f}%)")

        dest_lang = input("\nEnter target language code for translation (e.g., 'en', 'fr', 'hi', 'bn'): ").strip()
        translated = translate_text(text, dest_lang)

        if translated:
            print(f"📝 Translated Text: {translated}")

            # 🎭 Emotion-aware TTS
            tts = AdvancedEmotionalTTS()
            try:
                output_audio = tts.generate_emotional_speech(
                    translated,
                    lang=dest_lang,
                    emotion=emotion_label,
                    confidence=emotion_score
                )

                print("🔊 Playing emotional speech...")
                if platform.system() == "Windows":
                    os.system(f"start {output_audio}")
                elif platform.system() == "Darwin":
                    os.system(f"afplay {output_audio}")
                else:
                    os.system(f"mpg123 {output_audio}")

            finally:
                tts.cleanup()
        else:
            print("❌ Translation failed. Please try again.")
    else:
        print("❌ No speech detected. Please try again.")

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
