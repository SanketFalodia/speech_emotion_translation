"""
Enhanced version of speech_translate.py with emotional TTS support
This imports the original functionality and adds emotional speech
"""

import sounddevice as sd
from emotion_tts_windows import text_to_emotional_speech  # Windows-compatible version
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
    sd.wait()
    print("✅ Recording complete!")
    return audio, sample_rate

# Save audio to file
def save_audio(audio, sample_rate, filename="temp_audio.wav"):
    write(filename, sample_rate, audio)
    return filename

# Recognize speech using Google Speech API
def recognize_speech_from_file(audio_file, language="en-US"):
    try:
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        url = "http://www.google.com/speech-api/v2/recognize"
        params = {
            'output': 'json',
            'lang': language,
            'key': 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw'
        }
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

# Detect emotion from text and return clean emotion name
def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        label = result[0][0]['label']
        score = result[0][0]['score']
        return label, score
    except Exception as e:
        return "neutral", 0.0

# Play audio file
def play_audio(audio_file):
    """Play audio file using platform-specific command"""
    print(f"🔊 Playing emotional audio...")
    try:
        if platform.system() == "Windows":
            os.system(f"start {audio_file}")
        elif platform.system() == "Darwin":
            os.system(f"afplay {audio_file}")
        else:
            os.system(f"mpg123 {audio_file}")
    except Exception as e:
        print(f"❌ Could not play audio: {e}")

# Main workflow with emotional TTS
def main():
    print("=" * 60)
    print("🎙️  Multilingual Speech Emotion Recognition & Translation")
    print("    with Emotional Text-to-Speech")
    print("=" * 60)
    
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

        # Detect emotion
        emotion_label, emotion_score = detect_emotion(text)
        print(f"😊 Emotion: {emotion_label.capitalize()} ({emotion_score * 100:.1f}%)")

        # Translate text
        dest_lang = input("\nEnter target language code for translation (e.g., 'en', 'fr', 'hi', 'bn'): ").strip()
        translated = translate_text(text, dest_lang)
        
        if translated:
            print(f"📝 Translated Text: {translated}")
            
            # Ask user if they want emotional TTS
            use_emotional = input("\n🎭 Use emotional text-to-speech? (y/n, default=y): ").strip().lower()
            
            if use_emotional != 'n':
                print(f"\n🎨 Generating speech with {emotion_label.upper()} emotion...")
                try:
                    # Generate emotional speech
                    output_file = text_to_emotional_speech(translated, dest_lang, emotion_label)
                    play_audio(output_file)
                    print(f"✅ Emotional audio saved as: {output_file}")
                except Exception as e:
                    print(f"❌ Emotional TTS failed: {e}")
                    print("   Falling back to standard TTS...")
                    # Fallback to regular TTS
                    tts = gTTS(text=translated, lang=dest_lang)
                    tts.save("output.mp3")
                    play_audio("output.mp3")
            else:
                # Use standard TTS
                print("\n🔊 Generating standard speech...")
                tts = gTTS(text=translated, lang=dest_lang)
                tts.save("output.mp3")
                play_audio("output.mp3")
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