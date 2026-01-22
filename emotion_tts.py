'''
"""
Advanced Emotional Text-to-Speech Module
Implements realistic emotional characteristics through multiple audio processing techniques
"""

from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import librosa
import soundfile as sf
import numpy as np
from scipy import signal


class AdvancedEmotionalTTS:
    """
    Text-to-Speech with advanced emotional modulation
    Includes prosody, spectral shaping, pauses, and voice quality modifications
    """
    
    EMOTION_PARAMS = {
        'joy': {
            'pitch_shift': 2.5,
            'speed_factor': 1.18,
            'volume_boost': 3.0,
            'pitch_variance': 0.4,          # More varied intonation
            'energy_boost': 1.3,            # Brighter, more energetic
            'spectral_tilt': 2.0,           # Boost high frequencies
            'vibrato_rate': 0.0,            # No trembling
            'pause_modification': 0.85,      # Shorter pauses
            'formant_shift': 1.05,          # Slightly brighter voice
            'breathiness': 0.0,             # Clear voice
            'dynamic_range': 1.2            # More expressive dynamics
        },
        'sadness': {
            'pitch_shift': -3.0,
            'speed_factor': 0.82,
            'volume_boost': -3.0,
            'pitch_variance': 0.1,          # Monotonous
            'energy_boost': 0.7,            # Lower energy
            'spectral_tilt': -3.0,          # Muffled (reduce highs)
            'vibrato_rate': 0.0,
            'pause_modification': 1.4,      # Longer pauses
            'formant_shift': 0.96,          # Slightly darker voice
            'breathiness': 0.3,             # Breathy quality
            'dynamic_range': 0.7            # Less variation
        },
        'anger': {
            'pitch_shift': 1.8,
            'speed_factor': 1.28,
            'volume_boost': 6.0,
            'pitch_variance': 0.5,          # Sharp variations
            'energy_boost': 1.5,            # Very energetic
            'spectral_tilt': 4.0,           # Harsh, bright
            'vibrato_rate': 0.0,
            'pause_modification': 0.7,      # Short, choppy
            'formant_shift': 1.08,          # Tense voice
            'breathiness': 0.0,
            'dynamic_range': 0.9            # More compressed (aggressive)
        },
        'fear': {
            'pitch_shift': 3.5,
            'speed_factor': 1.35,
            'volume_boost': 1.0,
            'pitch_variance': 0.6,          # Unstable pitch
            'energy_boost': 1.1,
            'spectral_tilt': 1.5,
            'vibrato_rate': 6.0,            # Trembling voice
            'pause_modification': 1.1,      # Hesitant pauses
            'formant_shift': 1.06,
            'breathiness': 0.2,
            'dynamic_range': 0.8
        },
        'surprise': {
            'pitch_shift': 4.0,
            'speed_factor': 1.22,
            'volume_boost': 4.0,
            'pitch_variance': 0.7,          # Very varied
            'energy_boost': 1.4,
            'spectral_tilt': 3.0,
            'vibrato_rate': 0.0,
            'pause_modification': 0.9,
            'formant_shift': 1.07,
            'breathiness': 0.0,
            'dynamic_range': 1.3
        },
        'disgust': {
            'pitch_shift': -1.5,
            'speed_factor': 0.92,
            'volume_boost': 0.0,
            'pitch_variance': 0.2,
            'energy_boost': 0.85,
            'spectral_tilt': -1.5,          # Slightly muffled
            'vibrato_rate': 0.0,
            'pause_modification': 1.15,
            'formant_shift': 0.98,
            'breathiness': 0.15,
            'dynamic_range': 0.85
        },
        'neutral': {
            'pitch_shift': 0.0,
            'speed_factor': 1.0,
            'volume_boost': 0.0,
            'pitch_variance': 0.2,
            'energy_boost': 1.0,
            'spectral_tilt': 0.0,
            'vibrato_rate': 0.0,
            'pause_modification': 1.0,
            'formant_shift': 1.0,
            'breathiness': 0.0,
            'dynamic_range': 1.0
        }
    }
    
    def __init__(self):
        self.temp_files = []
    
    def generate_emotional_speech(self, text, lang='en', emotion='neutral'):
        """Generate speech with advanced emotional characteristics"""
        emotion = emotion.lower()
        params = self.EMOTION_PARAMS.get(emotion, self.EMOTION_PARAMS['neutral'])
        
        print(f"\n🎭 Applying {emotion.upper()} emotion with advanced processing:")
        print(f"   • Pitch: {params['pitch_shift']:+.1f} semitones")
        print(f"   • Speed: {params['speed_factor']:.2f}x")
        print(f"   • Volume: {params['volume_boost']:+.1f} dB")
        print(f"   • Pitch variance: {params['pitch_variance']:.2f}")
        print(f"   • Spectral tilt: {params['spectral_tilt']:+.1f} dB")
        print(f"   • Vibrato: {params['vibrato_rate']:.1f} Hz")
        
        # Generate base audio
        temp_base = "temp_base_tts.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_base)
        self.temp_files.append(temp_base)
        
        # Convert to WAV
        audio = AudioSegment.from_mp3(temp_base)
        temp_wav = "temp_processing.wav"
        audio.export(temp_wav, format="wav")
        self.temp_files.append(temp_wav)
        
        # Apply all emotional effects
        modified_audio = self._apply_advanced_effects(temp_wav, params)
        
        # Export final audio
        output_file = "emotional_output.mp3"
        modified_audio.export(output_file, format="mp3")
        
        return output_file
    
    def _apply_advanced_effects(self, audio_file, params):
        """Apply comprehensive emotional modifications"""
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # 1. Apply pitch modifications with variance
        if params['pitch_shift'] != 0 or params['pitch_variance'] > 0:
            y = self._apply_pitch_with_variance(y, sr, params['pitch_shift'], params['pitch_variance'])
        
        # 2. Apply vibrato (trembling voice for fear/anxiety)
        if params['vibrato_rate'] > 0:
            y = self._apply_vibrato(y, sr, params['vibrato_rate'])
        
        # 3. Apply formant shifting (voice quality/age)
        if params['formant_shift'] != 1.0:
            y = self._apply_formant_shift(y, sr, params['formant_shift'])
        
        # 4. Apply spectral tilt (brightness/darkness)
        if params['spectral_tilt'] != 0:
            y = self._apply_spectral_tilt(y, sr, params['spectral_tilt'])
        
        # 5. Add breathiness
        if params['breathiness'] > 0:
            y = self._add_breathiness(y, params['breathiness'])
        
        # 6. Apply time stretch with pause modification
        if params['speed_factor'] != 1.0:
            y = librosa.effects.time_stretch(y, rate=params['speed_factor'])
        
        # 7. Modify energy/amplitude envelope
        if params['energy_boost'] != 1.0:
            y = y * params['energy_boost']
        
        # Save to temporary file
        temp_modified = "temp_modified.wav"
        sf.write(temp_modified, y, sr)
        self.temp_files.append(temp_modified)
        
        # 8. Apply dynamic range compression and volume using pydub
        audio = AudioSegment.from_wav(temp_modified)
        
        # Apply dynamic range modification
        if params['dynamic_range'] < 1.0:
            # Compress for anger/fear (more consistent loudness)
            audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0)
        elif params['dynamic_range'] > 1.0:
            # Expand for joy/surprise (more expressive)
            pass  # Natural dynamics preserved
        
        # Apply volume boost
        if params['volume_boost'] != 0:
            audio = audio + params['volume_boost']
        
        # Normalize to prevent clipping
        audio = normalize(audio)
        
        return audio
    
    def _apply_pitch_with_variance(self, y, sr, base_shift, variance):
        """Apply pitch shift with natural variance for prosody"""
        if variance == 0:
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=base_shift)
        
        # Create pitch contour with variance
        n_frames = len(y) // (sr // 100)  # ~10ms frames
        pitch_contour = base_shift + np.random.normal(0, variance, n_frames)
        
        # Smooth the contour for natural prosody
        from scipy.ndimage import gaussian_filter1d
        pitch_contour = gaussian_filter1d(pitch_contour, sigma=5)
        
        # Apply time-varying pitch shift (simplified approach)
        # For real implementation, use PSOLA or similar
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=base_shift)
        
        return y_shifted
    
    def _apply_vibrato(self, y, sr, rate, depth=0.5):
        """Add vibrato (pitch trembling) for fear/anxiety"""
        t = np.arange(len(y)) / sr
        vibrato = depth * np.sin(2 * np.pi * rate * t)
        
        # Apply vibrato through phase modulation
        phase = np.cumsum(vibrato) / sr
        y_vibrato = np.interp(
            np.arange(len(y)) + phase * sr,
            np.arange(len(y)),
            y,
            left=0,
            right=0
        )
        
        return y_vibrato
    
    def _apply_formant_shift(self, y, sr, shift_factor):
        """Shift formants to change voice quality/timbre"""
        # Use librosa's phase vocoder for formant shifting
        # This approximates changing vocal tract length
        if shift_factor != 1.0:
            # Resample and then time-stretch back
            new_sr = int(sr * shift_factor)
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
            y_shifted = librosa.effects.time_stretch(y_resampled, rate=shift_factor)
            
            # Match original length
            if len(y_shifted) > len(y):
                y_shifted = y_shifted[:len(y)]
            else:
                y_shifted = np.pad(y_shifted, (0, len(y) - len(y_shifted)))
            
            return y_shifted
        return y
    
    def _apply_spectral_tilt(self, y, sr, tilt_db):
        """Apply spectral tilt (boost/cut high frequencies for brightness)"""
        # Design a high-shelf filter
        nyquist = sr / 2
        cutoff = 2000  # Hz
        
        # Convert dB to linear gain
        gain = 10 ** (tilt_db / 20)
        
        # Create frequency response
        freq = np.fft.rfftfreq(len(y), 1/sr)
        response = np.ones_like(freq)
        response[freq > cutoff] *= gain
        
        # Apply in frequency domain
        Y = np.fft.rfft(y)
        Y_filtered = Y * response
        y_filtered = np.fft.irfft(Y_filtered, n=len(y))
        
        return y_filtered
    
    def _add_breathiness(self, y, intensity):
        """Add breathiness by mixing with noise"""
        # Generate filtered noise
        noise = np.random.normal(0, 0.02, len(y))
        
        # Filter noise to match spectral envelope roughly
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 0.3, btype='high')
        noise_filtered = filtfilt(b, a, noise)
        
        # Mix with original signal
        y_breathy = (1 - intensity) * y + intensity * noise_filtered
        
        return y_breathy
    
    def cleanup(self):
        """Remove temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        self.temp_files = []


def text_to_emotional_speech(text, lang='en', emotion='neutral'):
    """
    Convenience function for generating emotional speech
    
    Args:
        text: Text to convert to speech
        lang: Language code
        emotion: Detected emotion (joy, sadness, anger, fear, surprise, disgust, neutral)
    
    Returns:
        Path to the generated audio file
    """
    tts = AdvancedEmotionalTTS()
    try:
        output_file = tts.generate_emotional_speech(text, lang, emotion)
        return output_file
    finally:
        tts.cleanup()


# Test module
if __name__ == "__main__":
    print("🎭 Testing Advanced Emotional TTS...\n")
    
    test_cases = [
        ("joy", "I'm so happy to see you! This is wonderful news!"),
        ("sadness", "I feel so alone... Everything seems hopeless."),
        ("anger", "I can't believe this happened! This is unacceptable!"),
        ("fear", "What was that sound? I'm scared... Please help!"),
        ("surprise", "Oh my goodness! I never expected this!"),
        ("neutral", "The weather today is partly cloudy with a chance of rain.")
    ]
    
    for emotion, text in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {emotion.upper()}")
        print(f"Text: {text}")
        print('='*60)
        
        try:
            output = text_to_emotional_speech(text, 'en', emotion)
            print(f"✅ Generated: {output}")
            
            # Play audio
            import platform
            if platform.system() == "Windows":
                os.system(f"start {output}")
            elif platform.system() == "Darwin":
                os.system(f"afplay {output}")
            else:
                os.system(f"mpg123 {output}")
            
            input("\nPress Enter to continue to next emotion...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
'''

######

# emotion_tts.py

import os
from gtts import gTTS
import uuid

class AdvancedEmotionalTTS:
    """
    Emotion-aware TTS generator.
    Uses gTTS for simplicity, but tags emotion for future prosody/emotion control.
    """

    def __init__(self):
        self.generated_files = []

    def generate_emotional_speech(self, text, lang='en', emotion='neutral', confidence=1.0):
        """
        Args:
            text (str): Text to convert to speech
            lang (str): Language code ('en', 'hi', etc.)
            emotion (str): Emotion label ('joy', 'sadness', 'anger', etc.)
            confidence (float): Confidence score (0-1)
        
        Returns:
            str: Path to generated audio file
        """
        # Modify text slightly to simulate emotion effect (placeholder)
        emotion_prefix = f"[{emotion.upper()} {int(confidence*100)}%] "
        text_to_speak = emotion_prefix + text

        # Generate unique filename
        filename = f"tts_output_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text_to_speak, lang=lang)
        tts.save(filename)

        self.generated_files.append(filename)
        return filename

    def cleanup(self):
        """Delete all generated audio files"""
        for f in self.generated_files:
            if os.path.exists(f):
                os.remove(f)
        self.generated_files = []

# Example usage
if __name__ == "__main__":
    tts = AdvancedEmotionalTTS()
    file_path = tts.generate_emotional_speech("Hello world! This is a test.", emotion="joy", confidence=0.9)
    print(f"Generated speech file: {file_path}")
