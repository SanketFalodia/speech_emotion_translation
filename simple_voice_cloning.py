"""
OpenVoice Voice Cloning Module
Integrates OpenVoice for voice cloning, then applies librosa-based emotional effects
Compatible with existing project structure
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt


class OpenVoiceCloner:
    """
    Voice cloning using OpenVoice with emotion post-processing
    """
    
    # Emotion parameters (same as original code)
    EMOTION_PARAMS = {
        'joy': {
            'pitch_adjust': 3.0,
            'speed_adjust': 1.20,
            'energy_adjust': 1.4,
            'vibrato': 0.05,
            'spectral_tilt': 2.0,
            'breathiness': 0.0
        },
        'sadness': {
            'pitch_adjust': -3.0,
            'speed_adjust': 0.80,
            'energy_adjust': 0.65,
            'vibrato': 0.0,
            'spectral_tilt': -3.0,
            'breathiness': 0.3
        },
        'anger': {
            'pitch_adjust': 2.0,
            'speed_adjust': 1.30,
            'energy_adjust': 1.6,
            'vibrato': 0.02,
            'spectral_tilt': 4.0,
            'breathiness': 0.0
        },
        'fear': {
            'pitch_adjust': 4.0,
            'speed_adjust': 1.35,
            'energy_adjust': 1.3,
            'vibrato': 0.08,
            'spectral_tilt': 1.5,
            'breathiness': 0.2
        },
        'surprise': {
            'pitch_adjust': 5.0,
            'speed_adjust': 1.25,
            'energy_adjust': 1.5,
            'vibrato': 0.03,
            'spectral_tilt': 3.0,
            'breathiness': 0.0
        },
        'disgust': {
            'pitch_adjust': -2.0,
            'speed_adjust': 0.90,
            'energy_adjust': 0.9,
            'vibrato': 0.0,
            'spectral_tilt': -1.5,
            'breathiness': 0.15
        },
        'neutral': {
            'pitch_adjust': 0.0,
            'speed_adjust': 1.0,
            'energy_adjust': 1.0,
            'vibrato': 0.0,
            'spectral_tilt': 0.0,
            'breathiness': 0.0
        }
    }
    
    # Language mapping for OpenVoice
    LANG_MAP = {
        'en': 'EN',
        'es': 'ES',
        'fr': 'FR',
        'de': 'EN',  # Use EN base for German (will be cloned)
        'it': 'EN',  # Use EN base for Italian
        'pt': 'EN',  # Use EN base for Portuguese
        'zh': 'ZH',
        'ja': 'JP',
        'ko': 'KR',
        'hi': 'EN',  # Use EN base for Hindi
        'bn': 'EN',  # Use EN base for Bengali
        'ar': 'EN',  # Use EN base for Arabic
    }
    
    def __init__(self):
        """Initialize OpenVoice components"""
        print("🔧 Initializing OpenVoice...")
        
        try:
            from melo.api import TTS
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   Using device: {self.device}")
            
            # Initialize tone color converter
            ckpt_converter = 'checkpoints/converter'
            
            # Check if checkpoints exist
            if not os.path.exists(ckpt_converter):
                print(f"   ⚠️  Checkpoints not found at {ckpt_converter}")
                print(f"   Downloading checkpoints...")
                self._download_checkpoints()
            
            self.tone_color_converter = ToneColorConverter(
                f'{ckpt_converter}/config.json', 
                device=self.device
            )
            self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
            
            # Store classes for later use
            self.TTS = TTS
            self.se_extractor = se_extractor
            
            print("✅ OpenVoice initialized successfully!")
            
        except ImportError as e:
            raise ImportError(
                "OpenVoice not installed. Install with:\n"
                "pip install git+https://github.com/myshell-ai/OpenVoice.git\n"
                "pip install git+https://github.com/myshell-ai/MeloTTS.git\n\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize OpenVoice: {e}")
        
        self.temp_files = []
    
    def _download_checkpoints(self):
        """Download OpenVoice checkpoints if not present"""
        print("   Downloading OpenVoice checkpoints (this may take a few minutes)...")
        os.system("python -m openvoice download")
    
    def clone_voice(self, text, reference_audio, language='en', emotion='neutral', 
                    output_file="cloned_voice.wav"):
        """
        Main method to clone voice with emotion
        
        Args:
            text: Text to speak
            reference_audio: Path to reference audio file
            language: Language code
            emotion: Emotion to apply
            output_file: Output file path
        
        Returns:
            Path to generated audio file
        """
        print("\n" + "="*70)
        print("🎤 OpenVoice Cloning with Emotional Expression")
        print("="*70)
        print(f"   🎭 Emotion: {emotion.upper()}")
        print(f"   🌐 Language: {language}")
        print(f"   🎵 Reference: {reference_audio}")
        
        # Map language code
        lang_code = self.LANG_MAP.get(language.lower(), 'EN')
        
        # Step 1: Extract speaker embedding from reference
        print("\n🔍 Extracting voice characteristics from reference...")
        target_se, _ = self.se_extractor.get_se(
            reference_audio, 
            self.tone_color_converter, 
            vad=True
        )
        
        # Step 2: Generate base speech
        print("🎵 Generating base speech...")
        temp_base = "tmp_base_openvoice.wav"
        self._generate_base_speech(text, lang_code, temp_base)
        self.temp_files.append(temp_base)
        
        # Step 3: Apply voice cloning (tone color conversion)
        print("🎨 Applying voice cloning...")
        temp_cloned = "tmp_cloned_openvoice.wav"
        
        # Get source speaker embedding (from base model)
        source_se = torch.load(f'checkpoints/base_speakers/ses/{lang_code}.pth').to(self.device)
        
        # Convert tone color
        self.tone_color_converter.convert(
            audio_src_path=temp_base,
            src_se=source_se,
            tgt_se=target_se,
            output_path=temp_cloned,
            message="@MyShell"
        )
        self.temp_files.append(temp_cloned)
        
        # Step 4: Apply emotional modulation using librosa (like original code)
        if emotion.lower() != 'neutral':
            print(f"\n🎭 Applying {emotion.upper()} emotion with librosa processing...")
            self._apply_emotion_effects(temp_cloned, emotion, output_file)
        else:
            import shutil
            shutil.copy(temp_cloned, output_file)
            print("✅ Neutral cloned voice saved!")
        
        print(f"\n✅ Final output saved: {output_file}")
        return output_file
    
    def _generate_base_speech(self, text, lang_code, output_file):
        """Generate base speech using MeloTTS"""
        # Initialize TTS for specific language
        tts = self.TTS(language=lang_code, device=self.device)
        
        # Get available speakers
        speaker_ids = tts.hps.data.spk2id
        
        # Use first available speaker
        if isinstance(speaker_ids, dict):
            speaker_id = list(speaker_ids.keys())[0]
        else:
            speaker_id = speaker_ids[0]
        
        # Generate speech
        tts.tts_to_file(text, speaker_id, output_file, speed=1.0)
    
    def _apply_emotion_effects(self, input_file, emotion, output_file):
        """
        Apply emotional effects using librosa (same as original code)
        """
        emotion = emotion.lower()
        params = self.EMOTION_PARAMS.get(emotion, self.EMOTION_PARAMS['neutral'])
        
        print(f"   • Pitch: {params['pitch_adjust']:+.1f} semitones")
        print(f"   • Speed: {params['speed_adjust']:.2f}x")
        print(f"   • Energy: {params['energy_adjust']:.2f}x")
        
        # Load audio
        y, sr = librosa.load(input_file, sr=None)
        
        # 1. Pitch shift
        if abs(params['pitch_adjust']) > 0.3:
            print(f"   • Applying pitch shift...")
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=params['pitch_adjust'])
        
        # 2. Time stretch (speed)
        if abs(params['speed_adjust'] - 1.0) > 0.05:
            print(f"   • Applying time stretch...")
            y = librosa.effects.time_stretch(y, rate=params['speed_adjust'])
        
        # 3. Energy/volume adjustment
        y = y * params['energy_adjust']
        
        # 4. Vibrato (for fear/anxiety)
        if params['vibrato'] > 0:
            print(f"   • Adding vibrato ({params['vibrato']:.2f})...")
            y = self._add_vibrato(y, sr, params['vibrato'])
        
        # 5. Spectral tilt (brightness/darkness)
        if abs(params['spectral_tilt']) > 0.1:
            print(f"   • Applying spectral tilt...")
            y = self._apply_spectral_tilt(y, sr, params['spectral_tilt'])
        
        # 6. Breathiness
        if params['breathiness'] > 0:
            print(f"   • Adding breathiness ({params['breathiness']:.2f})...")
            y = self._add_breathiness(y, params['breathiness'])
        
        # Normalize to prevent clipping
        max_val = np.abs(y).max()
        if max_val > 1.0:
            y = y / max_val * 0.95
        
        # Save output
        sf.write(output_file, y, sr)
        print(f"   ✅ Emotional processing complete!")
    
    def _add_vibrato(self, y, sr, amount=0.05):
        """Add vibrato effect (same as original code)"""
        vibrato_freq = 5.5
        t = np.arange(len(y)) / sr
        vibrato = 1 + amount * np.sin(2 * np.pi * vibrato_freq * t)
        return y * vibrato
    
    def _apply_spectral_tilt(self, y, sr, tilt_db):
        """Apply spectral tilt for brightness/darkness (same as original code)"""
        cutoff = 2000
        gain = 10 ** (tilt_db / 20)
        
        freq = np.fft.rfftfreq(len(y), 1/sr)
        response = np.ones_like(freq)
        response[freq > cutoff] *= gain
        
        Y = np.fft.rfft(y)
        Y_filtered = Y * response
        y_filtered = np.fft.irfft(Y_filtered, n=len(y))
        
        return y_filtered
    
    def _add_breathiness(self, y, intensity):
        """Add breathiness by mixing with filtered noise (same as original code)"""
        noise = np.random.normal(0, 0.02, len(y))
        b, a = butter(4, 0.3, btype='high')
        noise_filtered = filtfilt(b, a, noise)
        return (1 - intensity) * y + intensity * noise_filtered
    
    def cleanup(self):
        """Remove temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        self.temp_files = []


# Convenience function (same interface as original)
def clone_and_speak(text, reference_audio, language='en', emotion='neutral', 
                    output_file="cloned_output.wav"):
    """
    Convenience function for OpenVoice cloning with emotions
    
    Args:
        text: Text to speak
        reference_audio: Path to reference audio file
        language: Language code
        emotion: Emotion to apply
        output_file: Output file path
    
    Returns:
        Path to generated audio file
    """
    cloner = OpenVoiceCloner()
    try:
        output = cloner.clone_voice(text, reference_audio, language, emotion, output_file)
        return output
    finally:
        cloner.cleanup()


# Test the module
if __name__ == "__main__":
    print("=" * 70)
    print("🎤 OpenVoice Emotional Voice Cloning Test")
    print("="*70)
    
    reference = input("\nEnter path to reference audio (WAV file): ").strip()
    
    if os.path.exists(reference):
        test_text = input("Enter text to speak: ").strip() or "Hello! This is a test."
        
        print("\n🎭 Available emotions:")
        print("   joy, sadness, anger, fear, surprise, disgust, neutral")
        emotion = input("Choose emotion (default=neutral): ").strip() or "neutral"
        
        print("\n🌐 Supported languages:")
        print("   en, es, fr, de, it, pt, zh, ja, ko, hi, bn, ar")
        language = input("Choose language (default=en): ").strip() or "en"
        
        output = clone_and_speak(test_text, reference, language, emotion)
        
        # Play audio
        import platform
        print("\n🔊 Playing cloned audio...")
        if platform.system() == "Windows":
            os.system(f"start {output}")
        elif platform.system() == "Darwin":
            os.system(f"afplay {output}")
        else:
            os.system(f"mpg123 {output}")
    else:
        print(f"❌ Reference audio not found: {reference}")