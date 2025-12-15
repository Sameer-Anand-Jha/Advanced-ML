#!/usr/bin/env python3
import sys
import os
import tempfile

import numpy as np
import librosa
import pickle
import threading
import queue
import time
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# Import audio libraries
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("✅ sounddevice is available for audio recording")
except ImportError as e:
    print(f"❌ sounddevice import failed: {e}")
    SOUNDDEVICE_AVAILABLE = False

# Import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper is available for speech recognition")
except ImportError as e:
    print(f"❌ Whisper import failed: {e}")
    WHISPER_AVAILABLE = False

print("✅ All imports successful!")

class ContinuousVoiceSystem:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.users = {}
        self.model_file = "voice_models_continuous.pkl"
        
        # Audio streaming setup
        self.chunk_size = 1024
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.silence_threshold = 0.005
        self.silence_duration = 1.5
        self.stream = None
        self.selected_device = None
        self.processing_lock = threading.Lock()
        self.currently_processing = False
        self.audio_gain = 5.0
        
        # Check audio availability
        self.sounddevice_available = SOUNDDEVICE_AVAILABLE
        
        # Initialize Whisper
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                self.whisper_available = True
                print("✅ Whisper model loaded (base)")
            except Exception as e:
                print(f"❌ Whisper model loading failed: {e}")
                self.whisper_available = False
        else:
            self.whisper_available = False
            
        # List available audio devices
        if self.sounddevice_available:
            self.list_audio_devices()
    
    def list_audio_devices(self):
        """List available audio input devices"""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device
            
            print("\n🎤 Available Audio Input Devices:")
            print("=" * 60)
            
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append(i)
                    is_default = " (DEFAULT)" if i == default_input else ""
                    print(f"{i}: {device['name']}{is_default}")
            
            if not input_devices:
                print("❌ No input devices found!")
            else:
                print(f"✅ Found {len(input_devices)} input device(s)")
                self.selected_device = default_input
                print(f"🎯 Auto-selected device: {default_input}")
                
        except Exception as e:
            print(f"❌ Error querying audio devices: {e}")
    
    def audio_callback(self, indata, frames, time, status):
        """SoundDevice callback with volume boost"""
        if self.is_recording and not self.currently_processing:
            boosted_audio = indata * self.audio_gain
            boosted_audio = np.clip(boosted_audio, -1.0, 1.0)
            self.audio_queue.put(boosted_audio.copy())
    
    def start_continuous_listening(self, device_index=None):
        """Start continuous microphone monitoring"""
        if not self.sounddevice_available:
            print("❌ sounddevice not available for continuous listening")
            return False
        
        device_to_use = device_index if device_index is not None else self.selected_device
        
        print("🔊 Starting with volume boost...")
        
        try:
            self.is_recording = True
            self.currently_processing = False
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                device=device_to_use
            )
            self.stream.start()
            print("✅ Continuous listening started with volume boost")
            return True
                
        except Exception as e:
            print(f"❌ Failed to start continuous listening: {e}")
            return False
    
    def stop_continuous_listening(self):
        """Stop continuous microphone monitoring"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("🛑 Continuous listening stopped")
    
    def detect_speech_segment(self):
        """Detect speech segments with better volume handling"""
        if not self.sounddevice_available or not self.stream:
            return None
        
        print("👂 Listening for speech...")
        speech_buffer = []
        silence_counter = 0
        max_silence_frames = int(self.silence_duration * self.sample_rate / self.chunk_size)
        speech_started = False
        
        start_time = time.time()
        while self.is_recording and (time.time() - start_time < 30):
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                
                volume_rms = np.sqrt(np.mean(chunk**2))
                volume_peak = np.max(np.abs(chunk))
                
                if volume_rms > self.silence_threshold or volume_peak > 0.1:
                    speech_buffer.append(chunk)
                    silence_counter = 0
                    speech_started = True
                    print(f"🎙️  Speech detected (RMS: {volume_rms:.3f}, Peak: {volume_peak:.3f})", end='\r')
                else:
                    silence_counter += 1
                    if speech_started:
                        speech_buffer.append(chunk)
                    
                    if speech_started and silence_counter > max_silence_frames and len(speech_buffer) > 5:
                        print("\n✅ Speech segment complete")
                        audio_np = np.vstack(speech_buffer).flatten()
                        
                        if np.max(np.abs(audio_np)) < 0.05:
                            print("⚠️  Audio too quiet, ignoring")
                            speech_buffer = []
                            silence_counter = 0
                            speech_started = False
                            continue
                            
                        return audio_np
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in speech detection: {e}")
                break
        
        return None
    
    def transcribe_with_whisper(self, audio_data):
        """Transcribe audio using Whisper - FIXED VOLUME CHECK"""
        if not self.whisper_available:
            return None
        
        # temp_path = None
        try:
            # FIX: Check volume using the same method as detection
            volume_rms = np.sqrt(np.mean(audio_data**2))
            volume_peak = np.max(np.abs(audio_data))
            
            print(f"🔊 Audio for Whisper - RMS: {volume_rms:.4f}, Peak: {volume_peak:.4f}")
            
            # More lenient volume check - Whisper can handle quiet audio
            if volume_rms < 0.001:  # Much lower threshold
                print("📝 Audio extremely quiet, but trying Whisper anyway...")
                # Don't return None, let Whisper try
            
            # Normalize audio to optimal level for Whisper
            target_peak = 0.9  # Higher target for better transcription
            if volume_peak > 0:
                audio_data = audio_data * (target_peak / volume_peak)
            
            print(f"🔊 Normalized audio - RMS: {np.sqrt(np.mean(audio_data**2)):.4f}, Peak: {np.max(np.abs(audio_data)):.4f}")
            
            # Create a more reliable temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"whisper_temp_{int(time.time())}_{os.getpid()}.wav")
            
            # Ensure the audio data is in the correct format
            audio_data = audio_data.astype(np.float32)
            
            # Use soundfile to write WAV file (more reliable than scipy)
            try:
                import soundfile as sf
                sf.write(temp_path, audio_data, self.sample_rate)
                print(f"💾 Temporary audio file created: {temp_path}")
            except ImportError:
                # Fallback to scipy if soundfile not available
                print("⚠️  soundfile not available, using scipy fallback")
                import scipy.io.wavfile as wavfile
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wavfile.write(temp_path, self.sample_rate, audio_int16)
            
            # Verify the file was created and has content
            if not os.path.exists(temp_path):
                print("❌ Temporary file was not created")
                return None
            
            file_size = os.path.getsize(temp_path)
            if file_size < 1000:  # Less than 1KB is probably empty
                print(f"❌ Temporary file too small ({file_size} bytes)")
                return None
            
            print(f"📁 File size: {file_size} bytes")
            
            # Transcribe with Whisper
            print("🔍 Transcribing with Whisper...")
            result = self.whisper_model.transcribe(temp_path)
            transcription = result["text"].strip()
            
            if transcription and len(transcription) > 1:
                print(f"📝 {transcription}")
                return transcription
            else:
                print("📝 (No speech content detected)")
                return None
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return None
        finally:
            # Always clean up the temporary file
            if temp_path and os.path.exists(temp_path):
                print("OK")
                # print(f"{e}")
                # try:
                    # os.remove(temp_path)
                    # print(f"🧹 Cleaned up temporary file")
                # except Exception as e:
                    # print(f"⚠️  Could not remove temp file: {e}")
    
    def test_authentication_on_file(self, file_path):
        """Test voice authentication on a specific WAV file"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
    
        print(f"🔐 Testing authentication on: {file_path}")
    
        # Load and authenticate
        audio_data = self.load_audio_file(file_path)
        if audio_data is not None:
            user, confidence = self.authenticate_voice(audio_data)
            if user and confidence > 0.5:
                print(f"✅ Authenticated as: {user} (confidence: {confidence:.3f})")
            else:
                print(f"❌ Not authenticated (best: {user}, confidence: {confidence:.3f})")
        else:
            print("❌ Failed to load audio file")



    def extract_features(self, audio):
        """Extract MFCC features with audio enhancement"""
        if audio is None or len(audio) < 1.0 * self.sample_rate:
            return None
            
        try:
            # Enhanced audio preprocessing
            audio = audio.astype(np.float32)
            
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Normalize with headroom
            audio_max = np.max(np.abs(audio))
            if audio_max > 0:
                audio = audio / (audio_max * 1.1)
            
            # Extract features from cleaned audio
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=20,
                n_fft=2048,
                hop_length=512,
                fmin=50,
                fmax=8000
            )
            
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            # Additional spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            
            features = np.vstack([
                mfccs, 
                mfcc_delta, 
                mfcc_delta2,
                spectral_centroid,
                spectral_rolloff,
                spectral_bandwidth
            ])
            
            return features.T
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None
    
    def authenticate_voice(self, audio, threshold=0.7):
        """Authenticate speaker from audio"""
        if not self.users:
            return None, 0.0
        
        features = self.extract_features(audio)
        if features is None:
            return None, 0.0
        
        best_user = None
        best_score = -float('inf')
        
        for username, user_data in self.users.items():
            try:
                score = user_data['gmm'].score(features)
                if score > best_score:
                    best_score = score
                    best_user = username
            except:
                continue
        
        # Improved score normalization
        if best_score > -10:
            normalized_score = 0.9 + (best_score + 10) / 10 * 0.1
        elif best_score > -30:
            normalized_score = 0.7 + (best_score + 30) / 20 * 0.2
        elif best_score > -60:
            normalized_score = 0.4 + (best_score + 60) / 30 * 0.3
        elif best_score > -100:
            normalized_score = max(0.1, 0.1 + (best_score + 100) / 40 * 0.3)
        else:
            normalized_score = 0.0
        
        normalized_score = max(0.0, min(1.0, normalized_score))
        return best_user, normalized_score
    
    def continuous_processing_loop(self, authentication_threshold=0.7):
        """Main continuous processing loop"""
        if not self.start_continuous_listening():
            return
        
        print("\n" + "="*60)
        print("🎙️  CONTINUOUS VOICE PROCESSING ACTIVE")
        print(f"   - Volume Boost: {self.audio_gain}x")
        print("   - Press Ctrl+C to stop")
        print("="*60)
        
        processing_count = 0
        max_processes = 50
        
        try:
            while self.is_recording and processing_count < max_processes:
                if self.currently_processing:
                    time.sleep(0.1)
                    continue
                
                audio_segment = self.detect_speech_segment()
                
                if audio_segment is not None and len(audio_segment) > 0:
                    processing_count += 1
                    duration = len(audio_segment) / self.sample_rate
                    
                    self.currently_processing = True
                    
                    print(f"\n🔍 Processing segment #{processing_count} ({duration:.2f}s)...")
                    
                    try:
                        start_time = time.time()
                        
                        # Check audio quality
                        volume_rms = np.sqrt(np.mean(audio_segment**2))
                        volume_peak = np.max(np.abs(audio_segment))
                        print(f"📊 Audio - RMS: {volume_rms:.3f}, Peak: {volume_peak:.3f}")
                        
                        # Transcription
                        transcription = self.transcribe_with_whisper(audio_segment)
                        trans_time = time.time() - start_time
                        
                        # Authentication
                        auth_start = time.time()
                        user, score = self.authenticate_voice(audio_segment, authentication_threshold)
                        auth_time = time.time() - auth_start
                        
                        # Results
                        if transcription:
                            print(f"⏱️  Transcription: {trans_time:.2f}s")
                        
                        if user and score > authentication_threshold:
                            print(f"✅ Authenticated as: {user} (score: {score:.3f})")
                        elif user:
                            print(f"❌ Low confidence: {user} (score: {score:.3f})")
                        else:
                            print(f"❌ No match (best: {score:.3f})")
                            
                        print(f"⏱️  Auth: {auth_time:.2f}s")
                        
                    except Exception as e:
                        print(f"❌ Processing error: {e}")
                    
                    finally:
                        self.currently_processing = False
                    
                    print("-" * 40)
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping continuous processing...")
        except Exception as e:
            print(f"❌ Error in continuous processing: {e}")
        finally:
            self.currently_processing = False
            self.stop_continuous_listening()
            print(f"✅ Processed {processing_count} segments total")
    
    def find_audio_files(self, folder_path):
        """Find all audio files in a folder"""
        audio_extensions = {'.opus', '.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'}
        audio_files = []
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder '{folder_path}' does not exist!")
            return []
            
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in audio_extensions:
                    audio_files.append(file_path)
        
        return sorted(audio_files)
    
    def load_audio_file(self, file_path):
        """Load audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            duration = len(audio) / sr
            print(f"✓ {os.path.basename(file_path)} ({duration:.2f}s)")
            return audio
        except Exception as e:
            print(f"✗ {os.path.basename(file_path)}: {e}")
            return None
    
    def enroll_user_from_folder(self, username, folder_path, min_files=2):
        """Enroll user from folder of audio files"""
        print(f"🔍 Scanning: {folder_path}")
        
        audio_files = self.find_audio_files(folder_path)
        if not audio_files:
            print("❌ No audio files found!")
            return False
        
        print(f"📁 Found {len(audio_files)} files")
        
        all_features = []
        successful_files = 0
        
        for file_path in audio_files:
            audio = self.load_audio_file(file_path)
            if audio is None:
                continue
                
            features = self.extract_features(audio)
            if features is not None:
                all_features.append(features)
                successful_files += 1
        
        if successful_files < min_files:
            print(f"❌ Need {min_files} files, got {successful_files}")
            return False
        
        print(f"🤖 Training with {successful_files} files...")
        
        try:
            features_combined = np.vstack(all_features)
            n_components = min(8, max(2, successful_files // 2))
            
            gmm = GaussianMixture(
                n_components=n_components, 
                covariance_type='diag', 
                random_state=42
            )
            gmm.fit(features_combined)
            
            self.users[username] = {
                'gmm': gmm,
                'file_count': successful_files,
                'folder': folder_path
            }
            
            self.save_models()
            print(f"✅ Enrolled '{username}' with {successful_files} files!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def list_users(self):
        """List enrolled users"""
        if not self.users:
            print("📭 No users enrolled")
            return
        
        print(f"\n📋 Enrolled Users ({len(self.users)}):")
        for username, data in self.users.items():
            print(f"   👤 {username}: {data['file_count']} files")
    
    def save_models(self):
        """Save models to file"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.users, f)
        except Exception as e:
            print(f"❌ Error saving: {e}")
    
    def load_models(self):
        """Load models from file"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.users = pickle.load(f)
                print(f"📂 Loaded {len(self.users)} user(s)")
                return True
            except Exception as e:
                print(f"❌ Error loading: {e}")
                return False
        return False
    
    def test_microphone(self, duration=3):
        """Test microphone functionality with volume analysis"""
        if not self.sounddevice_available:
            return False
        
        try:
            print(f"🎤 Testing microphone for {duration} seconds...")
            print("💡 Speak at normal volume...")
            
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
            sd.wait()
            
            if audio is not None:
                audio_flat = audio.flatten()
                
                audio_boosted = audio_flat * self.audio_gain
                audio_boosted = np.clip(audio_boosted, -1.0, 1.0)
                
                volume_rms = np.sqrt(np.mean(audio_boosted**2))
                volume_peak = np.max(np.abs(audio_boosted))
                
                print(f"📊 Volume Analysis:")
                print(f"   RMS Volume: {volume_rms:.4f}")
                print(f"   Peak Volume: {volume_peak:.4f}")
                print(f"   Gain Applied: {self.audio_gain}x")
                
                if volume_rms < 0.01:
                    print("❌ Volume very low - check microphone settings")
                    print("💡 Try:")
                    print("   - Increase system microphone volume")
                    print("   - Move closer to microphone")
                    print("   - Check if microphone is muted")
                elif volume_rms < 0.05:
                    print("⚠️  Volume low - consider increasing gain")
                else:
                    print("✅ Volume good!")
                
                return volume_rms > 0.01
            return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

    def adjust_volume_settings(self):
        """Adjust volume and gain settings"""
        print("\n🔊 Volume Settings:")
        print(f"Current gain: {self.audio_gain}x")
        print(f"Current silence threshold: {self.silence_threshold}")
        
        new_gain = input(f"New gain (1.0-10.0) [{self.audio_gain}]: ").strip()
        if new_gain and new_gain.replace('.', '').isdigit():
            self.audio_gain = float(new_gain)
            print(f"✅ Gain set to {self.audio_gain}x")
        
        new_threshold = input(f"New silence threshold (0.001-0.1) [{self.silence_threshold}]: ").strip()
        if new_threshold and new_threshold.replace('.', '').isdigit():
            self.silence_threshold = float(new_threshold)
            print(f"✅ Silence threshold set to {self.silence_threshold}")

    def test_whisper_directly(self, audio_file_path=None):
        """Test Whisper transcription directly with a file - FIXED"""
        if not self.whisper_available:
            print("❌ Whisper not available")
            return
        
        if audio_file_path:
            # Test with existing audio file
            if not os.path.exists(audio_file_path):
                print(f"❌ File not found: {audio_file_path}")
                return
            
            print(f"🔍 Testing Whisper with file: {audio_file_path}")
            try:
                result = self.whisper_model.transcribe(audio_file_path)
                transcription = result["text"].strip()
                print(f"📝 Transcription: {transcription}")
                return transcription
            except Exception as e:
                print(f"❌ File transcription failed: {e}")
                return None
        else:
            # Test with microphone recording - FIXED TO USE BOOSTED AUDIO
            print("🎤 Testing Whisper with microphone recording...")
            try:
                audio = sd.rec(int(5 * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1, 
                              dtype='float32')
                sd.wait()
                
                if audio is not None:
                    audio_flat = audio.flatten()
                    
                    # Apply gain boost to match continuous processing
                    audio_boosted = audio_flat * self.audio_gain
                    audio_boosted = np.clip(audio_boosted, -1.0, 1.0)
                    
                    # Show volume of the actual audio being sent to Whisper
                    volume_rms = np.sqrt(np.mean(audio_boosted**2))
                    volume_peak = np.max(np.abs(audio_boosted))
                    print(f"📊 Audio sent to Whisper - RMS: {volume_rms:.4f}, Peak: {volume_peak:.4f}")
                    
                    transcription = self.transcribe_with_whisper(audio_boosted)
                    return transcription
                    
            except Exception as e:
                print(f"❌ Whisper test failed: {e}")
                return None

def main():
    print("🎙️  VOICE AUTHENTICATION SYSTEM")
    print("=" * 50)
    
    system = ContinuousVoiceSystem()
    
    if not system.sounddevice_available:
        print("❌ Audio system not available")
        return
    
    system.load_models()
    
    while True:
        print("\n" + "="*40)
        print("1. Enroll user")
        print("2. Start processing") 
        print("3. Test microphone")
        print("4. Test Whisper directly")
        print("5. Adjust volume settings")
        print("6. List users")
        print("7. Exit")
        print("8. Authenticate")
        
        choice = input("\nSelect: ").strip()
        
        if choice == '1':
            username = input("Username: ").strip()
            folder_path = input("Folder: ").strip()
            system.enroll_user_from_folder(username, folder_path)
            
        elif choice == '2':
            if not system.users:
                print("❌ No users enrolled")
                continue
            
            threshold = input("Threshold (0.1-0.9) [0.5]: ").strip()
            threshold = float(threshold) if threshold.replace('.', '').isdigit() else 0.5
            
            print("\n🚀 Starting with volume boost... (Ctrl+C to stop)")
            system.continuous_processing_loop(threshold)
            
        elif choice == '3':
            system.test_microphone()
            
        elif choice == '4':
            print("\n🎯 Test Whisper Transcription")
            print("1. Test with microphone")
            print("2. Test with audio file")
            test_choice = input("Select: ").strip()
            
            if test_choice == '1':
                system.test_whisper_directly()
            elif test_choice == '2':
                file_path = input("Audio file path: ").strip()
                system.test_whisper_directly(file_path)
            else:
                print("❌ Invalid choice")
            
        elif choice == '5':
            system.adjust_volume_settings()
            
        elif choice == '6':
            system.list_users()
            
        elif choice == '7':
            print("👋 Goodbye!")
        
        elif choice == '8':
            temp_file = input("Enter temp WAV file path: ").strip()
            system.test_authentication_on_file(temp_file)
            break
        

        else:
            print("❌ Invalid")

if __name__ == "__main__":
    main()