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

# Import face recognition libraries
try:
    from deepface import DeepFace
    import cv2
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace is available for face recognition")
except ImportError as e:
    print(f"❌ DeepFace import failed: {e}")
    DEEPFACE_AVAILABLE = False

# Import TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
    print("✅ pyttsx3 available for voice output")
except ImportError:
    print("❌ pyttsx3 not available (install with 'pip install pyttsx3')")
    TTS_AVAILABLE = False

print("✅ All imports successful!")


class VoiceControlledFaceSystem:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Audio streaming setup
        self.chunk_size = 2048
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.silence_threshold = 0.2
        self.silence_duration = 1
        self.stream = None
        self.selected_device = None
        self.selected_device_name = None
        self.processing_lock = threading.Lock()
        self.currently_processing = False
        self.audio_gain = 10.0
        
        # Face recognition system state
        self.guard_mode_active = False
        self.unauthorized_counter = 3
        self.face_system_running = False
        self.face_thread = None
        
        # Check audio availability
        self.sounddevice_available = SOUNDDEVICE_AVAILABLE
        self.deepface_available = DEEPFACE_AVAILABLE
        
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
            
        # Face recognition setup
        if self.deepface_available:
            self.setup_face_recognition()
            
        # List available audio devices
        if self.sounddevice_available:
            self.list_audio_devices()
        
        # Initialize TTS
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 175)
                self.tts_engine.setProperty('volume', 1.0)
                print("✅ TTS engine initialized successfully")
            except Exception as e:
                print(f"❌ TTS initialization failed: {e}")
                self.tts_engine = None
        else:
            self.tts_engine = None

    def setup_face_recognition(self):
        try:
            possible_paths = [
                "/home/mahendra/Documents/AML/Asiignment-2/AuthorizedPerson/face_embeddings.csv",
            ]
            
            self.face_embeddings_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.face_embeddings_path = path
                    break
            
            if self.face_embeddings_path:
                df = pd.read_csv(self.face_embeddings_path)
                self.image_names = df["Image"].values
                self.stored_embeddings = df.drop(columns=["Image"]).values
                print(f"✅ Loaded {len(self.stored_embeddings)} face embeddings from {self.face_embeddings_path}")
            else:
                print("⚠️  Face embeddings file not found. Face recognition will not work.")
                print("💡 To enable face recognition, create face embeddings first.")
                self.deepface_available = False
                
        except Exception as e:
            print(f"❌ Face recognition setup failed: {e}")
            self.deepface_available = False
    
    def create_sample_face_embeddings(self):
        try:
            print("🔄 Creating sample face embeddings for testing...")
            sample_embedding = np.random.randn(128)
            sample_embeddings = [sample_embedding]
            image_names = ["sample_person.jpg"]
            
            df = pd.DataFrame(sample_embeddings)
            df.insert(0, "Image", image_names)
            df.to_csv("sample_face_embeddings.csv", index=False)
            
            self.image_names = image_names
            self.stored_embeddings = np.array(sample_embeddings)
            self.face_embeddings_path = "sample_face_embeddings.csv"
            
            print("✅ Created sample face embeddings for testing")
            self.deepface_available = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to create sample embeddings: {e}")
            return False
    
    def is_authorized_face(self, face_embedding, threshold=0.7):
        if not self.deepface_available:
            return False, "Unknown", 0.0
            
        try:
            similarities = cosine_similarity([face_embedding], self.stored_embeddings)[0]
            max_sim = np.max(similarities)
            best_match = self.image_names[np.argmax(similarities)]
            if max_sim >= threshold:
                return True, best_match, max_sim
            else:
                return False, best_match, max_sim
        except Exception as e:
            print(f"❌ Face authentication error: {e}")
            return False, "Error", 0.0
    
    def list_audio_devices(self):
        try:
            devices = sd.query_devices()
            default_device = sd.default.device
            if isinstance(default_device, tuple):
                default_input = default_device[0]
            else:
                default_input = default_device
            
            print("\n🎤 Available Audio Input Devices:")
            print("=" * 60)
            
            input_devices = []
            self.device_names = {}
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append(i)
                    self.device_names[i] = device['name']
                    is_default = " (DEFAULT)" if i == default_input else ""
                    print(f"{i}: {device['name']}{is_default}")
            
            if not input_devices:
                print("❌ No input devices found!")
            else:
                print(f"✅ Found {len(input_devices)} input device(s)")
                self.selected_device = default_input
                self.selected_device_name = self.device_names.get(default_input, "Unknown")
                print(f"🎯 Auto-selected input device: {default_input} - {self.selected_device_name}")
                
        except Exception as e:
            print(f"❌ Error querying audio devices: {e}")
    
    def get_device_name(self, device_index):
        try:
            if isinstance(device_index, (list, tuple)):
                device_index = device_index[0]
            if hasattr(self, 'device_names') and device_index in self.device_names:
                return self.device_names[device_index]
            device_info = sd.query_devices(device_index)
            return device_info['name']
        except Exception as e:
            print(f"❌ Error getting device name for index {device_index}: {e}")
            return f"Unknown Device ({device_index})"
    
    def audio_callback(self, indata, frames, time, status):
        if self.is_recording and not self.currently_processing:
            boosted_audio = indata * self.audio_gain
            boosted_audio = np.clip(boosted_audio, -1.0, 1.0)
            self.audio_queue.put(boosted_audio.copy())
    
    def start_continuous_listening(self, device_index=None):
        if not self.sounddevice_available:
            print("❌ sounddevice not available for continuous listening")
            return False
        
        if device_index is None:
            device_to_use = self.selected_device
        else:
            device_to_use = device_index
            
        if isinstance(device_to_use, (list, tuple)):
            device_to_use = device_to_use[0]
        
        device_name = self.get_device_name(device_to_use)
        self.selected_device_name = device_name
        
        print(f"🔊 Starting with volume boost (10x)...")
        print(f"🎤 Using input device: {device_to_use} - {device_name}")
        
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
            print("✅ Continuous listening started with 10x volume boost")
            return True
                
        except Exception as e:
            print(f"❌ Failed to start continuous listening: {e}")
            return False
    
    def stop_continuous_listening(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("🛑 Continuous listening stopped")
    
    def speak_warning(self, message):
        """Speak system messages aloud and print them"""
        print(f"🔊 SYSTEM SAYS: {message}")
        if hasattr(self, 'tts_engine') and self.tts_engine:
            def _speak():
                try:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"❌ TTS playback failed: {e}")
            threading.Thread(target=_speak, daemon=True).start()
        else:
            print("⚠️  TTS not available, only printing message")

    # ---------------------------------------------------------
    # (Rest of your code below remains unchanged)
    # ---------------------------------------------------------
    # detect_speech_segment(), transcribe_with_whisper(),
    # start_face_recognition(), stop_face_recognition(),
    # _face_recognition_loop(), process_voice_command(),
    # continuous_processing_loop(), and main() stay as before.


    def detect_speech_segment(self):
        """Detect speech segments with improved algorithm"""
        if not self.sounddevice_available or not self.stream:
            return None
        
        print("👂 Listening for speech... (speak now)")
        
        speech_buffer = []
        silence_counter = 0
        max_silence_frames = int(self.silence_duration * self.sample_rate / self.chunk_size)
        speech_started = False
        min_speech_frames = 3  # Minimum frames to consider as speech
        
        start_time = time.time()
        frames_processed = 0
        
        while self.is_recording and (time.time() - start_time < 30):
            try:
                # Get audio chunk with shorter timeout
                chunk = self.audio_queue.get(timeout=0.5)
                frames_processed += 1
                
                # Calculate volume metrics
                volume_rms = np.sqrt(np.mean(chunk**2))
                volume_peak = np.max(np.abs(chunk))
                
                # Debug output every 20 frames
                if frames_processed % 20 == 0:
                    print(f"📊 Frame {frames_processed}: RMS: {volume_rms:.6f}, Peak: {volume_peak:.6f}")
                
                # Check if this is speech - SIMPLIFIED CONDITION
                is_speech = volume_rms > self.silence_threshold
                
                if is_speech:
                    speech_buffer.append(chunk)
                    silence_counter = 0
                    if not speech_started:
                        speech_started = True
                        print("🎤 Speech started! (adding to buffer)")
                else:
                    silence_counter += 1
                    if speech_started:
                        speech_buffer.append(chunk)  # Keep adding silence during speech
                
                # Check if we should return a speech segment
                if speech_started and silence_counter > max_silence_frames and len(speech_buffer) > min_speech_frames:
                    print(f"✅ Speech segment complete ({len(speech_buffer)} chunks, {silence_counter} silence frames)")
                    audio_np = np.vstack(speech_buffer).flatten()
                    
                    # Final volume check
                    final_rms = np.sqrt(np.mean(audio_np**2))
                    final_peak = np.max(np.abs(audio_np))
                    duration = len(audio_np) / self.sample_rate
                    print(f"📊 Final segment - Duration: {duration:.2f}s, RMS: {final_rms:.6f}, Peak: {final_peak:.6f}")
                    
                    if final_rms < 0.001:  # Very lenient threshold
                        print("⚠️  Segment too quiet, ignoring")
                        speech_buffer = []
                        silence_counter = 0
                        speech_started = False
                        continue
                        
                    return audio_np
                        
            except queue.Empty:
                # No audio data received, continue waiting
                if speech_started:
                    print("ℹ️  Queue empty but speech started, continuing...")
                continue
            except Exception as e:
                print(f"❌ Error in speech detection: {e}")
                break
        
        # If we exit the loop but have speech data, return it
        if speech_started and len(speech_buffer) > min_speech_frames:
            print(f"✅ Returning speech segment after timeout ({len(speech_buffer)} chunks)")
            audio_np = np.vstack(speech_buffer).flatten()
            return audio_np
        
        print("⏰ Speech detection timeout - no speech detected")
        return None
    
    def transcribe_with_whisper(self, audio_data):
        """Transcribe audio using Whisper - SIMPLIFIED VERSION"""
        if not self.whisper_available:
            return None
        
        temp_path = None
        try:
            # Basic audio preprocessing
            audio_data = audio_data.astype(np.float32)
            
            # Remove DC offset and normalize
            audio_data = audio_data - np.mean(audio_data)
            audio_max = np.max(np.abs(audio_data))
            if audio_max > 0:
                audio_data = audio_data / audio_max
            
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"whisper_temp_{int(time.time())}.wav")
            
            # Write with soundfile if available, else scipy
            try:
                import soundfile as sf
                sf.write(temp_path, audio_data, self.sample_rate)
            except ImportError:
                import scipy.io.wavfile as wavfile
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wavfile.write(temp_path, self.sample_rate, audio_int16)
            
            # Transcribe with Whisper
            print("🔍 Transcribing with Whisper...")
            result = self.whisper_model.transcribe(temp_path)
            transcription = result["text"].strip().upper()
            
            if transcription and len(transcription) > 1:
                print(f"📝 Transcription: {transcription}")
                return transcription
            else:
                print("📝 (No speech content detected)")
                return None
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    # def speak_warning(self, message):
    #     """Convert text to speech (placeholder - you can integrate with TTS library)"""
    #     print(f"🔊 SYSTEM SAYS: {message}")
    #     # You can integrate with pyttsx3 or gTTS here
    
    def start_face_recognition(self):
        """Start face recognition system in a separate thread"""
        if not self.deepface_available:
            print("❌ DeepFace not available for face recognition")
            # Try to create sample embeddings
            if self.create_sample_face_embeddings():
                print("🔄 Using sample embeddings for testing")
            else:
                return False
        
        self.face_system_running = True
        self.face_thread = threading.Thread(target=self._face_recognition_loop)
        self.face_thread.daemon = True
        self.face_thread.start()
        print("✅ Face recognition system started")
        return True
    
    def stop_face_recognition(self):
        """Stop face recognition system"""
        self.face_system_running = False
        if self.face_thread and self.face_thread.is_alive():
            self.face_thread.join(timeout=5)
        print("🛑 Face recognition system stopped")
    
    def _face_recognition_loop(self):
        """Main face recognition loop"""
        print("🎥 Starting camera for face recognition...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        while self.face_system_running:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            try:
                # Extract face embeddings from current frame
                result = DeepFace.represent(
                    img_path=frame,
                    model_name="Facenet",
                    detector_backend="retinaface",
                    enforce_detection=False
                )
                
                if len(result) > 0:
                    face_embedding = np.array(result[0]['embedding'])
                    authorized, name, score = self.is_authorized_face(face_embedding)
                    
                    if authorized:
                        text = f"Authorized: {name} ({score:.2f})"
                        color = (0, 255, 0)
                        # Reset counter on authorized access
                        self.unauthorized_counter = 3
                    else:
                        text = f"Unauthorized ({score:.2f})"
                        color = (0, 0, 255)
                        
                        # Handle unauthorized access
                        if self.unauthorized_counter > 0:
                            self.unauthorized_counter -= 1
                            warning_msg = f"Unauthorized, Go back! Attempts left: {self.unauthorized_counter}"
                            print(f"🔊 {warning_msg}")
                            self.speak_warning(warning_msg)
                            
                            if self.unauthorized_counter == 0:
                                fbi_msg = "You are being reported to FBI"
                                print(f"🔊 {fbi_msg}")
                                self.speak_warning(fbi_msg)
                else:
                    text = "No face detected"
                    color = (255, 255, 0)
                
                # Draw text overlay
                cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Guard Mode: {'ACTIVE' if self.guard_mode_active else 'INACTIVE'}", 
                           (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Attempts left: {self.unauthorized_counter}", 
                           (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Face Authorization - Press 'q' to stop", frame)
                
                # Check for 'q' key to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"❌ Face recognition error: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")
    
    def process_voice_command(self, transcription):
        """Process voice commands to control the system"""
        transcription_upper = transcription.upper()
        
        # Check for GUARD MODE ON 7667
        if "GUARD MODE ON 7667" in transcription_upper or "GUARD MODE ON" in transcription_upper or "GOD MODE ON" in transcription_upper or "GUIDE MODE ON" in transcription_upper:
            if not self.guard_mode_active:
                self.guard_mode_active = True
                self.unauthorized_counter = 3  # Reset counter
                print("🛡️  GUARD MODE ACTIVATED")
                self.speak_warning("Guard mode activated")
                
                # Start face recognition
                if self.deepface_available or self.create_sample_face_embeddings():
                    self.start_face_recognition()
                else:
                    print("❌ Face recognition not available")
            return True
        
        # Check for GUARD MODE OFF 7667
        elif "GUARD MODE OFF 7667" in transcription_upper or "GUARD MODE OF 7667" in transcription_upper or "GUIDE MODE OFF" in transcription_upper or "GOD MODE OFF" in transcription_upper:
            if self.guard_mode_active:
                self.guard_mode_active = False
                print("🛡️  GUARD MODE DEACTIVATED")
                self.speak_warning("Guard mode deactivated")
                
                # Stop face recognition
                self.stop_face_recognition()
                
                # Exit the program as requested
                print("👋 Exiting program...")
                self.stop_continuous_listening()
                sys.exit(0)
            return True
        
        return False
    
    def continuous_processing_loop(self):
        """Main continuous processing loop - listens for voice commands"""
        if not self.start_continuous_listening():
            return
        
        print("\n" + "="*60)
        print("🎙️  VOICE COMMAND PROCESSING ACTIVE")
        print(f"🎤 Using microphone: {self.selected_device_name}")
        print("   - Say 'GUARD MODE ON 7667' to start face recognition")
        print("   - Say 'GUARD MODE OFF 7667' to stop the system")
        print("   - Mic is continuously listening...")
        print("="*60)
        
        processing_count = 0
        
        try:
            while self.is_recording:
                if self.currently_processing:
                    time.sleep(0.1)
                    continue
                
                print("\n🔄 Waiting for speech...")
                audio_segment = self.detect_speech_segment()
                
                if audio_segment is not None and len(audio_segment) > 0:
                    processing_count += 1
                    duration = len(audio_segment) / self.sample_rate
                    print(f"\n🔍 Processing segment #{processing_count} ({duration:.2f}s)...")
                    
                    self.currently_processing = True
                    
                    try:
                        # Transcribe speech
                        transcription = self.transcribe_with_whisper(audio_segment)
                        
                        if transcription:
                            # Process voice command
                            command_processed = self.process_voice_command(transcription)
                            if not command_processed:
                                print("ℹ️  Command not recognized")
                        else:
                            print("❌ No transcription obtained")
                        
                    except Exception as e:
                        print(f"❌ Processing error: {e}")
                    
                    finally:
                        self.currently_processing = False
                    print("\n🔄 Returning to listening mode...")
                else:
                    print("❌ No speech segment detected")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping continuous processing...")
        except Exception as e:
            print(f"❌ Error in continuous processing: {e}")
        finally:
            self.currently_processing = False
            self.stop_continuous_listening()
            self.stop_face_recognition()
            print(f"✅ Processed {processing_count} voice segments total")

def main():
    print("🎙️  VOICE-CONTROLLED FACE RECOGNITION SYSTEM")
    print("=" * 50)
    
    system = VoiceControlledFaceSystem()
    
    if not system.sounddevice_available:
        print("❌ Audio system not available")
        return
    
    # Start the continuous voice processing
    system.continuous_processing_loop()

if __name__ == "__main__":
    main()
