#!/usr/bin/env python3
"""
Halloween Scary Monster - Raspberry Pi 3B
Detects kids with RCWL-1601 sensor, plays spooky music, and has AI conversations
"""
import RPi.GPIO as GPIO
import time
import queue
import pygame
import speech_recognition as sr
import pyttsx3
import anthropic
import os
import sys
import threading
import random
import tempfile
import logging
from logging.handlers import RotatingFileHandler
from ctypes import cdll, CFUNCTYPE, c_char_p, c_int
from pathlib import Path
from contextlib import contextmanager
from elevenlabs import ElevenLabs  # Import ElevenLabs class


_ALSA_ERR_HANDLER = None
_JACK_ERROR_HANDLER = None
_JACK_INFO_HANDLER = None


def _suppress_alsa_warnings():
    """Silence ALSA errors so the console stays quiet."""
    global _ALSA_ERR_HANDLER
    try:
        asound = cdll.LoadLibrary('libasound.so')
    except OSError:
        return

    def _py_error_handler(filename, line, function, err, fmt):
        return

    CALLBACK = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    c_handler = CALLBACK(_py_error_handler)
    try:
        asound.snd_lib_error_set_handler(c_handler)
    except Exception:
        pass
    _ALSA_ERR_HANDLER = c_handler


def _suppress_jack_messages():
    """Silence JACK info/error messages that otherwise spam stdout."""
    global _JACK_ERROR_HANDLER, _JACK_INFO_HANDLER
    try:
        jack = cdll.LoadLibrary('libjack.so.0')
    except OSError:
        return

    CALLBACK = CFUNCTYPE(None, c_char_p)

    def _noop(message):
        return

    err_cb = CALLBACK(_noop)
    info_cb = CALLBACK(_noop)
    try:
        if hasattr(jack, 'jack_set_error_function'):
            jack.jack_set_error_function(err_cb)
        if hasattr(jack, 'jack_set_info_function'):
            jack.jack_set_info_function(info_cb)
    except Exception:
        pass
    _JACK_ERROR_HANDLER = err_cb
    _JACK_INFO_HANDLER = info_cb

# Improve audio stability on Pi: increase SDL/pygame audio buffer to reduce ALSA underruns.
# These can be tuned via env vars: AUDIO_FREQ, AUDIO_CHANNELS, AUDIO_BUFFER
try:
    AUDIO_FREQ = int(os.environ.get('AUDIO_FREQ', '44100'))
    AUDIO_CHANNELS = int(os.environ.get('AUDIO_CHANNELS', '2'))
    AUDIO_BUFFER = int(os.environ.get('AUDIO_BUFFER', '4096'))
    # 16-bit signed samples (negative size) is common
    AUDIO_SIZE = -16
    # Pre-init must be called before mixer.init(); calling here ensures it runs early.
    pygame.mixer.pre_init(AUDIO_FREQ, AUDIO_SIZE, AUDIO_CHANNELS, AUDIO_BUFFER)
except Exception:
    # If anything goes wrong, ignore and let pygame choose defaults
    pass

# Configuration
SENSOR_TRIG = int(os.environ.get('SENSOR_TRIG', '23'))  # GPIO pin for sensor trigger
SENSOR_ECHO = int(os.environ.get('SENSOR_ECHO', '24'))  # GPIO pin for sensor echo
DETECTION_DISTANCE = int(os.environ.get('DETECTION_DISTANCE', '150'))  # cm - distance to trigger monster
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')  # Anthropic API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')  # OpenAI API key
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # Groq API key (FREE!)

# Sensor types supported
SENSOR_TYPE = os.environ.get('SENSOR_TYPE', 'auto')  # 'auto', 'rcwl-1601', 'us-100'

# AI Service to use: 'anthropic', 'openai', 'groq', 'none'
AI_SERVICE = os.environ.get('AI_SERVICE', 'auto')  # auto-detect by default

# Different types of responses when AI fails
FALLBACK_RESPONSES = {
    'greeting': [
        "Welcome to my lair... what brings you here, little morsel?",
        "I've been waiting for you... tell me, what is your name?",
        "Ah, fresh visitors! Do you dare enter my domain?",
        "What's this? New prey? Tell me, what do you fear most?",
        "Welcome, children... are you brave enough to answer my riddles?",
    ],
    'movement': [
        "Why do you approach me so boldly? Do you not fear?",
        "Where do you think you're going, little one?",
        "Running away already? What frightens you so?",
        "Dancing in my domain? Tell me, what makes you so brave?",
        "Creeping closer... do you seek candy or doom?",
    ],
    'taunt': [
        "Too scared to speak? What's got your tongue?",
        "Why so quiet? Are you planning your escape?",
        "What are you hiding from, little one?",
        "Is your candy worth the terror? Tell me!",
        "Frozen in fear? Or just thinking about running?",
    ],
    'engage': [
        "Tell me your name, and perhaps I'll spare you...",
        "What brings you to my haunted halls tonight?",
        "I smell fear and candy - which do you carry more of?",
        "What creature would you like to be on this dark night?",
        "Answer me this: what makes your heart race?",
    ],
    'leaving': [
        "Wait! Where are you going? Don't you want to know my secret?",
        "Stop! Tell me one more thing before you flee!",
        "Come back! Are you leaving because you're afraid?",
        "Leaving so soon? What if I promise you the best candy?",
        "Don't go! At least tell me your name before you escape!",
    ],
    'lure': [
        "Who dares approach my haunted domain tonight?",
        "What brave souls seek treats in the darkness?",
        "Do you dare enter? Tell me what you seek...",
        "Fresh visitors? What questions shall you answer first?",
        "Who wants to play my game of darkness and candy?",
    ]
}



class HalloweenMonster:
    def __init__(self):
        self.logger = None
        self._init_logging()
        _suppress_alsa_warnings()
        _suppress_jack_messages()
        self.verbose_console = bool(os.environ.get('HALLOWEEN_DEBUG'))
        # Initialize GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SENSOR_TRIG, GPIO.OUT)
        GPIO.setup(SENSOR_ECHO, GPIO.IN)
        
        # Detect sensor type
        self.sensor_type = self.detect_sensor_type()
        print(f"Detected sensor: {self.sensor_type}")
        self._log_event('sensor_initialized', sensor_type=self.sensor_type)
        
        # Initialize pygame for music - will use 3.5mm audio output
        pygame.mixer.init()
        
        # Initialize text-to-speech with spooky settings
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 120)  # Slower, creepier
            self.tts_engine.setProperty('volume', 1.0)
            
            # Try to set a deeper voice
            try:
                voices = self.tts_engine.getProperty('voices')
                if voices and len(voices) > 0:
                    for voice in voices:
                        if 'male' in voice.name.lower() or 'en' in voice.id.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
            except Exception as e:
                self._log_event('pyttsx3_voice_select_failed', level=logging.DEBUG, error=str(e))
        except Exception as e:
            self._log_event('pyttsx3_init_failed', level=logging.DEBUG, error=str(e))
            self.tts_engine = None
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.mic_available = False
        
        try:
            mic_list = sr.Microphone.list_microphone_names()
            mic_count = len(mic_list)
            self._log_event('mic_inputs_detected', level=logging.DEBUG, count=mic_count)
            self._console(f"Found {mic_count} audio input device(s)")
            
            if mic_list:
                self.microphone = sr.Microphone()
                self._console("Calibrating microphone for ambient noise...")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.mic_available = True
                self._console("Microphone initialized successfully!")
            else:
                self._log_event('mic_not_detected', level=logging.WARNING)
                self._console("No microphone detected. Monster will operate in display-only mode.")
        except Exception as e:
            self._log_event('mic_init_failed', level=logging.WARNING, error=str(e))
            self._console("Monster will operate without voice interaction.")
        
        # Initialize AI client
        self.client = None
        self.ai_service = None
        # ElevenLabs client and chosen voice id (optional)
        self.eleven_client = None
        self.eleven_voice_id = None
        # Attempt to initialize ElevenLabs voice selection (non-fatal)
        try:
            self._init_elevenlabs()
        except Exception as e:
            print(f"ElevenLabs init skipped/failed: {e}")
        self._initialize_ai_service()
        
        # State management
        self.is_active = False
        self.conversation_history = []
        self.music_playing = False
        self.current_volume = 0.6
        self.target_volume = 0.6
        self.music_thread = None
        self.stop_music_thread = False
        # TTS queue and worker (asynchronous TTS/playback)
        self.tts_queue = queue.Queue()
        self.tts_stop = threading.Event()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # Track when we are actively speaking to avoid the microphone re-hearing playback
        self.speaking_event = threading.Event()
        self.last_speech_time = 0.0
        try:
            self.speech_cooldown = float(os.environ.get('MIC_SPEECH_COOLDOWN', '1.0'))
        except Exception:
            self.speech_cooldown = 1.0
        self.last_user_speech_time = 0.0
        self.last_presence_time = 0.0
        self.last_engage_time = 0.0
        try:
            self.engage_idle_seconds = float(os.environ.get('ENGAGE_IDLE_SECONDS', '25'))
        except Exception:
            self.engage_idle_seconds = 25.0
        
        # Interaction state tracking
        self.interaction_level = 0  # 0=low, 1=medium, 2=high engagement
        self.conversation_count = 0  # Track number of exchanges in current interaction

        self._log_event('startup_complete', sensor_type=self.sensor_type, ai_service=self.ai_service or 'none')
        print("Halloween Monster initialized! Waiting for victims...")

    def _init_logging(self):
        try:
            log_dir = Path(os.environ.get('HALLOWEEN_LOG_DIR', 'dev'))
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / 'monster.log'
            handler = RotatingFileHandler(str(log_path), maxBytes=1_048_576, backupCount=3)
        except Exception:
            handler = logging.StreamHandler()
            log_path = None

        logger_name = f"HalloweenMonster.{os.getpid()}"
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        self.logger = logger
        if log_path:
            logger.debug(f"Logging initialized at {log_path}")

    def _log_event(self, event, level=logging.INFO, **data):
        if not self.logger:
            return
        parts = [f"event={event}"]
        for key, value in data.items():
            if value is None:
                safe_val = 'none'
            else:
                safe_val = str(value).replace('\n', ' ').replace('\r', ' ')
                if len(safe_val) > 120:
                    safe_val = safe_val[:117] + '...'
            parts.append(f"{key}={safe_val}")
        message = ' '.join(parts)
        self.logger.log(level, message)

    def _console(self, message):
        if self.verbose_console:
            print(message)

    # ------------------- AI Initialization -------------------
    def _initialize_ai_service(self):
        if AI_SERVICE.lower() == 'groq':
            if self._try_groq():
                return
        elif AI_SERVICE.lower() == 'openai':
            if self._try_openai():
                return
        elif AI_SERVICE.lower() == 'anthropic':
            if self._try_anthropic():
                return
        elif AI_SERVICE.lower() == 'none':
            print("AI service disabled. Using fallback responses.")
            return
        if self._try_groq(): return
        if self._try_openai(): return
        if self._try_anthropic(): return
        print("No AI service available. Using fallback responses.")
    
    def _try_groq(self):
        if GROQ_API_KEY:
            try:
                from groq import Groq
                self.client = Groq(api_key=GROQ_API_KEY)
                self.ai_service = 'groq'
                print("Using Groq API (FREE!)")
                return True
            except ImportError:
                print("groq package not installed. Run: pip3 install groq")
            except Exception as e:
                print(f"Groq initialization failed: {e}")
        return False
    
    def _try_openai(self):
        if OPENAI_API_KEY:
            try:
                import openai
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                self.ai_service = 'openai'
                print("Using OpenAI API")
                return True
            except ImportError:
                print("openai package not installed. Run: pip3 install openai")
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
        return False
    
    def _try_anthropic(self):
        if ANTHROPIC_API_KEY:
            try:
                self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                self.ai_service = 'anthropic'
                print("Using Anthropic API")
                return True
            except Exception as e:
                print(f"Anthropic initialization failed: {e}")
        return False

    # ------------------- ElevenLabs helpers -------------------
    def _init_elevenlabs(self):
        """Initialize ElevenLabs voice settings if API key is available."""
        if not os.environ.get("ELEVENLABS_API_KEY"):
            return

        # Initialize ElevenLabs client
        try:
            self.eleven = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
            self.eleven_available = True
            print("ElevenLabs API accessible")
            
            # Fetch available voices for random selection
            self.eleven_voices_list = []
            try:
                voices = None
                tts_client = getattr(self.eleven, 'text_to_speech', None)
                
                if os.environ.get('ELEVENLABS_DEBUG'):
                    print(f"TTS client: {tts_client}")
                    print(f"TTS client type: {type(tts_client)}")
                
                # Try multiple methods to get voices
                if tts_client and hasattr(tts_client, 'get_voices'):
                    if os.environ.get('ELEVENLABS_DEBUG'):
                        print("Trying tts_client.get_voices()...")
                    try:
                        voices = tts_client.get_voices()
                    except Exception as e:
                        if os.environ.get('ELEVENLABS_DEBUG'):
                            print(f"get_voices() failed: {e}")
                
                if not voices and tts_client and hasattr(tts_client, 'list_voices'):
                    if os.environ.get('ELEVENLABS_DEBUG'):
                        print("Trying tts_client.list_voices()...")
                    try:
                        voices = tts_client.list_voices()
                    except Exception as e:
                        if os.environ.get('ELEVENLABS_DEBUG'):
                            print(f"list_voices() failed: {e}")
                
                if not voices and hasattr(self.eleven, 'voices'):
                    voices_obj = getattr(self.eleven, 'voices')
                    if os.environ.get('ELEVENLABS_DEBUG'):
                        print(f"Voices object: {voices_obj}")
                    if hasattr(voices_obj, 'get_all'):
                        if os.environ.get('ELEVENLABS_DEBUG'):
                            print("Trying eleven.voices.get_all()...")
                        try:
                            voices = voices_obj.get_all()
                        except Exception as e:
                            if os.environ.get('ELEVENLABS_DEBUG'):
                                print(f"get_all() failed: {e}")
                    if not voices and hasattr(voices_obj, 'list'):
                        if os.environ.get('ELEVENLABS_DEBUG'):
                            print("Trying eleven.voices.list()...")
                        try:
                            voices = voices_obj.list()
                        except Exception as e:
                            if os.environ.get('ELEVENLABS_DEBUG'):
                                print(f"list() failed: {e}")
                
                if os.environ.get('ELEVENLABS_DEBUG'):
                    print(f"Voices result: {voices}")
                    print(f"Voices type: {type(voices)}")
                
                if voices:
                    # Handle different response types
                    voice_list = voices
                    if hasattr(voices, 'voices'):
                        voice_list = voices.voices
                    if hasattr(voices, '__iter__') and not isinstance(voices, (str, bytes)):
                        try:
                            voice_list = list(voices)
                        except Exception:
                            pass
                    
                    if os.environ.get('ELEVENLABS_DEBUG'):
                        print(f"Processing {len(voice_list) if hasattr(voice_list, '__len__') else '?'} voices")
                    
                    for v in voice_list:
                        if isinstance(v, dict):
                            vid = v.get("voice_id") or v.get("id") or v.get("name")
                            name = v.get("name") or vid
                        else:
                            vid = getattr(v, "voice_id", None) or getattr(v, "id", None) or getattr(v, "name", None)
                            name = getattr(v, "name", None) or vid
                        if vid:
                            self.eleven_voices_list.append((vid, name))
                            if os.environ.get('ELEVENLABS_DEBUG'):
                                print(f"  Added voice: {name} ({vid})")
                
                if self.eleven_voices_list:
                    # Pick initial random voice
                    chosen = random.choice(self.eleven_voices_list)
                    self.eleven_voice_id = chosen[0]
                    print(f"ElevenLabs initial voice: {chosen[1]} ({chosen[0]})")
                    print(f"Total voices available: {len(self.eleven_voices_list)}")
                else:
                    # Fallback to curated list of scary/monster Halloween voices
                    # These are pre-made voices available to most accounts
                    common_voices = [
                        ('2EiwWnXFnvU5JabPnv8n', 'Clyde'),       # Deep, masculine - Monster
                        ('AZnzlk1XvdvUeBnXmlld', 'Domi'),        # Strong, confident - Creature
                        ('D38z5RcWu1voky8WS1ja', 'Fin'),         # Elderly - Old witch/warlock
                        ('N2lVS1w4EtoT3dr4eOWO', 'Callum'),      # Hoarse, masculine - Demon
                        ('ODq5zmih8GrVes37Dizd', 'Patrick'),     # Shouty - Angry spirit
                        ('TxGEqnHWrfWFTfGW9XjX', 'Josh'),        # Deep, young - Beast
                        ('VR6AewLTigWG4xSOukaG', 'Arnold'),      # Crisp, middle-aged - Robot/AI
                        ('XB0fDUnXU5powFXDhCwa', 'Charlotte'),   # Seductive - Vampire/witch
                        ('onwK4e9ZLuTAKqWW03F9', 'Daniel'),      # Deep, authoritative - Monster lord
                        ('pNInz6obpgDQGcFmaJgB', 'Adam'),        # Deep, American - Classic monster
                        ('yoZ06aMxZJJ28mfd3POQ', 'Sam'),         # Raspy, young - Ghoul
                        ('g5CIjZEefAph4nQFvHAz', 'Ethan'),       # Whispery - Ghost/specter
                        ('JBFqnCBsd6RMkjVDRZzb', 'George'),      # British - Mad scientist
                        ('GBv7mTt0atIp3Br8iCZE', 'Thomas'),      # Calm, narrator - Sinister storyteller
                    ]
                    self.eleven_voices_list = common_voices
                    chosen = random.choice(common_voices)
                    self.eleven_voice_id = chosen[0]
                    print(f"API key lacks voices_read permission, using curated list")
                    print(f"ElevenLabs initial voice: {chosen[1]} ({chosen[0]})")
                    print(f"Total voices available: {len(common_voices)}")
                    print("To use custom voices, request voices_read permission for your API key")
            except Exception as e:
                print(f"Error fetching voices: {e}")
                if os.environ.get('ELEVENLABS_DEBUG'):
                    import traceback
                    traceback.print_exc()
                self.eleven_voice_id = os.environ.get('ELEVENLABS_VOICE', '2tTjAGX0n5ajDmazDcWk')
                
        except Exception as e:
            print(f"Could not initialize ElevenLabs: {e}")
            self.eleven_available = False
            self.eleven = None
            return

    def _randomize_voice(self):
        """Randomly change the ElevenLabs voice for variety"""
        if not self.eleven_available or not hasattr(self, 'eleven_voices_list') or not self.eleven_voices_list:
            return
        
        old_voice = self.eleven_voice_id
        chosen = random.choice(self.eleven_voices_list)
        self.eleven_voice_id = chosen[0]
        
        if old_voice != self.eleven_voice_id:
            print(f"Voice changed: {chosen[1]} ({chosen[0]})")
            self._log_event('voice_changed', new_voice=chosen[1], voice_id=chosen[0])

    def _list_elevenlabs_voices(self):
        """Return a list of (id, name) tuples for available ElevenLabs voices.

        This is used by the CLI helper.
        """
        if not os.environ.get("ELEVENLABS_API_KEY"):
            print("No ELEVENLABS_API_KEY set. Cannot query ElevenLabs.")
            return []

        try:
            client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        except Exception as e:
            print(f"Could not initialize ElevenLabs client: {e}")
            return []

        voices = None
        try:
            voices = client.text_to_speech.list_voices()
        except Exception:
            try:
                voices = client.voices.list()
            except Exception as e:
                print(f"Could not retrieve voices: {e}")
                return []

        result = []
        for v in voices:
            if isinstance(v, dict):
                vid = v.get("voice_id") or v.get("id") or v.get("name")
                name = v.get("name") or vid
            else:
                vid = getattr(v, "voice_id", None) or getattr(v, "id", None) or getattr(v, "name", None)
                name = getattr(v, "name", None) or vid
            if vid:
                result.append((vid, name))
        return result

    # ------------------- Sensor -------------------
    def detect_sensor_type(self):
        if SENSOR_TYPE.lower() in ['rcwl-1601', 'us-100']:
            return SENSOR_TYPE.upper()
        print("Auto-detecting sensor type...")
        valid_readings = []
        for _ in range(5):
            distance = self._measure_distance_raw(timeout=0.5)
            if distance > 0:
                valid_readings.append(distance)
            time.sleep(0.1)
        if not valid_readings:
            print("Warning: No valid readings. Defaulting to RCWL-1601.")
            return "RCWL-1601"
        return "RCWL-1601"
    
    def _measure_distance_raw(self, timeout=0.5):
        try:
            GPIO.output(SENSOR_TRIG, GPIO.LOW)
            time.sleep(0.000002)
            GPIO.output(SENSOR_TRIG, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(SENSOR_TRIG, GPIO.LOW)
            timeout_time = time.time() + timeout
            pulse_start = time.time()
            while GPIO.input(SENSOR_ECHO) == 0:
                pulse_start = time.time()
                if time.time() > timeout_time: return -1
            pulse_end = time.time()
            while GPIO.input(SENSOR_ECHO) == 1:
                pulse_end = time.time()
                if time.time() > timeout_time: return -1
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 34300 / 2
            return distance
        except Exception:
            return -1
    
    def measure_distance(self):
        timeout = 0.1 if self.sensor_type == "US-100" else 0.5
        max_range = 300 if self.sensor_type == "US-100" else 450
        distance = self._measure_distance_raw(timeout=timeout)
        if distance < 0:
            self._log_event('sensor_reading', level=logging.DEBUG, status='error')
            return -1

        status = 'valid'
        result = distance
        if distance < 2 or distance > max_range:
            status = 'clipped'
            result = -1
        self._log_event(
            'sensor_reading',
            level=logging.DEBUG,
            measured_cm=f"{distance:.2f}",
            status=status,
            result_cm=(f"{result:.2f}" if result >= 0 else 'invalid')
        )
        return result

    # ------------------- Music -------------------
    def play_spooky_music(self, music_dir='./music'):
        if self.music_playing: return
        self.music_playing = True
        self.stop_music_thread = False
        self.music_thread = threading.Thread(target=self._music_manager, args=(music_dir,))
        self.music_thread.daemon = True
        self.music_thread.start()
    
    def _music_manager(self, music_dir):
        music_path = Path(music_dir)
        if not music_path.exists():
            print(f"Warning: Music directory '{music_dir}' not found")
            self.music_playing = False
            return
        audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
        music_files = []
        for ext in audio_extensions:
            music_files.extend(list(music_path.glob(f'*{ext}')))
            music_files.extend(list(music_path.glob(f'*{ext.upper()}')))
        if not music_files:
            print(f"Warning: No music files found in '{music_dir}'")
            self.music_playing = False
            return
        while not self.stop_music_thread:
            try:
                selected_music = random.choice(music_files)
                print(f"Now playing: {selected_music.name}")
                pygame.mixer.music.load(str(selected_music))
                pygame.mixer.music.set_volume(self.current_volume)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not self.stop_music_thread:
                    if abs(self.current_volume - self.target_volume) > 0.01:
                        step = 0.05 if self.target_volume > self.current_volume else -0.05
                        self.current_volume += step
                        self.current_volume = max(0.0, min(1.0, self.current_volume))
                        pygame.mixer.music.set_volume(self.current_volume)
                    time.sleep(0.1)
                if self.stop_music_thread: break
                self._fade_out(duration=2.0)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in music playback: {e}")
                time.sleep(1)
    
    def _fade_out(self, duration=2.0):
        steps = int(duration * 10)
        volume_step = self.current_volume / steps
        for _ in range(steps):
            if self.stop_music_thread: break
            self.current_volume -= volume_step
            self.current_volume = max(0.0, self.current_volume)
            pygame.mixer.music.set_volume(self.current_volume)
            time.sleep(duration / steps)
    
    def _fade_in(self, target=0.6, duration=2.0):
        steps = int(duration * 10)
        volume_step = (target - self.current_volume) / steps
        for _ in range(steps):
            if self.stop_music_thread: break
            self.current_volume += volume_step
            self.current_volume = min(target, self.current_volume)
            pygame.mixer.music.set_volume(self.current_volume)
            time.sleep(duration / steps)
    
    def set_music_volume(self, volume, smooth=True):
        self.target_volume = max(0.0, min(1.0, volume))
        if not smooth:
            self.current_volume = self.target_volume
            if self.music_playing:
                pygame.mixer.music.set_volume(self.current_volume)
    
    def panic_music(self):
        """Temporarily boost music volume and tempo to create urgency"""
        if self.music_playing:
            print("PANIC MODE: Music intensifying!")
            self._log_event('panic_music_triggered')
            # Quickly boost volume
            self.set_music_volume(0.9, smooth=False)
            time.sleep(2.0)
            # Restore normal volume
            self.set_music_volume(0.6, smooth=True)
    
    def stop_music(self):
        if self.music_playing:
            self.stop_music_thread = True
            self._fade_out(duration=1.0)
            pygame.mixer.music.stop()
            self.music_playing = False
            if self.music_thread:
                self.music_thread.join(timeout=2)

    # ------------------- Text-to-Speech -------------------
    @contextmanager
    def _speaking(self):
        """Signal that we are playing audio so the mic can pause listening."""
        self.speaking_event.set()
        try:
            yield
        finally:
            self.last_speech_time = time.time()
            self.speaking_event.clear()

    def speak(self, text):
        """Convert text to spooky speech using ElevenLabs"""
        # Enqueue text for asynchronous TTS playback
        print(f"Monster (queued): {text}")
        try:
            self.tts_queue.put_nowait(text)
        except Exception as e:
            # If queueing fails for some reason, fall back to synchronous speak
            print(f"TTS queue failed ({e}), falling back to synchronous playback")
            try:
                self._speak_elevenlabs(text)
            except Exception as e2:
                print(f"ElevenLabs fallback triggered: {e2}")
                self._speak_espeak(text)

        return

    def wait_for_speech(self, timeout=30):
        """Wait for all queued speech to finish playing."""
        try:
            # Wait for queue to be empty and speaking to finish
            if self.verbose_console:
                print("Waiting for TTS queue to empty...")
            self.tts_queue.join()
            if self.verbose_console:
                print("TTS queue empty, waiting for playback to finish...")
            # Also wait for speaking_event to clear (actual playback done)
            deadline = time.time() + timeout
            while self.speaking_event.is_set() and time.time() < deadline:
                time.sleep(0.1)
            if self.verbose_console:
                print(f"Speech finished, speaking_event cleared: {not self.speaking_event.is_set()}")
            # Add small delay after speech for echo to dissipate
            time.sleep(0.5)
        except Exception as e:
            if os.environ.get('HALLOWEEN_DEBUG'):
                print(f"wait_for_speech error: {e}")

    def _tts_worker(self):
        """Background worker that consumes TTS queue and plays audio asynchronously."""
        while not self.tts_stop.is_set() or not self.tts_queue.empty():
            try:
                text = self.tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # Use the existing ElevenLabs/ESpeak routines; they handle ducking/restoring music
                try:
                    self._speak_elevenlabs(text)
                except Exception as e:
                    print(f"TTS worker ElevenLabs error: {e}")
                    try:
                        self._speak_espeak(text)
                    except Exception as e2:
                        print(f"TTS worker espeak error: {e2}")
            finally:
                try:
                    self.tts_queue.task_done()
                except Exception:
                    pass


    def _speak_elevenlabs(self, text):
        """High-quality spooky speech using ElevenLabs"""
        if not getattr(self, 'eleven_available', False) or not hasattr(self, 'eleven'):
            print("ElevenLabs not available, falling back to espeak.")
            self._speak_espeak(text)
            return

        tmp_path = None
        audio_bytes = None
        last_exc = None

        # Try multiple possible API shapes for ElevenLabs TTS
        try:
            # Helper to normalize any iterable/generator of bytes into a single bytes object
            def _collect_bytes(maybe_iter):
                if maybe_iter is None:
                    return None
                if isinstance(maybe_iter, (bytes, bytearray)):
                    return bytes(maybe_iter)
                # If it's file-like
                if hasattr(maybe_iter, 'read'):
                    try:
                        return maybe_iter.read()
                    except Exception:
                        pass
                # If it's an iterator/generator yielding bytes/chunks
                if hasattr(maybe_iter, '__iter__') and not isinstance(maybe_iter, (str, bytes, bytearray)):
                    parts = []
                    try:
                        chunk_count = 0
                        for chunk in maybe_iter:
                            chunk_count += 1
                            if isinstance(chunk, (bytes, bytearray)):
                                parts.append(bytes(chunk))
                            elif isinstance(chunk, str):
                                parts.append(chunk.encode('utf-8'))
                        if os.environ.get('ELEVENLABS_DEBUG') and chunk_count > 0:
                            print(f"_collect_bytes: consumed {chunk_count} chunks, {sum(len(p) for p in parts)} total bytes")
                        if parts:
                            return b''.join(parts)
                    except Exception as e:
                        if os.environ.get('ELEVENLABS_DEBUG'):
                            print(f"_collect_bytes iteration error: {e}")
                        # Check if this is an API error response
                        err_str = str(e)
                        if 'quota_exceeded' in err_str.lower():
                            print("ElevenLabs quota exceeded - please add credits or wait for quota reset")
                        elif '401' in err_str or 'unauthorized' in err_str.lower():
                            print("ElevenLabs authentication/quota issue")
                        return None
                if os.environ.get('ELEVENLABS_DEBUG'):
                    print(f"_collect_bytes: could not extract bytes from {type(maybe_iter)}")
                return None

            base_voice = os.environ.get('ELEVENLABS_VOICE') or getattr(self, 'eleven_voice_id', None) or 'Adam'
            base_model = os.environ.get('ELEVENLABS_MODEL', 'eleven_monolingual_v1')
            if os.environ.get('ELEVENLABS_DEBUG'):
                print(f"ElevenLabs voice: {base_voice}, model: {base_model}")

            # 0) Try convert() first (common in newer SDKs)
            tts = getattr(self.eleven, 'text_to_speech', None)
            if tts is not None and hasattr(tts, 'convert'):
                if os.environ.get('ELEVENLABS_DEBUG'):
                    print("Attempting text_to_speech.convert")
                try:
                    resp = tts.convert(
                        voice_id=base_voice,
                        text=text,
                        model_id=base_model,
                        optimize_streaming_latency=os.environ.get('ELEVENLABS_LATENCY', '0')
                    )
                    audio_bytes = _collect_bytes(resp)
                    if not audio_bytes and hasattr(resp, 'write_to_file'):
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpf:
                            tmp_name = tmpf.name
                        try:
                            resp.write_to_file(tmp_name)
                            with open(tmp_name, 'rb') as r:
                                audio_bytes = r.read()
                        finally:
                            try:
                                os.remove(tmp_name)
                            except Exception:
                                pass
                    if audio_bytes:
                        if os.environ.get('ELEVENLABS_DEBUG'):
                            print(f"convert() succeeded with {len(audio_bytes)} bytes")
                except Exception as e:
                    last_exc = e
                    if os.environ.get('ELEVENLABS_DEBUG'):
                        print(f"convert() failed: {e}")

            if not audio_bytes:
                tts_client = getattr(self.eleven, 'text_to_speech', None)
                if tts_client is not None:
                    stream_fn = getattr(tts_client, 'stream', None)
                    if callable(stream_fn):
                        # stream() requires voice_id as positional arg, text as keyword
                        stream_attempts = [
                            # (voice_id_positional, kwargs_dict)
                            (base_voice, {'text': text, 'model_id': base_model}),
                            (base_voice, {'text': text}),
                        ]
                        if os.environ.get('ELEVENLABS_DEBUG'):
                            print(f"Trying text_to_speech.stream with positional voice_id")
                        for voice_pos, kwargs in stream_attempts:
                            try:
                                if os.environ.get('ELEVENLABS_DEBUG'):
                                    print(f"Calling stream({voice_pos!r}, **{kwargs})")
                                resp = stream_fn(voice_pos, **kwargs)
                            except TypeError as e:
                                last_exc = e
                                if os.environ.get('ELEVENLABS_DEBUG'):
                                    print(f"stream TypeError: {e}")
                                continue
                            except Exception as e:
                                last_exc = e
                                if os.environ.get('ELEVENLABS_DEBUG'):
                                    print(f"stream error: {e}")
                                continue

                            audio_bytes = _collect_bytes(resp)
                            if not audio_bytes and hasattr(resp, 'write_to_file'):
                                try:
                                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpf:
                                        tmp_name = tmpf.name
                                    try:
                                        resp.write_to_file(tmp_name)
                                        with open(tmp_name, 'rb') as r:
                                            audio_bytes = r.read()
                                    finally:
                                        try:
                                            os.remove(tmp_name)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    last_exc = e
                                    audio_bytes = None
                            if audio_bytes:
                                if os.environ.get('ELEVENLABS_DEBUG'):
                                    print(f"stream() succeeded with {len(audio_bytes)} bytes")
                                break
                        if audio_bytes:
                            pass

                    if not audio_bytes:
                        for method_name in ('generate', 'synthesize', 'create', 'speak'):
                            fn = getattr(tts_client, method_name, None)
                            if not callable(fn):
                                continue
                            variants = [
                                {'text': text, 'voice_id': base_voice, 'model_id': base_model},
                                {'text': text, 'voice': base_voice, 'model_id': base_model},
                                {'text': text, 'voice': base_voice, 'model': base_model},
                                {'text': text, 'voice_id': base_voice},
                                {'text': text, 'voice': base_voice},
                                {'text': text},
                            ]
                            if os.environ.get('ELEVENLABS_DEBUG'):
                                print(f"Trying {method_name} variants: {variants}")
                            for kwargs in variants:
                                try:
                                    if os.environ.get('ELEVENLABS_DEBUG'):
                                        print(f"Calling {method_name} with kwargs: {kwargs}")
                                    resp = fn(**kwargs)
                                except Exception as e:
                                    last_exc = e
                                    continue
                                audio_bytes = _collect_bytes(resp)
                                if not audio_bytes and hasattr(resp, 'write_to_file'):
                                    try:
                                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpf:
                                            tmp_name = tmpf.name
                                        try:
                                            resp.write_to_file(tmp_name)
                                            with open(tmp_name, 'rb') as r:
                                                audio_bytes = r.read()
                                        finally:
                                            try:
                                                os.remove(tmp_name)
                                            except Exception:
                                                pass
                                    except Exception as e:
                                        last_exc = e
                                        audio_bytes = None
                                if audio_bytes:
                                    if os.environ.get('ELEVENLABS_DEBUG'):
                                        print(f"{method_name}() succeeded with {len(audio_bytes)} bytes")
                                    break
                            if audio_bytes:
                                break

            # Try top-level client methods: self.eleven.generate(...)
            if not audio_bytes and hasattr(self.eleven, 'generate'):
                try:
                    for kwargs in (
                        {'text': text, 'voice': 'Adam', 'model': 'eleven_monolingual_v1'},
                        {'text': text, 'voice': 'Adam'},
                        {'text': text},
                    ):
                        try:
                            resp = self.eleven.generate(**kwargs)
                        except TypeError:
                            try:
                                resp = self.eleven.generate(text)
                            except Exception as e:
                                last_exc = e
                                continue
                        except Exception as e:
                            last_exc = e
                            continue
                        audio_bytes = _collect_bytes(resp)
                        if audio_bytes:
                            break
                except Exception as e:
                    last_exc = e

            # Try the older top-level helper if present: elevenlabs.generate(...)
            if not audio_bytes:
                try:
                    import elevenlabs as el
                    if hasattr(el, 'generate'):
                        for kwargs in (
                            {'text': text, 'voice': 'Adam', 'model': 'eleven_monolingual_v1'},
                            {'text': text, 'voice': 'Adam'},
                            {'text': text},
                        ):
                            try:
                                resp = el.generate(**kwargs)
                            except TypeError:
                                try:
                                    resp = el.generate(text)
                                except Exception as e:
                                    last_exc = e
                                    continue
                            except Exception as e:
                                last_exc = e
                                continue
                            audio_bytes = _collect_bytes(resp)
                            if audio_bytes:
                                break
                except Exception as e:
                    last_exc = e

        except Exception as e:
            last_exc = e

        if not audio_bytes:
            if last_exc:
                print(f"ElevenLabs TTS error: {last_exc}")
                err_msg = str(last_exc).lower()
                if any(key in err_msg for key in ("voice_not_found", "voice_id", "not found")):
                    try:
                        print("Attempting to list available ElevenLabs voices...")
                        voices = None
                        tts_client = getattr(self.eleven, 'text_to_speech', None)
                        if tts_client and hasattr(tts_client, 'list_voices'):
                            voices = tts_client.list_voices()
                        elif hasattr(self.eleven, 'voices') and hasattr(self.eleven.voices, 'list'):
                            voices = self.eleven.voices.list()
                        if voices:
                            ids = []
                            for v in voices:
                                if isinstance(v, dict):
                                    ids.append(v.get('voice_id') or v.get('id') or v.get('name'))
                                else:
                                    ids.append(getattr(v, 'voice_id', None) or getattr(v, 'id', None) or getattr(v, 'name', None))
                            ids = [i for i in ids if i]
                            if ids:
                                print("Available ElevenLabs voices: " + ", ".join(ids))
                                print("Set ELEVENLABS_VOICE to one of these IDs, e.g.:\n  export ELEVENLABS_VOICE='Alloy'\n")
                    except Exception as voice_err:
                        print(f"Could not query ElevenLabs voices: {voice_err}")
            else:
                if os.environ.get('ELEVENLABS_DEBUG'):
                    print("ElevenLabs returned no audio but no exception was captured.")
            self._speak_espeak(text)
            return

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(audio_bytes)

            with self._speaking():
                if self.music_playing:
                    self.set_music_volume(0.15, smooth=True)
                    time.sleep(0.2)

                ch = pygame.mixer.find_channel()
                if not ch:
                    nc = pygame.mixer.get_num_channels() or 8
                    pygame.mixer.set_num_channels(nc + 1)
                    ch = pygame.mixer.find_channel()

                tts_sound = pygame.mixer.Sound(tmp_path)
                ch.play(tts_sound)

                while ch.get_busy():
                    time.sleep(0.1)

                if self.music_playing:
                    self.set_music_volume(0.6, smooth=True)

        except Exception as e:
            print(f"Error playing ElevenLabs audio: {e}")
            self._speak_espeak(text)
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return

    def _speak_espeak(self, text):
        import subprocess
        with self._speaking():
            try:
                subprocess.run([
                    'espeak',
                    '-s', '110',  # slower
                    '-p', '25',   # lower pitch
                    '-a', '200',  # amplitude
                    text
                ], check=False)
                # espeak returns immediately but audio continues playing
                # Add delay proportional to text length to ensure audio finishes
                estimated_duration = len(text) / 10.0  # rough estimate: 10 chars per second
                time.sleep(min(estimated_duration, 3.0))  # cap at 3 seconds
            except Exception as e:
                print(f"Espeak error: {e}")

    # ------------------- Speech Recognition -------------------
    def listen(self):
        if not self.mic_available or not self.microphone:
            print("No microphone available, skipping listen...")
            return None
        if self.speaking_event.is_set():
            self._log_event('listen_skipped', level=logging.DEBUG, reason='tts_active')
            if self.verbose_console:
                print("Listen skipped: TTS still active")
            time.sleep(0.2)
            return None

        if self.speech_cooldown > 0:
            elapsed = time.time() - self.last_speech_time
            if elapsed < self.speech_cooldown:
                remaining = self.speech_cooldown - elapsed
                self._log_event('listen_skipped', level=logging.DEBUG, reason='cooldown', remaining=f"{remaining:.2f}")
                if self.verbose_console:
                    print(f"Listen skipped: cooldown ({remaining:.1f}s remaining)")
                time.sleep(min(remaining, 0.5))
                return None
        print("Listening...")
        try:
            with self.microphone as source:
                if self.music_playing:
                    self.set_music_volume(0.1, smooth=True)
                    time.sleep(0.2)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                if self.music_playing:
                    self.set_music_volume(0.6, smooth=True)
                text = self.recognizer.recognize_google(audio)
                self.last_user_speech_time = time.time()
                print(f"Heard: {text}")
                return text
        except sr.WaitTimeoutError:
            print("No speech detected")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except Exception as e:
            print(f"Error listening: {e}")
        if self.music_playing:
            self.set_music_volume(0.6, smooth=True)
        return None

    # ------------------- AI Responses -------------------
    def get_ai_response(self, user_input=None, situation='greeting', context=None):
        """Get AI response based on the situation.
        
        Args:
            user_input: Optional user speech input
            situation: One of 'greeting', 'movement', 'taunt', 'lure', 'engage'
            context: Optional dict with additional context (distance, direction, etc.)
        """
        # Build context-aware prompt based on situation
        if user_input:
            prompt = f"You are a spooky but child-friendly Halloween monster in a haunted house. The visitor just said: '{user_input}'. Respond in character - be playful, spooky, and engaging. Ask them a follow-up question to keep the conversation going (about their costume, their fears, what candy they have, etc.). Keep it under 30 words."
        else:
            base_prompts = {
                'greeting': "You are a spooky Halloween monster greeting a new visitor who just arrived. Be dramatic, welcoming, and child-friendly. Introduce yourself in a creepy way and ask them a question (their name, what brings them here, etc.). Keep it under 30 words.",
                'movement': "You are a spooky monster reacting to someone moving. {movement_context} Be playful and curious - ask them a question about what they're doing or where they're going. Keep it under 25 words.",
                'taunt': "You are a Halloween monster. Someone is being still and quiet. Playfully question why they're so quiet or what they're afraid of. Taunt them with curiosity. Keep it under 20 words.",
                'lure': "You are a spooky monster trying to attract new visitors passing by. Ask them an enticing question - what do they seek, are they brave enough, do they want treats? Keep it under 20 words.",
                'engage': "You are a playful but menacing haunted house monster. Someone is nearby but quiet. Ask them something intriguing - what's their name, what do they fear most, what's their favorite scary creature, or if they'd make a tasty snack? Be darkly playful. Keep it under 25 words.",
                'leaving': "You are a desperate spooky monster - someone is leaving! Ask them urgently why they're leaving, what they're afraid of, or beg them to stay and answer one more question. Promise them something if they return. Sound urgent and dramatic but playful. Keep it under 30 words."
            }
            prompt = base_prompts.get(situation, base_prompts['greeting'])
            
            # Add movement context if available
            if situation == 'movement' and context and 'direction' in context:
                if context['direction'] == 'approaching':
                    movement_detail = "They're coming closer to you - react with dark curiosity and ask them why they dare approach!"
                elif context['direction'] == 'moving_away':
                    movement_detail = "They're moving away - ask them urgently where they think they're going or what they're afraid of!"
                else:
                    movement_detail = "They're moving around - question their movements with playful menace."
                prompt = prompt.format(movement_context=movement_detail)

        # Log the prompt we're sending to AI
        self._log_event('ai_request', situation=situation, prompt=prompt[:120])
        if self.verbose_console:
            print(f"AI Prompt ({situation}): {prompt[:100]}...")

        # Try AI services in order
        response = None
        if self.ai_service == 'groq':
            try:
                response = self._sanitize_response(self._get_groq_response(prompt, situation), situation)
            except Exception as e:
                print(f"Groq error: {e}")
        
        if not response and self.ai_service == 'openai':
            try:
                response = self._sanitize_response(self._get_openai_response(prompt), situation)
            except Exception as e:
                print(f"OpenAI error: {e}")
                
        if not response and self.ai_service == 'anthropic':
            try:
                response = self._sanitize_response(self._get_anthropic_response(prompt), situation)
            except Exception as e:
                print(f"Anthropic error: {e}")

        # If all AI services fail, use appropriate fallback
        if not response:
            response = self._fallback_line(situation)
        
        # Log the AI response
        self._log_event('ai_response', situation=situation, response=response[:120])
        if self.verbose_console:
            print(f"AI Response: {response[:100]}...")
        
        return response

    def _sanitize_response(self, response, situation):
        """Ensure AI responses are usable text; fall back if they look invalid."""
        try:
            if isinstance(response, bytes):
                response = response.decode('utf-8', errors='ignore')
        except Exception:
            response = None

        if response is None:
            return self._fallback_line(situation)

        if not isinstance(response, str):
            try:
                response = str(response)
            except Exception:
                return self._fallback_line(situation)

        cleaned = response.strip()
        if not cleaned:
            return self._fallback_line(situation)

        # Detect numeric-like outputs from guard models and discard them
        numeric_like = False
        try:
            float(cleaned)
            numeric_like = True
        except ValueError:
            pass
        if numeric_like and len(cleaned) <= 12:
            return self._fallback_line(situation)

        # Some models return JSON or other structures; keep only plain text segments
        if cleaned.startswith('{') and cleaned.endswith('}'):
            return self._fallback_line(situation)

        return cleaned

    def _fallback_line(self, situation):
        options = FALLBACK_RESPONSES.get(situation, FALLBACK_RESPONSES['greeting'])
        return random.choice(options)

    # ------------------- Groq handler (minimal) -------------------
    def _get_groq_response(self, user_input, situation='greeting'):
        """Minimal Groq response handler. Attempts to call common SDK methods, falls back to generic messages."""
        if not self.client:
            return self._fallback_line(situation)

        try:
            # Try a few common client patterns
            # 1) client.generate or client.predict
            if hasattr(self.client, 'generate'):
                try:
                    resp = self.client.generate(user_input)
                    # Extract text from common shapes
                    if isinstance(resp, dict) and 'text' in resp:
                        return resp['text']
                    try:
                        return str(resp)
                    except Exception:
                        return self._fallback_line(situation)
                except TypeError:
                    # try model parameter
                    try:
                        resp = self.client.generate(model='groq:latest', input=user_input)
                        if isinstance(resp, dict) and 'text' in resp:
                            return resp['text']
                        return str(resp)
                    except Exception:
                        pass

            if hasattr(self.client, 'predict'):
                try:
                    resp = self.client.predict(user_input)
                    return str(resp)
                except Exception:
                    pass

            # 2) client.chat or client.create_chat

            for name in ('chat', 'create_chat', 'send'):
                if hasattr(self.client, name):
                    fn = getattr(self.client, name)
                    try:
                        r = fn(user_input)
                        return str(r)
                    except Exception:
                        continue

            # 3) Try nested patterns commonly used by newer SDKs
            try_patterns = [
                (('responses', 'create'), {'input': user_input}),
                (('responses', 'generate'), {'prompt': user_input}),
                (('completions', 'create'), {'prompt': user_input}),
                (('completions', 'generate'), {'prompt': user_input}),
                (('create_completion',), {'prompt': user_input}),
                (('complete',), {'prompt': user_input}),
                (('run',), {'input': user_input}),
            ]

            for path, kwargs in try_patterns:
                obj = self.client
                ok = True
                for part in path:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                try:
                    resp = obj(**kwargs)
                    # try to extract text
                    if isinstance(resp, dict) and 'text' in resp:
                        return resp['text']
                    try:
                        return str(resp)
                    except Exception:
                        return self._fallback_line(situation)
                except Exception:
                    continue

            # 4) Try Groq-style client.chat.create(...) if available
            try:
                # First try to get available models
                model_to_use = None
                try:
                    if hasattr(self.client, 'models'):
                        models_list = self.client.models.list()
                        if hasattr(models_list, 'data'):
                            available_models = []
                            for entry in models_list.data:
                                candidate = None
                                for attr in ('id', 'model_id', 'name'):
                                    candidate = getattr(entry, attr, None)
                                    if candidate:
                                        break
                                if candidate and isinstance(candidate, str):
                                    available_models.append(candidate)

                            deny_keywords = ('tts', 'audio', 'vision', 'embed', 'whisper', 'speech', 'voice', 'spectra')
                            filtered_models = [
                                m for m in available_models
                                if not any(bad in m.lower() for bad in deny_keywords)
                            ]

                            allow_keywords = (
                                'gpt', 'llama', 'mixtral', 'mistral', 'moonshot',
                                'qwen', 'gemma', 'phi', 'command', 'sonnet', 'goliath',
                                'wizard', 'orca', 'granite', 'pplx', 'deepseek'
                            )
                            allowed_models = [
                                m for m in filtered_models
                                if any(key in m.lower() for key in allow_keywords)
                            ]

                            shortlisted = allowed_models or filtered_models or available_models

                            preferred_models = [
                                'gpt-4.1-mini',
                                'gpt-4.1',
                                'meta-llama/llama-4-sonnet-70b-128k-instruct',
                                'meta-llama/llama-4-maverick-17b-128e-instruct',
                                'moonshotai/kimi-k2-instruct-0905',
                                'mixtral-8x7b-32768',
                                'mixtral-8x7b',
                                'qwen/qwen3-32b',
                                'llama2-70b-4096'
                            ]
                            for m in preferred_models:
                                if m in shortlisted:
                                    model_to_use = m
                                    break
                            if not model_to_use and shortlisted:
                                model_to_use = shortlisted[0]
                except Exception as e:
                    if os.environ.get('GROQ_DEBUG'):
                        print(f"Could not list models, will try default: {e}")
                
                if not model_to_use:
                    model_to_use = "llama2-70b-4096"  # Fallback to a known model
                
                chat_obj = getattr(self.client, 'chat', None)
                # Include prompt instruction in the user message since system messages aren't supported
                prompt = (
                    "Respond as a spooky Halloween monster speaking to children and parents. "
                    "Keep the response short (1-2 short sentences), scary but child-appropriate, "
                    "and in character as a haunted creature. Use dramatic, eerie language "
                    "but avoid anything too frightening. Here is what the visitor said: " + user_input
                )
                msgs = [{"role": "user", "content": prompt}]

                # Prefer chat.completions if present (common in groq SDK)
                try:
                    comp = getattr(chat_obj, 'completions', None)
                    if comp is not None:
                        # try common method names on completions
                        for method_name in ('create', 'generate', 'complete'):
                            if hasattr(comp, method_name):
                                fn = getattr(comp, method_name)
                                try:
                                                        # try messages kw with required model parameter
                                    try:
                                        # Use the model we found earlier
                                        resp = fn(
                                            model=model_to_use,
                                            messages=msgs
                                        )
                                        print(f"Successfully using Groq model: {model_to_use}")
                                    except TypeError:
                                        resp = fn(prompt=user_input)

                                    # extract text from resp
                                    if isinstance(resp, dict):
                                        if 'text' in resp:
                                            return resp['text']
                                        if 'content' in resp:
                                            return resp['content']
                                        if 'choices' in resp and resp['choices']:
                                            first = resp['choices'][0]
                                            if isinstance(first, dict):
                                                for k in ('message', 'text', 'content'):
                                                    if k in first:
                                                        v = first[k]
                                                        if isinstance(v, dict) and 'content' in v:
                                                            return v['content']
                                                        return v
                                    # object-like
                                    if hasattr(resp, 'choices') and getattr(resp, 'choices'):
                                        first = getattr(resp, 'choices')[0]
                                        # handle objects with .message.content
                                        if hasattr(first, 'message') and hasattr(first.message, 'content'):
                                            return first.message.content
                                        for attr in ('text', 'content'):
                                            if hasattr(first, attr):
                                                return getattr(first, attr)
                                    if hasattr(resp, 'text'):
                                        return getattr(resp, 'text')
                                    if hasattr(resp, 'content'):
                                        return getattr(resp, 'content')
                                except Exception as e:
                                    if os.environ.get('GROQ_DEBUG'):
                                        print(f"Groq chat.completions.{method_name} failed: {e}")
                                    continue

                except Exception:
                    pass

                # As a final attempt, try callable chat or chat.create
                try:
                    if callable(chat_obj):
                        try:
                            resp = chat_obj(messages=msgs)
                        except TypeError:
                            resp = chat_obj(user_input)
                        if isinstance(resp, (str, bytes)):
                            return resp.decode() if isinstance(resp, bytes) else resp
                        if isinstance(resp, dict) and 'text' in resp:
                            return resp['text']
                        if hasattr(resp, 'text'):
                            return getattr(resp, 'text')
                    if hasattr(chat_obj, 'create'):
                        try:
                            resp = chat_obj.create(messages=msgs)
                            if isinstance(resp, dict) and 'text' in resp:
                                return resp['text']
                            if hasattr(resp, 'text'):
                                return getattr(resp, 'text')
                        except Exception as e:
                            if os.environ.get('GROQ_DEBUG'):
                                print(f"Groq chat.create failed: {e}")
                except Exception as e:
                    if os.environ.get('GROQ_DEBUG'):
                        print(f"Groq chat final attempts failed: {e}")
            except Exception:
                pass

            # As last resort, dump a compact diagnostic to help debugging
            public_attrs = [n for n in dir(self.client) if not n.startswith('_')]
            if os.environ.get('GROQ_DEBUG'):
                print("Groq client present but no supported method found; returning fallback response.")
                print("Groq client public attributes (sample):", ", ".join(public_attrs[:50]))
                print("If you paste this list I can adapt the handler to call the correct method.")
            return self._fallback_line(situation)

        except Exception as e:
            if os.environ.get('GROQ_DEBUG'):
                print(f"Groq handler error: {e}")
            return self._fallback_line(situation)

    # (AI methods remain unchanged: _get_groq_response, _get_openai_response, _get_anthropic_response)

    # ------------------- Interaction & Run Loop -------------------
    # (interact, run, cleanup methods remain unchanged)
    # Copy the rest of your original script for these methods.
    
    def interact(self):
        """Main interaction loop when someone is detected"""
        self.is_active = True
        self.conversation_history = []
        self.interaction_level = 0
        self.conversation_count = 0
        last_distance = self.measure_distance()
        last_movement_time = time.time()
        distance_samples = []  # Track recent distance readings
        leaving_warned = False  # Track if we've warned about leaving
        self._log_event('interaction_started', initial_distance_cm=(f"{last_distance:.2f}" if last_distance > 0 else 'unknown'))
        
        # Start spooky music
        self.play_spooky_music()
        
        # Randomly pick initial voice (20% chance to change from default)
        if random.random() < 0.2:
            self._randomize_voice()
        
        # Initial greeting
        greeting = self.get_ai_response(situation='greeting')
        self.speak(greeting)
        self._log_event('response_triggered', situation='greeting')
        self.last_engage_time = time.time()
        self.wait_for_speech()  # Wait for greeting to finish before listening
        
        while self.is_active:
            distance = self.measure_distance()
            now = time.time()

            # Check if they've left the detection zone
            if distance > DETECTION_DISTANCE:
                if not leaving_warned:
                    # Panic! They're leaving!
                    print("PANIC: Visitor is leaving!")
                    self._log_event('victim_leaving', distance_cm=f"{distance:.2f}" if distance > 0 else 'unknown')
                    self.panic_music()
                    response = self.get_ai_response(situation='leaving')
                    self.speak(response)
                    self._log_event('response_triggered', situation='leaving')
                    leaving_warned = True
                    self.wait_for_speech()
                    time.sleep(2)  # Give them a chance to return
                else:
                    # They really left
                    self._log_event('victim_escaped', distance_cm=f"{distance:.2f}" if distance > 0 else 'unknown')
                    break

            if 0 < distance < DETECTION_DISTANCE:
                self.last_presence_time = now
                leaving_warned = False  # Reset if they come back
                distance_samples.append(distance)
                # Keep only last 5 samples for trend detection
                if len(distance_samples) > 5:
                    distance_samples.pop(0)

            # Detect significant movement (approaching or leaving)
            if (
                distance > 0 and last_distance > 0 and
                (now - last_movement_time) >= 3.0  # Min 3 seconds between movement reactions
            ):
                delta = distance - last_distance
                
                # Check for instant movement (5cm) or gradual trend (10cm over samples)
                instant_movement = abs(delta) > 5
                trend_movement = False
                if len(distance_samples) >= 3:
                    oldest = distance_samples[0]
                    newest = distance_samples[-1]
                    trend_delta = newest - oldest
                    trend_movement = abs(trend_delta) > 10
                
                if instant_movement or trend_movement:
                    direction = "approaching" if delta < 0 else "moving_away"
                    
                    # Log with console output for visibility
                    print(f"Movement detected: {direction} (delta={delta:.1f}cm, distance={distance:.1f}cm)")
                    self._log_event(
                        'movement_detected',
                        distance_cm=f"{distance:.2f}",
                        delta_cm=f"{delta:.2f}",
                        direction=direction
                    )
                    
                    # Boost interaction level on approach
                    if direction == "approaching":
                        self.interaction_level = min(2, self.interaction_level + 1)
                        print(f"Interaction level increased to {self.interaction_level}")
                        # 20% chance to change voice on approach
                        if random.random() < 0.2:
                            self._randomize_voice()
                    
                    response = self.get_ai_response(situation='movement', context={'direction': direction, 'distance': distance})
                    self.speak(response)
                    self._log_event('response_triggered', situation='movement', direction=direction)
                    self.last_engage_time = time.time()
                    last_movement_time = now
                    distance_samples.clear()  # Reset samples after reaction
                    self.wait_for_speech()  # Wait before listening again

            if distance > 0:
                last_distance = distance

            if (
                self.engage_idle_seconds > 0 and
                0 < distance < DETECTION_DISTANCE and
                not self.speaking_event.is_set()
            ):
                since_user = None
                if self.last_user_speech_time:
                    since_user = now - self.last_user_speech_time
                since_engage = None
                if self.last_engage_time:
                    since_engage = now - self.last_engage_time
                user_ready = (self.last_user_speech_time == 0.0) or (since_user is not None and since_user >= self.engage_idle_seconds)
                engage_ready = (self.last_engage_time == 0.0) or (since_engage is not None and since_engage >= self.engage_idle_seconds)
                if user_ready and engage_ready and (now - self.last_speech_time) >= 2:
                    self._log_event(
                        'engagement_prompt',
                        idle_since_user=(f"{since_user:.2f}" if since_user is not None else 'unknown'),
                        idle_since_engage=(f"{since_engage:.2f}" if since_engage is not None else 'unknown')
                    )
                    response = self.get_ai_response(situation='engage')
                    self.speak(response)
                    self._log_event('response_triggered', situation='engage')
                    self.last_engage_time = time.time()
                    self.wait_for_speech()  # Wait before listening again
                    continue

            if self.mic_available:
                user_input = self.listen()
                if user_input:
                    # User spoke! Boost interaction and respond
                    self.interaction_level = 2  # Max engagement when they talk
                    self.conversation_count += 1
                    print(f"Conversation exchange #{self.conversation_count}, interaction level: {self.interaction_level}")
                    
                    # Randomly change voice every few exchanges (30% chance)
                    if self.conversation_count % 2 == 0 and random.random() < 0.3:
                        self._randomize_voice()
                    
                    self.last_engage_time = time.time()
                    self._log_event('speech_detected', transcript=user_input[:120], conversation_count=self.conversation_count)
                    response = self.get_ai_response(user_input=user_input)
                    self.speak(response)
                    self._log_event('response_triggered', situation='conversation', conversation_count=self.conversation_count)
                    self.wait_for_speech()  # Wait before listening again
                    
                    # High engagement: keep listening actively for follow-up
                    if self.conversation_count < 10:  # Limit to prevent infinite loops
                        time.sleep(0.3)  # Brief pause then listen again
                        continue
            else:
                time.sleep(0.5)

            time.sleep(0.2)
            
        # End interaction
        self.stop_music()
        self.is_active = False
        print("Interaction ended. Waiting for next victim...")
        self._log_event('interaction_finished')
        time.sleep(2)
    
    def run(self):
        """Main loop - monitor distance and trigger interactions"""
        try:
            while True:
                # Always measure distance (but only trigger interaction when idle)
                if not self.is_active:
                    distance = self.measure_distance()

                    # Display the current distance on a single console line (in-place update)
                    try:
                        if distance > 0:
                            print(f"\rDistance: {distance:.1f} cm", end='', flush=True)
                        else:
                            print(f"\rDistance: -- cm", end='', flush=True)
                    except Exception:
                        # Fallback to regular print if in-place update fails
                        if distance > 0:
                            print(f"Distance: {distance:.1f} cm")
                        else:
                            print("Distance: -- cm")

                    # Trigger interaction if within detection threshold
                    if 0 < distance < DETECTION_DISTANCE:
                        # move to next line so interaction prints are readable
                        print("")
                        print(f"Victim detected at {distance:.1f} cm!")
                        self._log_event('victim_detected', distance_cm=f"{distance:.2f}")
                        self.interact()

                # Sleep a short time between measurements
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\nShutting down Halloween Monster...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        # Stop TTS worker first
        try:
            self.tts_stop.set()
            # Wait briefly for pending items to be processed
            try:
                self.tts_queue.join()
            except Exception:
                pass
            if hasattr(self, 'tts_thread') and self.tts_thread:
                self.tts_thread.join(timeout=2)
        except Exception as e:
            print(f"Error stopping TTS worker: {e}")

        self.stop_music()
        pygame.mixer.quit()
        GPIO.cleanup()
        print("Cleanup complete. Happy Halloween!")


    @staticmethod
    def list_elevenlabs_voices_cli():
        """Standalone helper to list ElevenLabs voices without initializing hardware."""
        if not os.environ.get("ELEVENLABS_API_KEY"):
            print("No ELEVENLABS_API_KEY set. Cannot query ElevenLabs.")
            return []

        try:
            client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        except Exception as e:
            print(f"Could not initialize ElevenLabs client: {e}")
            return []

        voices = None
        try:
            voices = client.text_to_speech.list_voices()
        except Exception:
            try:
                voices = client.voices.list()
            except Exception as e:
                print(f"Could not retrieve voices: {e}")
                return []

        result = []
        for v in voices:
            if isinstance(v, dict):
                vid = v.get("voice_id") or v.get("id") or v.get("name")
                name = v.get("name") or vid
            else:
                vid = getattr(v, "voice_id", None) or getattr(v, "id", None) or getattr(v, "name", None)
                name = getattr(v, "name", None) or vid
            if vid:
                result.append((vid, name))
        return result

if __name__ == "__main__":
    print("\n" + "="*60)
    print("HALLOWEEN MONSTER CONFIGURATION")
    print("="*60)
    print(f"Sensor Trigger Pin (SENSOR_TRIG): GPIO {SENSOR_TRIG}")
    print(f"Sensor Echo Pin (SENSOR_ECHO): GPIO {SENSOR_ECHO}")
    print(f"Detection Distance: {DETECTION_DISTANCE} cm")
    print(f"Sensor Type: {SENSOR_TYPE}")
    print(f"AI Service: {AI_SERVICE}")
    print("="*60)
    if not ANTHROPIC_API_KEY:
        print("\nWARNING: ANTHROPIC_API_KEY not set! Monster will use fallback responses.")
    # CLI helper: list available ElevenLabs voices and exit
    if "--list-eleven-voices" in sys.argv or "--eleven-voices" in sys.argv:
        voices = HalloweenMonster.list_elevenlabs_voices_cli()
        if voices:
            print("Available ElevenLabs voices:")
            for vid, name in voices:
                print(f"  {vid}\t- {name}")
        else:
            print("No voices found or ELEVENLABS_API_KEY missing/invalid.")
        sys.exit(0)

    monster = HalloweenMonster()
    monster.run()
