import os

from dotenv import load_dotenv

load_dotenv()

# Audio
MIC_DEVICE     = 5       # Logi USB Headset (run: python -c "import sounddevice as sd; print(sd.query_devices())")
SAMPLE_RATE    = 16000
FRAME_MS       = 30
VAD_MODE       = 3       # 0-3, 3 = most aggressive noise rejection
SPEECH_FRAMES  = 8       # frames of speech to start recording
SILENCE_FRAMES = 30      # frames of silence to end utterance (~900ms)
MIN_SPEECH_FRAMES = 20   # discard utterances shorter than this

# Whisper / OpenVINO
WHISPER_MODEL  = "openai/whisper-base"
OV_DEVICE      = "GPU"   # "CPU" as fallback
ALLOWED_LANGUAGES = ("en", "de", "es")   # restrict decoding to these; rejects hallucinated languages (e.g. Korean on silence)
FALLBACK_LANGUAGE = "en"                 # used when detected language isn't in ALLOWED_LANGUAGES
OV_MODEL_DIR = ".ov_model"          # cached OpenVINO IR export; skips re-converting from PyTorch on every startup
OV_KERNEL_CACHE_DIR = ".ov_cache"   # OpenVINO compiled-kernel cache; skips GPU kernel recompilation on every startup

# LLM (via OpenRouter)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL      = "anthropic/claude-haiku-4-5"
COMMAND_PREFIXES = ("computer", "shooter")   # say "computer, ..." to trigger command mode; "shooter" is a common Whisper mis-hearing of "computer"
DICTATION_TRIGGERS = ("platypus", "platypurs", "platyposs", "platyposts")   # say "platypus, ..." to trigger plain-text typing; aliases cover observed word-ending mis-hearings; empty tuple lets the LLM judge intent instead

# X11
DISPLAY        = os.environ.get("DISPLAY", ":0")
