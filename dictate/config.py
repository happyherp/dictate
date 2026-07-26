import os

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

# LLM (via OpenRouter)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL      = "anthropic/claude-haiku-4-5"
COMMAND_PREFIX = "computer"   # say "computer, ..." to trigger command mode

# X11
DISPLAY        = os.environ.get("DISPLAY", ":0")
