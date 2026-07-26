import numpy as np
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor
from . import config


def load_model():
    print(f"Loading Whisper ({config.WHISPER_MODEL}) on {config.OV_DEVICE}...")
    processor = AutoProcessor.from_pretrained(config.WHISPER_MODEL)
    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            config.WHISPER_MODEL, export=True, device=config.OV_DEVICE,
            ov_config={"PERFORMANCE_HINT": "LATENCY"},
        )
    except Exception as e:
        print(f"GPU failed ({e}), falling back to CPU")
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            config.WHISPER_MODEL, export=True, device="CPU",
            ov_config={"PERFORMANCE_HINT": "LATENCY"},
        )
    print("Model ready.\n")
    return processor, model


def transcribe(processor, model, audio_bytes: bytes) -> str:
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    inputs = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    generated = model.generate(inputs["input_features"])
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def is_hallucination(text: str) -> bool:
    if not text or len(text) < 2:
        return True
    for ch in set(text.lower()):
        if ch.isalpha() and text.lower().count(ch) / len(text) > 0.4:
            return True
    noise_patterns = ["thank you", "thanks for watching", "www.", ".com", "♪", "♫"]
    low = text.lower()
    if any(p in low for p in noise_patterns) and len(text.split()) <= 3:
        return True
    return False
