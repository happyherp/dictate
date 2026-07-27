import os
from pathlib import Path

import numpy as np
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor
from . import config


def load_model():
    print(f"Loading Whisper ({config.WHISPER_MODEL}) on {config.OV_DEVICE}...")

    # Export to OpenVINO IR once and cache it on disk — re-exporting from the
    # PyTorch checkpoint on every startup is what makes startup slow.
    model_dir = Path(config.OV_MODEL_DIR)
    cached = model_dir.exists()
    source = model_dir if cached else config.WHISPER_MODEL

    if cached:
        # Everything needed is already on disk; skip the Hub network check.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    processor = AutoProcessor.from_pretrained(config.WHISPER_MODEL)
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": config.OV_KERNEL_CACHE_DIR}

    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            source, export=not cached, device=config.OV_DEVICE, ov_config=ov_config,
        )
    except Exception as e:
        print(f"GPU failed ({e}), falling back to CPU")
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            source, export=not cached, device="CPU", ov_config=ov_config,
        )

    if not cached:
        model.save_pretrained(model_dir)

    print("Model ready.\n")
    return processor, model


def _detected_language(processor, generated_ids) -> str:
    if generated_ids.shape[1] < 2:
        return config.FALLBACK_LANGUAGE
    token = processor.tokenizer.convert_ids_to_tokens([generated_ids[0][1].item()])[0]
    return token.strip("<|>")


def transcribe(processor, model, audio_bytes: bytes) -> str:
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    inputs = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    generated = model.generate(inputs["input_features"])
    if _detected_language(processor, generated) not in config.ALLOWED_LANGUAGES:
        generated = model.generate(inputs["input_features"], language=config.FALLBACK_LANGUAGE)
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
