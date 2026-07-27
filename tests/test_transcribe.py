import wave
from pathlib import Path

import pytest

from dictate.transcribe import is_hallucination, load_model, transcribe

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    return load_model()


def _load_wav_bytes(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def test_transcribe_pangram_fixture(model):
    processor, m = model
    audio = _load_wav_bytes(FIXTURES / "pangram_en.wav")
    text = transcribe(processor, m, audio)
    assert text.strip(" .").lower() == "the quick brown fox jumps over the lazy dog"


def test_is_hallucination_rejects_empty():
    assert is_hallucination("")
    assert is_hallucination("a")


def test_is_hallucination_rejects_repeated_char():
    assert is_hallucination("sssssss")


def test_is_hallucination_rejects_noise_phrases():
    assert is_hallucination("thanks for watching")


def test_is_hallucination_accepts_real_text():
    assert not is_hallucination("The quick brown fox jumps over the lazy dog.")
