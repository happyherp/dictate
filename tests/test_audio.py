from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dictate import config
from dictate.audio import FRAME_SAMPLES, AudioListener


def _frame(speech: bool):
    """One silent float32 frame; speech-ness is faked via the mocked VAD."""
    return np.zeros((FRAME_SAMPLES, 1), dtype=np.float32)


@pytest.fixture
def listener():
    with patch("dictate.audio.webrtcvad.Vad") as mock_vad_cls:
        mock_vad = MagicMock()
        mock_vad_cls.return_value = mock_vad
        l = AudioListener(on_utterance=MagicMock())
        yield l, mock_vad


def _feed(listener, vad, speech_flags):
    for is_speech in speech_flags:
        vad.is_speech.return_value = is_speech
        listener._callback(_frame(is_speech), FRAME_SAMPLES, None, None)


def test_stays_idle_without_enough_speech_frames(listener):
    l, vad = listener
    _feed(l, vad, [True] * (config.SPEECH_FRAMES - 1))
    assert l.recording is False
    l.on_utterance.assert_not_called()


def test_starts_recording_after_speech_frames(listener):
    l, vad = listener
    _feed(l, vad, [True] * config.SPEECH_FRAMES)
    assert l.recording is True


def test_emits_utterance_after_sufficient_speech_then_silence(listener):
    l, vad = listener
    speech_count = config.SPEECH_FRAMES + config.MIN_SPEECH_FRAMES
    _feed(l, vad, [True] * speech_count)
    _feed(l, vad, [False] * config.SILENCE_FRAMES)

    assert l.recording is False
    l.on_utterance.assert_called_once()
    (audio_bytes,) = l.on_utterance.call_args.args
    # The frame that flips recording on isn't itself buffered, so only
    # (speech_count - SPEECH_FRAMES) speech frames get appended, plus all
    # SILENCE_FRAMES trailing silence frames (also buffered while recording).
    expected_frames = speech_count - config.SPEECH_FRAMES + config.SILENCE_FRAMES
    assert len(audio_bytes) == expected_frames * FRAME_SAMPLES * 2


def test_discards_short_utterance(listener, monkeypatch):
    l, vad = listener
    # With real config, trailing silence alone (SILENCE_FRAMES=30) already
    # exceeds MIN_SPEECH_FRAMES=20, so the discard path can only be reached
    # by raising the threshold above SILENCE_FRAMES for this test.
    monkeypatch.setattr("dictate.audio.config.MIN_SPEECH_FRAMES", config.SILENCE_FRAMES + 5)

    _feed(l, vad, [True] * config.SPEECH_FRAMES)
    _feed(l, vad, [False] * config.SILENCE_FRAMES)

    assert l.recording is False
    l.on_utterance.assert_not_called()


def test_silence_count_resets_on_intermittent_speech(listener):
    l, vad = listener
    speech_count = config.SPEECH_FRAMES + config.MIN_SPEECH_FRAMES
    _feed(l, vad, [True] * speech_count)
    _feed(l, vad, [False] * (config.SILENCE_FRAMES - 1))
    assert l.recording is True
    _feed(l, vad, [True])
    assert l.silence_count == 0
    assert l.recording is True
    l.on_utterance.assert_not_called()
