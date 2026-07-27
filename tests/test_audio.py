import time

import numpy as np

from dictate import config
from dictate.audio import FRAME_SAMPLES, AudioListener


def test_callback_does_not_block_on_slow_handler(monkeypatch):
    calls = []

    def slow_handler(audio):
        time.sleep(0.3)
        calls.append(audio)

    listener = AudioListener(on_utterance=slow_handler)
    listener._worker.start()

    # Drive the VAD deterministically: enough "speech" frames to start
    # recording and pass MIN_SPEECH_FRAMES, then enough "silence" frames
    # to end the utterance.
    speech_frames = config.SPEECH_FRAMES + config.MIN_SPEECH_FRAMES + 2
    silence_frames = config.SILENCE_FRAMES + 1
    pattern = [True] * speech_frames + [False] * silence_frames
    pattern_iter = iter(pattern)
    monkeypatch.setattr(listener.vad, "is_speech", lambda *a, **kw: next(pattern_iter, False))

    indata = np.zeros((FRAME_SAMPLES, 1), dtype="float32")

    start = time.monotonic()
    for _ in pattern:
        listener._callback(indata, FRAME_SAMPLES, None, None)
    elapsed = time.monotonic() - start

    # The callback loop must stay fast even though the handler is slow —
    # proves heavy processing happens off the audio thread, not inline.
    assert elapsed < 0.3

    for _ in range(50):
        if calls:
            break
        time.sleep(0.02)
    assert len(calls) == 1
