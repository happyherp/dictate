"""Record a WAV fixture from the configured mic for use in transcription tests.

Usage: uv run python scripts/record_fixture.py <output.wav> [duration_s] [countdown_s]
"""

import sys
import time
import wave

import numpy as np
import sounddevice as sd

from dictate import config


def main():
    out_path = sys.argv[1]
    duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    countdown_s = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

    print(f"Recording starts in {countdown_s:.0f}s on device {config.MIC_DEVICE}...", flush=True)
    time.sleep(countdown_s)
    print("RECORDING NOW", flush=True)

    frames = sd.rec(
        int(duration_s * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=config.MIC_DEVICE,
    )
    sd.wait()
    print("Done recording.", flush=True)

    audio_i16 = (frames[:, 0] * 32767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(audio_i16.tobytes())

    peak = np.abs(audio_i16).max()
    print(f"Saved to {out_path} (peak amplitude {peak}/32767)")


if __name__ == "__main__":
    main()
