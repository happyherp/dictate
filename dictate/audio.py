import collections
import queue
import threading

import numpy as np
import sounddevice as sd
import webrtcvad
from . import config

FRAME_SAMPLES = int(config.SAMPLE_RATE * config.FRAME_MS / 1000)


class AudioListener:
    def __init__(self, on_utterance):
        self.on_utterance = on_utterance
        self.vad = webrtcvad.Vad(config.VAD_MODE)
        self.ring = collections.deque(maxlen=config.SPEECH_FRAMES)
        self.recording = False
        self.silence_count = 0
        self.audio_buffer = b""
        self._utterances = queue.Queue()
        self._worker = threading.Thread(target=self._process_utterances, daemon=True)

    def _callback(self, indata, frames, time, status):
        raw = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        is_speech = self.vad.is_speech(raw, config.SAMPLE_RATE)
        self.ring.append(is_speech)

        if not self.recording:
            if sum(self.ring) >= config.SPEECH_FRAMES:
                self.recording = True
                self.silence_count = 0
                self.audio_buffer = b""
                print("▶", end="", flush=True)
        else:
            self.audio_buffer += raw
            if not is_speech:
                self.silence_count += 1
                if self.silence_count >= config.SILENCE_FRAMES:
                    self.recording = False
                    frames_recorded = len(self.audio_buffer) // (FRAME_SAMPLES * 2)
                    if frames_recorded >= config.MIN_SPEECH_FRAMES:
                        print(" ◼", flush=True)
                        self._utterances.put(self.audio_buffer)
                    else:
                        print(" (short, skipped)", flush=True)
            else:
                self.silence_count = 0

    def _process_utterances(self):
        while True:
            audio = self._utterances.get()
            self.on_utterance(audio)

    def run(self):
        self._worker.start()
        with sd.InputStream(
            device=config.MIC_DEVICE,
            channels=1,
            samplerate=config.SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="float32",
            callback=self._callback,
        ):
            print(f"Listening on device {config.MIC_DEVICE} (Ctrl+C to stop)...")
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print("\nStopped.")
