"""An audio or video file, released at the pace it would play.

`--file` exists so a clip can be watched being transcribed exactly as live
audio would be — same gain, same streamer, same timing — and so a problem seen
live can be reproduced from the recording.
"""

import time
from pathlib import Path

import numpy as np

from src.config import SAMPLE_RATE


class FileSource:
    def __init__(self, path: str | Path, speed: float = 1.0):
        from faster_whisper.audio import decode_audio

        self.path = Path(path)
        self.speed = speed
        self.audio = decode_audio(str(self.path), sampling_rate=SAMPLE_RATE)
        self._pos = 0
        self._t0 = 0.0

    @property
    def label(self) -> str:
        return f"file {self.path.name} ({len(self.audio) / SAMPLE_RATE:.0f}s)"

    def start(self):
        self._t0 = time.monotonic()

    def stop(self):
        pass

    def read(self, timeout: float) -> np.ndarray | None:
        """The audio that has 'played' since the last read; None once it has all played."""
        if self._pos >= len(self.audio):
            return None
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            due = min(len(self.audio),
                      int((time.monotonic() - self._t0) * self.speed * SAMPLE_RATE))
            if due > self._pos or time.monotonic() >= deadline:
                break
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        out = self.audio[self._pos:due]
        self._pos = max(self._pos, due)
        return out
