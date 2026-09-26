"""Make-up gain: bring quiet playback up to a level Whisper hears well.

Local-Live-Captions measured the failure this prevents: the same recording
played at 0.05x — peak ~0.01, an ordinary low volume — loses the first word of
each utterance and garbles the rest, and make-up gain recovers the sentences
exactly. At normal levels the gain is ~1 and changes nothing.

The gain comes from the loudest peak over the last GAIN_WINDOW_S, not from
the current block. A pause between sentences is two orders of magnitude
quieter than speech, and a per-block gain would pump it up to speech level.
"""

from collections import deque

import numpy as np

from src.config import (
    SAMPLE_RATE, CAPTURE_CHUNK_MS, GAIN_TARGET_PEAK, GAIN_MAX, GAIN_WINDOW_S,
    GAIN_FLOOR_PEAK,
)


class AutoGain:
    """Stateful gain over a stream of float32 blocks.

    Works in fixed CAPTURE_CHUNK_MS blocks internally, so the result does not
    depend on how the caller happened to slice the stream — a live session and
    its replay get the same gain.
    """

    def __init__(self, target_peak=GAIN_TARGET_PEAK, max_gain=GAIN_MAX,
                 window_s=GAIN_WINDOW_S, floor_peak=GAIN_FLOOR_PEAK,
                 sample_rate=SAMPLE_RATE, block_ms=CAPTURE_CHUNK_MS):
        self.target_peak = target_peak
        self.max_gain = max_gain
        self.floor_peak = floor_peak
        self._block = sample_rate * block_ms // 1000
        self._window_blocks = max(1, int(window_s * 1000 // block_ms))
        self._peaks: deque[float] = deque(maxlen=self._window_blocks)
        self._pending = np.zeros(0, dtype=np.float32)
        self.gain = 1.0
        self.peak = 0.0      # loudest peak in the window, before gain

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        whole = len(audio) - len(audio) % self._block
        self._pending = audio[whole:]
        out = np.empty(whole, dtype=np.float32)
        for i in range(0, whole, self._block):
            block = audio[i:i + self._block]
            self._peaks.append(float(np.abs(block).max()))
            self.peak = max(self._peaks)
            if self.peak < self.floor_peak:
                self.gain = 1.0
            else:
                self.gain = min(self.max_gain, max(1.0, self.target_peak / self.peak))
            np.clip(block * self.gain, -1.0, 1.0, out=out[i:i + self._block])
        return out

    def flush(self) -> np.ndarray:
        """The last partial block, at the current gain."""
        rest, self._pending = self._pending, np.zeros(0, dtype=np.float32)
        return np.clip(rest * self.gain, -1.0, 1.0).astype(np.float32)
