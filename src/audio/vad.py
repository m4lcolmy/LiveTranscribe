"""Speech probability for every 32 ms frame of the stream, computed once.

Silero, as bundled with faster-whisper. The model is recurrent: each 512-sample
frame is scored from the frame, the 64 samples before it, and an LSTM state
(h, c) carried over from the frames before. faster-whisper's wrapper starts
that state from zero on every call, which is right for a whole file and wrong
for a stream fed a piece at a time — every piece would begin with no memory of
the last, and the scores would differ from a single pass (the test caught
exactly that). So the session is driven here directly, carrying h, c and the
64-sample context from one piece to the next. A stream scored in pieces then
gets the same probabilities as the whole recording scored at once.

Frames are indexed from the start of the stream, so the streamer can ask about
any stretch of it by sample index.
"""

import numpy as np

from src.config import VAD_THRESHOLD

FRAME = 512
CONTEXT = 64


class StreamingVad:
    def __init__(self, threshold: float = VAD_THRESHOLD, model=None):
        if model is None:
            from faster_whisper.vad import get_vad_model
            model = get_vad_model()
        self._session = model.session
        self.threshold = threshold
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._probs = np.zeros(0, dtype=np.float32)
        self._first_frame = 0     # stream frame index of _probs[0]

    # ── Feeding ────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray):
        self._pending = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        n = len(self._pending) // FRAME
        if n == 0:
            return
        frames = self._pending[: n * FRAME].reshape(n, FRAME)
        self._pending = self._pending[n * FRAME:]

        # Each frame's context is the tail of the frame before it — the same
        # layout faster-whisper's wrapper builds, continued across pieces.
        contexts = np.concatenate([self._context[None, :], frames[:-1, -CONTEXT:]], axis=0)
        batch = np.concatenate([contexts, frames], axis=1)
        out, self._h, self._c = self._session.run(
            None, {"input": batch, "h": self._h, "c": self._c}
        )
        self._context = frames[-1, -CONTEXT:].copy()
        self._probs = np.concatenate([self._probs, out.reshape(-1).astype(np.float32)])

    def forget_before(self, sample: int):
        """Drop scores for audio before `sample`; the streamer no longer holds it."""
        # Never past the last scored frame, or later scores would be filed
        # under the wrong frame index.
        drop = min(sample // FRAME - self._first_frame, len(self._probs))
        if drop > 0:
            self._probs = self._probs[drop:]
            self._first_frame += drop

    # ── Questions ──────────────────────────────────────────────────────

    @property
    def scored_until(self) -> int:
        """Stream sample index up to which every frame has a score."""
        return (self._first_frame + len(self._probs)) * FRAME

    def _window(self, start: int, end: int) -> tuple[int, np.ndarray]:
        lo = max(start // FRAME, self._first_frame)
        hi = min(end // FRAME, self._first_frame + len(self._probs))
        if hi <= lo:
            return lo, np.zeros(0, dtype=np.float32)
        return lo, self._probs[lo - self._first_frame: hi - self._first_frame]

    def speech_samples(self, start: int, end: int) -> int:
        """How much of [start, end) is speech, in samples."""
        _, probs = self._window(start, end)
        return int((probs >= self.threshold).sum()) * FRAME

    def last_speech_end(self, start: int, end: int) -> int | None:
        """End sample of the last speech frame in [start, end), or None if none."""
        lo, probs = self._window(start, end)
        idx = np.flatnonzero(probs >= self.threshold)
        if len(idx) == 0:
            return None
        return (lo + int(idx[-1]) + 1) * FRAME

    def last_pause(self, start: int, end: int, min_frames: int = 2) -> int | None:
        """Middle sample of the last run of >= min_frames silent frames in [start, end)."""
        lo, probs = self._window(start, end)
        silent = np.concatenate([[False], probs < self.threshold, [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(silent))       # alternating run starts and ends
        starts, ends = edges[0::2], edges[1::2]
        long_enough = np.flatnonzero(ends - starts >= min_frames)
        if len(long_enough) == 0:
            return None
        k = long_enough[-1]
        return (lo * 2 + int(starts[k]) + int(ends[k])) * FRAME // 2

    def probability(self, start: int, end: int) -> float:
        """Mean speech probability over [start, end) — for the status line."""
        _, probs = self._window(start, end)
        return float(probs.mean()) if len(probs) else 0.0
