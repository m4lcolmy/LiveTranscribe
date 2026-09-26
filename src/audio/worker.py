"""The live loop: source → gain → streamer → sinks, paced by the wall clock.

A pass runs every `step_s`. A pass that takes longer than the step is not a
problem in itself: the next one starts as soon as it returns and simply covers
more audio — nothing is dropped, updates just come less often. The session log's
summary says how often that happened.

The terminal mode runs this on the main thread (Ctrl+C stops it); the overlay
runs it on a worker thread and drives it with `stop()` and `set_paused()`.
Both only set flags: the loop applies them itself, so the streamer is only
ever touched from the one thread that runs it.

Pausing stops the capture process itself, so nothing is being recorded while
paused, and closes the utterance in progress, so its words are not lost.
"""

import threading
import time

import numpy as np

from src.audio.gain import AutoGain
from src.audio.streamer import Streamer, Update
from src.core.debug import log
from src.sinks import clock

STATUS_EVERY_S = 0.5


def _quote(text: str) -> str:
    return f'"{text}"'


def _db(peak: float) -> str:
    return f"{20 * np.log10(peak):.0f} dB" if peak > 1e-5 else "silent"


class Pipeline:
    def __init__(self, source, streamer: Streamer, sinks: list, step_s: float,
                 recorder=None, gain: AutoGain | None = None):
        self.source = source
        self.streamer = streamer
        self.sinks = sinks
        self.step_s = step_s
        self.recorder = recorder
        self.gain = gain or AutoGain()
        self.interrupted = False
        self.ended = False             # a file source played to its end
        self.paused = False
        self._want_paused = False
        self._quit = threading.Event()
        self._next_pass = self._next_status = 0.0

    # ── Control, from any thread ───────────────────────────────────────

    def stop(self):
        self._quit.set()

    def set_paused(self, paused: bool):
        self._want_paused = paused

    # ── The loop ───────────────────────────────────────────────────────

    def run(self):
        self._resume()
        try:
            while not self._quit.is_set():
                if self._want_paused != self.paused:
                    self._suspend() if self._want_paused else self._resume()
                    self.paused = self._want_paused
                if self.paused:
                    self._quit.wait(0.1)
                    continue

                wait = min(self._next_pass, self._next_status) - time.monotonic()
                audio = self.source.read(timeout=max(0.0, wait))
                if audio is None:              # a file that has finished playing
                    self.ended = True
                    break
                if audio.size:
                    if self.recorder is not None:
                        self.recorder.audio(audio)
                    self.streamer.feed(self.gain(audio))

                now = time.monotonic()
                if now >= self._next_pass:
                    update = self.streamer.step()
                    self._next_pass = now + self.step_s
                    if update is not None:
                        self._publish(update)
                if now >= self._next_status:
                    self._status()
                    self._next_status = now + STATUS_EVERY_S
        except KeyboardInterrupt:
            self.interrupted = True
        finally:
            try:
                if not self.paused:
                    self._suspend()
            except KeyboardInterrupt:
                pass
            for sink in self.sinks:
                sink.close()
            if self.recorder is not None:
                self.recorder.close()

    def _resume(self):
        self.source.start()
        log.event("START", f"{self.source.label} step={self.step_s:.2f}s")
        now = time.monotonic()
        self._next_pass, self._next_status = now + self.step_s, now

    def _suspend(self):
        """Stop capturing and close the utterance in progress."""
        self.source.stop()
        log.event("STOP", self.source.label)
        self.streamer.feed(self.gain.flush())
        update = self.streamer.finish()
        if update is not None:
            self._publish(update)

    # ── Output ─────────────────────────────────────────────────────────

    def _publish(self, update: Update):
        r = update.report
        if r is not None:
            log.pass_latencies.append(r.latency_s)
            log.event("PASS", f"t={r.at:.2f} buf={r.at - r.buffer_start:.2f}s "
                              f"lat={r.latency_s * 1000:.0f}ms"
                              f"{' FINAL' if r.final else ''}"
                              f"{f' prompt={len(r.prompt)}ch' if r.prompt else ''}")
            log.detail("heard", _quote(r.heard))
            if r.committed:
                log.detail("commit", _quote(" ".join(w.text for w in r.committed)))
            if r.tentative:
                log.detail("tentative", _quote(" ".join(w.text for w in r.tentative)))
            for reason, detail, text in r.dropped:
                log.count(f"drop_{reason}")
                log.detail("DROP", f"{reason} {detail} {_quote(text)}")
        for line in update.finished:
            log.count("lines")
            log.count("words_committed", len(line.words))
            log.event("LINE", f"[{clock(line.start)}-{clock(line.end)}] {_quote(line.text)}")

        for sink in self.sinks:
            sink.update(update)
        if self.recorder is not None:
            self.recorder.update(update)

    def _status(self):
        s = self.streamer
        speaking = s.vad.probability(s.fed - 8000, s.fed) >= s.vad.threshold
        text = (f"● listening — level {_db(self.gain.peak)}, gain {self.gain.gain:.1f}×"
                f"{', speech…' if speaking else ''}")
        for sink in self.sinks:
            sink.status(text)
