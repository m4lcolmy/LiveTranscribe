"""What the computer is playing, via pw-record on the default output's monitor.

    pw-record -P '{ stream.capture.sink=true }' --rate 16000 --channels 1 --format s16 -

`stream.capture.sink=true` with no --target makes WirePlumber link the stream
to the monitor of the *default output* — checked on this machine: it attached
to the same sink Chrome plays into. PipeWire does the 48 kHz → 16 kHz
resampling and the downmix to mono, so what arrives on stdout is exactly what
Whisper wants. No PulseAudio tools, no loopback module, no Python audio
library, none of which can see a PipeWire monitor as cleanly.

pw-record is a child process, so two things are handled here:
  * it can die (PipeWire restarts, a device disappears): it is restarted with
    backoff, and every exit is logged;
  * the default output can change (headphones, Bluetooth): a watcher notices
    and restarts capture on the new one, rather than trusting the stream to
    follow — it may or may not, depending on WirePlumber's policy.
"""

import json
import queue
import shutil
import subprocess
import threading
import time

import numpy as np

from src.config import (
    SAMPLE_RATE, CAPTURE_CHUNK_MS, CAPTURE_RESTART_BACKOFF_S, SINK_WATCH_S,
)
from src.core.debug import log

_CHUNK_BYTES = SAMPLE_RATE * CAPTURE_CHUNK_MS // 1000 * 2


class CaptureError(RuntimeError):
    """Capture cannot start at all — says what to install or check."""


def default_sink() -> str | None:
    """node.name of the default output, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["pw-metadata", "0", "default.audio.sink"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if "value:'" in line:
            try:
                return json.loads(line.split("value:'", 1)[1].rsplit("'", 1)[0])["name"]
            except (ValueError, KeyError, IndexError):
                return None
    return None


class SystemAudioSource:
    """Float32 16 kHz mono blocks of the system's output, as they are played."""

    def __init__(self, sink: str | None = None):
        if shutil.which("pw-record") is None:
            raise CaptureError(
                "pw-record not found. It ships with PipeWire: sudo apt install pipewire-bin"
            )
        # An explicit sink is captured and never switched; None follows the default.
        self.sink = sink
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._current_sink: str | None = None
        self._reader: threading.Thread | None = None
        self._watcher: threading.Thread | None = None

    @property
    def label(self) -> str:
        target = self.sink or self._current_sink or "default output"
        return f"system audio ({target})"

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self):
        """Start capturing; also after a stop() (resume from pause)."""
        self._stop.clear()
        self._queue = queue.Queue()
        self._current_sink = self.sink or default_sink()
        self._reader = threading.Thread(target=self._read_loop, name="capture", daemon=True)
        self._reader.start()
        if self.sink is None:
            self._watcher = threading.Thread(target=self._watch_loop, name="sink-watch", daemon=True)
            self._watcher.start()

    def stop(self):
        self._stop.set()
        self._kill()
        for thread in (self._reader, self._watcher):
            if thread is not None:
                thread.join(timeout=2)

    def read(self, timeout: float) -> np.ndarray | None:
        """Everything captured since the last read; waits up to `timeout` for the first block."""
        try:
            blocks = [self._queue.get(timeout=max(0.0, timeout))]
        except queue.Empty:
            return np.zeros(0, dtype=np.float32)
        while True:
            try:
                blocks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(blocks)

    # ── pw-record ──────────────────────────────────────────────────────

    def _command(self) -> list[str]:
        cmd = [
            "pw-record",
            "-P", "{ stream.capture.sink=true node.name=livetranscribe media.name=LiveTranscribe }",
            "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "s16",
            "--latency", f"{CAPTURE_CHUNK_MS}ms",
        ]
        if self.sink:
            cmd += ["--target", self.sink]
        return cmd + ["-"]

    def _read_loop(self):
        failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with self._proc_lock:
                    self._proc = subprocess.Popen(
                        self._command(), stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, bufsize=0,
                    )
                    proc = self._proc
            except OSError as e:
                log.event("CAPTURE", f"cannot start pw-record: {e!r}")
                proc = None

            if proc is not None:
                log.event("CAPTURE", f"pw-record pid={proc.pid} → {self.label}")
                self._pump(proc)
                code = proc.wait()
                if self._stop.is_set():
                    break
                log.event("CAPTURE", f"pw-record exited code={code} after "
                                     f"{time.monotonic() - started:.1f}s")
                log.count("capture_restarts")

            # A process that ran a while was restarted on purpose (sink change)
            # or died once; one that keeps dying at once backs off.
            failures = 0 if time.monotonic() - started > 5 else failures + 1
            if failures:
                wait = CAPTURE_RESTART_BACKOFF_S[min(failures, len(CAPTURE_RESTART_BACKOFF_S)) - 1]
                self._stop.wait(wait)

    def _pump(self, proc: subprocess.Popen):
        carry = b""
        while not self._stop.is_set():
            data = proc.stdout.read(_CHUNK_BYTES)
            if not data:
                return
            data = carry + data
            usable = len(data) - len(data) % 2
            carry = data[usable:]
            if usable:
                pcm = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
                self._queue.put(pcm)

    def _kill(self):
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ── Following the default output ───────────────────────────────────

    def _watch_loop(self):
        while not self._stop.wait(SINK_WATCH_S):
            sink = default_sink()
            if sink and sink != self._current_sink:
                log.event("CAPTURE", f"default output changed: {self._current_sink} → {sink}")
                self._current_sink = sink
                self._kill()      # the read loop starts pw-record again, on the new default
