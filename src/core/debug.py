"""Session log: every pass, every commit, every drop, for post-mortem.

On by default. Every run writes ``logs/live-YYYYmmdd-HHMMSS.log`` and links
the newest as ``logs/live-session.log``; only the newest LOG_KEEP are kept.
A session that went wrong can always be read afterwards, instead of being
lost because nobody thought to pass a flag. (hifz's rule.)

    [  12.431s] PASS     t=12.40 buf=3.20s lat=412ms
                  heard      "..."
                  commit     "..."
                  tentative  "..."
                  drop       low_logprob -1.31 "..."
    [  13.902s] LINE     [00:10.2-00:13.1] "..."

The summary written on close says whether the machine kept up: pass latency
percentiles against the step, and how often a pass overran it.
"""

import threading
import time
from collections import Counter
from pathlib import Path

from src.config import LOGS_DIR, LOG_KEEP


class SessionLog:
    """Append-only session log. Thread-safe — the capture thread writes too."""

    def __init__(self):
        self._fh = None
        self._path: Path | None = None
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self.counters: Counter = Counter()
        self.pass_latencies: list[float] = []

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def open(self, path: Path, header: str = ""):
        self._path = Path(path)
        self._fh = open(self._path, "w", encoding="utf-8", buffering=1)
        self._t0 = time.monotonic()
        # A settings change reloads the model and opens a new log in the same
        # process; its summary must count only its own session.
        self.counters = Counter()
        self.pass_latencies = []
        self._raw(f"# LiveTranscribe session log — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if header:
            for line in header.splitlines():
                self._raw(f"# {line}")
        self._raw(f"# {'-' * 70}")

    def close(self):
        if self._fh is not None:
            with self._lock:
                self._fh.close()
                self._fh = None

    def _raw(self, line: str):
        if self._fh is None:
            return
        with self._lock:
            self._fh.write(line + "\n")

    def event(self, kind: str, message: str):
        """One timestamped line."""
        if self._fh is None:
            return
        elapsed = time.monotonic() - self._t0
        self._raw(f"[{elapsed:8.3f}s] {kind:<8} {message}")

    def detail(self, label: str, message: str):
        """An indented continuation line under the previous event."""
        if self._fh is None:
            return
        self._raw(f"{'':>13} {label:<10} {message}")

    def count(self, name: str, n: int = 1):
        self.counters[name] += n

    def summary(self, step_s: float) -> str:
        """Whether the machine kept up, and what was dropped. Also written to the log."""
        lat = sorted(self.pass_latencies)
        lines = []
        if lat:
            p50 = lat[len(lat) // 2]
            p90 = lat[min(len(lat) - 1, int(len(lat) * 0.9))]
            late = sum(1 for x in lat if x > step_s)
            lines.append(
                f"passes {len(lat)}  latency p50={p50 * 1000:.0f}ms p90={p90 * 1000:.0f}ms "
                f"max={lat[-1] * 1000:.0f}ms  step={step_s * 1000:.0f}ms  "
                f"overran={late} ({100 * late / len(lat):.0f}%)"
            )
        else:
            lines.append("passes 0 — no speech reached the model")
        c = self.counters
        lines.append(
            f"lines {c['lines']}  words {c['words_committed']}  "
            f"dropped segments: phantom={c['drop_phantom']} low_logprob={c['drop_low_logprob']} "
            f"repetitive={c['drop_repetitive']} loops cut={c['drop_loop_cut']}  "
            f"short blips={c['blips']}"
        )
        if c["capture_restarts"]:
            lines.append(f"capture restarts {c['capture_restarts']}")
        for line in lines:
            self._raw(f"# {line}")
        return "\n".join(lines)


log = SessionLog()


def new_log_path(prefix: str = "live") -> Path:
    """A fresh log path in LOGS_DIR, the newest linked, old ones pruned."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.log"

    link = LOGS_DIR / f"{prefix}-session.log"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.name)
    except OSError:
        pass

    old = sorted(LOGS_DIR.glob(f"{prefix}-2*.log"))
    for stale in old[: max(0, len(old) - (LOG_KEEP - 1))]:
        try:
            stale.unlink()
        except OSError:
            pass
    return path
