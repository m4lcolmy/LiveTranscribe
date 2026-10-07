"""Model downloads the user starts in the settings, and the row that shows one.

A download runs as its own process (`python -m src.core.models <name>`, see
src/core/models.py), not a thread: Hugging Face's downloader cannot be
interrupted, a process can — Cancel stops it at once. The files it finished
are kept; the half-written one is removed, and starts over next time. The app
stays offline (HF_HUB_OFFLINE=1); only that process may reach the Hub.

Downloads belong to the app, not to the dialog: closing the settings leaves
one running, and opening them again shows it where it got to. Progress is the
bytes of the model in the Hugging Face cache, read a few times a second —
which counts the files earlier, stopped attempts finished.
"""

import json
import sys
import time
from dataclasses import dataclass, field

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from src.config import PROJECT_ROOT
from src.core import models
from src.ui.theme import ACCENT, ERROR, LINE, LIVE, SMALL_PX, TEXT_2, rgba

POLL_MS = 400


def command(name: str) -> tuple[str, list[str]]:
    """The process that downloads a model: program, arguments."""
    return sys.executable, ["-m", "src.core.models", name]


@dataclass
class Job:
    model: models.Model
    process: QProcess
    total: int = 0                  # bytes; 0 until the Hub has said
    done: int = 0
    speed: float = 0.0              # bytes a second, smoothed
    error: str = ""
    cancelled: bool = False
    _last: tuple[float, int] = field(default=(0.0, 0))
    _out: bytes = b""

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning


class Downloads(QObject):
    """Every download of this run. One per model at a time."""

    changed = pyqtSignal(str)            # a model's name: progress, or its state
    finished = pyqtSignal(str, str)      # name, error ("" when the model is ready)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs: dict[str, Job] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

    def job(self, name: str) -> Job | None:
        return self.jobs.get(name)

    def running(self, name: str) -> bool:
        job = self.jobs.get(name)
        return job is not None and job.running

    def start(self, name: str):
        model = models.find(name)
        if model is None or self.running(name):
            return
        process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.remove("HF_HUB_OFFLINE")            # the app's rule; this process is the exception
        env.remove("TRANSFORMERS_OFFLINE")
        # Plain HTTP downloads grow a file in the cache as they go, which is
        # what the progress bar reads; Xet assembles the file elsewhere first.
        env.insert("HF_HUB_DISABLE_XET", "1")
        env.insert("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        job = Job(model, process, done=models.bytes_in_cache(model))
        job._last = (time.monotonic(), job.done)
        self.jobs[name] = job
        process.readyReadStandardOutput.connect(lambda: self._read(job))
        process.finished.connect(lambda code, _status: self._ended(job, code))
        process.errorOccurred.connect(lambda e: self._failed_to_start(job, e))
        process.start(*command(name))
        self._timer.start()
        self.changed.emit(name)

    def cancel(self, name: str):
        job = self.jobs.get(name)
        if job is not None and job.running:
            job.cancelled = True
            job.process.kill()

    def cancel_all(self):
        for name in list(self.jobs):
            self.cancel(name)
        for job in self.jobs.values():
            job.process.waitForFinished(2000)

    def _read(self, job: Job):
        job._out += bytes(job.process.readAllStandardOutput())
        *lines, job._out = job._out.split(b"\n")
        for line in lines:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if "total" in message:
                job.total = int(message["total"])
            if "error" in message:
                job.error = str(message["error"])
        self.changed.emit(job.model.name)

    def _poll(self):
        now = time.monotonic()
        busy = False
        for name, job in self.jobs.items():
            if not job.running:
                continue
            busy = True
            job.done = models.bytes_in_cache(job.model)
            if job.total:
                job.done = min(job.done, job.total)
            then, before = job._last
            if now - then >= 1.0:
                rate = max(0.0, (job.done - before) / (now - then))
                job.speed = rate if job.speed == 0 else 0.7 * job.speed + 0.3 * rate
                job._last = (now, job.done)
            self.changed.emit(name)
        if not busy:
            self._timer.stop()

    def _ended(self, job: Job, code: int):
        self._read(job)
        models.remove_partial(job.model)        # the process is gone; nothing else will finish them
        name = job.model.name
        if job.cancelled:
            del self.jobs[name]
            self.changed.emit(name)
            return
        if code == 0 and models.is_downloaded(job.model):
            del self.jobs[name]
            self.changed.emit(name)
            self.finished.emit(name, "")
            return
        job.error = job.error or "The download stopped unexpectedly"
        self.changed.emit(name)
        self.finished.emit(name, job.error)

    def _failed_to_start(self, job: Job, error):
        if error == QProcess.ProcessError.FailedToStart:
            job.error = "Could not start the download"
            self.changed.emit(job.model.name)
            self.finished.emit(job.model.name, job.error)


_downloads: Downloads | None = None


def downloads() -> Downloads:
    """The app's one set of downloads; stopped when the app quits."""
    global _downloads
    if _downloads is None or sip.isdeleted(_downloads):     # its application has gone (tests)
        app = QCoreApplication.instance()
        _downloads = Downloads(app)
        if app is not None:
            app.aboutToQuit.connect(_downloads.cancel_all)
    return _downloads


# ── The row in the settings ────────────────────────────────────────────

def _duration(seconds: float) -> str:
    if seconds < 60:
        return "< 1 min"
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes / 60:.1f} h"


MAX_LINE = 42       # characters: the row's text is kept to lines this short, never wrapped


class DownloadRow(QWidget):
    """Under a drop-down whose choice is not on this computer: what it costs, and the way to get it.

        Not downloaded yet · 1.5 GB                       [Download]

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        420 MB of 1.5 GB · 6.1 MB/s · 3 min left              [Cancel]
        Keeps going if you close Settings.

    Hidden when the choice is already there (or is not a model at all).
    `ready` says when the model it shows has arrived. Its text is never
    wrapped — a wrapped label in the dialog's grid was cut off rather than
    given a second line — so each line is kept short, and a long error shows
    in full as a tooltip.
    """

    ready = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.name: str | None = None
        self._just_done = False
        self.label = QLabel()
        self.label.setProperty("role", "hint")
        self.button = QPushButton()
        self.button.setStyleSheet("padding: 4px 12px; min-width: 0;")
        self.button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.button.clicked.connect(self._clicked)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(4)
        self.bar.setRange(0, 1000)
        self._bar_colour(ACCENT)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(self.label, 1)
        top.addWidget(self.button)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 2, 0, 2)
        column.setSpacing(6)
        column.addWidget(self.bar)
        column.addLayout(top)
        self.manager = downloads()
        self.manager.changed.connect(self._changed)
        self.hide()

    def _bar_colour(self, colour):
        self.bar.setStyleSheet(
            f"QProgressBar {{ background: {rgba(LINE)}; border: none; border-radius: 2px; }}"
            f"QProgressBar::chunk {{ background: {rgba(colour)}; border-radius: 2px; }}")

    def show_for(self, name: str | None):
        """The model this row is about: a name from src.core.models, or None for none."""
        if name != self.name:
            self._just_done = False
        self.name = name
        self._refresh()

    @property
    def missing(self) -> bool:
        """The model shown is not on this computer (and is not arriving right now)."""
        model = models.find(self.name) if self.name else None
        return model is not None and not models.is_downloaded(model)

    def _changed(self, name: str):
        if name != self.name:
            return
        model = models.find(name)
        if model is not None and self.manager.job(name) is None and models.is_downloaded(model):
            self._just_done = True
            self._refresh()
            self.ready.emit(name)
        else:
            self._refresh()

    def _refresh(self):
        model = models.find(self.name) if self.name else None
        if model is None:
            self.hide()
            return
        job = self.manager.job(model.name)
        if job is None and models.is_downloaded(model):
            if not self._just_done:
                self.hide()
                return
            self._show_state("✓ Downloaded — ready to use", None, None, LIVE)
            return
        if job is not None and job.running:
            if job.total:
                left = max(0, job.total - job.done)
                text = f"{models.size_text(job.done)} of {models.size_text(job.total)}"
                if job.speed > 0:
                    text += f" · {job.speed / 1e6:.1f} MB/s · {_duration(left / job.speed)} left"
                fraction = job.done / job.total
            else:
                text, fraction = "Starting the download…", None
            self._show_state(text + "\nKeeps going if you close Settings.",
                             "Cancel", fraction, ACCENT)
            return
        if job is not None and job.error:
            self._show_state(f"✕ {job.error}", "Try again", None, ERROR)
            return
        self._show_state(f"Not downloaded yet · {model.size}", "Download", None, ACCENT)

    def _show_state(self, text: str, button: str | None, fraction: float | None, colour):
        lines = [line if len(line) <= MAX_LINE else line[:MAX_LINE - 1].rstrip() + "…"
                 for line in text.split("\n")]
        self.label.setText("\n".join(lines))
        self.label.setToolTip(text if lines != text.split("\n") else "")
        self.label.setStyleSheet(f"color: {rgba(ERROR if colour == ERROR else TEXT_2)};"
                                 f" font-size: {SMALL_PX}px;")
        self.button.setVisible(button is not None)
        if button is not None:
            self.button.setText(button)
            self.button.setProperty("primary", button == "Download")
            self.button.style().unpolish(self.button)
            self.button.style().polish(self.button)
        self.bar.setVisible(fraction is not None)
        if fraction is not None:
            self.bar.setValue(int(fraction * 1000))
        self.show()

    def _clicked(self):
        if self.name is None:
            return
        if self.manager.running(self.name):
            self.manager.cancel(self.name)
        else:
            self.manager.start(self.name)
