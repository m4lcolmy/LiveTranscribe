"""The overlay app: settings, the model, the live loop, the window and the menu.

Threads:
  * the GUI thread — the window, the tray, the menu, the settings dialog;
  * a loader thread whenever a model is loaded, so the window keeps working
    while Whisper loads;
  * the pipeline thread — capture, streamer, Whisper; it only talks to the GUI
    through Qt signals (queued across threads), and the GUI only talks to it
    through Pipeline.stop() and set_paused(), which just set flags.

Settings are loaded at start and saved when the dialog is saved (settings.py).
A saved change is applied at the smallest scope that works: a new model,
device or precision reloads the engine; a new source, interval or language
filter restarts the listening loop on the same engine; everything else is the
window's and changes on the spot. The transcript window keeps its text across
restarts, with a note where the setup changed.

The tray is the way back from click-through: a window that ignores the mouse
cannot be clicked. Without a tray (no AppIndicator support), click-through is
not offered at all.
"""

import gc
import sys
import threading

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import QApplication, QDialog, QMenu, QSystemTrayIcon

from src.core.debug import log
from src.session import effective_args
from src.ui.overlay import TranscriptWindow
from src.ui.settings import AppSettings, SettingsDialog, open_store
from src.ui.tray import make_icon


class OverlaySink(QObject):
    """A pipeline sink that hands updates and the level line to the GUI thread."""

    updated = pyqtSignal(object)
    level = pyqtSignal(str)

    def update(self, update):
        self.updated.emit(update)

    def status(self, text: str):
        self.level.emit(text)

    def close(self):
        pass


class Controller(QObject):
    engine_ready = pyqtSignal(object)
    engine_failed = pyqtSignal(str)
    pipeline_ended = pyqtSignal()

    def __init__(self, cli, app: QApplication):
        super().__init__()
        self.cli = cli
        self.app = app
        self.store = open_store()
        self.prefs = AppSettings.load(self.store)
        self.args = effective_args(cli, self.prefs)

        self.engine = None
        self.session = None
        self.pipeline = None
        self.thread = None
        self.paused = False
        self._restart_pending: bool | None = None    # None, or: reload the engine?
        self._quitting = False
        self._finished = False
        self._device_label = ""

        self.tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(make_icon(), self)
            self.tray.setToolTip("LiveTranscribe")
        if self.tray is None and QGuiApplication.platformName() != "xcb":
            self.prefs.click_through = False      # nothing would be left to click

        self.window = TranscriptWindow(self.store, bypass_wm=cli.bypass_wm)
        self.window.apply_settings(self.prefs)
        self.sink = OverlaySink()
        self.sink.updated.connect(self.window.show_update)
        self.sink.level.connect(self._show_level)
        self.window.view.translation_changed.connect(self._translation_chosen)
        self.window.click_through_toggled.connect(self._set_click_through)
        self.window.pause_clicked.connect(self.toggle_pause)
        self.window.settings_clicked.connect(self.open_settings)
        self.window.quit_clicked.connect(self.quit)

        self.menu = self._build_menu()
        if self.tray is not None:
            self.tray.setContextMenu(self.menu)
            self.tray.show()
        self.window.extra_actions = [self.pause_action, self.settings_action, self.quit_action]

        self.engine_ready.connect(self._start)
        self.engine_failed.connect(self._failed)
        self.pipeline_ended.connect(self._pipeline_ended)

        self.window.show_window()
        self._load_engine()

    # ── Menu ───────────────────────────────────────────────────────────

    def _build_menu(self) -> QMenu:
        menu = QMenu()
        self.pause_action = QAction("Pause", menu)
        self.pause_action.triggered.connect(self.toggle_pause)
        self.pause_action.setEnabled(False)
        menu.addAction(self.pause_action)
        self.settings_action = QAction("Settings…", menu)
        self.settings_action.triggered.connect(self.open_settings)
        menu.addAction(self.settings_action)
        self.click_action = QAction("Click-through", menu, checkable=True)
        self.click_action.setChecked(self.prefs.click_through)
        self.click_action.toggled.connect(self._set_click_through)
        self.click_action.setEnabled(self.tray is not None or QGuiApplication.platformName() == "xcb")
        menu.addAction(self.click_action)
        menu.addSeparator()
        menu.addAction("Copy whole transcript",
                       lambda: QGuiApplication.clipboard().setText(self.window.view.full_text()))
        self.transcript_action = menu.addAction("Open transcript file", self._open_transcript)
        self.transcript_action.setEnabled(False)
        menu.addAction("Reset window position", self.window.reset_position)
        menu.addSeparator()
        self.quit_action = QAction("Quit", menu)
        self.quit_action.triggered.connect(self.quit)
        menu.addAction(self.quit_action)
        return menu

    def _open_transcript(self):
        if self.session is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session.transcript)))

    def _set_click_through(self, on: bool):
        """From the header's lock button or the tray: saved, and all three kept in step."""
        self.click_action.blockSignals(True)
        self.click_action.setChecked(on)
        self.click_action.blockSignals(False)
        if self.prefs.click_through == on:
            return
        self.prefs.click_through = on
        self.prefs.save(self.store)
        self.window.set_click_through(on)

    def _translation_chosen(self, mode: str, target: str):
        """Chosen in the transcript's right-click menu: saved like any setting."""
        self.prefs.translate, self.prefs.translate_to = mode, target
        self.prefs.save(self.store)

    # ── Settings ───────────────────────────────────────────────────────

    def open_settings(self):
        # Show what is in effect: saved settings with this run's flags on top.
        shown = AppSettings(**{**self.prefs.__dict__,
                               "model": self.args.model, "device": self.args.device})
        # No parent: a child dialog would be placed over the transcript window,
        # at the bottom of the screen. It belongs in the middle.
        dialog = SettingsDialog(shown, None, click_through_allowed=(
            self.tray is not None or QGuiApplication.platformName() == "xcb"))
        dialog.adjustSize()
        screen = self.window.screen() or QGuiApplication.primaryScreen()
        dialog.move(screen.availableGeometry().center() - dialog.rect().center())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new = dialog.result_settings()
        needs = shown.needs(new)
        self.prefs = new
        new.save(self.store)
        # From now on the saved settings rule; the command line applied to the start only.
        for flag in ("model", "device", "step", "sink"):
            setattr(self.cli, flag, None)
        self.args = effective_args(self.cli, new)

        self.window.apply_settings(new)
        self.click_action.blockSignals(True)
        self.click_action.setChecked(new.click_through)
        self.click_action.blockSignals(False)
        if needs == "engine":
            self.window.add_note(f"— model: {new.model}, {new.device}, {new.precision} —")
            self._restart(reload_engine=True)
        elif needs == "pipeline":
            self.window.add_note("— listening restarted with new settings —")
            self._restart(reload_engine=False)

    def _restart(self, reload_engine: bool):
        if self._restart_pending is not None:
            self._restart_pending = self._restart_pending or reload_engine
            return
        self._restart_pending = reload_engine
        if self.pipeline is not None and self.thread is not None and self.thread.is_alive():
            self.window.set_status("Stopping…")
            self.pipeline.stop()        # _pipeline_ended carries on
        else:
            self._continue_restart()

    def _continue_restart(self):
        reload_engine, self._restart_pending = self._restart_pending, None
        self._close_session()
        self.pipeline, self.thread = None, None
        if self.paused:
            self._show_paused(False)
        if reload_engine or self.engine is None:
            # Free the old model before loading the next: two models do not
            # fit on a 4 GB GPU together.
            self.engine = None
            gc.collect()
            self._load_engine()
        else:
            self._start(self.engine)

    # ── Loading and running ────────────────────────────────────────────

    def _load_engine(self):
        a = self.args
        self.window.set_status(f"Loading Whisper {a.model}…")
        if self.tray is not None:
            self.tray.setToolTip(f"LiveTranscribe — loading {a.model}")
        threading.Thread(target=self._load, args=(a.model, a.device, a.beam, a.precision),
                         name="model-loader", daemon=True).start()

    def _load(self, model, device, beam, precision):
        from src.audio.engine import WhisperEngine
        try:
            engine = WhisperEngine(model, device, beam, precision).load()
        except Exception as e:  # a missing model, a broken CUDA install, no GPU memory
            self.engine_failed.emit(f"Could not load {model}: {e}")
            return
        self.engine_ready.emit(engine)

    def _failed(self, message: str):
        print(message, file=sys.stderr)
        self.window.set_status(message)
        self.window.add_note(f"— {message} —")
        if self.tray is not None:
            self.tray.setToolTip(f"LiveTranscribe — {message}")

    def _start(self, engine):
        if self._quitting:
            return
        from src.audio.capture import CaptureError
        from src.audio.streamer import Streamer
        from src.audio.worker import Pipeline
        from src.session import open_session
        from src.sinks import ConsoleSink, TranscriptSink

        self.engine = engine
        try:
            self.session = open_session(self.args, engine)
        except (CaptureError, OSError, RuntimeError) as e:
            self._failed(f"Cannot read audio: {e}")
            return

        sinks = [self.sink, TranscriptSink(self.session.transcript, self.session.source.label)]
        if sys.stdout.isatty():
            sinks.append(ConsoleSink())
            print("LiveTranscribe — transcript window")
            print("\n".join(self.session.header()))
            print("Settings: ⚙ in the window, or the tray icon.  Ctrl+C here also quits.\n")

        streamer = Streamer(engine, language_check=self.args.arabic_only)
        self.pipeline = Pipeline(self.session.source, streamer, sinks,
                                 self.session.step_s, self.session.recorder)
        self.thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self.thread.start()

        gpu = engine.device == "cuda"
        self._device_label = f"{self.args.model} · {'GPU' if gpu else 'CPU'}"
        self.window.set_status(f"● Listening — {self._device_label}"
                               + ("" if gpu else " (slower)"), paused=False)
        self.pause_action.setEnabled(True)
        self.transcript_action.setEnabled(True)
        if self.tray is not None:
            self.tray.setToolTip(f"LiveTranscribe — {self._device_label} — "
                                 f"{self.session.source.label}")

    def _run(self):
        try:
            self.pipeline.run()
        finally:
            self.pipeline_ended.emit()

    def _pipeline_ended(self):
        if self._quitting:
            self._finish()
        elif self._restart_pending is not None:
            self._continue_restart()
        elif self.pipeline is not None and self.pipeline.ended:
            self.window.set_status("End of file")
            self.pause_action.setEnabled(False)

    def _show_level(self, text: str):
        if self.paused or self._restart_pending is not None or self._quitting:
            return
        detail = text.split("— ", 1)[-1]           # "level -23 dB, gain 1.0×, speech…"
        self.window.set_status(f"● {self._device_label} — {detail}")

    def _close_session(self):
        if self.session is not None and log.enabled:
            log.summary(self.session.step_s)
            log.close()

    # ── Pause and quit ─────────────────────────────────────────────────

    def toggle_pause(self):
        if self.pipeline is None:
            return
        self._show_paused(not self.paused)
        self.pipeline.set_paused(self.paused)

    def _show_paused(self, paused: bool):
        self.paused = paused
        self.pause_action.setText("Resume" if paused else "Pause")
        if self.tray is not None:
            self.tray.setIcon(make_icon(paused=paused))
        self.window.set_status("⏸ Paused — not listening" if paused
                               else f"● Listening — {self._device_label}", paused=paused)

    def quit(self):
        if self._quitting:
            return
        self._quitting = True
        self.window.hide()
        if self.pipeline is not None and self.thread is not None and self.thread.is_alive():
            self.pipeline.stop()        # the thread closes the utterance, then signals
            QTimer.singleShot(15000, self._finish)   # never hang on a stuck pass
        else:
            self._finish()

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        if self.session is not None and log.enabled:
            summary = log.summary(self.session.step_s)
            if sys.stdout.isatty():
                print("\n" + summary)
        log.close()
        if self.tray is not None:
            self.tray.hide()
        self.app.quit()
