"""The transcript window and settings, offscreen; and the live loop's pause/stop control."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false;qt.qpa.*=false")

import numpy as np
import pytest
from PyQt6.QtCore import QRect, QSettings, Qt
from PyQt6.QtWidgets import QApplication

from src.audio.streamer import Line, Streamer, Update
from src.audio.worker import Pipeline
from src.ui.settings import AppSettings
from tests.test_streamer import FakeVad, ScriptedEngine


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(app, store):
    from src.ui.overlay import TranscriptWindow
    w = TranscriptWindow(store)
    w.apply_settings(AppSettings())
    w.show_window()
    yield w
    w.close()


def update(finished=(), committed="", tentative=""):
    return Update(tuple(Line(t, 0, 1) for t in finished), committed, tentative, (), (), None, 0.0)


def settle(app):
    for _ in range(3):
        app.processEvents()


def paragraphs(window):
    return window.view.toPlainText().split("\n")


# ── The transcript ─────────────────────────────────────────────────────

def test_finished_lines_stack_and_the_live_line_is_replaced_not_repeated(window):
    window.show_update(update(committed="قال", tentative="الرئيس"))
    window.show_update(update(committed="قال الرئيس", tentative="إن"))
    assert paragraphs(window) == ["قال الرئيس إن"]
    window.show_update(update(["قال الرئيس إن الحكومة"], "", "ستعلن"))
    window.show_update(update(["ستعلن القرار غدا"], "وأضاف", ""))
    assert paragraphs(window) == ["قال الرئيس إن الحكومة", "ستعلن القرار غدا", "وأضاف"]


def test_words_that_may_change_can_be_hidden_and_are_never_copied(window):
    window.show_update(update(["جملة أولى"], "قال", "الرئيس"))
    assert paragraphs(window)[-1] == "قال الرئيس"
    assert window.view.full_text() == "جملة أولى\nقال"
    window.apply_settings(AppSettings(show_tentative=False))
    assert paragraphs(window)[-1] == "قال"


def test_the_view_follows_new_text_only_while_at_the_bottom(app, window):
    for i in range(60):
        window.show_update(update([f"السطر رقم {i} من النشرة"]))
    settle(app)
    bar = window.view.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()

    bar.setValue(0)                                  # the user scrolls up to read
    for i in range(10):
        window.show_update(update([f"سطر جديد {i}"]))
    settle(app)
    assert bar.value() == 0                          # left where they were reading

    bar.setValue(bar.maximum())                      # back at the bottom: follows again
    window.show_update(update(["وسطر أخير"]))
    settle(app)
    assert bar.value() == bar.maximum()


def test_text_can_be_selected_and_copied(app, window):
    window.show_update(update(["الجملة الأولى", "الجملة الثانية"], "والثالثة", ""))
    window.view.selectAll()
    window.view.copy()
    assert QApplication.clipboard().text() == "الجملة الأولى\nالجملة الثانية\nوالثالثة"
    assert window.view.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse


def test_notes_sit_above_the_live_line_and_stay_out_of_copies(window):
    window.show_update(update(["قبل التغيير"], "كلمة", ""))
    window.add_note("— model: large-v3 —")
    assert paragraphs(window) == ["قبل التغيير", "— model: large-v3 —", "كلمة"]
    assert "large-v3" not in window.view.full_text()


def test_history_is_capped_without_breaking_the_live_line(window, monkeypatch):
    import src.ui.overlay as overlay
    monkeypatch.setattr(overlay, "OVERLAY_HISTORY_LINES", 20)
    for i in range(50):
        window.show_update(update([f"سطر {i}"], "الحالي", ""))
    assert window.view.document().blockCount() <= 21
    assert paragraphs(window)[-1] == "الحالي"
    assert paragraphs(window)[-2] == "سطر 49"


# ── The window ─────────────────────────────────────────────────────────

def test_the_window_comes_back_where_the_user_left_it(app, store):
    from src.ui.overlay import TranscriptWindow
    first = TranscriptWindow(store)
    first.setGeometry(QRect(100, 120, 500, 300))
    first._save_geometry()
    first.close()
    second = TranscriptWindow(store)
    assert second.geometry() == QRect(100, 120, 500, 300)
    second.close()


def test_text_size_changes_do_not_resize_the_window(window):
    before = window.geometry()
    window.apply_settings(AppSettings(font_px=40))
    assert window.geometry() == before
    assert window.view.font().pixelSize() == 40


def test_click_through_is_a_window_flag(window):
    window.apply_settings(AppSettings(click_through=True))
    assert window.windowFlags() & Qt.WindowType.WindowTransparentForInput
    window.apply_settings(AppSettings(click_through=False))
    assert not window.windowFlags() & Qt.WindowType.WindowTransparentForInput


# ── Settings ───────────────────────────────────────────────────────────

def test_settings_survive_a_restart(store):
    chosen = AppSettings(model="large-v3", device="cuda", precision="int8_float16", step_s=1.5,
                         sink="alsa_output.test", font_px=30, opacity=60,
                         show_tentative=False, arabic_only=False, click_through=True)
    chosen.save(store)
    assert AppSettings.load(QSettings(store.fileName(), QSettings.Format.IniFormat)) == chosen


def test_nothing_saved_gives_the_defaults_and_a_corrupt_value_falls_back_alone(store):
    assert AppSettings.load(store) == AppSettings()
    store.setValue("settings/font_px", "huge")
    store.setValue("settings/model", "large-v3")
    loaded = AppSettings.load(store)
    assert loaded.font_px == AppSettings().font_px and loaded.model == "large-v3"


def test_each_change_restarts_only_what_it_must():
    base = AppSettings()
    assert base.needs(AppSettings(model="large-v3")) == "engine"
    assert base.needs(AppSettings(precision="int8")) == "engine"
    assert base.needs(AppSettings(sink="alsa_output.x")) == "pipeline"
    assert base.needs(AppSettings(step_s=0.5)) == "pipeline"
    assert base.needs(AppSettings(font_px=40, opacity=50, show_tentative=False)) == "window"


def test_the_dialog_shows_the_current_settings_and_returns_them_unchanged(app):
    from src.ui.settings import SettingsDialog
    current = AppSettings(font_px=30, opacity=55, step_s=1.5, show_tentative=False)
    dialog = SettingsDialog(current)
    assert dialog.model.currentData() == current.model
    assert dialog.result_settings() == current
    dialog.close()


def test_command_line_flags_override_saved_settings_for_one_run():
    from types import SimpleNamespace
    from src.session import effective_args
    saved = AppSettings(model="large-v3", device="cuda", step_s=2.0, sink="alsa_output.x")
    none = SimpleNamespace(model=None, device=None, beam=None, step=None, sink=None)
    args = effective_args(none, saved)
    assert (args.model, args.device, args.step, args.sink) == ("large-v3", "cuda", 2.0, "alsa_output.x")
    flags = SimpleNamespace(model="small", device="cpu", beam=None, step=None, sink=None)
    assert effective_args(flags, saved).model == "small"


# ── The loop behind it: pause and stop ─────────────────────────────────

class FakeSource:
    label = "fake"

    def __init__(self):
        self.starts = self.stops = 0
        self.running = False

    def start(self):
        self.starts += 1
        self.running = True

    def stop(self):
        self.stops += 1
        self.running = False

    def read(self, timeout):
        time.sleep(min(timeout, 0.01))
        return np.zeros(160 if self.running else 0, dtype=np.float32)


class Sink:
    closed = False

    def update(self, _):
        pass

    def status(self, _):
        pass

    def close(self):
        self.closed = True


def wait_until(condition, app, timeout=3.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def test_pause_stops_capture_resume_restarts_it_and_stop_ends_the_thread(app):
    source, sink = FakeSource(), Sink()
    pipeline = Pipeline(source, Streamer(ScriptedEngine([]), FakeVad([])), [sink], step_s=0.05)
    thread = threading.Thread(target=pipeline.run)
    thread.start()
    try:
        assert wait_until(lambda: source.starts == 1, app)
        pipeline.set_paused(True)
        assert wait_until(lambda: pipeline.paused and not source.running, app)
        pipeline.set_paused(False)
        assert wait_until(lambda: source.starts == 2 and source.running, app)
    finally:
        pipeline.stop()
        thread.join(3)
    assert not thread.is_alive()
    assert sink.closed
    assert source.stops == 2
