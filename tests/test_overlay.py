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


def test_click_through_leaves_the_header_clickable_under_x11(window, monkeypatch):
    import src.ui.x11 as x11
    regions = []
    monkeypatch.setattr(x11, "set_input_region", lambda wid, rects: regions.append(rects) or True)
    window._shape_input = True                       # as under XWayland
    window.set_click_through(True)
    assert not window.windowFlags() & Qt.WindowType.WindowTransparentForInput
    (x, y, w, h), = regions[-1]
    header = window.header.geometry()
    assert x <= header.x() and w >= header.width() and y + h >= header.bottom()
    assert h < window.height() / 2                   # the transcript itself passes clicks
    assert window.header.lock_button.isChecked()
    window.set_click_through(False)
    assert regions[-1] is None                       # the whole window again
    assert not window.header.lock_button.isChecked()


def test_the_lock_button_asks_for_the_opposite_of_the_current_state(window):
    asked = []
    window.click_through_toggled.connect(asked.append)
    window.header.lock_button.click()
    assert asked == [True]


def test_the_controls_show_under_the_mouse_and_the_status_line_when_it_matters(app, window):
    settle(app)
    opacity = lambda w: w.graphicsEffect().opacity()      # noqa: E731
    window.reveal_controls(False, animate=False)
    window.set_status("small · GPU · -18 dB", "listening", speech=True)
    assert wait_until(lambda: opacity(window.header.status) == 0, app)
    assert opacity(window.header.controls) == 0
    assert window.header.dot.speech
    window.set_status("Loading Whisper small…", "loading")
    assert wait_until(lambda: opacity(window.header.status) == 1, app)   # something to know
    window.reveal_controls(True, animate=False)
    assert opacity(window.header.controls) == 1


def test_a_paused_state_shows_the_play_button(window):
    window.set_status("Paused — not listening", "paused")
    assert window.header.pause_button.icon().cacheKey() == window.header._icons["play"].cacheKey()
    window.set_status("small · GPU", "listening")
    assert window.header.pause_button.icon().cacheKey() == window.header._icons["pause"].cacheKey()


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


def test_the_dialog_previews_the_look_and_says_what_saving_will_restart(app):
    from src.ui.settings import SettingsDialog
    dialog = SettingsDialog(AppSettings())
    previewed = []
    dialog.appearance_changed.connect(previewed.append)
    dialog.font_px.setValue(36)
    assert previewed[-1].font_px == 36 and dialog.note.text() == ""
    dialog.step.setCurrentIndex(dialog.step.findData(2.0))
    assert dialog.note.text() == "Saving restarts listening"
    dialog.precision.setCurrentIndex(dialog.precision.findData("int8"))
    assert dialog.note.text() == "Saving reloads the model"
    dialog.close()


def test_preview_changes_only_the_look(window):
    window.preview(AppSettings(font_px=40, opacity=50, click_through=True, translate="off"))
    assert window.view.font().pixelSize() == 40
    assert not window.click_through and window.view.translate_mode == "button"


def test_icon_files_stay_clear_of_the_single_instance_socket(app):
    # A folder at the socket's path made every start think another was
    # running, and quit (2026-09-26).
    from src.ui import icons
    from src.ui.single import SingleInstance
    from src.ui.theme import TEXT
    folder = os.path.dirname(icons.file(icons.CHECK, 16, TEXT))
    assert os.path.basename(folder) != SingleInstance().name


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


# ── No caret; selection, copy, and translation ─────────────────────────

from PyQt6.QtGui import QTextCursor                     # noqa: E402
from PyQt6.QtTest import QTest                          # noqa: E402


def select(view, start=0, end=None):
    cur = QTextCursor(view.document())
    cur.setPosition(start)
    cur.setPosition(end if end is not None else view.document().characterCount() - 1,
                    QTextCursor.MoveMode.KeepAnchor)
    view.setTextCursor(cur)


def test_there_is_no_text_cursor_and_ctrl_c_still_copies(app, window):
    view = window.view
    assert view.cursorWidth() == 0
    assert not view.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByKeyboard
    window.show_update(update(["نص للنسخ"]))
    select(view, 0, 4)
    QApplication.clipboard().clear()
    QTest.keyClick(view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == "نص ل"


def test_the_translate_button_shows_only_on_a_selection_in_button_mode(app, window):
    view = window.view
    window.show_update(update(["جملة يمكن ترجمتها"]))
    view.set_translation("button", "en")
    select(view, 0, 4)
    assert view.translate_button.isVisible()
    view.setTextCursor(QTextCursor(view.document()))       # selection cleared
    assert not view.translate_button.isVisible()
    view.set_translation("off", "en")
    select(view, 0, 4)
    assert not view.translate_button.isVisible()


def test_automatic_mode_translates_on_release_and_shows_the_result(app, window, monkeypatch):
    import src.ui.translate as tr
    asked = []
    monkeypatch.setattr(tr, "translate", lambda text, target, **_: asked.append((text, target)) or "hello")
    view = window.view
    window.show_update(update(["مرحبا"]))
    view.set_translation("auto", "en")
    select(view, 0, 5)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton)
    assert asked == [("مرحبا", "en")]
    assert wait_until(lambda: view.popup.text.toPlainText() == "hello", app)
    view.popup.hide()


def test_a_refusal_from_google_is_said_plainly(app, window, monkeypatch):
    import src.ui.translate as tr

    def refuse(text, target, **_):
        raise tr.TranslationError(tr.RATE_LIMITED)
    monkeypatch.setattr(tr, "translate", refuse)
    view = window.view
    window.show_update(update(["مرحبا"]))
    view.set_translation("button", "tr")
    select(view, 0, 5)
    view.translate_selection()
    assert wait_until(lambda: "too many requests" in view.popup.text.toPlainText(), app)
    view.popup.hide()


def test_the_right_click_menu_offers_translation_and_remembers_the_choice(app, window):
    view = window.view
    chosen = []
    view.translation_changed.connect(lambda m, t: chosen.append((m, t)))
    view.set_translation("button", "en")
    menu = view.build_context_menu()
    google = next(a.menu() for a in menu.actions() if a.menu() and a.text() == "Google Translate")
    auto = next(a for a in google.actions() if a.text().startswith("Translate as soon"))
    auto.trigger()
    assert chosen[-1] == ("auto", "en") and view.translate_mode == "auto"
    into = next(a.menu() for a in google.actions() if a.menu())
    next(a for a in into.actions() if a.text() == "Türkçe").trigger()
    assert chosen[-1] == ("auto", "tr")


def test_translation_settings_are_saved_and_need_no_restart(store):
    s = AppSettings(translate="auto", translate_to="tr")
    s.save(store)
    assert AppSettings.load(store).translate == "auto"
    assert AppSettings().needs(s) == "window"


# ── The endpoint's answers ─────────────────────────────────────────────

def test_the_translation_is_read_out_of_googles_nested_lists():
    from src.ui.translate import parse
    body = '[[["Hello, ","مرحبا، ",null,null,10],["how are you?","كيف حالك؟",null,null,10]],null,"ar"]'
    assert parse(body) == "Hello, how are you?"


def test_the_sorry_page_and_http_429_mean_rate_limited(monkeypatch):
    import io
    import urllib.error
    import src.ui.translate as tr
    with pytest.raises(tr.TranslationError, match="too many requests"):
        tr.parse("<html><head><title>Sorry...</title>")

    def http_429(*_a, **_k):
        raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b""))
    monkeypatch.setattr(tr.urllib.request, "urlopen", http_429)
    with pytest.raises(tr.TranslationError, match="too many requests"):
        tr.translate("مرحبا", "en")


def test_a_second_start_shows_the_running_one_instead(app):
    from src.ui.single import SingleInstance
    first = SingleInstance("livetranscribe-test-instance")
    assert first.claim()
    shown = []
    first.activated.connect(lambda: shown.append(True))
    second = SingleInstance("livetranscribe-test-instance")
    assert not second.claim()
    assert wait_until(lambda: shown, app)
    first.server.close()
