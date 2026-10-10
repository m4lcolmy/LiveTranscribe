"""The translation history: what is kept, the exports word-learning apps import, and its window."""

import csv
import io
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false;qt.qpa.*=false")

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from src.core import history as h
from src.ui.settings import AppSettings


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def kept(history_of_its_own):
    return history_of_its_own


def wait_until(condition, app, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    return condition()


# ── What is kept ───────────────────────────────────────────────────────

def test_a_translation_is_kept_on_disk_and_read_back(kept):
    kept.add("مرحبا", "Hello", "en", "google")
    kept.add("كتاب", "kitap", "tr", "offline")
    again = h.History(kept.path).entries()
    assert [(e.text, e.translation, e.target, e.engine, e.source, e.starred) for e in again] == [
        ("مرحبا", "Hello", "en", "google", "ar", False),
        ("كتاب", "kitap", "tr", "offline", "ar", False)]
    assert again[0].time[:4].isdigit() and "T" in again[0].time


def test_the_same_translation_again_moves_to_the_top_with_its_star_and_is_not_doubled(kept):
    kept.add("مرحبا", "Hello", "en", "google")
    kept.add("كتاب", "Book", "en", "google")
    kept.set_starred(("مرحبا", "en", "google"), True)
    kept.add("مرحبا", "Hello!", "en", "google")
    entries = h.History(kept.path).entries()
    assert [e.text for e in entries] == ["كتاب", "مرحبا"]
    assert entries[-1].starred and entries[-1].translation == "Hello!"
    # Another language or engine is another card.
    kept.add("مرحبا", "Merhaba", "tr", "google")
    kept.add("مرحبا", "Hello", "en", "offline")
    assert len(kept.entries()) == 4


def test_a_selection_across_lines_is_kept_as_one_line_and_empty_answers_not_at_all(kept):
    entry = kept.add("السلام\nعليكم  ", "Peace\nbe upon you", "en", "google")
    assert (entry.text, entry.translation) == ("السلام عليكم", "Peace be upon you")
    assert kept.find("السلام \n عليكم", "en", "google") is not None
    assert kept.add("   ", "x", "en", "google") is None
    assert kept.add("نص", "", "en", "google") is None
    assert len(kept.entries()) == 1


def test_a_broken_line_costs_only_that_line(kept):
    kept.add("واحد", "one", "en", "google")
    good = kept.entries()[0].to_json().replace("واحد", "اثنان")
    with open(kept.path, "a", encoding="utf-8") as f:
        f.write('{"text": "cut off in the middle\n')
        f.write("not json at all\n")
        f.write('{"text": "", "translation": "x"}\n')
        f.write(good + "\n")
    assert [e.text for e in h.History(kept.path).entries()] == ["واحد", "اثنان"]


def test_stars_deletions_and_clearing_are_saved(kept):
    for word, meaning in (("واحد", "one"), ("اثنان", "two"), ("ثلاثة", "three")):
        kept.add(word, meaning, "en", "google")
    kept.set_starred(("اثنان", "en", "google"), True)
    kept.remove([("واحد", "en", "google")])
    again = h.History(kept.path)
    assert [(e.text, e.starred) for e in again.entries()] == [("اثنان", True), ("ثلاثة", False)]
    assert not kept.path.with_name(kept.path.name + ".tmp").exists()
    again.clear()
    assert h.History(kept.path).entries() == [] and not kept.path.exists()


def test_listeners_hear_every_change(kept):
    heard = []
    kept.listeners.append(lambda: heard.append(1))
    kept.add("واحد", "one", "en", "google")
    kept.set_starred(("واحد", "en", "google"), True)
    kept.set_starred(("واحد", "en", "google"), True)        # no change, nothing to hear
    kept.remove([("واحد", "en", "google")])
    assert len(heard) == 3


# ── Exports ────────────────────────────────────────────────────────────

def entries():
    return [
        h.Entry("2026-10-10T14:03:22+03:00", "كتاب, قلم", 'book, "pen"', "en", "google",
                starred=True),
        h.Entry("2026-10-10T14:05:00+03:00", "مرحبا", "Merhaba", "tr", "offline"),
        h.Entry("2026-10-10T14:06:00+03:00", "شكرا", "谢谢", "zh-CN", "google"),
    ]


def test_the_csv_has_google_translates_four_columns_and_no_header():
    text = h.google_csv(entries())
    assert text.splitlines()[0] == 'Arabic,English,"كتاب, قلم","book, ""pen"""'
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [["Arabic", "English", "كتاب, قلم", 'book, "pen"'],
                    ["Arabic", "Turkish", "مرحبا", "Merhaba"],
                    ["Arabic", "Chinese (Simplified)", "شكرا", "谢谢"]]


def test_the_flashcard_csv_is_front_and_back_only():
    rows = list(csv.reader(io.StringIO(h.pairs_csv(entries()))))
    assert rows == [["كتاب, قلم", 'book, "pen"'], ["مرحبا", "Merhaba"], ["شكرا", "谢谢"]]


def test_the_json_names_the_same_fields_and_keeps_the_rest():
    data = json.loads(h.as_json(entries()))
    assert data[0] == {
        "source_language": "Arabic", "target_language": "English",
        "source_text": "كتاب, قلم", "translated_text": 'book, "pen"',
        "source_code": "ar", "target_code": "en", "engine": "google",
        "time": "2026-10-10T14:03:22+03:00", "starred": True,
    }
    assert "كتاب" in h.as_json(entries())               # Arabic as it is, not \\u escapes


def test_every_language_offered_has_googles_english_name():
    from src.ui.translate import LANGUAGES
    assert all(code in h.LANGUAGE_NAMES for code, _ in LANGUAGES)


def test_an_export_is_utf8_without_a_byte_order_mark(tmp_path):
    for fmt, _, ext in h.FORMATS:
        path = tmp_path / f"out{ext}"
        h.export(entries(), path, fmt)
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert "مرحبا" in raw.decode("utf-8")
    assert (tmp_path / "out.csv").read_bytes().count(b"\r\n") == 3     # one row a line, no blanks


# ── The transcript window keeps what it translates ─────────────────────

@pytest.fixture
def window(app, tmp_path):
    from src.ui.overlay import TranscriptWindow
    w = TranscriptWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    w.apply_settings(AppSettings())
    w.show_window()
    yield w
    if w.view.popup is not None:
        w.view.popup.hide()
    w.close()


def translate_all(app, view, text="مرحبا"):
    from PyQt6.QtGui import QTextCursor
    from src.audio.streamer import Line, Update
    view.window().show_update(Update((Line(text, 0, 1),), "", "", (), (), None, 0.0))
    cur = QTextCursor(view.document())
    cur.setPosition(0)
    cur.setPosition(len(text), QTextCursor.MoveMode.KeepAnchor)
    view.setTextCursor(cur)
    view.translate_selection()


def test_a_translation_is_kept_and_can_be_starred_from_the_popup(app, window, kept, monkeypatch):
    import src.ui.translate as tr
    monkeypatch.setattr(tr, "translate", lambda text, target, **_: "Hello")
    view = window.view
    view.set_translation("button", "en", "google")
    translate_all(app, view)
    assert wait_until(lambda: view.popup.text.toPlainText() == "Hello", app)
    assert [(e.text, e.translation, e.target) for e in kept.entries()] == [("مرحبا", "Hello", "en")]
    star = view.popup.star_button
    assert not star.isHidden() and not star.isChecked()
    star.click()
    assert kept.entries()[0].starred
    # The same text again — from the translator's cache — is still one entry, still starred.
    view.popup.hide()
    view.translate_selection()
    assert wait_until(lambda: view.popup.star_button.isChecked(), app)
    assert len(kept.entries()) == 1


def test_a_failed_translation_is_not_kept(app, window, kept, monkeypatch):
    import src.ui.translate as tr

    def refuse(text, target, **_):
        raise tr.TranslationError(tr.RATE_LIMITED)
    monkeypatch.setattr(tr, "translate", refuse)
    translate_all(app, window.view)
    assert wait_until(lambda: "too many" in window.view.popup.text.toPlainText(), app)
    assert kept.entries() == [] and window.view.popup.star_button.isHidden()


def test_with_history_off_nothing_is_kept_and_there_is_no_star(app, window, kept, monkeypatch):
    import src.ui.translate as tr
    monkeypatch.setattr(tr, "translate", lambda text, target, **_: "Hello")
    window.apply_settings(AppSettings(translate_history=False))
    translate_all(app, window.view)
    assert wait_until(lambda: window.view.popup.text.toPlainText() == "Hello", app)
    assert kept.entries() == [] and window.view.popup.star_button.isHidden()


def test_the_right_click_menu_turns_history_off_and_opens_it(app, window):
    view = window.view
    kept_choice, opened = [], []
    view.history_kept.connect(kept_choice.append)
    view.history_requested.connect(lambda: opened.append(True))
    menu = view.build_context_menu()
    options = next(a.menu() for a in menu.actions() if a.menu() and a.text() == "Translation")
    keep = next(a for a in options.actions() if a.text() == "Keep translation history")
    assert keep.isChecked()
    keep.trigger()
    assert kept_choice == [False] and not view.keep_history
    next(a for a in menu.actions() if a.text() == "Translation history…").trigger()
    assert opened == [True]


def test_the_history_setting_is_saved_and_needs_no_restart(app, tmp_path):
    from src.ui.settings import SettingsDialog
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    AppSettings(translate_history=False).save(store)
    loaded = AppSettings.load(store)
    assert loaded.translate_history is False and AppSettings().translate_history is True
    assert AppSettings().needs(loaded) == "window"
    dialog = SettingsDialog(loaded)
    assert not dialog.translate_history.isChecked()
    dialog.translate_history.setChecked(True)
    assert dialog.result_settings().translate_history is True


# ── The history window ─────────────────────────────────────────────────

@pytest.fixture
def history_window(app, kept):
    from src.ui.history import HistoryWindow
    for word, meaning in (("واحد", "one"), ("اثنان", "two"), ("ثلاثة", "three")):
        kept.add(word, meaning, "en", "google")
    w = HistoryWindow(kept)
    w.show()
    yield w
    # The search field's clear button fades in and out; a window thrown away
    # mid-fade stalls Qt's animation timer offscreen, and the next test's fades with it.
    w.search.clear()
    wait_until(lambda: False, app, timeout=0.4)
    w.close()


def column(w, c):
    return [w.table.item(r, c).text() for r in range(w.table.rowCount())]


def test_the_window_lists_newest_first_and_searches_both_sides(app, history_window):
    from src.ui.history import TEXT, TRANSLATION
    w = history_window
    assert column(w, TEXT) == ["ثلاثة", "اثنان", "واحد"]
    w.search.setText("TWO")
    assert column(w, TRANSLATION) == ["two"] and "1 of 3" in w.note.text()
    w.search.setText("واحد")
    assert column(w, TRANSLATION) == ["one"]


def test_the_star_column_stars_and_starred_only_filters(app, history_window, kept):
    from src.ui.history import STAR, TEXT
    w = history_window
    w._clicked(1, STAR)
    assert [e.starred for e in kept.entries()] == [False, True, False]
    w.starred_only.setChecked(True)
    assert column(w, TEXT) == ["اثنان"]


def test_delete_removes_the_selected_rows(app, history_window, kept):
    w = history_window
    assert not w.delete_button.isEnabled()
    w.table.selectRow(0)
    assert w.delete_button.isEnabled()
    w.delete_selected()
    assert [e.text for e in kept.entries()] == ["واحد", "اثنان"]


def test_the_window_follows_translations_made_while_it_is_open(app, history_window, kept):
    from src.ui.history import TEXT
    kept.add("أربعة", "four", "en", "google")
    assert column(history_window, TEXT)[0] == "أربعة"
    history_window.close()
    assert history_window.refresh not in kept.listeners


def test_export_writes_what_is_shown_and_adds_the_extension(app, history_window, kept, tmp_path):
    w = history_window
    kept.set_starred(("واحد", "en", "google"), True)
    w.starred_only.setChecked(True)
    path = w.export_to(str(tmp_path / "words"), "google")
    assert path.name == "words.csv"
    assert list(csv.reader(open(path, encoding="utf-8"))) == [["Arabic", "English", "واحد", "one"]]
    w.starred_only.setChecked(False)
    path = w.export_to(str(tmp_path / "all.json"), "json")
    assert [d["translated_text"] for d in json.loads(path.read_text(encoding="utf-8"))] == [
        "three", "two", "one"]
    assert "Exported 3" in w.note.text()
