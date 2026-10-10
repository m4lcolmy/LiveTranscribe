"""The translation history window: the translations kept so far, starred, searched, exported.

Opened from the tray menu or the transcript's right-click menu. Newest first;
a click on the star column stars or unstars a row, Delete removes the
selected rows. Export writes the rows on show — the search and "Starred
only" apply — in one of history.FORMATS, for word-learning apps.

It follows the history while it is open: a translation made meanwhile shows
up at the top.
"""

from datetime import date, datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.config import OVERLAY_FONTS
from src.core import history
from src.core.history import FORMATS, Entry, History
from src.ui import icons
from src.ui.theme import STARRED, dialog_palette, dialog_stylesheet
from src.ui.translate import ENGINE_SOURCE, LANGUAGES

STAR, TEXT, TRANSLATION, INTO, WHEN = range(5)
KEY = Qt.ItemDataRole.UserRole


def when_text(stamp: str) -> str:
    """The time alone for today, the date and time before."""
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    if moment.date() == date.today():
        return moment.strftime("%H:%M")
    return moment.strftime("%Y-%m-%d %H:%M")


class HistoryWindow(QDialog):
    def __init__(self, store: History | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store or history.shared()
        self.setWindowTitle("Translation history")
        # Like the settings: over the transcript window, which stays on top.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog_palette(self)
        self.setStyleSheet(dialog_stylesheet())
        self.resize(780, 500)
        self._format = FORMATS[0][0]

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search the Arabic or the translation")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.refresh)
        self.starred_only = QCheckBox("Starred only")
        self.starred_only.toggled.connect(self.refresh)
        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self.search, 1)
        top.addWidget(self.starred_only)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Arabic", "Translation", "Into", "When"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        header = self.table.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionResizeMode(STAR, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(STAR, 34)
        for column in (TEXT, TRANSLATION):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        for column in (INTO, WHEN):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellClicked.connect(self._clicked)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        QShortcut(QKeySequence.StandardKey.Delete, self.table, activated=self.delete_selected)

        self._arabic = QFont()
        self._arabic.setFamilies(OVERLAY_FONTS)
        self._arabic.setPixelSize(15)
        self._star_on = icons.icon(icons.STAR_ON, 16, rest=STARRED, hover=STARRED)
        self._star_off = icons.icon(icons.STAR, 16)

        self.note = QLabel("")
        self.note.setProperty("role", "note")
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Delete the selected translations")
        self.delete_button.clicked.connect(self.delete_selected)
        self.clear_button = QPushButton("Clear all…")
        self.clear_button.clicked.connect(self.clear_all)
        self.export_button = QPushButton("Export…")
        self.export_button.setProperty("primary", True)
        self.export_button.setToolTip("Save the translations shown, for Anki, Quizlet and other "
                                      "word-learning apps")
        self.export_button.clicked.connect(self.export)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self.note, 1)
        footer.addWidget(self.delete_button)
        footer.addWidget(self.clear_button)
        footer.addSpacing(8)
        footer.addWidget(close)
        footer.addWidget(self.export_button)

        body = QVBoxLayout(self)
        body.setContentsMargins(20, 16, 20, 16)
        body.setSpacing(12)
        body.addLayout(top)
        body.addWidget(self.table, 1)
        body.addLayout(footer)

        self.store.listeners.append(self.refresh)
        self.finished.connect(self._stop_following)
        self.refresh()

    # ── What is shown ──────────────────────────────────────────────────

    def shown(self) -> list[Entry]:
        """The entries on show, newest first: the search and "Starred only" applied."""
        words = self.search.text().strip().casefold().split()
        out = []
        for entry in reversed(self.store.entries()):
            if self.starred_only.isChecked() and not entry.starred:
                continue
            haystack = f"{entry.text} {entry.translation}".casefold()
            if all(w in haystack for w in words):
                out.append(entry)
        return out

    def refresh(self, *_):
        entries = self.shown()
        selected = {self.table.item(i.row(), STAR).data(KEY)
                    for i in self.table.selectionModel().selectedRows()}
        self.table.setUpdatesEnabled(False)
        self.table.clearSelection()
        self.table.setRowCount(len(entries))
        languages = dict(LANGUAGES)
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignAbsolute | Qt.AlignmentFlag.AlignVCenter
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        for row, entry in enumerate(entries):
            star = QTableWidgetItem(self._star_on if entry.starred else self._star_off, "")
            star.setData(KEY, entry.key)
            star.setToolTip("Unstar" if entry.starred else "Star")
            text = QTableWidgetItem(entry.text)
            text.setFont(self._arabic)
            text.setTextAlignment(right)
            text.setToolTip(entry.text)
            translation = QTableWidgetItem(entry.translation)
            translation.setTextAlignment(left)
            translation.setToolTip(entry.translation)
            into = QTableWidgetItem(languages.get(entry.target, entry.target))
            into.setToolTip(ENGINE_SOURCE.get(entry.engine, entry.engine))
            when = QTableWidgetItem(when_text(entry.time))
            when.setToolTip(entry.time)
            for column, item in enumerate((star, text, translation, into, when)):
                self.table.setItem(row, column, item)
            if entry.key in selected:
                self.table.selectRow(row)
        self.table.setUpdatesEnabled(True)
        total = len(self.store.entries())
        if not total:
            self.note.setText("Translations you make are kept here")
        elif len(entries) == total:
            self.note.setText(f"{total} translation{'s' if total != 1 else ''}")
        else:
            self.note.setText(f"{len(entries)} of {total} shown")
        self.export_button.setEnabled(bool(entries))
        self.clear_button.setEnabled(bool(total))
        self._selection_changed()

    def _selected_keys(self) -> list[tuple[str, str, str]]:
        return [self.table.item(i.row(), STAR).data(KEY)
                for i in self.table.selectionModel().selectedRows()]

    def _selection_changed(self):
        self.delete_button.setEnabled(bool(self._selected_keys()))

    def _stop_following(self, *_):
        if self.refresh in self.store.listeners:
            self.store.listeners.remove(self.refresh)

    def closeEvent(self, event):
        self._stop_following()
        super().closeEvent(event)

    # ── Changing ───────────────────────────────────────────────────────

    def _clicked(self, row: int, column: int):
        if column != STAR:
            return
        key = self.table.item(row, STAR).data(KEY)
        entry = next((e for e in self.store.entries() if e.key == key), None)
        if entry is not None:
            self._write(lambda: self.store.set_starred(key, not entry.starred))

    def delete_selected(self):
        keys = self._selected_keys()
        if keys:
            self._write(lambda: self.store.remove(keys))

    def clear_all(self):
        answer = QMessageBox.question(
            self, "Clear the translation history",
            "Delete every translation in the history, starred ones too? This cannot be undone.")
        if answer == QMessageBox.StandardButton.Yes:
            self._write(self.store.clear)

    def _write(self, change):
        try:
            change()
        except OSError as e:
            self.note.setText(f"Could not save the history: {e.strerror or e}")

    # ── Export ─────────────────────────────────────────────────────────

    def export(self):
        entries = self.shown()
        if not entries:
            return
        filters = {label: (fmt, ext) for fmt, label, ext in FORMATS}
        chosen = next(label for fmt, label, _ in FORMATS if fmt == self._format)
        ext = filters[chosen][1]
        suggested = Path.home() / f"livetranscribe-translations-{date.today().isoformat()}{ext}"
        path, label = QFileDialog.getSaveFileName(self, "Export translations", str(suggested),
                                                  ";;".join(filters), chosen)
        if not path:
            return
        self.export_to(path, filters.get(label, filters[chosen])[0])

    def export_to(self, path: str, fmt: str) -> Path | None:
        """Writes the rows on show; the file's extension is added when it has none of its own."""
        entries = self.shown()
        target = Path(path)
        ext = next(e for f, _, e in FORMATS if f == fmt)
        if target.suffix.lower() != ext:
            target = target.with_name(target.name + ext)
        try:
            history.export(entries, target, fmt)
        except OSError as e:
            self.note.setText(f"Could not export: {e.strerror or e}")
            return None
        self._format = fmt
        self.note.setText(f"Exported {len(entries)} to {target.name}")
        return target
