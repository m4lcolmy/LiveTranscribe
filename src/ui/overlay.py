"""The transcript window: a fixed-size panel, above everything, scrollable and copyable.

GNOME on Wayland lets no app place its own window or keep it above others, so
the window runs through XWayland (app.py sets QT_QPA_PLATFORM=xcb): a
frameless, always-on-top tool window, raised again on a timer against windows
that raise themselves, and made visible on every workspace through the X11
hint mutter honours (src/ui/x11.py) — the recipe Local-Live-Captions falls back
to on GNOME (PLAN.md §6, option A).

It began as a two-line subtitle box that hugged its text and faded when the
speaker stopped. It is now a panel of fixed size (the user may resize it; it
never resizes itself):

  * a slim header — status, pause, settings, close — which is also the handle
    the window is dragged by;
  * the transcript: every finished line a right-to-left paragraph, the line
    being spoken last, its tentative words dimmed. It follows new text while
    scrolled to the bottom and stays where it is while the user reads further
    up. Text can be selected and copied (Ctrl+C, or the right-click menu, which
    also copies the whole transcript).

It is shown without taking focus, but it can be clicked into — copying needs
the keyboard. Click-through (from the tray) makes it ignore the mouse entirely.
"""

from PyQt6.QtCore import (
    QPoint, QPointF, QRect, QRectF, QSettings, QSize, Qt, QTimer, pyqtSignal,
)
from PyQt6.QtGui import (
    QAction, QActionGroup, QColor, QFont, QGuiApplication, QIcon, QPainter, QPainterPath, QPen,
    QPixmap, QPolygonF, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextOption,
)
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMenu, QSizeGrip, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from src.ui.translate import (
    LANGUAGES, MODES, TranslatePopup, Translator, translate_icon,
)
from src.config import (
    OVERLAY_BOTTOM_MARGIN, OVERLAY_FONTS, OVERLAY_HEIGHT_PX, OVERLAY_HISTORY_LINES,
    OVERLAY_MAX_WIDTH_PX, OVERLAY_RAISE_EVERY_MS, OVERLAY_STATUS, OVERLAY_TENTATIVE,
    OVERLAY_TEXT, OVERLAY_WIDTH_FRACTION,
)

RADIUS = 12


def _rgba(values) -> str:
    r, g, b, a = values
    return f"rgba({r},{g},{b},{a})"


# ── The transcript ─────────────────────────────────────────────────────

class TranscriptView(QTextEdit):
    """Read-only, selectable, right to left. The live line is always the last paragraph.

    Selected by mouse only, with no caret: a blinking text cursor in a
    transcript nobody can type into reads as a bug (it did, 2026-09-26).
    Ctrl+C still copies — Qt handles Copy before it asks about the caret.

    Selected text can go to Google Translate: a button beside the selection,
    or at once, as the user chose (translation_changed carries a menu choice
    back to the settings).
    """

    translation_changed = pyqtSignal(str, str)      # mode, target language

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setCursorWidth(0)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.viewport().setAutoFillBackground(False)
        self.setStyleSheet(
            "QTextEdit { background: transparent; color: white;"
            " selection-background-color: rgba(70,130,230,190); selection-color: white; }"
            "QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }"
            "QScrollBar::handle:vertical { background: rgba(255,255,255,70);"
            " border-radius: 3px; min-height: 24px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }"
            "QScrollBar::add-page, QScrollBar::sub-page { background: none; }"
        )
        self.setPlaceholderText("…")

        # In a right-to-left paragraph a plain AlignRight means "the starting
        # edge" — which Qt then puts on the left. Absolute means the right edge.
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignAbsolute
        option = QTextOption(right)
        option.setTextDirection(Qt.LayoutDirection.RightToLeft)
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        self.document().setDefaultTextOption(option)

        self._block = QTextBlockFormat()
        self._block.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._block.setAlignment(right)
        self._block.setBottomMargin(4)
        self._note_block = QTextBlockFormat()
        self._note_block.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self._note_block.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._note_block.setBottomMargin(4)

        self._final = QTextCharFormat()
        self._final.setForeground(QColor(*OVERLAY_TEXT))
        self._tentative = QTextCharFormat()
        self._tentative.setForeground(QColor(*OVERLAY_TENTATIVE))
        self._note = QTextCharFormat()
        self._note.setForeground(QColor(*OVERLAY_STATUS))
        self._note.setFontItalic(True)

        QTextCursor(self.document()).setBlockFormat(self._block)
        self._live_start = 0          # where the live paragraph begins
        self._lines: list[str] = []   # finished lines, for "copy whole transcript"
        self._committed = ""
        self._tentative_text = ""
        self.show_tentative = True
        # Owned by the view, so a pending scroll dies with it — a bare lambda
        # holding the scroll bar outlived a closed window and crashed the tests.
        self._follow_later = QTimer(self)
        self._follow_later.setSingleShot(True)
        self._follow_later.setInterval(0)
        self._follow_later.timeout.connect(self._scroll_to_bottom)

        self.translate_mode = "button"
        self.translate_to = "en"
        self.translator = Translator(self)
        self.translator.finished.connect(self._translated)
        self.popup: TranslatePopup | None = None
        self.translate_button = QToolButton(self.viewport())
        self.translate_button.setIcon(translate_icon(22))
        self.translate_button.setIconSize(QSize(22, 22))
        self.translate_button.setToolTip("Translate the selection with Google Translate")
        self.translate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_button.setStyleSheet(
            "QToolButton { background: rgba(255,255,255,235); border-radius: 7px; padding: 2px; }"
            "QToolButton:hover { background: white; }")
        self.translate_button.hide()
        self.translate_button.clicked.connect(self.translate_selection)
        self.selectionChanged.connect(self._selection_changed)

    # ── Content ────────────────────────────────────────────────────────

    def at_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def _scroll_to_bottom(self):
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _follow(self, was_at_bottom: bool):
        if was_at_bottom:
            self._scroll_to_bottom()
            # The layout can settle after this returns; follow it there too.
            self._follow_later.start()

    def _rewrite_live(self, before_live=None):
        """Replace the live paragraph; `before_live(cursor)` inserts above it first."""
        follow = self.at_bottom()
        cur = QTextCursor(self.document())
        cur.setPosition(self._live_start)
        cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cur.removeSelectedText()
        cur.setBlockFormat(self._block)
        if before_live is not None:
            before_live(cur)
        self._live_start = cur.position()
        if self._committed:
            cur.insertText(self._committed, self._final)
        if self.show_tentative and self._tentative_text:
            cur.insertText((" " if self._committed else "") + self._tentative_text, self._tentative)
        self._trim()
        self._follow(follow)

    def add_update(self, finished: list[str], committed: str, tentative: str):
        def insert_finished(cur):
            for text in finished:
                cur.insertText(text, self._final)
                cur.insertBlock(self._block, self._final)
        self._lines.extend(finished)
        del self._lines[:-OVERLAY_HISTORY_LINES]
        self._committed, self._tentative_text = committed, tentative
        self._rewrite_live(insert_finished if finished else None)

    def add_note(self, text: str):
        """A dim line of the app's own — 'model changed to large-v3' — above the live line."""
        def insert_note(cur):
            cur.setBlockFormat(self._note_block)
            cur.insertText(text, self._note)
            cur.insertBlock(self._block, self._final)
        self._rewrite_live(insert_note)

    def set_show_tentative(self, on: bool):
        self.show_tentative = on
        self._rewrite_live()

    def clear_all(self):
        self.clear()
        QTextCursor(self.document()).setBlockFormat(self._block)
        self._live_start, self._lines = 0, []
        self._committed = self._tentative_text = ""

    def full_text(self) -> str:
        """Everything final: finished lines and the committed part of the live one."""
        return "\n".join(self._lines + ([self._committed] if self._committed else []))

    def _trim(self):
        doc = self.document()
        excess = doc.blockCount() - OVERLAY_HISTORY_LINES
        if excess <= 0:
            return
        before = doc.characterCount()
        cur = QTextCursor(doc)
        cur.movePosition(QTextCursor.MoveOperation.Start)
        cur.movePosition(QTextCursor.MoveOperation.NextBlock, QTextCursor.MoveMode.KeepAnchor, excess)
        cur.removeSelectedText()
        self._live_start -= before - doc.characterCount()

    # ── Translation ────────────────────────────────────────────────────

    def selected_text(self) -> str:
        # Qt separates paragraphs in a selection with U+2029.
        return self.textCursor().selectedText().replace("\u2029", "\n").strip()

    def set_translation(self, mode: str, target: str):
        self.translate_mode, self.translate_to = mode, target
        self._selection_changed()

    def _selection_rect(self):
        cur = self.textCursor()
        end = QTextCursor(cur)
        end.setPosition(cur.selectionEnd())
        return self.cursorRect(end)

    def _selection_changed(self):
        if self.translate_mode != "button" or not self.selected_text():
            self.translate_button.hide()
            return
        rect, b = self._selection_rect(), self.translate_button
        b.adjustSize()
        x = min(max(0, rect.center().x() - b.width() // 2), self.viewport().width() - b.width())
        y = rect.top() - b.height() - 2
        if y < 0:
            y = min(rect.bottom() + 2, self.viewport().height() - b.height())
        b.move(x, y)
        b.show()
        b.raise_()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.translate_mode == "auto" and self.selected_text():
            self.translate_selection()

    def translate_selection(self):
        text = self.selected_text()
        if not text:
            return
        self.translate_button.hide()
        if self.popup is None:
            self.popup = TranslatePopup()
        self.popup.set_waiting(dict(LANGUAGES).get(self.translate_to, self.translate_to))
        # Above the transcript window, over the selection.
        window = self.window()
        x = self.viewport().mapTo(window, self._selection_rect().center()).x()
        self.popup.show_near(window.mapToGlobal(QPoint(x, 0)), window.width())
        self.translator.request(text, self.translate_to)

    def _translated(self, _token: int, text: str, error: str):
        if self.popup is None:
            return
        if error:
            self.popup.set_error(error)
        else:
            self.popup.set_result(text, self.translate_to)

    # ── Menu ───────────────────────────────────────────────────────────

    def build_context_menu(self) -> QMenu:
        menu = self.createStandardContextMenu()     # Copy, Select All
        menu.addSeparator()
        copy_all = QAction("Copy whole transcript", menu)
        copy_all.triggered.connect(lambda: QGuiApplication.clipboard().setText(self.full_text()))
        menu.addAction(copy_all)
        clear = QAction("Clear window", menu)
        clear.triggered.connect(self.clear_all)
        menu.addAction(clear)

        menu.addSeparator()
        if self.translate_mode != "off":
            now = QAction(translate_icon(16), "Translate selection", menu)
            now.setEnabled(bool(self.selected_text()))
            now.triggered.connect(self.translate_selection)
            menu.addAction(now)
        google = menu.addMenu(translate_icon(16), "Google Translate")
        modes = QActionGroup(google)
        for value, label in MODES:
            a = QAction(label, google, checkable=True)
            a.setChecked(value == self.translate_mode)
            a.triggered.connect(lambda _=False, v=value: self._choose(v, self.translate_to))
            modes.addAction(a)
            google.addAction(a)
        into = google.addMenu("Translate into")
        targets = QActionGroup(into)
        for code, name in LANGUAGES:
            a = QAction(name, into, checkable=True)
            a.setChecked(code == self.translate_to)
            a.triggered.connect(lambda _=False, c=code: self._choose(self.translate_mode, c))
            targets.addAction(a)
            into.addAction(a)

        extra = getattr(self.window(), "extra_actions", [])
        if extra:
            menu.addSeparator()
            for action in extra:
                menu.addAction(action)
        return menu

    def _choose(self, mode: str, target: str):
        self.set_translation(mode, target)
        self.translation_changed.emit(mode, target)

    def contextMenuEvent(self, event):
        self.build_context_menu().exec(event.globalPos())


# ── The header: status, buttons, and the handle to drag ────────────────

def _glyph(draw, size: int) -> QIcon:
    """A flat white glyph on a 24-unit grid: dimmed at rest, full white under the mouse.

    Drawn rather than typed: the emoji 🔒 🔓 ⏸ came out in colour, or small,
    depending on the font the system picked.
    """
    icon = QIcon()
    for s in (size, size * 2):
        full = QPixmap(s, s)
        full.fill(Qt.GlobalColor.transparent)
        p = QPainter(full)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(s / 24, s / 24)
        draw(p, QColor(255, 255, 255))
        p.end()
        dim = QPixmap(s, s)
        dim.fill(Qt.GlobalColor.transparent)
        p = QPainter(dim)
        p.setOpacity(190 / 255)
        p.drawPixmap(0, 0, full)
        p.end()
        icon.addPixmap(dim, QIcon.Mode.Normal)
        icon.addPixmap(full, QIcon.Mode.Active)
    return icon


def _lock(closed: bool):
    def draw(p: QPainter, color: QColor):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(5, 10.5, 14, 10.5), 2, 2)
        shackle = QPainterPath(QPointF(8, 11))
        shackle.lineTo(8, 7)
        shackle.arcTo(QRectF(8, 3, 8, 8), 180, -180)
        if closed:
            shackle.lineTo(16, 11)
        p.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(shackle)
    return draw


def _pause(p: QPainter, color: QColor):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawRoundedRect(QRectF(6, 4.5, 4.5, 15), 1.2, 1.2)
    p.drawRoundedRect(QRectF(13.5, 4.5, 4.5, 15), 1.2, 1.2)


def _play(p: QPainter, color: QColor):
    p.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    p.setBrush(color)
    p.drawPolygon(QPolygonF([QPointF(8, 5), QPointF(8, 19), QPointF(19, 12)]))


class Header(QWidget):
    def __init__(self, window: "TranscriptWindow"):
        super().__init__(window)
        self._window = window
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(2)
        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {_rgba(OVERLAY_STATUS)}; font-size: 12px;")
        layout.addWidget(self.status, 1)
        self._icons = {
            "locked": _glyph(_lock(True), 18), "unlocked": _glyph(_lock(False), 18),
            "pause": _glyph(_pause, 22), "play": _glyph(_play, 22),
        }
        self.lock_button = self._button("", "")
        self.lock_button.setIconSize(QSize(18, 18))
        self.lock_button.setCheckable(True)
        self.set_locked(False)
        self.pause_button = self._button("", "Pause / resume listening")
        self.pause_button.setIconSize(QSize(22, 22))
        self.set_paused(False)
        self.settings_button = self._button("⚙", "Settings")
        self.quit_button = self._button("✕", "Quit LiveTranscribe")
        for button in (self.lock_button, self.pause_button, self.settings_button, self.quit_button):
            layout.addWidget(button)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def set_locked(self, on: bool):
        self.lock_button.setChecked(on)
        self.lock_button.setIcon(self._icons["locked" if on else "unlocked"])
        self.lock_button.setToolTip(
            "Click-through is on: the transcript lets clicks pass to the window under it. "
            "Click to turn it off." if on else
            "Click-through: let clicks on the transcript pass to the window under it "
            "(this header stays clickable)")

    def set_paused(self, paused: bool):
        self.pause_button.setIcon(self._icons["play" if paused else "pause"])

    def _button(self, text: str, tip: str) -> QToolButton:
        b = QToolButton(self)
        b.setText(text)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setAutoRaise(True)
        b.setStyleSheet(
            "QToolButton { color: rgba(255,255,255,190); border: none; padding: 1px 6px;"
            " font-size: 14px; border-radius: 6px; }"
            "QToolButton:hover { background: rgba(255,255,255,40); color: white; }"
        )
        return b

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._window.start_move(event)

    def mouseMoveEvent(self, event):
        self._window.continue_move(event)

    def mouseReleaseEvent(self, _event):
        self._window.end_move()


# ── The window ─────────────────────────────────────────────────────────

class TranscriptWindow(QWidget):
    pause_clicked = pyqtSignal()
    click_through_toggled = pyqtSignal(bool)
    settings_clicked = pyqtSignal()
    quit_clicked = pyqtSignal()

    def __init__(self, store: QSettings, bypass_wm: bool = False):
        super().__init__(None)
        self._store = store
        self._bypass_wm = bypass_wm
        self._click_through = False
        # Where X11 is underneath, click-through can spare the header.
        self._shape_input = QGuiApplication.platformName() == "xcb"
        self._opacity = 75
        self._drag_offset = None
        self.extra_actions: list[QAction] = []

        self.setWindowTitle("LiveTranscribe")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._apply_flags()

        self.header = Header(self)
        self.view = TranscriptView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 6, 10, 12)
        layout.setSpacing(2)
        layout.addWidget(self.header)
        layout.addWidget(self.view, 1)
        self.grip = QSizeGrip(self)
        self.grip.setFixedSize(14, 14)
        self.grip.setStyleSheet("background: transparent;")

        self.header.lock_button.clicked.connect(
            lambda: self.click_through_toggled.emit(not self._click_through))
        self.header.pause_button.clicked.connect(self.pause_clicked)
        self.header.settings_button.clicked.connect(self.settings_clicked)
        self.header.quit_button.clicked.connect(self.quit_clicked)

        self._save_geometry_later = QTimer(self)
        self._save_geometry_later.setSingleShot(True)
        self._save_geometry_later.setInterval(500)
        self._save_geometry_later.timeout.connect(self._save_geometry)

        self._raiser = QTimer(self)
        self._raiser.setInterval(OVERLAY_RAISE_EVERY_MS)
        self._raiser.timeout.connect(self._raise_if_visible)
        self._raiser.start()

        self._restore_geometry()

    # ── What the controller drives ─────────────────────────────────────

    def show_update(self, update):
        self.view.add_update([line.text for line in update.finished],
                             update.committed, update.tentative)

    def set_status(self, text: str, paused: bool | None = None):
        self.header.status.setText(text)
        if paused is not None:
            self.header.set_paused(paused)

    def add_note(self, text: str):
        self.view.add_note(text)

    def apply_settings(self, prefs):
        font = QFont()
        font.setFamilies(OVERLAY_FONTS)
        font.setPixelSize(prefs.font_px)
        font.setWeight(QFont.Weight.DemiBold)
        self.view.setFont(font)
        self.view.document().setDefaultFont(font)
        self._opacity = prefs.opacity
        if self.view.show_tentative != prefs.show_tentative:
            self.view.set_show_tentative(prefs.show_tentative)
        self.view.set_translation(prefs.translate, prefs.translate_to)
        if self._click_through != prefs.click_through:
            self.set_click_through(prefs.click_through)
        self.update()

    def show_window(self):
        if not self.isVisible():
            self.show()
            if not self._bypass_wm:
                # The window manager acts on the request only once the window
                # is mapped, which happens after show() returns.
                QTimer.singleShot(300, self._make_sticky)
            QTimer.singleShot(300, self._apply_input_region)
        self.raise_()

    @property
    def click_through(self) -> bool:
        return self._click_through

    def set_click_through(self, on: bool):
        """The transcript lets clicks through; the header never does.

        Under X11 (XWayland — how the app runs on GNOME) the window's input
        region is set to the header alone. Elsewhere Qt's all-or-nothing flag
        is the only way, and the tray menu is the way back.
        """
        self._click_through = on
        self.header.set_locked(on)
        if self._shape_input:
            self._apply_input_region()
            return
        visible = self.isVisible()
        geometry = self.geometry()
        self._apply_flags()             # changing flags unmaps the window
        self.setGeometry(geometry)
        if visible:
            self.show_window()

    def _input_rects(self) -> list[tuple[int, int, int, int]] | None:
        """What takes the mouse: all of the window, or in click-through only the header."""
        if not self._click_through:
            return None
        ratio = self.devicePixelRatioF()
        h = self.header.geometry().adjusted(-4, -6, 4, 2)   # a little margin to grab it by
        return [(int(h.x() * ratio), int(h.y() * ratio),
                 int(h.width() * ratio), int(h.height() * ratio))]

    def _apply_input_region(self):
        if self._shape_input and self.isVisible():
            from src.ui.x11 import set_input_region
            set_input_region(int(self.winId()), self._input_rects())

    def reset_position(self):
        self._store.remove("window/geometry")
        self.setGeometry(self._default_geometry())

    # ── Window setup ───────────────────────────────────────────────────

    def _apply_flags(self):
        flags = (Qt.WindowType.Tool
                 | Qt.WindowType.FramelessWindowHint
                 | Qt.WindowType.WindowStaysOnTopHint)
        if self._bypass_wm:
            # Override-redirect: no window manager at all — above fullscreen
            # windows and on every workspace, moved only by the app itself.
            flags |= Qt.WindowType.X11BypassWindowManagerHint
        if self._click_through and not self._shape_input:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)

    def _default_geometry(self) -> QRect:
        screen = QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        width = int(min(avail.width() * OVERLAY_WIDTH_FRACTION, OVERLAY_MAX_WIDTH_PX))
        height = OVERLAY_HEIGHT_PX
        bottom = avail.bottom() - OVERLAY_BOTTOM_MARGIN * screen.geometry().height()
        return QRect(int(avail.center().x() - width / 2), int(bottom - height), width, height)

    def _restore_geometry(self):
        saved = self._store.value("window/geometry")
        if isinstance(saved, QRect) and saved.isValid() and QGuiApplication.screenAt(saved.center()):
            self.setGeometry(saved)
        else:
            self.setGeometry(self._default_geometry())

    def _save_geometry(self):
        self._store.setValue("window/geometry", self.geometry())

    def _make_sticky(self):
        if self.isVisible() and QGuiApplication.platformName() == "xcb":
            from src.ui.x11 import make_sticky
            make_sticky(int(self.winId()))

    def _raise_if_visible(self):
        if self.isVisible():
            self.raise_()

    # ── Moving, by the header ──────────────────────────────────────────

    def start_move(self, event):
        handle = self.windowHandle()
        if self._bypass_wm or handle is None or not handle.startSystemMove():
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def continue_move(self, event):
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def end_move(self):
        self._drag_offset = None

    def moveEvent(self, _event):
        if self.isVisible():
            self._save_geometry_later.start()

    def resizeEvent(self, _event):
        self.grip.move(self.width() - self.grip.width() - 2, self.height() - self.grip.height() - 2)
        self._apply_input_region()      # the header's width follows the window's
        if self.isVisible():
            self._save_geometry_later.start()

    # ── Painting ───────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(12, 14, 18, int(255 * self._opacity / 100)))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), RADIUS, RADIUS)
        p.end()
