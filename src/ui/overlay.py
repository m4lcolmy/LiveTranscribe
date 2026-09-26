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

  * a slim header — a status dot, then lock, pause, settings and close — which
    is also the handle the window is dragged by. Only the dot stays; the
    buttons and the status line fade in while the mouse is over the window
    (the status line also when something needs attention: loading, paused,
    an error, running on the CPU), so what shows over a video is the text;
  * the transcript: every finished line a right-to-left paragraph, the line
    being spoken last, its tentative words dimmed. It follows new text while
    scrolled to the bottom and stays where it is while the user reads further
    up. Text can be selected and copied (Ctrl+C, or the right-click menu, which
    also copies the whole transcript).

It is shown without taking focus, but it can be clicked into — copying needs
the keyboard. Click-through (from the tray) makes it ignore the mouse entirely.
"""

from PyQt6.QtCore import (
    QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSettings, QSize, Qt,
    QTimer, pyqtSignal,
)
from PyQt6.QtGui import (
    QAction, QActionGroup, QFont, QGuiApplication, QIcon, QPainter, QPen, QTextBlockFormat,
    QTextCharFormat, QTextCursor, QTextFormat, QTextOption,
)
from PyQt6.QtWidgets import (
    QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMenu, QSizeGrip, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)

from src.config import (
    OVERLAY_BOTTOM_MARGIN, OVERLAY_FONTS, OVERLAY_HEIGHT_PX, OVERLAY_HISTORY_LINES,
    OVERLAY_MAX_WIDTH_PX, OVERLAY_RAISE_EVERY_MS, OVERLAY_WIDTH_FRACTION,
)
from src.ui import icons
from src.ui.theme import (
    ACCENT, ACCENT_HOVER, BUSY, ERROR, FIELD_HOVER, HAIRLINE, IDLE, LINE, LIVE, RADIUS,
    RADIUS_CONTROL, RAISED, SELECTION, SMALL_PX, SURFACE, TENTATIVE, TEXT, TEXT_2, TEXT_3,
    icon_button_stylesheet, polish_menu, qcolor, rgba,
)
from src.ui.translate import LANGUAGES, MODES, TranslatePopup, Translator

FADE_MS = 160            # controls fading in and out
HIDE_AFTER_MS = 600      # the mouse gone this long before they fade
FIRST_SHOW_MS = 3000     # shown this long at the start, so they can be found


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
        # Always on, and invisible until the mouse is over the window: its
        # space is kept, so the text does not reflow when history first
        # overflows or when the bar is shown.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.viewport().setAutoFillBackground(False)
        self.setStyleSheet(
            f"QTextEdit {{ background: transparent; color: {rgba(TEXT)};"
            f" selection-background-color: {rgba(SELECTION)}; selection-color: {rgba(TEXT)}; }}"
            "QScrollBar:vertical { background: transparent; width: 8px; margin: 2px 1px 2px 3px; }"
            "QScrollBar::handle:vertical { background: transparent; border-radius: 2px;"
            " min-height: 24px; }"
            f'QScrollBar[revealed="true"]::handle:vertical {{ background: {rgba(TEXT_3)}; }}'
            "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }"
            "QScrollBar::add-page, QScrollBar::sub-page { background: none; }"
        )
        self.setPlaceholderText("…")
        self._hovered = False
        self.verticalScrollBar().rangeChanged.connect(self._range_changed)

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
        self._note_block.setTopMargin(4)
        self._note_block.setBottomMargin(8)

        self._final = QTextCharFormat()
        self._final.setForeground(qcolor(TEXT))
        self._tentative = QTextCharFormat()
        self._tentative.setForeground(qcolor(TENTATIVE))
        # A caption, not a subtitle: small, light and grey whatever the text size.
        self._note = QTextCharFormat()
        self._note.setForeground(qcolor(TEXT_3))
        self._note.setProperty(QTextFormat.Property.FontPixelSize, SMALL_PX)
        self._note.setFontWeight(QFont.Weight.Normal)

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
        self.translate_button.setIcon(icons.icon(icons.TRANSLATE, 16))
        self.translate_button.setIconSize(QSize(16, 16))
        self.translate_button.setFixedSize(28, 26)
        self.translate_button.setAutoRaise(True)
        self.translate_button.setToolTip("Translate the selection with Google Translate")
        self.translate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_button.setStyleSheet(
            f"QToolButton {{ background: {rgba(RAISED)}; border: 1px solid {rgba(LINE)};"
            f" border-radius: {RADIUS_CONTROL}px; }}"
            f"QToolButton:hover {{ background: {rgba(FIELD_HOVER)}; }}")
        self.translate_button.hide()
        self.translate_button.clicked.connect(self.translate_selection)
        self.selectionChanged.connect(self._selection_changed)

    # ── Content ────────────────────────────────────────────────────────

    def set_hovered(self, on: bool):
        """The scroll bar shows only under the mouse, and only when there is history."""
        self._hovered = on
        self._range_changed()

    def _range_changed(self, *_):
        bar = self.verticalScrollBar()
        revealed = self._hovered and bar.maximum() > 0
        if bar.property("revealed") != revealed:
            bar.setProperty("revealed", revealed)
            bar.style().unpolish(bar)
            bar.style().polish(bar)
            bar.update()

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
        # Text only: the desktop theme's colour icons, and the mnemonic
        # underlines ("C̲opy"), were the loudest things in a quiet menu.
        for action in menu.actions():
            action.setIcon(QIcon())
            action.setText(action.text().replace("&", ""))
        menu.addSeparator()
        copy_all = QAction("Copy whole transcript", menu)
        copy_all.triggered.connect(lambda: QGuiApplication.clipboard().setText(self.full_text()))
        menu.addAction(copy_all)
        clear = QAction("Clear window", menu)
        clear.triggered.connect(self.clear_all)
        menu.addAction(clear)

        menu.addSeparator()
        if self.translate_mode != "off":
            now = QAction("Translate selection", menu)
            now.setEnabled(bool(self.selected_text()))
            now.triggered.connect(self.translate_selection)
            menu.addAction(now)
        google = menu.addMenu("Google Translate")
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
        return polish_menu(menu)

    def _choose(self, mode: str, target: str):
        self.set_translation(mode, target)
        self.translation_changed.emit(mode, target)

    def contextMenuEvent(self, event):
        self.build_context_menu().exec(event.globalPos())


# ── The header: status, buttons, and the handle to drag ────────────────

class StatusDot(QWidget):
    """What the app is doing, at a glance: a small coloured dot, lit fully while speech is heard."""

    COLORS = {"loading": BUSY, "slow": BUSY, "listening": LIVE, "paused": IDLE, "ended": IDLE,
              "error": ERROR}

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.state, self.speech = "loading", False

    def set_state(self, state: str, speech: bool):
        if (state, speech) != (self.state, self.speech):
            self.state, self.speech = state, speech
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        centre = QPointF(self.width() / 2, self.height() / 2)
        colour = qcolor(self.COLORS.get(self.state, IDLE))
        if self.state == "listening" and not self.speech:
            colour.setAlpha(140)             # listening to silence: the same light, dimmer
        p.setBrush(colour)
        p.drawEllipse(centre, 3, 3)
        p.end()


class Header(QWidget):
    # States whose line stays on screen without the mouse: something to know.
    ATTENTION = {"loading", "slow", "paused", "ended", "error"}

    def __init__(self, window: "TranscriptWindow"):
        super().__init__(window)
        self._window = window
        self._state = "loading"
        self._revealed = False
        self.setFixedHeight(26)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.dot = StatusDot(self)
        layout.addWidget(self.dot)
        layout.addSpacing(6)
        self.status = QLabel("")
        self.status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.status.setStyleSheet(f"color: {rgba(TEXT_2)}; font-size: {SMALL_PX}px;")
        layout.addWidget(self.status, 1)

        self.controls = QWidget(self)
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self._icons = {
            "locked": icons.icon(icons.LOCK, 16, rest=ACCENT, hover=ACCENT_HOVER),
            "unlocked": icons.icon(icons.UNLOCK, 16),
            "pause": icons.icon(icons.PAUSE, 16), "play": icons.icon(icons.PLAY, 16),
        }
        self.lock_button = self._button("")
        self.lock_button.setCheckable(True)
        self.set_locked(False)
        self.pause_button = self._button("Pause / resume listening")
        self.set_paused(False)
        self.settings_button = self._button("Settings")
        self.settings_button.setIcon(icons.icon(icons.SETTINGS, 16))
        self.quit_button = self._button("Quit LiveTranscribe")
        self.quit_button.setIcon(icons.icon(icons.CLOSE, 16))
        for button in (self.lock_button, self.pause_button, self.settings_button, self.quit_button):
            row.addWidget(button)
        layout.addWidget(self.controls)

        # The buttons and the status line fade; the dot never does.
        self._fades = {}
        for widget in (self.controls, self.status):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(0.0)
            widget.setGraphicsEffect(effect)
            fade = QPropertyAnimation(effect, b"opacity", self)
            fade.setDuration(FADE_MS)
            fade.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._fades[widget] = fade
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def set_status(self, text: str, state: str, speech: bool = False, tip: str = ""):
        self._state = state
        self.status.setText(text)
        self.setToolTip(tip or text)
        self.dot.set_state(state, speech)
        if state == "paused":
            self.set_paused(True)
        elif state in ("listening", "slow"):
            self.set_paused(False)
        self._fade(self.status, self._revealed or state in self.ATTENTION)

    def reveal(self, on: bool, animate: bool = True):
        self._revealed = on
        self._fade(self.controls, on, animate)
        self._fade(self.status, on or self._state in self.ATTENTION, animate)

    def _fade(self, widget: QWidget, visible: bool, animate: bool = True):
        fade, target = self._fades[widget], 1.0 if visible else 0.0
        effect = widget.graphicsEffect()
        if fade.endValue() == target and fade.state() == QPropertyAnimation.State.Running:
            return
        fade.stop()
        if not animate or not self.isVisible():
            effect.setOpacity(target)
            return
        fade.setStartValue(effect.opacity())
        fade.setEndValue(target)
        fade.start()

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

    def _button(self, tip: str) -> QToolButton:
        b = QToolButton(self.controls)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setAutoRaise(True)                 # the brighter icon under the mouse
        b.setFixedSize(26, 26)
        b.setIconSize(QSize(16, 16))
        b.setStyleSheet(icon_button_stylesheet())
        return b

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._window.start_move(event)

    def mouseMoveEvent(self, event):
        self._window.continue_move(event)

    def mouseReleaseEvent(self, _event):
        self._window.end_move()


# ── The window ─────────────────────────────────────────────────────────

class _Grip(QSizeGrip):
    """The resize corner, unpainted: the window draws its mark, and only under the mouse."""

    def paintEvent(self, _event):
        pass



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
        self._hovered = False
        self._revealed = False
        self.extra_actions: list[QAction] = []

        self.setWindowTitle("LiveTranscribe")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._apply_flags()

        self.header = Header(self)
        self.view = TranscriptView(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 6, 10)
        layout.setSpacing(2)
        layout.addWidget(self.header)
        layout.addWidget(self.view, 1)
        self.grip = _Grip(self)
        self.grip.setFixedSize(14, 14)

        self.header.lock_button.clicked.connect(
            lambda: self.click_through_toggled.emit(not self._click_through))
        self.header.pause_button.clicked.connect(self.pause_clicked)
        self.header.settings_button.clicked.connect(self.settings_clicked)
        self.header.quit_button.clicked.connect(self.quit_clicked)

        self._save_geometry_later = QTimer(self)
        self._save_geometry_later.setSingleShot(True)
        self._save_geometry_later.setInterval(500)
        self._save_geometry_later.timeout.connect(self._save_geometry)

        # The controls fade out a moment after the mouse leaves, not at once:
        # crossing the edge on the way to a button must not flicker them.
        self._conceal_later = QTimer(self)
        self._conceal_later.setSingleShot(True)
        self._conceal_later.timeout.connect(lambda: self.reveal_controls(self._hovered))

        self._raiser = QTimer(self)
        self._raiser.setInterval(OVERLAY_RAISE_EVERY_MS)
        self._raiser.timeout.connect(self._raise_if_visible)
        self._raiser.start()

        self._restore_geometry()

    # ── What the controller drives ─────────────────────────────────────

    def show_update(self, update):
        self.view.add_update([line.text for line in update.finished],
                             update.committed, update.tentative)

    def set_status(self, text: str, state: str = "listening", speech: bool = False, tip: str = ""):
        """`state`: loading, listening, slow (on the CPU), paused, ended or error — the dot's colour.

        While listening the line shows under the mouse only; any other state
        keeps it on screen.
        """
        self.header.set_status(text, state, speech, tip)

    def add_note(self, text: str):
        self.view.add_note(text)

    def apply_settings(self, prefs):
        self.preview(prefs)
        self.view.set_translation(prefs.translate, prefs.translate_to)
        if self._click_through != prefs.click_through:
            self.set_click_through(prefs.click_through)

    def preview(self, prefs):
        """Only the look — text size, background, tentative words: the settings
        dialog shows them live, and puts the saved ones back on Cancel."""
        font = QFont()
        font.setFamilies(OVERLAY_FONTS)
        font.setPixelSize(prefs.font_px)
        font.setWeight(QFont.Weight.DemiBold)
        self.view.setFont(font)
        self.view.document().setDefaultFont(font)
        self._opacity = prefs.opacity
        if self.view.show_tentative != prefs.show_tentative:
            self.view.set_show_tentative(prefs.show_tentative)
        self.update()

    def show_window(self):
        if not self.isVisible():
            self.show()
            # The controls hide until the mouse comes; show them at the start
            # so there is a first chance to see that they exist.
            self.reveal_controls(True, animate=False)
            self._conceal_later.start(FIRST_SHOW_MS)
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

    # ── The mouse over it: controls in, controls out ───────────────────

    def enterEvent(self, _event):
        self._hovered = True
        self._conceal_later.stop()
        self.reveal_controls(True)

    def leaveEvent(self, _event):
        self._hovered = False
        self._conceal_later.start(HIDE_AFTER_MS)

    def reveal_controls(self, on: bool, animate: bool = True):
        self._revealed = on
        self.header.reveal(on, animate)
        self.view.set_hovered(on)
        self.update()               # the resize mark

    # ── Painting ───────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(qcolor(HAIRLINE), 1))
        p.setBrush(qcolor(SURFACE, int(255 * self._opacity / 100)))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
        if self._revealed:
            # Where to drag to resize: two short strokes in the corner.
            p.setPen(QPen(qcolor(TEXT_3), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            x, y = self.width() - 6.0, self.height() - 6.0
            p.drawLine(QPointF(x - 8, y), QPointF(x, y - 8))
            p.drawLine(QPointF(x - 4, y), QPointF(x, y - 4))
        p.end()
