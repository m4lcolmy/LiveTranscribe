"""Translate a selection of the transcript with Google Translate's free endpoint.

This is the one thing in LiveTranscribe that leaves the machine, so it is
narrow on purpose: only the text the user selected, and only when they ask —
by clicking the translate button, or by having chosen "translate as soon as
text is selected" themselves. Transcription stays offline either way.

The endpoint (translate.googleapis.com/translate_a/single, client=gtx) is the
unofficial one the web widget uses: no key, no account, no guarantee. Google
answers too many requests from one network with a "Sorry…" page instead of
JSON (seen 2026-09-26, after the corpus fetch had leaned on YouTube), and that
is reported as what it is rather than as a broken translation.
"""

import json
import math
import threading
import urllib.error
import urllib.parse
import urllib.request

from PyQt6.QtCore import QObject, QPoint, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QPainter, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QTextBrowser, QToolButton, QVBoxLayout,
)

from src.ui import icons
from src.ui.theme import (
    LINE, RADIUS, RAISED, SELECTION, SMALL_PX, TEXT, TEXT_2, TEXT_3,
    icon_button_stylesheet, qcolor, rgba,
)

ENDPOINT = "https://translate.googleapis.com/translate_a/single"
MAX_CHARS = 4500            # Google refuses much more than ~5000 characters at once
TIMEOUT_S = 8.0

MODES = [("off", "Off"),
         ("button", "Show a translate button on selected text"),
         ("auto", "Translate as soon as text is selected")]

LANGUAGES = [("en", "English"), ("tr", "Türkçe"), ("fr", "Français"), ("de", "Deutsch"),
             ("es", "Español"), ("it", "Italiano"), ("ru", "Русский"), ("fa", "فارسی"),
             ("ur", "اردو"), ("id", "Bahasa Indonesia"), ("ms", "Bahasa Melayu"),
             ("zh-CN", "中文"), ("ja", "日本語")]
RIGHT_TO_LEFT = {"fa", "ur", "ar", "he"}

RATE_LIMITED = ("Google is refusing requests from this network right now "
                "(too many requests). Try again later.")


class TranslationError(Exception):
    """Something to show the user, in words they can act on."""


def default_target() -> str:
    """The system's language when there is a translation into it; else English."""
    from PyQt6.QtCore import QLocale
    code = QLocale.system().name().split("_")[0]
    return code if code != "ar" and code in {c for c, _ in LANGUAGES} else "en"


def parse(body: str) -> str:
    """The translated text out of the endpoint's nested-list JSON."""
    if body.lstrip().startswith("<"):
        raise TranslationError(RATE_LIMITED)      # the "Sorry…" page, not JSON
    try:
        data = json.loads(body)
        return "".join(part[0] for part in data[0] if part and part[0]).strip()
    except (ValueError, TypeError, IndexError) as e:
        raise TranslationError(f"Google Translate answered in an unexpected form ({e})")


def translate(text: str, target: str, source: str = "auto", timeout: float = TIMEOUT_S) -> str:
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""
    query = urllib.parse.urlencode({"client": "gtx", "sl": source, "tl": target, "dt": "t"})
    request = urllib.request.Request(
        f"{ENDPOINT}?{query}",
        data=urllib.parse.urlencode({"q": text}).encode("utf-8"),
        headers={"User-Agent": "Mozilla/5.0",
                 "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise TranslationError(RATE_LIMITED)
        raise TranslationError(f"Google Translate answered HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, OSError):
        raise TranslationError("Could not reach Google Translate — is this computer online?")
    return parse(body)


class Translator(QObject):
    """Runs translate() off the GUI thread; only the newest request's answer is delivered."""

    finished = pyqtSignal(int, str, str)       # token, translation, error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._latest = 0

    def request(self, text: str, target: str) -> int:
        self._latest += 1
        token = self._latest
        threading.Thread(target=self._run, args=(token, text, target),
                         name="translate", daemon=True).start()
        return token

    def _run(self, token: int, text: str, target: str):
        try:
            result, error = translate(text, target), ""
        except TranslationError as e:
            result, error = "", str(e)
        if token == self._latest:
            self.finished.emit(token, result, error)


# ── The popup that shows a translation ─────────────────────────────────

class TranslatePopup(QFrame):
    """A small panel above the transcript: Translating… → the translation, or why not.

    As tall as its text, up to MAX_TEXT_PX, and kept on the side of the
    transcript it opened on as it grows or shrinks.
    """

    MAX_TEXT_PX = 280
    MARGINS = (14, 6, 6, 12)        # left, top, right, bottom — the buttons sit near the edge
    TEXT_RIGHT = 8                  # ...the text keeps as far from it as from the left

    def __init__(self):
        super().__init__(None, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("translatePopup")
        # The rounded background is painted in paintEvent: a stylesheet
        # background is not drawn for a translucent top-level window.
        self.setStyleSheet(
            f"QLabel {{ color: {rgba(TEXT_2)}; font-size: {SMALL_PX}px; }}"
            f"QTextBrowser {{ background: transparent; color: {rgba(TEXT)}; border: none;"
            f" font-size: 15px; selection-background-color: {rgba(SELECTION)};"
            f" selection-color: {rgba(TEXT)}; }}"
            + icon_button_stylesheet()
        )
        self.title = QLabel()
        self.source = QLabel("Google Translate")
        self.source.setStyleSheet(f"color: {rgba(TEXT_3)};")
        self.copy_button = self._button(icons.COPY, "Copy the translation")
        self.copy_button.clicked.connect(self._copy)
        self.close_button = self._button(icons.CLOSE, "Close")
        self.close_button.clicked.connect(self.hide)
        top = QHBoxLayout()
        top.setSpacing(2)
        top.addWidget(self.title, 1)
        top.addWidget(self.source)
        top.addSpacing(8)
        top.addWidget(self.copy_button)
        top.addWidget(self.close_button)
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.setFrameShape(QFrame.Shape.NoFrame)
        self.text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text.document().setDocumentMargin(0)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, self.TEXT_RIGHT, 0)
        body.addWidget(self.text)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*self.MARGINS)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addLayout(body)

        self._title_text = ""
        self._anchor: QPoint | None = None
        self._restore_title = QTimer(self)
        self._restore_title.setSingleShot(True)
        self._restore_title.setInterval(1200)
        self._restore_title.timeout.connect(lambda: self.title.setText(self._title_text))
        self.resize(460, 100)

    def _button(self, glyph, tip: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(icons.icon(glyph, 16))
        b.setIconSize(QSize(16, 16))
        b.setFixedSize(26, 26)
        b.setAutoRaise(True)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(qcolor(LINE))
        p.setBrush(qcolor(RAISED, 250))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS, RADIUS)
        p.end()

    def show_near(self, anchor: QPoint, width: int):
        """Just above `anchor` (global), kept on the screen."""
        self._anchor = anchor
        self.resize(max(320, min(width, 560)), self.height())
        self._fit()
        self.show()

    def _fit(self):
        """As tall as the text, then back in place above (or below) the anchor."""
        left, _, right, _ = self.MARGINS
        doc = self.text.document()
        doc.setTextWidth(self.width() - left - right - self.TEXT_RIGHT)
        self.text.setFixedHeight(max(22, min(self.MAX_TEXT_PX, math.ceil(doc.size().height()) + 2)))
        self.layout().activate()
        self.resize(self.width(), self.layout().sizeHint().height())
        if self._anchor is None:
            return
        screen = QGuiApplication.screenAt(self._anchor) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(max(area.left(), self._anchor.x() - self.width() // 2), area.right() - self.width())
        y = self._anchor.y() - self.height() - 8
        if y < area.top():
            y = self._anchor.y() + 8
        self.move(x, y)

    def _set_text(self, text: str, colour=TEXT, rtl: bool = False):
        self.text.setLayoutDirection(Qt.LayoutDirection.RightToLeft if rtl
                                     else Qt.LayoutDirection.LeftToRight)
        self.text.setPlainText(text)
        if colour != TEXT:
            cursor = QTextCursor(self.text.document())
            cursor.select(QTextCursor.SelectionType.Document)
            fmt = QTextCharFormat()
            fmt.setForeground(qcolor(colour))
            cursor.mergeCharFormat(fmt)
        self._fit()

    def set_waiting(self, target_name: str):
        self._title_text = f"Arabic to {target_name}"
        self._restore_title.stop()
        self.title.setText(self._title_text)
        self._set_text("Translating…", TEXT_3)
        self.copy_button.setEnabled(False)

    def set_result(self, text: str, target: str):
        self._set_text(text, rtl=target in RIGHT_TO_LEFT)
        self.copy_button.setEnabled(bool(text))

    def set_error(self, message: str):
        self._set_text(message, TEXT_2)
        self.copy_button.setEnabled(False)

    def _copy(self):
        QGuiApplication.clipboard().setText(self.text.toPlainText())
        self.title.setText("Copied")
        self._restore_title.start()
