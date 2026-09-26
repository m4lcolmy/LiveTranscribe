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
import threading
import urllib.error
import urllib.parse
import urllib.request

from PyQt6.QtCore import QObject, QPoint, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QTextBrowser, QToolButton, QVBoxLayout,
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


# ── The icon: a blue tile with G over a grey one with 文 ───────────────

def translate_icon(size: int = 22) -> QIcon:
    """Drawn here, in the spirit of Google Translate's two-tile mark (not its artwork)."""
    icon = QIcon()
    for s in (size, size * 2):
        pix = QPixmap(s, s)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        back = QRectF(s * 0.34, s * 0.30, s * 0.64, s * 0.64)
        p.setBrush(QColor(236, 239, 241))
        p.drawRoundedRect(back, s * 0.10, s * 0.10)
        front = QRectF(s * 0.02, s * 0.04, s * 0.60, s * 0.60)
        p.setBrush(QColor(66, 133, 244))
        p.drawRoundedRect(front, s * 0.10, s * 0.10)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(int(s * 0.44))
        p.setFont(font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(front, Qt.AlignmentFlag.AlignCenter, "G")
        font.setPixelSize(int(s * 0.36))
        p.setFont(font)
        p.setPen(QColor(95, 99, 104))
        p.drawText(back.adjusted(s * 0.20, s * 0.18, 0, 0), Qt.AlignmentFlag.AlignCenter, "文")
        p.end()
        icon.addPixmap(pix)
    return icon


# ── The popup that shows a translation ─────────────────────────────────

class TranslatePopup(QFrame):
    """A small panel above the transcript: Translating… → the translation, or why not."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("translatePopup")
        # The rounded background is painted in paintEvent: a stylesheet
        # background is not drawn for a translucent top-level window.
        self.setStyleSheet(
            "QLabel { color: rgba(255,255,255,170); font-size: 12px; }"
            "QTextBrowser { background: transparent; color: white; border: none; font-size: 16px;"
            " selection-background-color: rgba(70,130,230,190); }"
            "QToolButton { color: rgba(255,255,255,190); border: none; padding: 2px 6px;"
            " border-radius: 6px; }"
            "QToolButton:hover { background: rgba(255,255,255,40); color: white; }"
        )
        self.title = QLabel("Google Translate")
        self.copy_button = QToolButton()
        self.copy_button.setText("Copy")
        self.copy_button.clicked.connect(self._copy)
        self.close_button = QToolButton()
        self.close_button.setText("✕")
        self.close_button.clicked.connect(self.hide)
        top = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(translate_icon(16).pixmap(16, 16))
        top.addWidget(icon)
        top.addWidget(self.title, 1)
        top.addWidget(self.copy_button)
        top.addWidget(self.close_button)
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.setFrameShape(QFrame.Shape.NoFrame)
        self.text.setMinimumHeight(40)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.addLayout(top)
        layout.addWidget(self.text, 1)
        self.resize(460, 150)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(255, 255, 255, 40))
        p.setBrush(QColor(24, 26, 32, 245))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)
        p.end()

    def show_near(self, anchor: QPoint, width: int):
        """Just above `anchor` (global), kept on the screen."""
        self.resize(max(320, min(width, 560)), self.height())
        screen = QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(max(area.left(), anchor.x() - self.width() // 2), area.right() - self.width())
        y = anchor.y() - self.height() - 8
        if y < area.top():
            y = anchor.y() + 8
        self.move(x, y)
        self.show()

    def set_waiting(self, target_name: str):
        self.title.setText(f"Google Translate → {target_name}")
        self.text.setPlainText("Translating…")
        self.copy_button.setEnabled(False)

    def set_result(self, text: str, target: str):
        rtl = target in RIGHT_TO_LEFT
        self.text.setLayoutDirection(Qt.LayoutDirection.RightToLeft if rtl
                                     else Qt.LayoutDirection.LeftToRight)
        self.text.setPlainText(text)
        self.copy_button.setEnabled(bool(text))

    def set_error(self, message: str):
        self.text.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.text.setPlainText(message)
        self.copy_button.setEnabled(False)

    def _copy(self):
        QGuiApplication.clipboard().setText(self.text.toPlainText())
