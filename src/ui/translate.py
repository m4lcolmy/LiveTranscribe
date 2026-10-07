"""Translate a selection of the transcript: with Google Translate, or offline.

Google is the one thing in LiveTranscribe that leaves the machine, so it is
narrow on purpose: only the text the user selected, and only when they ask —
by clicking the translate button, or by having chosen "translate as soon as
text is selected" themselves. Offline translation (src/core/nllb.py) sends
nothing anywhere; transcription stays offline either way.

Google's free endpoints — translate_a/single with client=gtx, which the web
widget uses, and translate_a/t with client=dict-chrome-ex, which the Chrome
extension uses — are unofficial: no key, no account, no guarantee. Google
answers too many requests from one network with a "Sorry…" page instead of
JSON (seen 2026-09-26, after the corpus fetch had leaned on YouTube, and
often since). So:

  * an answer is kept (TRANSLATE_CACHE_SIZE), and the same text again is
    shown from it without asking;
  * a refused endpoint hands over to the other, which is often still open,
    and the one that answered is asked first from then on;
  * when both refuse, nothing is sent for TRANSLATE_COOLDOWN_S — asking
    while refused can keep the block in place — and the popup says how long
    is left, and that offline translation is in the settings.
"""

import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

from PyQt6.QtCore import QObject, QPoint, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QPainter, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QTextBrowser, QToolButton, QVBoxLayout,
)

from src.config import TRANSLATE_CACHE_SIZE, TRANSLATE_COOLDOWN_S, TRANSLATE_TIMEOUT_S
from src.ui import icons
from src.ui.theme import (
    LINE, RADIUS, RAISED, SELECTION, SMALL_PX, TEXT, TEXT_2, TEXT_3,
    icon_button_stylesheet, qcolor, rgba,
)

ENDPOINT = "https://translate.googleapis.com/translate_a/single"
FALLBACK = "https://clients5.google.com/translate_a/t"
MAX_CHARS = 4500            # Google refuses much more than ~5000 characters at once
TIMEOUT_S = TRANSLATE_TIMEOUT_S

MODES = [("off", "Off"),
         ("button", "Show a translate button on selected text"),
         ("auto", "Translate as soon as text is selected")]
ENGINES = [("google", "Google Translate", "Online: only the text you select is sent to Google"),
           ("offline", "Offline (NLLB-200)", "On this computer: nothing is sent anywhere")]
ENGINE_SOURCE = {"google": "Google Translate", "offline": "Offline · NLLB-200"}

LANGUAGES = [("en", "English"), ("tr", "Türkçe"), ("fr", "Français"), ("de", "Deutsch"),
             ("es", "Español"), ("it", "Italiano"), ("ru", "Русский"), ("fa", "فارسی"),
             ("ur", "اردو"), ("id", "Bahasa Indonesia"), ("ms", "Bahasa Melayu"),
             ("zh-CN", "中文"), ("ja", "日本語")]
RIGHT_TO_LEFT = {"fa", "ur", "ar", "he"}

RATE_LIMITED = ("Google is refusing requests from this network right now "
                "(too many requests). Try again later.")


class TranslationError(Exception):
    """Something to show the user, in words they can act on."""


class RateLimited(TranslationError):
    """Google said no: the "Sorry…" page, or HTTP 429."""


def default_target() -> str:
    """The system's language when there is a translation into it; else English."""
    from PyQt6.QtCore import QLocale
    code = QLocale.system().name().split("_")[0]
    return code if code != "ar" and code in {c for c, _ in LANGUAGES} else "en"


# ── Google ─────────────────────────────────────────────────────────────

def parse(body: str) -> str:
    """The translated text out of translate_a/single's nested-list JSON."""
    if body.lstrip().startswith("<"):
        raise RateLimited(RATE_LIMITED)           # the "Sorry…" page, not JSON
    try:
        data = json.loads(body)
        return "".join(part[0] for part in data[0] if part and part[0]).strip()
    except (ValueError, TypeError, IndexError) as e:
        raise TranslationError(f"Google Translate answered in an unexpected form ({e})")


def parse_fallback(body: str) -> str:
    """translate_a/t's answer: [[text, detected language]] with sl=auto, [text] without."""
    if body.lstrip().startswith("<"):
        raise RateLimited(RATE_LIMITED)
    try:
        first = json.loads(body)[0]
        return (first[0] if isinstance(first, list) else first).strip()
    except (ValueError, TypeError, IndexError, AttributeError) as e:
        raise TranslationError(f"Google Translate answered in an unexpected form ({e})")


def _post(url: str, params: dict, text: str, timeout: float) -> str:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        data=urllib.parse.urlencode({"q": text}).encode("utf-8"),
        headers={"User-Agent": "Mozilla/5.0",
                 "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):           # 503 comes with the "Sorry…" page too
            raise RateLimited(RATE_LIMITED)
        raise TranslationError(f"Google Translate answered HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError, OSError):
        raise TranslationError("Could not reach Google Translate — is this computer online?")


def _single(text: str, target: str, source: str, timeout: float) -> str:
    return parse(_post(ENDPOINT, {"client": "gtx", "sl": source, "tl": target, "dt": "t"},
                       text, timeout))


def _extension(text: str, target: str, source: str, timeout: float) -> str:
    return parse_fallback(_post(FALLBACK, {"client": "dict-chrome-ex", "sl": source,
                                           "tl": target}, text, timeout))


class _Google:
    """Which endpoint to ask first, and until when to ask none."""

    def __init__(self):
        self.lock = threading.Lock()
        self.endpoints = [_single, _extension]
        self.until = 0.0            # time.monotonic() before which nothing is sent
        self.refusals = 0           # in a row, both endpoints

    def wait_left(self) -> float:
        return max(0.0, self.until - time.monotonic())

    def reset(self):
        with self.lock:
            self.endpoints = [_single, _extension]
            self.until, self.refusals = 0.0, 0


_google = _Google()


def wait_text(seconds: float) -> str:
    seconds = math.ceil(seconds)
    if seconds >= 90:
        return f"{math.ceil(seconds / 60)} min"
    return f"{seconds} s"


def _refused_message(seconds: float) -> str:
    return (f"Google is refusing requests from this network for now (too many requests). "
            f"Trying again is possible in {wait_text(seconds)} — or switch Translation "
            f"to Offline in the settings.")


def google(text: str, target: str, source: str = "auto", timeout: float = TIMEOUT_S) -> str:
    left = _google.wait_left()
    if left > 0:
        raise RateLimited(_refused_message(left))
    with _google.lock:
        order = list(_google.endpoints)
    for endpoint in order:
        try:
            result = endpoint(text, target, source, timeout)
        except RateLimited:
            continue
        with _google.lock:
            if order[0] is not endpoint:                # this one is open: ask it first
                _google.endpoints = [endpoint] + [e for e in _google.endpoints if e is not endpoint]
            _google.refusals = 0
        return result
    with _google.lock:
        wait = TRANSLATE_COOLDOWN_S[min(_google.refusals, len(TRANSLATE_COOLDOWN_S) - 1)]
        _google.refusals += 1
        _google.until = time.monotonic() + wait
    raise RateLimited(_refused_message(wait))


# ── Either ─────────────────────────────────────────────────────────────

def translate(text: str, target: str, engine: str = "google", source: str = "auto",
              timeout: float = TIMEOUT_S) -> str:
    text = text.strip()[:MAX_CHARS]
    if not text:
        return ""
    if engine == "offline":
        from src.core import nllb
        try:
            return nllb.translate(text, target)
        except nllb.OfflineError as e:
            raise TranslationError(str(e))
    return google(text, target, source, timeout)


class Translator(QObject):
    """Runs translate() off the GUI thread; only the newest request's answer is delivered.

    Answers are kept per (engine, language, text): asked again, they come back
    at once, from here, without a thread or a request.
    """

    finished = pyqtSignal(int, str, str)       # token, translation, error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._latest = 0
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._lock = threading.Lock()

    def cached(self, text: str, target: str, engine: str) -> str | None:
        key = (engine, target, text.strip())
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _keep(self, text: str, target: str, engine: str, result: str):
        with self._lock:
            self._cache[(engine, target, text.strip())] = result
            while len(self._cache) > TRANSLATE_CACHE_SIZE:
                self._cache.popitem(last=False)

    def request(self, text: str, target: str, engine: str = "google") -> int:
        self._latest += 1
        token = self._latest
        hit = self.cached(text, target, engine)
        if hit is not None:
            self.finished.emit(token, hit, "")
            return token
        threading.Thread(target=self._run, args=(token, text, target, engine),
                         name="translate", daemon=True).start()
        return token

    def _deliver(self, token: int, result: str, error: str):
        if token == self._latest:
            self.finished.emit(token, result, error)

    def _run(self, token: int, text: str, target: str, engine: str):
        try:
            result, error = translate(text, target, engine=engine), ""
            if result:
                self._keep(text, target, engine, result)
        except TranslationError as e:
            result, error = "", str(e)
        self._deliver(token, result, error)


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

    def set_waiting(self, target_name: str, engine: str = "google", note: str = "Translating…"):
        self._title_text = f"Arabic to {target_name}"
        self._restore_title.stop()
        self.title.setText(self._title_text)
        self.source.setText(ENGINE_SOURCE.get(engine, ENGINE_SOURCE["google"]))
        self._set_text(note, TEXT_3)
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
