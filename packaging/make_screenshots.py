"""Render the README's pictures from the app's real widgets.

    python packaging/make_screenshots.py      # → docs/images/{banner,window,settings,settings-deepgram}.png

Nothing here is mocked up: the transcript panel, the translate button, the
Google Translate popup and the settings dialog are the classes the app runs,
drawn offscreen over a plain backdrop. The Arabic is the app's own output on
an Al Jazeera report from the test corpus; the English under it is what Google
Translate returned for that selection (2026-09-26).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false;qt.qpa.*=false")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, QSettings       # noqa: E402
from PyQt6.QtGui import (                                                 # noqa: E402
    QColor, QFont, QImage, QLinearGradient, QPainter, QRadialGradient, QTextCursor,
)
from PyQt6.QtWidgets import QApplication, QWidget                         # noqa: E402

OUT = ROOT / "docs" / "images"

LINES = [
    "ولد كير ستارمر في الثاني من ديسمبر عام 1962 لأب يعمل صانع أدوات في مصنع",
    "وأم ممرضة في هيئة الصحة الوطنية تصارع مرضا نادرا طوال حياتها",
    "نشأ وسط أسرة من الطبقة العاملة وسماه والداه تيمنا بأول زعيم برلماني لحزب العمال",
]
LIVE_COMMITTED = "قضى طفولته المبكرة في الستينيات"
LIVE_TENTATIVE = "بينما كانت بريطانيا تودع"
SELECTED = "ولد كير ستارمر في الثاني من ديسمبر عام 1962 لأب يعمل صانع أدوات في مصنع"
TRANSLATION = ("Keir Starmer was born on December 2, 1962, to a father who worked "
               "as a tool maker in a factory")


def backdrop(w: int, h: int) -> QImage:
    """A plain 'video' behind the panel: a dark gradient with two soft lights."""
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    p = QPainter(img)
    g = QLinearGradient(QPointF(0, 0), QPointF(w, h))
    g.setColorAt(0, QColor(18, 40, 70))
    g.setColorAt(1, QColor(52, 28, 58))
    p.fillRect(img.rect(), g)
    for cx, cy, r, colour in ((0.25, 0.35, 0.45, QColor(90, 150, 220, 90)),
                              (0.78, 0.25, 0.35, QColor(230, 150, 90, 70))):
        rg = QRadialGradient(QPointF(w * cx, h * cy), w * r)
        rg.setColorAt(0, colour)
        rg.setColorAt(1, QColor(0, 0, 0, 0))
        p.fillRect(img.rect(), rg)
    p.end()
    return img


def panel(width: int, height: int, select: bool = False, hovered: bool = False):
    from src.audio.streamer import Line, Update
    from src.ui.overlay import TranscriptWindow
    from src.ui.settings import AppSettings

    store = QSettings(str(OUT / ".render.ini"), QSettings.Format.IniFormat)
    store.clear()
    w = TranscriptWindow(store)
    w.apply_settings(AppSettings(font_px=22, opacity=82))
    w.setGeometry(QRect(0, 0, width, height))
    w.show_window()
    w.show_update(Update(tuple(Line(t, 0, 1) for t in LINES), LIVE_COMMITTED, LIVE_TENTATIVE,
                         (), (), None, 0.0))
    w.set_status("small · GPU · -18 dB", "listening", speech=True)
    for _ in range(5):
        QApplication.processEvents()
    # Under the mouse the controls show; otherwise only the status dot does.
    # Set once events are done: offscreen, the window gets the mouse's Enter.
    w.reveal_controls(hovered, animate=False)
    if select:
        # Scroll first: the translate button is placed beside the selection
        # as it is on screen at the moment it is selected.
        w.view.verticalScrollBar().setValue(0)
        QApplication.processEvents()
        doc = w.view.document()
        start = doc.toPlainText().index(SELECTED)
        cur = QTextCursor(doc)
        cur.setPosition(start)
        cur.setPosition(start + len(SELECTED), QTextCursor.MoveMode.KeepAnchor)
        w.view.setTextCursor(cur)
        w.view.verticalScrollBar().setValue(0)
        for _ in range(3):
            QApplication.processEvents()
    return w


def draw_widget(p: QPainter, widget: QWidget, at: QPoint):
    # grab(), not render(): render() draws children that have a graphics
    # effect (the header's fading controls and status line) out of place.
    p.drawPixmap(at, widget.grab())


def window_png():
    from src.ui.translate import TranslatePopup
    W, H = 1280, 720
    img = backdrop(W, H)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    w = panel(820, 230, select=True, hovered=True)
    at = QPoint((W - w.width()) // 2, H - w.height() - 40)
    draw_widget(p, w, at)
    pop = TranslatePopup()
    pop.show_near(QPoint(0, 0), 520)
    pop.set_waiting("English")
    pop.set_result(TRANSLATION, "en")
    QApplication.processEvents()
    draw_widget(p, pop, QPoint(at.x() + 250, at.y() - pop.height() - 12))
    p.end()
    img.save(str(OUT / "window.png"))


def banner_png():
    from src.ui.tray import ICONS
    W, H = 1400, 500
    img = backdrop(W, H)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    icon = QImage(str(ICONS / "livetranscribe-256.png"))
    p.drawImage(QRectF(70, 70, 140, 140), icon)
    title = QFont()
    title.setPixelSize(64)
    title.setBold(True)
    p.setFont(title)
    p.setPen(QColor(255, 255, 255))
    p.drawText(QPointF(235, 150), "LiveTranscribe")
    body = QFont()
    body.setPixelSize(26)
    p.setFont(body)
    p.setPen(QColor(255, 255, 255, 215))
    p.drawText(QPointF(238, 195), "Live Arabic subtitles for anything your computer plays")
    small = QFont()
    small.setPixelSize(20)
    p.setFont(small)
    p.setPen(QColor(255, 255, 255, 160))
    p.drawText(QPointF(72, 290), "Offline  ·  on your own GPU  ·  Whisper  ·  GNOME")
    w = panel(640, 200)
    draw_widget(p, w, QPoint(W - w.width() - 60, H - w.height() - 40))
    p.end()
    img.save(str(OUT / "banner.png"))


def settings_png(name: str = "settings.png", **chosen):
    from src.ui.settings import AppSettings, SettingsDialog
    os.environ.pop("DEEPGRAM_API_KEY", None)        # the hint as most people see it
    d = SettingsDialog(AppSettings(**chosen))
    d.adjustSize()
    d.show()
    QApplication.processEvents()
    img = QImage(d.size(), QImage.Format.Format_ARGB32)
    img.fill(QColor(27, 30, 36))
    p = QPainter(img)
    d.render(p)
    p.end()
    img.save(str(OUT / name))


def main():
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    OUT.mkdir(parents=True, exist_ok=True)
    banner_png()
    window_png()
    settings_png()
    # A made-up key: the field shows it masked, as it is shown.
    settings_png("settings-deepgram.png", model="deepgram", deepgram_key="0" * 40)
    (OUT / ".render.ini").unlink(missing_ok=True)
    print(f"→ {OUT.relative_to(ROOT)}/ banner.png window.png settings.png settings-deepgram.png")


if __name__ == "__main__":
    main()
