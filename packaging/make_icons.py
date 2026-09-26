"""Draw the app icon at every size a desktop asks for, and the tray's symbolic icon.

    python packaging/make_icons.py        # → packaging/icons/livetranscribe-<size>.png

The app icon, in GNOME's app-icon manner: a dark plate with a darker lower
edge for depth — the transcript panel itself — holding a blue waveform (the
speech) over two caption lines (the text), aligned right as Arabic is, the
second grey as the words that may still change are. Small sizes keep the
plate and the lines and drop detail.

The tray icon is symbolic, as GNOME's top bar expects: a white glyph, the
captions mark from the app's icon set (Lucide), dimmed while paused.

Drawn with Qt rather than shipped as artwork, so it can be regenerated and
changed in one place; the PNGs are committed.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QByteArray, QPointF, QRectF, Qt       # noqa: E402
from PyQt6.QtGui import (                                      # noqa: E402
    QColor, QGuiApplication, QLinearGradient, QPainter, QPainterPath, QPixmap,
)
from PyQt6.QtSvg import QSvgRenderer                           # noqa: E402

HERE = Path(__file__).resolve().parent
SIZES = (16, 22, 24, 32, 48, 64, 128, 256, 512)
TRAY_SIZES = (16, 22, 24, 32, 48, 64)

BLUE_LIGHT, BLUE = QColor(98, 160, 234), QColor(53, 132, 228)      # GNOME blue 2 and 3
GREY_LIGHT, GREY = QColor(154, 153, 150), QColor(119, 118, 123)    # paused


def _gradient(top: float, bottom: float, c0: QColor, c1: QColor) -> QLinearGradient:
    g = QLinearGradient(QPointF(0, top), QPointF(0, bottom))
    g.setColorAt(0, c0)
    g.setColorAt(1, c1)
    return g


def draw(size: int, grey: bool = False) -> QPixmap:
    """On a 128-unit canvas, scaled to `size`."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 128, size / 128)
    p.setPen(Qt.PenStyle.NoPen)
    small = size < 32

    # The plate: its lower edge first, then the face over it, lit from above.
    left, top, right, bottom, lip, radius = (4, 10, 124, 118, 0, 16) if small else \
                                            (8, 14, 120, 114, 6, 13)
    p.setBrush(QColor(22, 22, 24))
    p.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)
    face = QPainterPath()
    face.addRoundedRect(QRectF(left, top, right - left, bottom - top - lip), radius, radius)
    p.fillPath(face, _gradient(top, bottom - lip, QColor(62, 62, 66), QColor(42, 42, 46)))
    if not small:
        # A hairline of light along the top edge, inside the corners.
        p.save()
        p.setClipPath(face)
        p.fillRect(QRectF(left, top, right - left, 2), QColor(255, 255, 255, 34))
        p.restore()

    # The waveform: the speech being heard.
    light, dark = (GREY_LIGHT, GREY) if grey else (BLUE_LIGHT, BLUE)
    heights = [22, 44, 30] if small else [14, 26, 44, 34, 50, 28, 18]
    width, gap = (16, 12) if small else (7, 6)
    centre = 44 if small else 48
    x = 64 - (len(heights) * width + (len(heights) - 1) * gap) / 2
    for h in heights:
        p.setBrush(_gradient(centre - h / 2, centre + h / 2, light, dark))
        p.drawRoundedRect(QRectF(x, centre - h / 2, width, h), width / 2, width / 2)
        x += width + gap

    # The captions, right-aligned: the final line, and the tentative one.
    edge = right - (18 if small else 20)
    rows = [(76, 96, 255)] if small else [(82, 78, 245), (50, 92, 120)]
    thick = 14 if small else 8
    for length, y, alpha in rows:
        p.setBrush(QColor(255, 255, 255, alpha))
        p.drawRoundedRect(QRectF(edge - length, y, length, thick), thick / 2, thick / 2)
    p.end()
    return pix


# Lucide's "captions" (lucide.dev, ISC licence), the tray's mark.
CAPTIONS = ('<rect width="18" height="14" x="3" y="5" rx="2" ry="2"/>'
            '<path d="M7 15h4M15 15h2M7 11h2M13 11h4"/>')


def draw_tray(size: int, paused: bool = False) -> QPixmap:
    alpha = 0.5 if paused else 1.0
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"'
           f' stroke="#ffffff" stroke-opacity="{alpha}" stroke-width="2"'
           f' stroke-linecap="round" stroke-linejoin="round">{CAPTIONS}</svg>')
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(p, QRectF(0, 0, size, size))
    p.end()
    return pix


def main():
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    out = HERE / "icons"
    out.mkdir(exist_ok=True)
    for size in SIZES:
        draw(size).save(str(out / f"livetranscribe-{size}.png"))
        draw(size, grey=True).save(str(out / f"livetranscribe-paused-{size}.png"))
    for size in TRAY_SIZES:
        draw_tray(size).save(str(out / f"livetranscribe-tray-{size}.png"))
        draw_tray(size, paused=True).save(str(out / f"livetranscribe-tray-paused-{size}.png"))
    print(f"{len(SIZES) * 2 + len(TRAY_SIZES) * 2} icons → {out.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
