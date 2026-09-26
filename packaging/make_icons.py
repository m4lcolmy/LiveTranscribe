"""Draw the app icon at every size a desktop asks for.

    python packaging/make_icons.py        # → packaging/icons/livetranscribe-<size>.png

A blue tile, the letter ع, and a dark caption bar with two lines of
"subtitles" under it. Drawn with Qt rather than shipped as artwork, so it can
be regenerated and changed in one place; the PNGs are committed.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QPointF, QRectF, Qt                   # noqa: E402
from PyQt6.QtGui import (                                      # noqa: E402
    QColor, QFont, QGuiApplication, QLinearGradient, QPainter, QPixmap,
)

HERE = Path(__file__).resolve().parent
SIZES = (16, 22, 24, 32, 48, 64, 128, 256, 512)


def draw(size: int, grey: bool = False) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    s = float(size)
    tile = QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92)
    gradient = QLinearGradient(QPointF(0, 0), QPointF(0, s))
    if grey:
        gradient.setColorAt(0, QColor(150, 150, 150))
        gradient.setColorAt(1, QColor(95, 95, 95))
    else:
        gradient.setColorAt(0, QColor(40, 132, 255))
        gradient.setColorAt(1, QColor(18, 72, 170))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(gradient)
    p.drawRoundedRect(tile, s * 0.22, s * 0.22)

    # The letter, in the upper part of the tile.
    font = QFont()
    font.setFamilies(["Noto Sans Arabic", "Noto Naskh Arabic", "Sans"])
    font.setBold(True)
    font.setPixelSize(max(8, int(s * 0.50)))
    p.setFont(font)
    p.setPen(QColor(255, 255, 255))
    p.drawText(QRectF(0, s * 0.02, s, s * 0.60), Qt.AlignmentFlag.AlignCenter, "ع")

    # The caption bar: two lines of subtitles, the second shorter and dimmer.
    bar = QRectF(s * 0.16, s * 0.64, s * 0.68, s * 0.22)
    p.setBrush(QColor(8, 20, 45, 190))
    p.drawRoundedRect(bar, s * 0.06, s * 0.06)
    line_h = max(1.0, s * 0.045)
    p.setBrush(QColor(255, 255, 255, 235))
    p.drawRoundedRect(QRectF(s * 0.24, s * 0.690, s * 0.52, line_h), line_h / 2, line_h / 2)
    p.setBrush(QColor(255, 255, 255, 140))
    p.drawRoundedRect(QRectF(s * 0.36, s * 0.775, s * 0.40, line_h), line_h / 2, line_h / 2)
    p.end()
    return pix


def main():
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    out = HERE / "icons"
    out.mkdir(exist_ok=True)
    for size in SIZES:
        draw(size).save(str(out / f"livetranscribe-{size}.png"))
        draw(size, grey=True).save(str(out / f"livetranscribe-paused-{size}.png"))
    print(f"{len(SIZES) * 2} icons → {out.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
