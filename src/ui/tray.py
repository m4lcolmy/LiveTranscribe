"""The tray icon, drawn at run time: a dark tile, the letter ع, a subtitle bar.

Paused, the tile is grey, so the state is visible without opening the menu.
"""

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap


def make_icon(paused: bool = False) -> QIcon:
    icon = QIcon()
    for size in (16, 22, 24, 32, 48, 64):
        pix = QPixmap(size, size)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(110, 110, 110) if paused else QColor(24, 88, 160))
        p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)

        font = QFont()
        font.setFamilies(["Noto Sans Arabic", "Noto Naskh Arabic", "Sans"])
        font.setPixelSize(int(size * 0.62))
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(0, -size * 0.08, size, size * 0.8), Qt.AlignmentFlag.AlignCenter, "ع")

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 200))
        p.drawRoundedRect(QRectF(size * 0.2, size * 0.76, size * 0.6, size * 0.1),
                          size * 0.05, size * 0.05)
        p.end()
        icon.addPixmap(pix)
    return icon
