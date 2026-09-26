"""The app's icons: Lucide's line glyphs (lucide.dev, ISC licence), rendered with QtSvg.

One family, one stroke, on a 24-unit grid — a set drawn by people who draw
icons, rather than glyphs typed from fonts (the emoji 🔒 ⏸ ⚙ came out in
colour, or small, or not at all, depending on the font the system picked).

icon() makes a QIcon for a button: dimmed at rest, full under the mouse.
file() writes a PNG for a stylesheet's url(), which cannot take a QIcon.
"""

import os
import tempfile
from typing import NamedTuple

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from src.ui.theme import TEXT, TEXT_2, TEXT_3, qcolor

STROKE = 1.75           # Lucide's default is 2; a little finer at the 16 px they show at


class Glyph(NamedTuple):
    name: str
    svg: str            # the elements inside Lucide's <svg>, stroked in currentColor


LOCK = Glyph("lock", '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>'
                     '<path d="M7 11V7a5 5 0 0 1 10 0v4"/>')
UNLOCK = Glyph("lock-open", '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>'
                            '<path d="M7 11V7a5 5 0 0 1 9.9-1"/>')
PAUSE = Glyph("pause", '<rect x="14" y="4" width="4" height="16" rx="1"/>'
                       '<rect x="6" y="4" width="4" height="16" rx="1"/>')
PLAY = Glyph("play", '<polygon points="6 3 20 12 6 21 6 3"/>')
SETTINGS = Glyph("settings", (
    '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08'
    'a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74'
    'l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25'
    'a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25'
    'a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08'
    'a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38'
    'a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>'
    '<circle cx="12" cy="12" r="3"/>'))
CLOSE = Glyph("x", '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>')
COPY = Glyph("copy", '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
                     '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>')
TRANSLATE = Glyph("languages", '<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/>'
                               '<path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>')
CHECK = Glyph("check", '<path d="M20 6 9 17l-5-5"/>')
CHEVRON_DOWN = Glyph("chevron-down", '<path d="m6 9 6 6 6-6"/>')
CHEVRON_RIGHT = Glyph("chevron-right", '<path d="m9 18 6-6-6-6"/>')


# ── Rendering ──────────────────────────────────────────────────────────

def pixmap(glyph: Glyph, px: int, color: QColor, stroke: float = STROKE) -> QPixmap:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"'
           f' stroke="{color.name()}" stroke-opacity="{color.alphaF():.3f}" stroke-width="{stroke}"'
           f' stroke-linecap="round" stroke-linejoin="round">{glyph.svg}</svg>')
    pix = QPixmap(px, px)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(p, QRectF(0, 0, px, px))
    p.end()
    return pix


def icon(glyph: Glyph, px: int = 16, rest=TEXT_2, hover=TEXT) -> QIcon:
    """Normal at `rest`, Active (under the mouse, on an auto-raise button) at `hover`."""
    result = QIcon()
    for s in (px, px * 2):
        result.addPixmap(pixmap(glyph, s, qcolor(rest)), QIcon.Mode.Normal)
        result.addPixmap(pixmap(glyph, s, qcolor(hover)), QIcon.Mode.Active)
        result.addPixmap(pixmap(glyph, s, qcolor(TEXT_3)), QIcon.Mode.Disabled)
    return result


_files: dict[tuple, str] = {}


def file(glyph: Glyph, px: int, color, stroke: float = STROKE) -> str:
    """A PNG of the glyph (and its @2x) in a private folder; the path, for url()."""
    key = (glyph.name, px, tuple(color), stroke)
    if key not in _files:
        # Not /tmp/livetranscribe-<uid>: that is the single-instance socket
        # (src/ui/single.py). A folder there made every start think another
        # was running, and quit (2026-09-26).
        base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
        folder = os.path.join(base, f"livetranscribe-icons-{os.getuid()}")
        os.makedirs(folder, exist_ok=True)
        name = os.path.join(folder, f"{glyph.name}-{px}-{stroke}-{'-'.join(map(str, color))}")
        pixmap(glyph, px, qcolor(color), stroke).save(f"{name}.png")
        pixmap(glyph, px * 2, qcolor(color), stroke).save(f"{name}@2x.png")
        _files[key] = f"{name}.png".replace(os.sep, "/")
    return _files[key]
