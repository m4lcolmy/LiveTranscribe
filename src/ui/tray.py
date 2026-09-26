"""The app's icons: the full-colour one for the window and the app menu, and
the tray's symbolic one — a white mark, as GNOME's top bar expects — dimmed
while paused.

The artwork is packaging/icons/*.png (drawn by packaging/make_icons.py), so
the menu entry and the window show the same picture. Without those files it
is drawn on the spot, by the same code.
"""

import importlib.util
from pathlib import Path

from PyQt6.QtGui import QIcon

PACKAGING = Path(__file__).resolve().parent.parent.parent / "packaging"
ICONS = PACKAGING / "icons"


def _drawing():
    spec = importlib.util.spec_from_file_location("make_icons", PACKAGING / "make_icons.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_icon(paused: bool = False) -> QIcon:
    name = "livetranscribe-paused" if paused else "livetranscribe"
    icon = QIcon()
    for png in sorted(ICONS.glob(f"{name}-[0-9]*.png")):
        icon.addFile(str(png))
    if icon.isNull():
        for size in (16, 22, 32, 64, 128):
            icon.addPixmap(_drawing().draw(size, grey=paused))
    return icon


def make_tray_icon(paused: bool = False) -> QIcon:
    name = "livetranscribe-tray-paused" if paused else "livetranscribe-tray"
    icon = QIcon()
    for png in sorted(ICONS.glob(f"{name}-[0-9]*.png")):
        icon.addFile(str(png))
    if icon.isNull():
        for size in (16, 22, 24, 32):
            icon.addPixmap(_drawing().draw_tray(size, paused=paused))
    return icon
