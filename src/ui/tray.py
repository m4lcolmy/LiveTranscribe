"""The app icon: the tray, the window, the app menu. Grey while paused.

The artwork is packaging/icons/*.png (drawn by packaging/make_icons.py), so
the menu entry, the tray and the window show the same picture. Without those
files it is drawn on the spot, by the same code.
"""

import importlib.util
from pathlib import Path

from PyQt6.QtGui import QIcon

PACKAGING = Path(__file__).resolve().parent.parent.parent / "packaging"
ICONS = PACKAGING / "icons"


def make_icon(paused: bool = False) -> QIcon:
    name = "livetranscribe-paused" if paused else "livetranscribe"
    icon = QIcon()
    for png in sorted(ICONS.glob(f"{name}-[0-9]*.png")):
        icon.addFile(str(png))
    if icon.isNull():
        spec = importlib.util.spec_from_file_location("make_icons", PACKAGING / "make_icons.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for size in (16, 22, 32, 64, 128):
            icon.addPixmap(module.draw(size, grey=paused))
    return icon
