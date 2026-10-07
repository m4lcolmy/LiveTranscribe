"""The look: one set of colours, sizes and stylesheets for every window the app draws.

Neutral greys, one accent, small radii. The greys are GNOME's dark ones and
the accent its blue (#3584E4), so the app sits in the desktop it runs on
rather than bringing a palette of its own. The transcript window, the
translate popup, the menus and the settings dialog all take their colours
from here; a colour typed anywhere else is a bug.

Colours are (r, g, b) or (r, g, b, alpha 0-255) tuples, so they drop into
QColor(*c) and into stylesheets through rgba() alike.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QMenu, QWidget

# ── Colours ────────────────────────────────────────────────────────────

SURFACE = (18, 18, 18)                 # the transcript panel, under the user's opacity
WINDOW = (36, 36, 36)                  # the settings dialog
RAISED = (46, 46, 46)                  # menus, the translate popup, tooltips
FIELD = (52, 52, 52)                   # drop-downs, buttons, the slider's track
FIELD_HOVER = (62, 62, 62)
LINE = (68, 68, 68)                    # borders of fields and popups, dividers
HAIRLINE = (255, 255, 255, 22)         # the transcript panel's edge, over any video
HOVER = (255, 255, 255, 22)            # under an icon button or a menu item
PRESSED = (255, 255, 255, 36)

TEXT = (255, 255, 255, 255)            # final words, values
TEXT_2 = (255, 255, 255, 178)          # labels, the status line, icons at rest
TEXT_3 = (255, 255, 255, 110)          # hints, notes, disabled
TENTATIVE = (255, 255, 255, 128)       # words that may still change

ACCENT = (53, 132, 228)                # GNOME blue: the primary button, selection, focus
ACCENT_HOVER = (74, 146, 234)
SELECTION = (53, 132, 228, 120)

# The status dot.
LIVE = (46, 194, 126)                  # listening
BUSY = (229, 165, 10)                  # loading, stopping, running on the CPU
ERROR = (237, 51, 59)
IDLE = (255, 255, 255, 90)             # paused, ended

# ── Sizes ──────────────────────────────────────────────────────────────

RADIUS = 6             # the panel, popups, menus
RADIUS_CONTROL = 4     # buttons, fields
UI_PX = 13             # text of the app's own: menus, the dialog
SMALL_PX = 12          # the status line, hints, notes


def rgba(c) -> str:
    r, g, b, *a = c
    return f"rgba({r},{g},{b},{a[0] if a else 255})"


def qcolor(c, alpha: int | None = None) -> QColor:
    color = QColor(*c)
    if alpha is not None:
        color.setAlpha(alpha)
    return color


# ── Stylesheets ────────────────────────────────────────────────────────

def app_stylesheet() -> str:
    """Menus and tooltips, application-wide. The tray's menu is drawn by the
    desktop when it has AppIndicator support; this reaches every other menu."""
    from src.ui import icons
    return f"""
    QMenu {{
        background: {rgba(RAISED)}; color: {rgba(TEXT)};
        border: 1px solid {rgba(LINE)}; border-radius: {RADIUS}px;
        padding: 4px; font-size: {UI_PX}px;
    }}
    QMenu::item {{ padding: 5px 28px 5px 10px; border-radius: 3px; background: transparent; }}
    QMenu::item:selected {{ background: {rgba(ACCENT)}; }}
    QMenu::item:disabled {{ color: {rgba(TEXT_3)}; background: transparent; }}
    QMenu::indicator {{ width: 14px; height: 14px; padding-left: 8px; }}
    QMenu::indicator:checked {{ image: url("{icons.file(icons.CHECK, 14, TEXT, 2.25)}"); }}
    QMenu::separator {{ height: 1px; background: {rgba(LINE)}; margin: 4px 6px; }}
    QMenu::right-arrow {{ image: url("{icons.file(icons.CHEVRON_RIGHT, 14, TEXT_2)}");
                         width: 14px; height: 14px; padding-right: 6px; }}
    QToolTip {{
        background: {rgba(RAISED)}; color: {rgba(TEXT)};
        border: 1px solid {rgba(LINE)}; padding: 4px 7px; font-size: {SMALL_PX}px;
    }}
    """


def polish_menu(menu: QMenu) -> QMenu:
    """Rounded corners need a see-through window behind them: this menu and its submenus."""
    for m in [menu, *menu.findChildren(QMenu)]:
        m.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        m.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
    return menu


def icon_button_stylesheet() -> str:
    """A bare icon that shows a soft square under the mouse."""
    return (f"QToolButton {{ border: none; border-radius: {RADIUS_CONTROL}px;"
            f" background: transparent; padding: 0; }}"
            f"QToolButton:hover {{ background: {rgba(HOVER)}; }}"
            f"QToolButton:pressed {{ background: {rgba(PRESSED)}; }}")


def dialog_palette(widget: QWidget):
    """Dark roles for whatever the stylesheet does not reach (focus frames, the file dialog)."""
    p = widget.palette()
    for role, c in ((QPalette.ColorRole.Window, WINDOW), (QPalette.ColorRole.WindowText, TEXT),
                    (QPalette.ColorRole.Base, FIELD), (QPalette.ColorRole.AlternateBase, WINDOW),
                    (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, FIELD),
                    (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
                    (QPalette.ColorRole.HighlightedText, TEXT),
                    (QPalette.ColorRole.PlaceholderText, TEXT_3),
                    (QPalette.ColorRole.ToolTipBase, RAISED), (QPalette.ColorRole.ToolTipText, TEXT)):
        p.setColor(role, qcolor(c))
    widget.setPalette(p)


def dialog_stylesheet() -> str:
    from src.ui import icons
    chevron = icons.file(icons.CHEVRON_DOWN, 14, TEXT_2)
    check = icons.file(icons.CHECK, 12, TEXT, 3.0)
    return f"""
    QDialog {{ background: {rgba(WINDOW)}; }}
    QLabel {{ color: {rgba(TEXT_2)}; font-size: {UI_PX}px; }}
    QLabel[role="section"] {{ color: {rgba(TEXT)}; font-size: {UI_PX}px; font-weight: 600; }}
    QFrame[role="rule"] {{ background: {rgba(LINE)}; border: none; }}
    QLabel[role="hint"] {{ color: {rgba(TEXT_3)}; font-size: {SMALL_PX}px; }}
    QLabel[role="value"] {{ color: {rgba(TEXT)}; font-size: {UI_PX}px; }}
    QLabel[role="note"] {{ color: {rgba(TEXT_2)}; font-size: {SMALL_PX}px; }}

    QComboBox {{
        background: {rgba(FIELD)}; color: {rgba(TEXT)}; font-size: {UI_PX}px;
        border: 1px solid {rgba(LINE)}; border-radius: {RADIUS_CONTROL}px;
        padding: 4px 8px; min-height: 18px; combobox-popup: 0;
    }}
    QComboBox:hover {{ background: {rgba(FIELD_HOVER)}; }}
    QComboBox:focus, QComboBox:on {{ border-color: {rgba(ACCENT)}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox::down-arrow {{ image: url("{chevron}"); width: 14px; height: 14px; }}
    QComboBox QAbstractItemView {{
        background: {rgba(RAISED)}; color: {rgba(TEXT)}; outline: 0; padding: 3px;
        border: 1px solid {rgba(LINE)};
        selection-background-color: {rgba(ACCENT)}; selection-color: {rgba(TEXT)};
    }}
    QComboBox QAbstractItemView::item {{ min-height: 24px; padding: 0 6px; border-radius: 3px; }}
    QComboBox QAbstractItemView::item:disabled {{ color: {rgba(TEXT_3)}; }}

    QLineEdit {{
        background: {rgba(FIELD)}; color: {rgba(TEXT)}; font-size: {UI_PX}px;
        border: 1px solid {rgba(LINE)}; border-radius: {RADIUS_CONTROL}px;
        padding: 4px 8px; min-height: 18px; selection-background-color: {rgba(ACCENT)};
    }}
    QLineEdit:hover {{ background: {rgba(FIELD_HOVER)}; }}
    QLineEdit:focus {{ border-color: {rgba(ACCENT)}; }}

    QCheckBox {{ color: {rgba(TEXT)}; font-size: {UI_PX}px; spacing: 8px; }}
    QCheckBox:disabled {{ color: {rgba(TEXT_3)}; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px; border-radius: 3px;
        border: 1px solid {rgba(TEXT_3)}; background: {rgba(FIELD)};
    }}
    QCheckBox::indicator:hover {{ border-color: {rgba(TEXT_2)}; }}
    QCheckBox::indicator:checked {{
        background: {rgba(ACCENT)}; border-color: {rgba(ACCENT)}; image: url("{check}");
    }}
    QCheckBox::indicator:disabled {{ background: transparent; border-color: {rgba(LINE)}; }}

    QSlider {{ min-height: 18px; }}
    QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {rgba(LINE)}; }}
    QSlider::sub-page:horizontal {{ height: 4px; border-radius: 2px; background: {rgba(ACCENT)}; }}
    QSlider::handle:horizontal {{
        width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {rgba(TEXT)};
    }}

    QPushButton {{
        background: {rgba(FIELD)}; color: {rgba(TEXT)}; font-size: {UI_PX}px;
        border: 1px solid {rgba(LINE)}; border-radius: {RADIUS_CONTROL}px;
        padding: 5px 16px; min-width: 64px;
    }}
    QPushButton:hover {{ background: {rgba(FIELD_HOVER)}; }}
    QPushButton[primary="true"] {{ background: {rgba(ACCENT)}; border-color: {rgba(ACCENT)}; }}
    QPushButton[primary="true"]:hover {{ background: {rgba(ACCENT_HOVER)}; }}
    QPushButton:disabled, QPushButton[primary="true"]:disabled {{
        background: {rgba(FIELD)}; border-color: {rgba(LINE)}; color: {rgba(TEXT_3)};
    }}

    QToolButton#disclosure {{
        color: {rgba(TEXT)}; font-size: {UI_PX}px; font-weight: 600;
        border: none; background: transparent; padding: 0;
    }}

    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {rgba(LINE)}; border-radius: 2px; min-height: 24px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    """
