"""The one X11 hint Qt has no API for: be on every workspace.

The overlay runs under XWayland (see overlay.py). Qt already asks for "always
on top" (_NET_WM_STATE_ABOVE); it has nothing for "on all workspaces", so the
subtitles would stay behind on the workspace they started on. EWMH asks for it
with two client messages to the root window, which mutter honours for X11
windows — the same thing the Live Captions GNOME extension gets by calling
mutter's window.stick().

Plain libX11 through ctypes: no new dependency, and when there is no X server
(a native Wayland run) or no libX11, it quietly does nothing.
"""

import ctypes
import ctypes.util

_CLIENT_MESSAGE = 33
_SUBSTRUCTURE_NOTIFY = 1 << 19
_SUBSTRUCTURE_REDIRECT = 1 << 20
_ALL_DESKTOPS = 0xFFFFFFFF
_NET_WM_STATE_ADD = 1
_SOURCE_PAGER = 2


class _ClientMessage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", ctypes.c_long * 5),
    ]


class _XEvent(ctypes.Union):
    _fields_ = [("xclient", _ClientMessage), ("pad", ctypes.c_long * 24)]


def _libx11():
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
    except OSError:
        return None
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XInternAtom.restype = ctypes.c_ulong
    lib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                               ctypes.c_long, ctypes.POINTER(_XEvent)]
    lib.XFlush.argtypes = [ctypes.c_void_p]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    return lib


def make_sticky(window_id: int) -> bool:
    """Ask the window manager to show this X11 window on every workspace."""
    lib = _libx11()
    if lib is None:
        return False
    display = lib.XOpenDisplay(None)
    if not display:
        return False
    try:
        root = lib.XDefaultRootWindow(display)

        def atom(name: str) -> int:
            return lib.XInternAtom(display, name.encode(), 0)

        def send(message_type: int, *data: int):
            event = _XEvent()
            msg = event.xclient
            msg.type, msg.send_event, msg.display = _CLIENT_MESSAGE, 1, display
            msg.window, msg.message_type, msg.format = window_id, message_type, 32
            for i, value in enumerate(data):
                msg.data[i] = value
            lib.XSendEvent(display, root, 0, _SUBSTRUCTURE_REDIRECT | _SUBSTRUCTURE_NOTIFY,
                           ctypes.byref(event))

        send(atom("_NET_WM_DESKTOP"), _ALL_DESKTOPS, _SOURCE_PAGER)
        send(atom("_NET_WM_STATE"), _NET_WM_STATE_ADD, atom("_NET_WM_STATE_STICKY"), 0, _SOURCE_PAGER)
        lib.XFlush(display)
        return True
    finally:
        lib.XCloseDisplay(display)


# ── Which part of the window takes the mouse ───────────────────────────

class _XRectangle(ctypes.Structure):
    _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short),
                ("width", ctypes.c_ushort), ("height", ctypes.c_ushort)]


_SHAPE_INPUT = 2
_SHAPE_SET = 0
_UNSORTED = 0


def set_input_region(window_id: int, rects: list[tuple[int, int, int, int]] | None) -> bool:
    """Let only `rects` (x, y, w, h in window pixels) take the mouse; None restores the whole window.

    Qt's WindowTransparentForInput is all or nothing — a click-through window
    whose own buttons ignore the mouse is a trap (a user checked the box and
    could not click anything in the app again, 2026-09-26). The X Shape
    extension's input region can leave a part clickable, and mutter honours it
    for XWayland windows, as it honours Qt's all-or-nothing version.
    """
    lib = _libx11()
    try:
        ext = ctypes.CDLL(ctypes.util.find_library("Xext") or "libXext.so.6")
    except OSError:
        return False
    if lib is None:
        return False
    ext.XShapeCombineRectangles.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(_XRectangle), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    ext.XShapeCombineMask.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_ulong, ctypes.c_int]
    display = lib.XOpenDisplay(None)
    if not display:
        return False
    try:
        if rects is None:
            ext.XShapeCombineMask(display, window_id, _SHAPE_INPUT, 0, 0, 0, _SHAPE_SET)
        else:
            array = (_XRectangle * len(rects))(*[_XRectangle(*r) for r in rects])
            ext.XShapeCombineRectangles(display, window_id, _SHAPE_INPUT, 0, 0,
                                        array, len(rects), _SHAPE_SET, _UNSORTED)
        lib.XFlush(display)
        return True
    finally:
        lib.XCloseDisplay(display)
