"""LiveTranscribe — live Arabic subtitles for whatever the computer plays.

    ./run.sh                          # subtitles at the bottom of the screen
    ./run.sh --console                # step 1's terminal mode, no window
    ./run.sh --file lecture.mp4       # a file, at the pace it would play
    ./run.sh --record                 # also keep the session for scripts/replay.py
    ./run.sh --sink <node.name>       # a specific output instead of the default
"""

import argparse
import os
import sys
import warnings

# Offline, always: a model missing from the cache is an error, not a download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import ASR_STEP_S_GPU, ASR_STEP_S_CPU, OVERLAY_FORCE_XWAYLAND
from src.core.debug import log


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="livetranscribe",
        description="Live Arabic subtitles for the system's audio output.",
    )
    p.add_argument("--console", action="store_true",
                   help="print to the terminal instead of showing subtitles on screen")
    p.add_argument("--file", help="transcribe this audio/video file at playback pace")
    p.add_argument("--sink", help="capture this output (node.name, see `wpctl status`) "
                                  "instead of following the default one")
    # Model, device, step and output default to the saved settings (the ⚙
    # dialog); a flag given here overrides them for this run and is not saved.
    p.add_argument("--model", help="model size or folder (default: the saved setting)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--beam", type=int, help="beam size")
    p.add_argument("--step", type=float, help="seconds between passes "
                                              f"(default {ASR_STEP_S_GPU} GPU / {ASR_STEP_S_CPU} CPU)")
    p.add_argument("--record", action="store_true",
                   help="keep audio + every pass under logs/session-<time>/ for replay")
    p.add_argument("--no-debug", action="store_true", help="no session log")
    p.add_argument("--bypass-wm", action="store_true",
                   help="overlay: bypass the window manager (above fullscreen video, "
                        "on every workspace; moved only by dragging)")
    p.add_argument("--reset-settings", action="store_true",
                   help="forget all saved settings and the window's place, then start")
    p.add_argument("--wayland", action="store_true",
                   help="overlay: run as a native Wayland window (GNOME will not keep it on top)")
    return p.parse_args(argv)


# ── Terminal (step 1) ──────────────────────────────────────────────────

def run_console(args) -> int:
    from src.audio.capture import CaptureError
    from src.audio.engine import WhisperEngine
    from src.audio.streamer import Streamer
    from src.audio.worker import Pipeline
    from src.session import effective_args, open_session
    from src.sinks import ConsoleSink, TranscriptSink
    from src.ui.settings import AppSettings, open_store

    args = effective_args(args, AppSettings.load(open_store()))
    print(f"Loading Whisper {args.model}…", end="", flush=True)
    try:
        engine = WhisperEngine(args.model, args.device, args.beam, args.precision).load()
    except Exception as e:  # a missing model or a broken CUDA install: say which
        print(f"\nCould not load the model: {e}")
        return 1
    print("\r\033[K", end="")

    try:
        session = open_session(args, engine)
    except (CaptureError, OSError, RuntimeError) as e:
        print(f"Cannot read audio: {e}")
        return 1

    print("LiveTranscribe — terminal")
    print("\n".join(session.header()))
    print("Bold = final, grey = may still change.  Ctrl+C to stop.\n")

    pipeline = Pipeline(
        session.source, Streamer(engine, language_check=args.arabic_only),
        [ConsoleSink(), TranscriptSink(session.transcript, session.source.label)],
        session.step_s, session.recorder,
    )
    pipeline.run()

    summary = log.summary(session.step_s)
    log.close()
    print("\n" + summary)
    return 0


# ── Overlay (step 2) ───────────────────────────────────────────────────

def choose_platform(args):
    """XWayland on a Wayland desktop: the only way GNOME lets a window stay on top.

    An explicit QT_QPA_PLATFORM is the user's choice and always wins.
    """
    if os.environ.get("QT_QPA_PLATFORM") or args.wayland or not OVERLAY_FORCE_XWAYLAND:
        return
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def run_overlay(args) -> int:
    choose_platform(args)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    import signal

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from src.ui.controller import Controller

    from src.ui.single import SingleInstance
    from src.ui.tray import make_icon

    app = QApplication(sys.argv)
    app.setApplicationName("LiveTranscribe")
    app.setDesktopFileName("livetranscribe")
    app.setWindowIcon(make_icon())
    app.setQuitOnLastWindowClosed(False)     # the tray keeps it going

    # Started a second time (from the app menu, say): show the running one
    # instead of opening a second window that captures the same audio again.
    instance = SingleInstance()
    if not instance.claim():
        print("LiveTranscribe is already running — showing its window.")
        return 0

    controller = Controller(args, app)
    instance.activated.connect(controller.window.show_window)

    # Ctrl+C in the terminal quits cleanly. Python only runs signal handlers
    # between bytecodes, so a timer keeps the interpreter waking up.
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, controller.quit))
    wake = QTimer()
    wake.start(250)
    wake.timeout.connect(lambda: None)

    return app.exec()


def reset_settings():
    from src.ui.settings import open_store
    store = open_store()
    store.clear()
    store.sync()
    print(f"Saved settings cleared ({store.fileName()}).")


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.reset_settings:
        reset_settings()
    return run_console(args) if args.console else run_overlay(args)


if __name__ == "__main__":
    sys.exit(main())
