"""Where step 1's text goes: the terminal, and a transcript file.

Both consume the streamer's `Update`s. Step 2's overlay will be one more
consumer of the same updates; nothing upstream changes for it.

Why a file as well as the terminal: the VS Code terminal does not lay Arabic
out right to left reliably (GNOME Terminal does). The transcript file opens
correctly in any editor, and it is what to read when judging whether the
transcription is *right*.
"""

import shutil
import sys
import time
from pathlib import Path

from src.audio.streamer import Line, Update

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREY = "\033[90m"
_DIM = "\033[2m"
_CLEAR = "\r\033[K"


def clock(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


class ConsoleSink:
    """Finished lines scroll up; the current line is redrawn in place.

    Committed words are bold, tentative words grey: grey means the next pass
    may still change it. With nothing to show, a dim status line says whether
    audio is arriving at all — the first question when no text appears.
    """

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.live = self.stream.isatty()
        self._committed = ""
        self._tentative = ""
        self._status = ""

    def update(self, upd: Update):
        for line in upd.finished:
            self._print_line(line)
        self._committed, self._tentative = upd.committed, upd.tentative
        self._draw()

    def status(self, text: str):
        self._status = text
        if not self._committed and not self._tentative:
            self._draw()

    def close(self):
        if self.live:
            self.stream.write(_CLEAR)
            self.stream.flush()

    def _print_line(self, line: Line):
        prefix = f"{_DIM}[{clock(line.start)}]{_RESET} " if self.live else f"[{clock(line.start)}] "
        self.stream.write((_CLEAR if self.live else "") + prefix + line.text + "\n")
        self.stream.flush()

    def _draw(self):
        if not self.live:
            return
        width = max(20, shutil.get_terminal_size((100, 20)).columns - 2)
        committed, tentative = self._committed, self._tentative
        if not committed and not tentative:
            self.stream.write(_CLEAR + _DIM + self._status[:width] + _RESET)
            self.stream.flush()
            return
        # Too long for one row: keep the end, which is what is being said now.
        # A wrapped row would break the in-place redraw.
        joined = f"{committed} {tentative}".strip()
        if len(joined) > width:
            cut = len(joined) - (width - 1)
            if committed and cut <= len(committed):
                committed = "…" + committed[cut:]
            else:
                cut -= len(committed) + (1 if committed else 0)
                committed, tentative = "", "…" + tentative[max(0, cut):]
        out = _CLEAR + _BOLD + committed + _RESET
        if tentative:
            out += (" " if committed else "") + _GREY + tentative + _RESET
        self.stream.write(out)
        self.stream.flush()


class TranscriptSink:
    """Finished lines, timestamped, flushed as they finish."""

    def __init__(self, path: Path, header: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8", buffering=1)
        self._fh.write(f"# LiveTranscribe — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if header:
            self._fh.write(f"# {header}\n")

    def update(self, upd: Update):
        for line in upd.finished:
            self._fh.write(f"[{clock(line.start)}] {line.text}\n")

    def status(self, text: str):
        pass

    def close(self):
        self._fh.close()
