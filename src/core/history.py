"""The translations the user made, kept on this computer, and exported for word-learning apps.

Every translation that comes back (src/ui/translate.py) is kept here, unless
the user turned history off. One line of JSON per translation, in
HISTORY_FILE: appended as they come, rewritten whole (through a temporary
file, so a crash cannot leave half of it) only when one is starred, deleted,
or translated again. A line that cannot be read is skipped, not fatal.

It is a word list, so each (text, language, engine) is in it once: the same
selection translated again moves to the top and keeps its star, rather than
becoming a second card in a flashcard deck. Whitespace is folded to single
spaces for the same reason — a selection across two transcript lines is one
phrase, and many importers break a row at a newline even when it is quoted.

Exports, all UTF-8, newest first, without a header row:

  * google — the columns of Google Translate's saved words ("Saved" /
    phrasebook) export: source language, target language, source text,
    translation, languages named in English. Anki, and the apps that offer
    "import from Google Translate", take it as it is;
  * pairs — just the text and its translation, two columns: what Quizlet,
    Brainscape, Knowt and the like expect for front and back;
  * json — a list of objects with the same four fields under plain names,
    and the rest of what is kept: language codes, engine, time, star.

Nothing here is sent anywhere.
"""

import csv
import io
import json
import os
import threading
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from src.config import HISTORY_FILE

# Google Translate's own English names, as its export writes them.
LANGUAGE_NAMES = {
    "ar": "Arabic", "en": "English", "tr": "Turkish", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "ru": "Russian", "fa": "Persian", "ur": "Urdu",
    "id": "Indonesian", "ms": "Malay", "zh-CN": "Chinese (Simplified)", "ja": "Japanese",
}

# (format, file-dialog filter, extension)
FORMATS = [
    ("google", "Google Translate saved words (*.csv)", ".csv"),
    ("pairs", "Flashcards: text, translation (*.csv)", ".csv"),
    ("json", "JSON (*.json)", ".json"),
]


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def fold(text: str) -> str:
    """Runs of whitespace, newlines included, to one space."""
    return " ".join(text.split())


@dataclass
class Entry:
    time: str                   # local, ISO 8601 to the second, with the UTC offset
    text: str                   # what was selected (Arabic)
    translation: str
    target: str                 # a code from translate.LANGUAGES
    engine: str                 # google | offline
    source: str = "ar"
    starred: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.text, self.target, self.engine)

    @classmethod
    def from_json(cls, line: str) -> "Entry":
        data = json.loads(line)
        names = {f.name for f in fields(cls)}
        entry = cls(**{k: v for k, v in data.items() if k in names})
        if not (isinstance(entry.text, str) and isinstance(entry.translation, str)
                and entry.text and entry.translation):
            raise ValueError("an entry without its text")
        entry.starred = bool(entry.starred)
        return entry

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class History:
    """The kept translations, oldest first, read from `path` the first time they are needed.

    `listeners` are called, without arguments, after every change.
    """

    def __init__(self, path: Path | str = HISTORY_FILE):
        self.path = Path(path)
        self.listeners: list = []
        self._entries: list[Entry] | None = None
        self._lock = threading.Lock()

    # ── Reading ────────────────────────────────────────────────────────

    def entries(self) -> list[Entry]:
        with self._lock:
            return list(self._load())

    def find(self, text: str, target: str, engine: str) -> Entry | None:
        key = (fold(text), target, engine)
        with self._lock:
            return next((e for e in self._load() if e.key == key), None)

    def _load(self) -> list[Entry]:
        if self._entries is None:
            self._entries = []
            try:
                with open(self.path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            self._entries.append(Entry.from_json(line))
                        except (ValueError, TypeError):
                            continue        # a broken line costs that line only
            except FileNotFoundError:
                pass
        return self._entries

    # ── Changing ───────────────────────────────────────────────────────

    def add(self, text: str, translation: str, target: str, engine: str) -> Entry | None:
        """Keep a translation; the same one again moves to the top with its star. None if empty."""
        text, translation = fold(text), fold(translation)
        if not text or not translation:
            return None
        entry = Entry(time=datetime.now().astimezone().isoformat(timespec="seconds"),
                      text=text, translation=translation, target=target, engine=engine)
        with self._lock:
            entries = self._load()
            old = next((e for e in entries if e.key == entry.key), None)
            if old is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(entry.to_json() + "\n")
                entries.append(entry)
            else:
                entry.starred = old.starred
                entries.remove(old)
                entries.append(entry)
                self._write()
        self._changed()
        return entry

    def set_starred(self, key: tuple[str, str, str], on: bool):
        with self._lock:
            entry = next((e for e in self._load() if e.key == key), None)
            if entry is None or entry.starred == on:
                return
            entry.starred = on
            self._write()
        self._changed()

    def remove(self, keys):
        keys = set(keys)
        with self._lock:
            before = len(self._load())
            self._entries = [e for e in self._entries if e.key not in keys]
            if len(self._entries) == before:
                return
            self._write()
        self._changed()

    def clear(self):
        with self._lock:
            self._entries = []
            self.path.unlink(missing_ok=True)
        self._changed()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            f.writelines(e.to_json() + "\n" for e in self._entries)
        os.replace(temporary, self.path)

    def _changed(self):
        for listener in list(self.listeners):
            listener()


_shared: History | None = None


def shared() -> History:
    """The app's one history, in HISTORY_FILE."""
    global _shared
    if _shared is None:
        _shared = History()
    return _shared


# ── Exports ────────────────────────────────────────────────────────────

def _csv(rows) -> str:
    out = io.StringIO()
    csv.writer(out).writerows(rows)          # quoted only where needed; \r\n, as RFC 4180
    return out.getvalue()


def google_csv(entries) -> str:
    return _csv([language_name(e.source), language_name(e.target), e.text, e.translation]
                for e in entries)


def pairs_csv(entries) -> str:
    return _csv([e.text, e.translation] for e in entries)


def as_json(entries) -> str:
    return json.dumps([{
        "source_language": language_name(e.source),
        "target_language": language_name(e.target),
        "source_text": e.text,
        "translated_text": e.translation,
        "source_code": e.source,
        "target_code": e.target,
        "engine": e.engine,
        "time": e.time,
        "starred": e.starred,
    } for e in entries], ensure_ascii=False, indent=2) + "\n"


def export(entries, path: Path | str, fmt: str):
    text = {"google": google_csv, "pairs": pairs_csv, "json": as_json}[fmt](entries)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
