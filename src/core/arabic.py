"""Arabic text keys for comparing two transcriptions of the same speech.

Two passes over the same audio often write the same word differently: with
or without a hamza seat, with a trailing ة or ه, with punctuation attached.
Those are the same word, and treating them as different would keep it
tentative forever. So hypotheses are compared on a key, never on the raw
text. The text shown is always Whisper's own.

Adapted from hifz's src/core/arabic.py, without the Quran-specific parts.
"""

import re

# Tashkeel, dagger alef, Quranic annotation marks and their extended forms.
_DIACRITICS = re.compile(
    "["
    "ً-ٕ"      # fathatan … hamza below
    "ٰ"             # superscript (dagger) alef
    "ۖ-ۭ"      # Quranic annotation marks
    "ؐ-ؚ"      # more combining marks
    "࣓-࣡"
    "ࣣ-ࣿ"
    "]"
)

_TATWEEL = "ـ"

# Punctuation Whisper attaches to words, Arabic and Latin.
_PUNCTUATION = re.compile(r"[،؛؟٪-٭۔.,;:!?\"'«»()\[\]{}\-–—…/\\]")

# Arabic-Indic and Persian digits: a caption writes ٢٠٢٦ where Whisper writes 2026.
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ALEF_FORMS = str.maketrans({
    "ٱ": "ا",  # ٱ → ا
    "آ": "ا",  # آ → ا
    "أ": "ا",  # أ → ا
    "إ": "ا",  # إ → ا
    "ة": "ه",  # ة → ه
    "ى": "ي",  # ى → ي
})

_SPACES = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    """Remove tashkeel and Quranic marks."""
    return _DIACRITICS.sub("", text)


def normalize(text: str) -> str:
    """Diacritics, tatweel and letter-form variants removed; spacing collapsed."""
    t = strip_diacritics(text).replace(_TATWEEL, "")
    t = t.translate(_ALEF_FORMS)
    return _SPACES.sub(" ", t).strip()


def compare_key(text: str) -> str:
    """normalize() without punctuation, digits unified. Two spellings of one word share a key."""
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", normalize(text).translate(_DIGITS))).strip()
