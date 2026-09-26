"""Scoring a streamed session: how right, how complete, how fast, how steady.

The numbers are always reported together, because each one alone rewards the
wrong thing (hifz's lesson):

    COVERAGE    reference words that were shown at all. A setting that shows
                less text makes fewer mistakes; this is what catches it.
    CER / WER   against a reference, after compare_key(). CER leads: Arabic
                attaches clitics (و، ف، ب، ال) that make one letter of
                disagreement a whole wrong word.
    STREAM GAP  CER of the stream minus CER of the same model on the whole
                file. What streaming costs, separate from what the model costs.
    LATENCY     word end → first shown (tentative) and → committed.
    FLICKER     tentative words shown and then not confirmed by the next pass —
                hifz's "retracted reds": invisible in the final text, obvious to
                anyone watching.
    REAL-TIME   pass latency against the step.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from src.audio.engine import Word
from src.audio.streamer import Update
from src.core.arabic import compare_key

# ── Text ───────────────────────────────────────────────────────────────

_SRT_TIMING = re.compile(r"^\d\d:\d\d:\d\d[,.]\d+\s*-->")
_SRT_TIMES = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d+)\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d+)")
_TRANSCRIPT_STAMP = re.compile(r"^\[[\d:.]+\]\s*")
_TAGS = re.compile(r"<[^>]+>|\{[^}]+\}")
# [موسيقى], [تصفيق], [Music]: annotations a captioner adds, not speech.
_ANNOTATION = re.compile(r"\[[^\]]*\]")


def _clean(line: str) -> str:
    return _ANNOTATION.sub(" ", _TAGS.sub("", _TRANSCRIPT_STAMP.sub("", line))).strip()


def load_cues(path) -> list[tuple[float, float, str]]:
    """(start, end, text) for every cue of an SRT."""
    cues, times, text = [], None, []
    for raw in list(open(path, encoding="utf-8-sig")) + [""]:
        line = raw.strip()
        m = _SRT_TIMES.search(line)
        if m:
            g = [int(x) for x in m.groups()]
            times = (g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000)
            text = []
        elif not line:
            if times is not None and text:
                cues.append((times[0], times[1], _clean(" ".join(text))))
            times, text = None, []
        elif times is not None:
            text.append(line)
    return cues


def load_reference(path, start: float | None = None, end: float | None = None
                   ) -> tuple[str, float | None, float | None]:
    """The reference text, and the span it covers.

    Plain text or a LiveTranscribe transcript: all of it, span unknown. An SRT
    with a window: only the cues wholly inside [start, end], and the span from
    the first of them to the last — the audio to score is cut to that span, so
    no half-cue sits at either edge.
    """
    if str(path).lower().endswith(".srt"):
        cues = load_cues(path)
        if start is not None or end is not None:
            lo, hi = start or 0.0, end if end is not None else float("inf")
            cues = [c for c in cues if c[0] >= lo and c[1] <= hi]
        if not cues:
            return "", None, None
        return " ".join(c[2] for c in cues), cues[0][0], cues[-1][1]

    lines = []
    for raw in open(path, encoding="utf-8-sig"):
        line = raw.strip()
        if not line or line.startswith("#") or line.isdigit() or _SRT_TIMING.match(line):
            continue
        lines.append(_clean(line))
    return " ".join(lines), None, None


def levenshtein(a, b) -> int:
    """Edit distance between two sequences, one numpy row at a time."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    b_arr = np.array([hash(x) for x in b])
    idx = np.arange(len(b) + 1)
    prev = idx.copy()
    for i, x in enumerate(a, 1):
        cost = (b_arr != hash(x)).astype(np.int64)
        row = np.empty_like(prev)
        row[0] = i
        row[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        # Insertions run along the row: d[j] = min_k<=j (d[k] + j - k).
        row = np.minimum.accumulate(row - idx) + idx
        prev = row
    return int(prev[-1])


def word_alignment(hyp: list[str], ref: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of the best word alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = dl = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dl, ins


@dataclass
class TextScore:
    cer: float
    wer: float
    coverage: float
    ref_words: int
    hyp_words: int
    ref_chars: int = 0


def score_text(hyp: str, ref: str) -> TextScore:
    h, r = compare_key(hyp), compare_key(ref)
    hw, rw = h.split(), r.split()
    s, dl, ins = word_alignment(hw, rw)
    cer = levenshtein(list(h), list(r)) / max(1, len(r))
    return TextScore(
        cer=cer, wer=(s + dl + ins) / max(1, len(rw)),
        coverage=(len(rw) - dl) / max(1, len(rw)),
        ref_words=len(rw), hyp_words=len(hw), ref_chars=len(r),
    )


# ── A session ──────────────────────────────────────────────────────────

def _pct(values, q) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


@dataclass
class SessionRecord:
    """Every update of a streamed session, with the (simulated) time it was shown."""

    duration_s: float = 0.0
    step_s: float = 0.0
    shown: list[tuple[float, Update]] = field(default_factory=list)

    def add(self, shown_at: float, update: Update):
        self.shown.append((shown_at, update))

    @property
    def text(self) -> str:
        return " ".join(line.text for _, u in self.shown for line in u.finished)

    def latencies(self) -> tuple[list[float], list[float]]:
        """(first shown, committed) latencies per committed word, from the word's end.

        Committed text only ever grows, in order: finished lines, then the
        current line. So the k-th committed word was committed by the first
        update whose committed text reached k+1 words.
        """
        commit_time: list[tuple[Word, float]] = []
        done: list[Word] = []
        for at, u in self.shown:
            for line in u.finished:
                done.extend(line.words)
            for w in (done + list(u.line_words))[len(commit_time):]:
                commit_time.append((w, at))

        # When each word was first on screen as a tentative guess.
        seen: dict[str, list[tuple[float, float]]] = {}
        for at, u in self.shown:
            for w in u.tentative_words:
                seen.setdefault(compare_key(w.text), []).append((at, w.start))

        first, committed = [], []
        for w, at in commit_time:
            guesses = [t for t, start in seen.get(compare_key(w.text), ())
                       if t <= at and abs(start - w.start) < 0.5]
            first.append(max(0.0, min(guesses + [at]) - w.end))
            committed.append(max(0.0, at - w.end))
        return first, committed

    def flicker(self) -> int:
        """Tentative words the very next pass did not confirm."""
        passes = [u for _, u in self.shown if u.report is not None]
        count = 0
        for a, b in zip(passes, passes[1:]):
            confirmed = [(compare_key(c.text), c.start)
                         for c in list(b.report.committed) + list(b.tentative_words)]
            for w in a.tentative_words:
                key = compare_key(w.text)
                if not any(k == key and abs(start - w.start) < 0.5 for k, start in confirmed):
                    count += 1
        return count

    def pass_latencies(self) -> list[float]:
        return [u.report.latency_s for _, u in self.shown if u.report is not None]

    def summary(self) -> dict:
        first, committed = self.latencies()
        lat = self.pass_latencies()
        minutes = max(self.duration_s / 60, 1e-9)
        return {
            "words": len(self.text.split()),
            "shown_p50": _pct(first, 0.5), "shown_p90": _pct(first, 0.9),
            "commit_p50": _pct(committed, 0.5), "commit_p90": _pct(committed, 0.9),
            "flicker_per_min": self.flicker() / minutes,
            "passes": len(lat),
            "pass_p50": _pct(lat, 0.5), "pass_p90": _pct(lat, 0.9),
            "overran": sum(1 for x in lat if x > self.step_s),
        }
