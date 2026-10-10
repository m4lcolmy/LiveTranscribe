"""The streaming policy: when Whisper runs, which words are final, where to cut.

Nothing here knows about Qt, PipeWire or the wall clock. The live pipeline,
the --file mode, the replay benchmark and the unit tests all drive this one
class with `feed()` and `step()`, so what the benchmark measures is what the
app does.

**Why a growing buffer.** faster-whisper pads every input to 30 s, so a pass
over 2 s of audio costs about what a pass over 15 s does. Short windows buy no
speed; they only take context away, and hifz measured what that does to a
small model (23.4% false alarms against 4.6%). So every pass transcribes
*everything since the last commit point*.

**Which words are final.** A word is committed when two consecutive passes
agree on it — the common prefix of the previous hypothesis and this one
(LocalAgreement-2, Macháček et al., whisper_streaming), made tolerant of one
short disagreement so a single unstable word cannot hold back the sentence
after it (see AGREE_MAX_GAP). Words after the agreed part are tentative:
shown, but free to change on the next pass. A pass that stops short of the
last one changes only what it heard: guesses past its end stay shown.

**When a line ends.** When the speaker stops (END_SILENCE_S of trailing
silence), one closing pass hears the whole utterance, every word is committed,
the line is closed and the buffer starts afresh. For a short utterance that
pass sees exactly what an offline whole-utterance transcription would.

**Where the buffer is cut.** Past MAX_BUFFER_S it is cut at the last pause
the VAD found before the last committed word — never inside a word — or,
without one, at the end of the last committed Whisper segment. Committed text whose audio has been
cut away becomes the prompt for the next pass; text still in the buffer never
does, or Whisper would hear it twice and repeat it.
"""

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from src.audio.engine import Engine, Segment, Word
from src.config import (
    SAMPLE_RATE, MIN_FIRST_PASS_S, MIN_UTTERANCE_SPEECH_S, LEAD_IN_S,
    END_SILENCE_S, TAIL_PAD_S, MAX_BUFFER_S, HARD_MAX_BUFFER_S,
    KEEP_AFTER_HARD_TRIM_S, PROMPT_CHARS, LINE_MAX_CHARS,
    COMMIT_TIME_TOLERANCE_S, ECHO_MAX_WORDS, AGREE_MAX_GAP, AGREE_MIN_RUN,
    CUT_PAUSE_FRAMES, TENTATIVE_GUARD_S, LANGUAGE_CHECK, LANGUAGE_MIN_ARABIC,
    LANGUAGE_RECHECK_S, LANGUAGE_MIN_AUDIO_S, LANGUAGE_CONFIRMATIONS, LANGUAGE_CONFIRM_AFTER_S,
)
from src.core.arabic import compare_key


@dataclass(frozen=True)
class Line:
    """A finished subtitle line. Times are seconds from the start of the stream."""
    text: str
    start: float
    end: float
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class PassReport:
    """One Whisper pass, for the session log and the benchmark."""
    at: float                    # stream time the pass's audio ended at
    buffer_start: float
    latency_s: float
    heard: str                   # every word the kept segments contained
    dropped: tuple               # (reason, detail, text) per gated segment
    committed: tuple[Word, ...]  # committed by this pass
    tentative: tuple[Word, ...]
    final: bool                  # the utterance's closing pass
    prompt: str


@dataclass(frozen=True)
class Update:
    """What a step changed. `committed` and `tentative` are the current line."""
    finished: tuple[Line, ...]
    committed: str
    tentative: str
    line_words: tuple[Word, ...]       # the committed words of the current line
    tentative_words: tuple[Word, ...]
    report: PassReport | None
    now: float


def agreed_prefix(previous: list[Word], current: list[Word],
                  max_gap: int = AGREE_MAX_GAP, min_run: int = AGREE_MIN_RUN) -> int:
    """How many leading words of `current` the previous pass agrees with.

    A common prefix, compared on keys — except that a disagreement of at most
    `max_gap` words is passed over when at least `min_run` words agree right
    after it. The words committed across such a gap are the current pass's.
    """
    a = [compare_key(w.text) for w in previous]
    b = [compare_key(w.text) for w in current]
    agreed = through_a = 0
    for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        if block.size == 0:
            break
        gap_a, gap_b = block.a - through_a, block.b - agreed
        if gap_a or gap_b:
            if gap_a > max_gap or gap_b > max_gap or block.size < min_run:
                break
        agreed, through_a = block.b + block.size, block.a + block.size
    return agreed


class Streamer:
    def __init__(self, engine: Engine, vad=None, *, sample_rate=SAMPLE_RATE,
                 min_first_pass_s=MIN_FIRST_PASS_S,
                 min_utterance_speech_s=MIN_UTTERANCE_SPEECH_S,
                 lead_in_s=LEAD_IN_S, end_silence_s=END_SILENCE_S,
                 tail_pad_s=TAIL_PAD_S, max_buffer_s=MAX_BUFFER_S,
                 hard_max_buffer_s=HARD_MAX_BUFFER_S,
                 keep_after_hard_trim_s=KEEP_AFTER_HARD_TRIM_S,
                 prompt_chars=PROMPT_CHARS, line_max_chars=LINE_MAX_CHARS,
                 commit_tolerance_s=COMMIT_TIME_TOLERANCE_S,
                 echo_max_words=ECHO_MAX_WORDS, tentative_guard_s=TENTATIVE_GUARD_S,
                 language_check=LANGUAGE_CHECK, min_arabic=LANGUAGE_MIN_ARABIC,
                 language_recheck_s=LANGUAGE_RECHECK_S,
                 language_min_audio_s=LANGUAGE_MIN_AUDIO_S,
                 language_confirmations=LANGUAGE_CONFIRMATIONS,
                 language_confirm_after_s=LANGUAGE_CONFIRM_AFTER_S):
        if vad is None:
            from src.audio.vad import StreamingVad
            vad = StreamingVad()
        self.engine = engine
        self.vad = vad
        self.sr = sample_rate
        self.min_first_pass_s = min_first_pass_s
        self.min_utterance_speech_s = min_utterance_speech_s
        self.lead_in_s = lead_in_s
        self.end_silence_s = end_silence_s
        self.tail_pad_s = tail_pad_s
        self.max_buffer_s = max_buffer_s
        self.hard_max_buffer_s = hard_max_buffer_s
        self.keep_after_hard_trim_s = keep_after_hard_trim_s
        self.prompt_chars = prompt_chars
        self.line_max_chars = line_max_chars
        self.commit_tolerance_s = commit_tolerance_s
        self.echo_max_words = echo_max_words
        self.tentative_guard_s = tentative_guard_s
        self.language_check = language_check
        self.min_arabic = min_arabic
        self.language_recheck_s = language_recheck_s
        self.language_min_audio_s = language_min_audio_s
        self.language_confirmations = language_confirmations
        self.language_confirm_after_s = language_confirm_after_s
        self._foreign_votes = 0

        self._arabic = True
        self._language_checked_at: int | None = None   # stream sample of the last check
        self.last_arabic_probability = 1.0
        self._audio = np.zeros(0, dtype=np.float32)
        self._start = 0                  # stream sample index of _audio[0]
        self._in_buffer: list[Word] = []  # committed, audio still in the buffer
        self._line: list[Word] = []       # committed words of the current line
        self._tentative: list[Word] = []  # last pass's words after the committed ones
        self._history: list[str] = []     # committed words whose audio is gone: the prompt
        self._last_committed_end = 0.0
        self._segments: list[Segment] = []
        self.stats: Counter = Counter()

    # ── Time ───────────────────────────────────────────────────────────

    @property
    def fed(self) -> int:
        """Samples fed so far: the stream's 'now', in samples."""
        return self._start + len(self._audio)

    @property
    def now(self) -> float:
        return self.fed / self.sr

    @property
    def buffer_seconds(self) -> float:
        return len(self._audio) / self.sr

    # ── Driving ────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray):
        if audio.size == 0:
            return
        self._audio = np.concatenate([self._audio, audio.astype(np.float32, copy=False)])
        self.vad.feed(audio)

    def step(self) -> Update | None:
        """Run a pass if one is due. None when nothing ran and nothing changed."""
        scored = min(self.fed, self.vad.scored_until)
        last_speech = self.vad.last_speech_end(self._start, scored)

        if last_speech is None:
            if self._line or self._tentative:
                return self._close_utterance(None)
            self._keep_last(self.lead_in_s)
            return None

        if (scored - last_speech) / self.sr >= self.end_silence_s:
            return self._close_utterance(last_speech)

        starting = not self._tentative and not self._in_buffer
        if starting and self.vad.speech_samples(self._start, scored) / self.sr < self.min_first_pass_s:
            return None

        if not self._is_arabic():
            return self._drop_foreign()
        return self._pass()

    def finish(self) -> Update | None:
        """End of stream: close whatever utterance is open."""
        scored = min(self.fed, self.vad.scored_until)
        return self._close_utterance(self.vad.last_speech_end(self._start, scored))

    # ── Language ───────────────────────────────────────────────────────

    def _is_arabic(self) -> bool:
        """Whisper's language detection, once the buffer is long enough to trust it.

        A "not Arabic" verdict counts only when LANGUAGE_CONFIRMATIONS checks in
        a row agree; after a first one the next check comes sooner.
        """
        if not self.language_check:
            return True
        if self.buffer_seconds < self.language_min_audio_s:
            return self._arabic         # too little to judge: keep the last verdict
        wait = self.language_confirm_after_s if self._foreign_votes else self.language_recheck_s
        due = (self._language_checked_at is None
               or self.fed - self._language_checked_at >= wait * self.sr)
        if due:
            p = self.engine.arabic_probability(self._audio[-10 * self.sr:])
            self.last_arabic_probability = p
            self._language_checked_at = self.fed
            self.stats["language_checks"] += 1
            if p >= self.min_arabic:
                self._foreign_votes, self._arabic = 0, True
            else:
                self.stats["foreign_checks"] += 1
                self._foreign_votes += 1
                self._arabic = self._foreign_votes < self.language_confirmations
        return self._arabic

    def _drop_foreign(self) -> Update | None:
        """Not Arabic: none of it is transcribed. What was already final stays final.

        The last few seconds of audio are kept, untranscribed, so the verdict can
        be taken again on enough of it — cutting back to a lead-in would leave
        the buffer too short to judge, and the verdict stuck until a pause.
        """
        had_tentative = bool(self._tentative)
        self._tentative = []
        finished = [self._end_line()]
        self._history.extend(w.text for w in self._in_buffer)
        self._in_buffer = []
        self._keep_last(self.language_min_audio_s + 1.0)
        finished = [line for line in finished if line is not None]
        if not finished and not had_tentative:
            return None
        return self._update(finished, None)

    # ── Passes ─────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray):
        prompt = self._prompt()
        result = self.engine.transcribe(audio, prompt or None)
        offset = self._start / self.sr
        words = [w.shifted(offset) for w in result.words]
        segments = [s.shifted(offset) for s in result.segments]
        return result, words, segments, prompt

    def _pass(self) -> Update:
        buffer_start = self._start / self.sr
        result, words, segments, prompt = self._transcribe(self._audio)
        self._segments = segments
        self.stats["passes"] += 1

        new = self._fresh(words)
        n = agreed_prefix(self._tentative, new)
        stable = new[:n]
        # A pass can stop short of the buffer's end — Whisper ends the decode
        # early, or a loop is cut. It did not hear the words after that point;
        # it did not un-hear them either. The last pass's guesses there stay on
        # screen, still tentative, until a pass reaches them. Erasing them is
        # what made words appear and vanish on fast speech (2026-09-27).
        heard_until = max((w.end for w in words), default=buffer_start)
        unreached = [w for w in self._tentative if w.start >= heard_until - self.commit_tolerance_s]
        self.stats["unreached_kept"] += len(unreached)
        self._tentative = new[n:] + unreached
        finished = self._commit(stable)
        finished += self._trim()

        report = PassReport(
            at=self.now, buffer_start=buffer_start, latency_s=result.latency_s,
            heard=" ".join(w.text for w in words), dropped=tuple(result.dropped),
            committed=tuple(stable), tentative=tuple(self._tentative),
            final=False, prompt=prompt,
        )
        return self._update(finished, report)

    def _close_utterance(self, last_speech: int | None) -> Update | None:
        """The speaker stopped: hear the utterance once more, commit it all, close the line."""
        report = None
        scored = min(self.fed, self.vad.scored_until)
        speech_s = self.vad.speech_samples(self._start, scored) / self.sr
        underway = bool(self._tentative or self._in_buffer)

        foreign = False
        if last_speech is not None and (underway or speech_s >= self.min_utterance_speech_s):
            foreign = not self._is_arabic()
        if foreign:
            self.stats["foreign_utterances"] += 1
            new = []
        elif last_speech is not None and (underway or speech_s >= self.min_utterance_speech_s):
            buffer_start = self._start / self.sr
            cut = min(self.fed, last_speech + int(self.tail_pad_s * self.sr))
            result, words, segments, prompt = self._transcribe(self._audio[: cut - self._start])
            self.stats["passes"] += 1
            new = self._fresh(words)
            report = PassReport(
                at=self.now, buffer_start=buffer_start, latency_s=result.latency_s,
                heard=" ".join(w.text for w in words), dropped=tuple(result.dropped),
                committed=tuple(new), tentative=(), final=True, prompt=prompt,
            )
        elif last_speech is not None:
            # A click or a cough: not worth a pass, and Whisper would name it.
            self.stats["blips"] += 1
            new = []
        else:
            # No speech left in the buffer to hear them again: keep what was heard.
            new = list(self._tentative)

        self._tentative = []
        finished = self._commit(new)
        finished.append(self._end_line())
        # Every committed word has now left the buffer; only a lead-in stays.
        self._history.extend(w.text for w in self._in_buffer)
        self._in_buffer = []
        self._keep_last(self.lead_in_s)
        self._language_checked_at = None          # the next utterance is checked afresh
        self._arabic, self._foreign_votes = True, 0   # ...and presumed Arabic until then

        finished = [line for line in finished if line is not None]
        if not finished and report is None:
            return None
        return self._update(finished, report)

    # ── Commit bookkeeping ─────────────────────────────────────────────

    def _fresh(self, words: list[Word]) -> list[Word]:
        """Drop words that are the already-committed ones heard again."""
        end = self._last_committed_end
        cutoff = end - self.commit_tolerance_s
        # A word starting before the cutoff is a committed one heard again —
        # unless most of it lies past the committed end and the word before it
        # is the last committed one, heard again at its own time: then it is
        # the next word, its start pulled early by Whisper's wobble on fast
        # speech. Dropping it lost words that had been shown grey (2026-10-07).
        # Without that word before it, it is the last one itself heard late,
        # perhaps spelled anew (تهموا → تهمواء): keeping it doubled the word.
        new = [w for i, w in enumerate(words)
               if w.start > cutoff or ((w.start + w.end) / 2 > end and i > 0
                                       and self._at_last_committed(words[i - 1]))]
        if new and new[0].start <= cutoff:
            self.stats["early_words_kept"] += 1
        before = words[:words.index(new[0])] if new else words

        # A committed word can come back at the head of the new words: its time
        # pushed late (a word cut at the trim point does this), or simply past
        # the cutoff. If the new head repeats the committed tail, it is that
        # echo — longest match first. Unless this pass already heard those
        # words again, at their own time, just before it: then the speaker
        # said them twice ("ايه ايه"), and dropping the second lost it.
        if new and self._in_buffer and abs(new[0].start - end) < 1.0:
            tail = [compare_key(w.text) for w in self._in_buffer[-self.echo_max_words:]]
            # Only the committed words this pass has not heard again can echo:
            # in "ايه ايه ايه" a longest match would take the real ones too.
            unheard = max(1, len(self._in_buffer) - len(before))
            for n in range(min(len(tail), len(new), unheard), 0, -1):
                if tail[-n:] == [compare_key(w.text) for w in new[:n]]:
                    if self._heard_again(before, n):
                        self.stats["repeats_kept"] += n
                    else:
                        self.stats["echo_words"] += n
                        new = new[n:]
                    break
        return new

    def _heard_again(self, before: list[Word], n: int) -> bool:
        """Whether `before` ends with the last n committed words, the last one at its own time."""
        if len(before) < n or ([compare_key(w.text) for w in before[-n:]]
                               != [compare_key(w.text) for w in self._in_buffer[-n:]]):
            return False
        return self._at_last_committed(before[-1])

    def _at_last_committed(self, word: Word) -> bool:
        """Whether `word` sits where the last committed word still in the buffer is."""
        if not self._in_buffer:
            return False
        last = self._in_buffer[-1]
        middle = (word.start + word.end) / 2
        return last.start - self.commit_tolerance_s <= middle <= last.end + self.commit_tolerance_s

    def _commit(self, words: list[Word]) -> list[Line | None]:
        finished: list[Line | None] = []
        for w in words:
            self._line.append(w)
            self._in_buffer.append(w)
            self._last_committed_end = max(self._last_committed_end, w.end)
            self.stats["words"] += 1
            if len(" ".join(x.text for x in self._line)) >= self.line_max_chars:
                finished.append(self._end_line())
        return finished

    def _end_line(self) -> Line | None:
        if not self._line:
            return None
        line = Line(
            " ".join(w.text for w in self._line),
            self._line[0].start, self._line[-1].end, tuple(self._line),
        )
        self._line = []
        self.stats["lines"] += 1
        return line

    # ── The buffer ─────────────────────────────────────────────────────

    def _trim(self) -> list[Line | None]:
        finished: list[Line | None] = []
        if self.buffer_seconds > self.max_buffer_s and self._in_buffer:
            self._drop_before(self._cut_point())

        if self.buffer_seconds > self.hard_max_buffer_s:
            # Nothing agreed for too long. Commit what there is rather than
            # let the buffer reach Whisper's 30 s limit.
            self.stats["hard_trims"] += 1
            finished += self._commit(self._tentative)
            self._tentative = []
            keep_from = self.fed - int(self.keep_after_hard_trim_s * self.sr)
            if self._in_buffer:
                keep_from = max(int(self._last_committed_end * self.sr), keep_from)
            self._drop_before(keep_from)
        return finished

    def _cut_point(self) -> int:
        """A pause before the last committed word's end; else a committed segment end."""
        limit = int(self._last_committed_end * self.sr)
        pause = self.vad.last_pause(self._start, limit, CUT_PAUSE_FRAMES)
        # A pause right at the buffer's head would barely shorten it.
        if pause is not None and pause >= self._start + self.sr:
            self.stats["cuts_at_pause"] += 1
            return pause
        ends = [seg.end for seg in self._segments if seg.end <= self._last_committed_end + 1e-6]
        cut = max(ends) if ends else self._last_committed_end
        if cut * self.sr <= self._start:
            cut = self._last_committed_end
        self.stats["cuts_at_segment"] += 1
        return int(cut * self.sr)

    def _keep_last(self, seconds: float):
        self._drop_before(self.fed - int(seconds * self.sr))

    def _drop_before(self, sample: int):
        sample = min(max(sample, self._start), self.fed)
        if sample <= self._start:
            return
        self._audio = self._audio[sample - self._start:]
        self._start = sample
        t = sample / self.sr + 1e-6
        gone = [w for w in self._in_buffer if w.end <= t]
        if gone:
            self._in_buffer = [w for w in self._in_buffer if w.end > t]
            self._history.extend(w.text for w in gone)
        del self._history[:-400]
        self.vad.forget_before(sample)

    def _prompt(self) -> str:
        if self.prompt_chars <= 0 or not self._history:
            return ""
        text = " ".join(self._history)
        if len(text) > self.prompt_chars:
            text = text[-self.prompt_chars:].split(" ", 1)[-1]
        return text

    def _shown_tentative(self) -> list[Word]:
        """The tentative words worth showing: not the half-heard one at the very end."""
        if self.tentative_guard_s <= 0:
            return list(self._tentative)
        limit = self.now - self.tentative_guard_s
        return [w for w in self._tentative if w.end <= limit]

    def _update(self, finished, report) -> Update:
        shown = self._shown_tentative()
        return Update(
            finished=tuple(finished),
            committed=" ".join(w.text for w in self._line),
            tentative=" ".join(w.text for w in shown),
            line_words=tuple(self._line),
            tentative_words=tuple(shown),
            report=report,
            now=self.now,
        )
