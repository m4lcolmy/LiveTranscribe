"""The streaming policy against a scripted engine.

The fake engine knows the whole script — every word with its time — and
answers each pass with the words that lie inside the audio it was handed,
the way Whisper would if it were never wrong about a whole word. What it does
get wrong is what Whisper really gets wrong at a buffer edge: the word the
buffer ends in the middle of comes back mangled. So these tests are about the
policy, not the model: what gets committed, when, and whether any word is ever
lost or said twice.

The audio carries its own position: sample i of the stream has the value i,
so the engine can tell where in the stream the buffer it was handed begins.
"""

import numpy as np
import pytest

from src.audio.engine import Engine, Segment, Transcription, Word
from src.audio.streamer import Streamer, agreed_prefix

SR = 16000


class FakeVad:
    """Speech exactly where the test says it is."""

    threshold = 0.5

    def __init__(self, spans):
        self.spans = [(int(a * SR), int(b * SR)) for a, b in spans]
        self.fed = 0

    def feed(self, audio):
        self.fed += len(audio)

    @property
    def scored_until(self):
        return self.fed

    def speech_samples(self, start, end):
        return sum(max(0, min(b, end) - max(a, start)) for a, b in self.spans)

    def last_speech_end(self, start, end):
        ends = [min(b, end) for a, b in self.spans if a < end and b > start]
        return max(ends) if ends else None

    def forget_before(self, sample):
        pass

    def last_pause(self, start, end, min_frames=2):
        """Middle of the last gap between spans in [start, end) at least min_frames long."""
        edges = sorted((max(a, start), min(b, end)) for a, b in self.spans if a < end and b > start)
        gaps, cursor = [], start
        for a, b in edges:
            if a > cursor:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < end:
            gaps.append((cursor, end))
        gaps = [g for g in gaps if g[1] - g[0] >= min_frames * 512]
        return (gaps[-1][0] + gaps[-1][1]) // 2 if gaps else None

    def probability(self, start, end):
        return 1.0 if self.speech_samples(start, end) else 0.0


class ScriptedEngine(Engine):
    """Answers with the script's words inside the buffer; mangles the one cut at its end.

    echo=True also re-hears a committed word that straddles the buffer's start,
    with its time pushed late — Whisper's tail echo, which the time filter
    alone lets through. Segment ends sit 0.15 s past their last word, inside the
    next word, so a cut at a segment end really does straddle a word.
    """

    def __init__(self, script, echo=False, segment_size=4):
        self.script = script
        self.echo = echo
        self.segment_size = segment_size
        self.calls = []

    def transcribe(self, audio, prompt=None):
        start = float(audio[0]) / SR
        end = start + len(audio) / SR
        self.calls.append((start, end, prompt or ""))
        words = []
        for w in self.script:
            if w.start >= start - 1e-6 and w.end <= end + 1e-6:
                words.append(Word(w.text, w.start - start, w.end - start))
            elif w.start < end < w.end and w.start >= start:
                words.append(Word(w.text + "~", w.start - start, end - start))
            elif self.echo and w.start < start < w.end:
                words.append(Word(w.text, 0.25, w.end - start + 0.1))
        segments = [
            Segment(chunk[0].start, chunk[-1].end + 0.15, " ".join(x.text for x in chunk))
            for chunk in (words[i:i + self.segment_size]
                          for i in range(0, len(words), self.segment_size))
        ]
        return Transcription(words=words, segments=segments)


def script(n, start=0.0, every=0.4, length=0.3):
    return [Word(f"w{i}", start + i * every, start + i * every + length) for i in range(n)]


def audio_between(a, b):
    return np.arange(a, b, dtype=np.float32)


def run(streamer, until_s, chunk_s=1.0):
    """Feed the stream in chunks, one step per chunk. Returns every Update."""
    updates = []
    fed = streamer.fed
    target = int(until_s * SR)
    while fed < target:
        nxt = min(target, fed + int(chunk_s * SR))
        streamer.feed(audio_between(fed, nxt))
        fed = nxt
        u = streamer.step()
        if u is not None:
            updates.append(u)
    return updates


def committed_words(updates):
    return [w.text for u in updates if u is not None for line in u.finished for w in line.words]


# ── Agreement ──────────────────────────────────────────────────────────

def test_first_pass_only_suggests_and_second_pass_commits():
    words = script(10)
    engine = ScriptedEngine(words)
    s = Streamer(engine, FakeVad([(0, 4.0)]))

    first = run(s, 2.0)                         # 2 s of speech: one pass
    assert len(engine.calls) == 1
    assert first[-1].committed == ""
    assert first[-1].tentative.startswith("w0 w1")

    second = run(s, 3.0)
    assert second[-1].committed.startswith("w0 w1 w2")


def test_a_word_cut_by_the_buffer_edge_is_never_committed():
    s = Streamer(ScriptedEngine(script(40)), FakeVad([(0, 16.0)]))
    updates = run(s, 16.0, chunk_s=0.7)
    for u in updates:
        assert "~" not in u.committed
        for line in u.finished:
            assert "~" not in line.text


def test_the_guard_hides_only_the_words_at_the_very_end_and_agreement_is_unchanged():
    words = script(10)
    plain = Streamer(ScriptedEngine(words), FakeVad([(0, 4.0)]), tentative_guard_s=0.0)
    guarded = Streamer(ScriptedEngine(words), FakeVad([(0, 4.0)]), tentative_guard_s=0.5)
    a, b = run(plain, 3.0), run(guarded, 3.0)
    assert b[-1].committed == a[-1].committed            # what becomes final is the same
    shown_a, shown_b = a[-1].tentative.split(), b[-1].tentative.split()
    assert shown_b == shown_a[: len(shown_b)] and len(shown_b) < len(shown_a)
    assert all(w.end <= guarded.now - 0.5 for w in b[-1].tentative_words)


def test_nothing_runs_before_enough_speech():
    engine = ScriptedEngine(script(3))
    s = Streamer(engine, FakeVad([(0, 5.0)]), min_first_pass_s=1.2)
    run(s, 1.0)
    assert engine.calls == []


def W(*texts):
    return [Word(t, i * 0.4, i * 0.4 + 0.3) for i, t in enumerate(texts)]


def test_agreement_is_a_common_prefix():
    assert agreed_prefix(W("a", "b", "c"), W("a", "b", "x")) == 2
    assert agreed_prefix([], W("a", "b")) == 0
    assert agreed_prefix(W("a", "b"), W("a", "b", "c", "d")) == 2


def test_one_unstable_word_does_not_hold_back_the_words_after_it():
    # Whisper spells one hard word differently each pass; three words after it agree.
    prev = W("ولهن", "نربع", "مما", "تركتم", "إن", "لم")
    cur = W("ولهن", "الربع", "مما", "تركتم", "إن", "لم", "يكن")
    n = agreed_prefix(prev, cur)
    assert [w.text for w in cur[:n]] == ["ولهن", "الربع", "مما", "تركتم", "إن", "لم"]


def test_a_disagreement_is_passed_over_only_when_enough_agrees_after_it():
    assert agreed_prefix(W("a", "x", "c", "d"), W("a", "y", "c", "d")) == 1       # 2 agree after: not enough
    assert agreed_prefix(W("a", "x", "y", "z", "c", "d", "e"),
                         W("a", "p", "q", "r", "c", "d", "e")) == 1                 # gap of 3: too long
    assert agreed_prefix(W("a", "x", "c", "d", "e"), W("a", "c", "d", "e")) == 4    # a dropped word


# ── Closing an utterance ───────────────────────────────────────────────

def test_silence_closes_the_line_with_every_word_once():
    words = script(10)                           # w9 ends at 3.9 s
    s = Streamer(ScriptedEngine(words), FakeVad([(0, 4.0)]))
    updates = run(s, 6.0, chunk_s=0.5)
    assert committed_words(updates) == [w.text for w in words]
    assert updates[-1].tentative == ""
    assert updates[-1].report is not None and updates[-1].report.final


class StopsShort(ScriptedEngine):
    """Whisper ending a decode early (or a loop cut, or every segment gated):
    on the given passes, nothing after `keep_s` seconds into the buffer."""

    def __init__(self, script, short_passes, keep_s):
        super().__init__(script)
        self.short_passes = short_passes
        self.keep_s = keep_s

    def transcribe(self, audio, prompt=None):
        result = super().transcribe(audio, prompt)
        if len(self.calls) in self.short_passes:
            result.words = [w for w in result.words if w.end <= self.keep_s]
            result.segments = [s for s in result.segments if s.end <= self.keep_s]
        return result


@pytest.mark.parametrize("keep_s", [0.0, 0.5])
def test_a_pass_that_stops_short_leaves_the_words_after_it_on_screen(keep_s):
    # Measured on fast speech (2026-09-27): grey words shown, then gone on the
    # next pass, then back on the one after — the pass had simply stopped early.
    words = script(10)
    s = Streamer(StopsShort(words, short_passes={3}, keep_s=keep_s), FakeVad([(0, 6.0)]),
                 tentative_guard_s=0.0)          # every grey word shown, the half-heard one too
    updates = run(s, 3.0)
    before = updates[-1].tentative
    assert before == "w5 w6 w7~"
    short = run(s, 4.0)[-1]
    assert short.tentative == before and short.committed == updates[-1].committed
    after = run(s, 5.0)[-1]
    assert after.committed.endswith("w5 w6")                 # the next full pass agrees with them
    updates += [short, after] + run(s, 8.0)
    assert committed_words(updates) == [w.text for w in words]


def test_finish_closes_speech_still_going_at_the_end_of_the_stream():
    words = script(10)
    s = Streamer(ScriptedEngine(words), FakeVad([(0, 4.0)]))
    updates = run(s, 4.0)
    updates.append(s.finish())
    assert committed_words(updates) == [w.text for w in words]


def test_a_blip_is_not_transcribed():
    engine = ScriptedEngine([Word("x", 1.0, 1.1)])
    s = Streamer(engine, FakeVad([(1.0, 1.1)]))
    updates = run(s, 3.0, chunk_s=0.5)
    assert engine.calls == []
    assert updates == []
    assert s.stats["blips"] == 1


def test_no_speech_never_reaches_the_model_and_the_buffer_stays_small():
    engine = ScriptedEngine([])
    s = Streamer(engine, FakeVad([]))
    run(s, 30.0)
    assert engine.calls == []
    assert s.buffer_seconds <= s.lead_in_s + 1e-6


def test_two_utterances_make_two_lines():
    words = script(5) + script(5, start=5.0)
    words = [Word(f"u{i}", w.start, w.end) for i, w in enumerate(words)]
    s = Streamer(ScriptedEngine(words), FakeVad([(0, 2.0), (5.0, 7.0)]))
    updates = run(s, 9.0, chunk_s=0.5)
    lines = [line for u in updates for line in u.finished]
    assert [line.text for line in lines] == ["u0 u1 u2 u3 u4", "u5 u6 u7 u8 u9"]


# ── Language ───────────────────────────────────────────────────────────

class Bilingual(ScriptedEngine):
    """Arabic except inside the given spans, where the audio is 'English'."""

    def __init__(self, script, english):
        super().__init__(script)
        self.english = english
        self.checks = 0

    def arabic_probability(self, audio):
        self.checks += 1
        end = (float(audio[-1]) + 1) / SR
        return 0.01 if any(a <= end - 0.5 <= b for a, b in self.english) else 0.95


def test_a_long_english_utterance_is_dropped_once_there_is_enough_audio_to_judge():
    engine = Bilingual(script(40), english=[(0, 20)])
    s = Streamer(engine, FakeVad([(0, 16.0)]), language_min_audio_s=6.0)
    updates = run(s, 19.0, chunk_s=0.5)
    words = committed_words(updates)
    assert s.stats["foreign_checks"] >= 1 and s.stats["foreign_utterances"] == 1
    # Before 6 s there was nothing to judge on; after it, nothing more gets through.
    assert all(float(w[1:]) * 0.4 < 8.0 for w in words)


def test_arabic_resumes_after_english_without_a_pause_between():
    # 0-12 s English, then Arabic straight on: no silence to close the utterance.
    words = [Word(f"e{i}", i * 0.4, i * 0.4 + 0.3) for i in range(30)] + \
            [Word(f"a{i}", 12.0 + i * 0.4, 12.3 + i * 0.4) for i in range(40)]
    engine = Bilingual(words, english=[(0, 12.0)])
    s = Streamer(engine, FakeVad([(0, 28.0)]), language_min_audio_s=6.0)
    updates = run(s, 30.0, chunk_s=0.5)
    arabic = [w for w in committed_words(updates) if w.startswith("a")]
    assert len(arabic) >= 30                     # the verdict came back, speech resumed


def test_one_foreign_verdict_alone_drops_nothing():
    class OneBadCheck(Bilingual):
        def arabic_probability(self, audio):
            self.checks += 1
            return 0.01 if self.checks == 1 else 0.95   # a single false alarm
    engine = OneBadCheck(script(40), english=[])
    s = Streamer(engine, FakeVad([(0, 16.0)]), language_min_audio_s=6.0)
    updates = run(s, 19.0, chunk_s=0.5)
    assert committed_words(updates) == [w.text for w in script(40)]


def test_a_short_utterance_is_never_judged():
    engine = Bilingual(script(10), english=[(0, 10)])
    s = Streamer(engine, FakeVad([(0, 4.0)]), language_min_audio_s=6.0)
    run(s, 7.0, chunk_s=0.5)
    assert engine.checks == 0


def test_arabic_after_english_is_transcribed_once_rechecked():
    words = script(5) + [Word(f"a{i}", 6.0 + i * 0.4, 6.3 + i * 0.4) for i in range(5)]
    engine = Bilingual(words, english=[(0, 3.0)])
    s = Streamer(engine, FakeVad([(0, 2.0), (6.0, 8.0)]), language_min_audio_s=1.0,
                 language_confirmations=1)
    updates = run(s, 10.0, chunk_s=0.5)
    assert committed_words(updates) == ["a0", "a1", "a2", "a3", "a4"]


def test_the_language_is_checked_at_the_start_and_then_only_every_few_seconds():
    engine = Bilingual(script(150), english=[])
    s = Streamer(engine, FakeVad([(0, 60.0)]), language_recheck_s=5.0, language_min_audio_s=1.0)
    run(s, 60.0)
    assert 10 <= engine.checks <= 14                     # ~60 s / 5 s, not once per pass


def test_the_check_can_be_turned_off():
    engine = Bilingual(script(10), english=[(0, 10)])
    s = Streamer(engine, FakeVad([(0, 4.0)]), language_check=False, language_min_audio_s=1.0)
    updates = run(s, 7.0, chunk_s=0.5)
    assert engine.checks == 0 and committed_words(updates)


# ── Long speech: trimming must lose nothing and repeat nothing ─────────

@pytest.mark.parametrize("pauses", [False, True])
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("chunk_s", [0.5, 1.0, 3.0])
def test_a_minute_of_unbroken_speech_loses_and_repeats_nothing(echo, chunk_s, pauses):
    words = script(150)                          # 0 – 60 s, no pause long enough to close
    engine = ScriptedEngine(words, echo=echo)
    # pauses: the VAD hears the 0.1 s gap between words, so cuts land in them;
    # otherwise speech is solid and cuts fall back to segment ends.
    spans = [(w.start, w.end) for w in words] if pauses else [(0, 60.0)]
    s = Streamer(engine, FakeVad(spans))

    updates = run(s, 60.0, chunk_s=chunk_s)
    updates.append(s.finish())

    assert committed_words(updates) == [w.text for w in words]
    assert s.stats["hard_trims"] == 0
    assert s.stats["cuts_at_pause" if pauses else "cuts_at_segment"] >= 1
    longest = max(end - start for start, end, _ in engine.calls)
    assert longest <= s.max_buffer_s + chunk_s + 1.0


def test_a_cut_at_a_pause_never_lands_inside_a_word():
    words = script(150)
    engine = ScriptedEngine(words)
    s = Streamer(engine, FakeVad([(w.start, w.end) for w in words]))
    run(s, 60.0)
    starts = sorted({start for start, _, _ in engine.calls if start > 0})
    assert starts, "a minute of speech must have been cut"
    for start in starts:
        assert not any(w.start < start < w.end for w in words), f"cut at {start:.3f}s is inside a word"


def test_the_prompt_only_holds_text_whose_audio_has_left_the_buffer():
    words = script(150)
    ends = {w.text: w.end for w in words}
    engine = ScriptedEngine(words)
    s = Streamer(engine, FakeVad([(0, 60.0)]))
    run(s, 60.0)

    prompted = [(start, prompt) for start, _, prompt in engine.calls if prompt]
    assert prompted, "a minute of speech must have trimmed and prompted at least once"
    for start, prompt in prompted:
        for text in prompt.split():
            assert ends[text] <= start + 1e-3, f"{text} is still in the buffer at {start:.2f}s"


def test_the_prompt_is_capped():
    engine = ScriptedEngine(script(150))
    s = Streamer(engine, FakeVad([(0, 60.0)]), prompt_chars=20)
    run(s, 60.0)
    assert all(len(prompt) <= 20 for _, _, prompt in engine.calls)


def test_no_agreement_at_all_is_cut_before_whisper_limit():
    """An engine that never says the same thing twice: the hard limit must still hold."""

    class Fickle(ScriptedEngine):
        def transcribe(self, audio, prompt=None):
            result = super().transcribe(audio, prompt)
            n = len(self.calls)
            result.words = [Word(f"{w.text}#{n}", w.start, w.end) for w in result.words]
            return result

    engine = Fickle(script(150))
    s = Streamer(engine, FakeVad([(0, 60.0)]))
    run(s, 60.0)
    assert s.stats["hard_trims"] >= 1
    assert max(end - start for start, end, _ in engine.calls) <= s.hard_max_buffer_s + 1.0 + 1e-6


# ── A new word, or a committed one heard again ─────────────────────────

def _after(committed):
    """A streamer whose last committed words, still in its buffer, are `committed`."""
    s = Streamer(ScriptedEngine([]), FakeVad([]))
    s._in_buffer = list(committed)
    s._last_committed_end = committed[-1].end
    return s


def test_a_word_said_twice_keeps_both():
    # Fast Gulf speech is full of "ايه ايه" and "لا لا لا". The second one,
    # right after a commit, was taken for the first heard again (2026-10-07).
    words = [Word("ايه" if 4 <= i < 12 else f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(20)]
    s = Streamer(ScriptedEngine(words), FakeVad([(0, 8.0)]))
    updates = run(s, 10.0, chunk_s=0.5)
    assert committed_words(updates) == [w.text for w in words]
    assert s.stats["repeats_kept"] >= 1


def test_a_new_word_whose_start_wobbles_before_the_committed_end_is_kept():
    s = _after([Word("أ", 4.6, 5.0), Word("ب", 5.0, 5.4)])
    heard = [Word("أ", 4.62, 5.02), Word("ب", 5.02, 5.36), Word("ج", 5.22, 5.8), Word("د", 5.8, 6.2)]
    assert [w.text for w in s._fresh(heard)] == ["ج", "د"]


def test_the_last_committed_word_heard_late_and_spelled_anew_is_not_a_new_word():
    # Corpus, Wz4BCyCl5nk at 1.35x: "تهموا" committed, heard on the next pass
    # as "تهمواء" with most of it past the committed end. It is no new word:
    # this pass has nothing at the committed word's own time before it.
    s = _after([Word("خيما", 4.6, 5.0), Word("تهموا", 5.0, 5.4)])
    heard = [Word("خيما", 4.6, 5.0), Word("تهمواء", 5.15, 5.75), Word("نعرفكم", 5.8, 6.2)]
    assert [w.text for w in s._fresh(heard)] == ["نعرفكم"]


def test_a_committed_word_heard_late_is_still_an_echo():
    s = _after([Word("أ", 4.6, 5.0), Word("ب", 5.0, 5.4)])
    late = [Word("أ", 4.6, 5.0), Word("ب", 5.32, 5.7), Word("ج", 5.7, 6.0)]
    assert [w.text for w in s._fresh(late)] == ["ج"]
    # The trim-point echo: the committed word is the buffer's first, heard late.
    s = _after([Word("ب", 5.0, 5.4)])
    assert [w.text for w in s._fresh([Word("ب", 5.45, 5.6), Word("ج", 5.7, 6.0)])] == ["ج"]


def test_a_repeat_said_twice_is_kept_and_heard_three_times_is_not():
    s = _after([Word("قال", 4.6, 5.0), Word("ايه", 5.0, 5.4)])
    twice = [Word("قال", 4.6, 5.0), Word("ايه", 5.0, 5.4), Word("ايه", 5.4, 5.8), Word("ج", 5.8, 6.1)]
    assert [w.text for w in s._fresh(twice)] == ["ايه", "ج"]
    # Both said already; this pass hears the first at its time and the second
    # late. The late one is the second heard again, not a third.
    s = _after([Word("ايه", 5.0, 5.4), Word("ايه", 5.4, 5.8)])
    again = [Word("ايه", 5.0, 5.42), Word("ايه", 5.72, 6.1), Word("ج", 6.1, 6.4)]
    assert [w.text for w in s._fresh(again)] == ["ج"]


# ── Lines ──────────────────────────────────────────────────────────────

def test_a_long_line_is_split_without_losing_words():
    words = script(30)
    s = Streamer(ScriptedEngine(words), FakeVad([(0, 12.0)]), line_max_chars=12)
    updates = run(s, 14.0, chunk_s=0.5)
    lines = [line for u in updates for line in u.finished]
    assert len(lines) > 3
    assert all(len(line.text) <= 12 + 4 for line in lines)
    assert committed_words(updates) == [w.text for w in words]
