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


# ── Lines ──────────────────────────────────────────────────────────────

def test_a_long_line_is_split_without_losing_words():
    words = script(30)
    s = Streamer(ScriptedEngine(words), FakeVad([(0, 12.0)]), line_max_chars=12)
    updates = run(s, 14.0, chunk_s=0.5)
    lines = [line for u in updates for line in u.finished]
    assert len(lines) > 3
    assert all(len(line.text) <= 12 + 4 for line in lines)
    assert committed_words(updates) == [w.text for w in words]
