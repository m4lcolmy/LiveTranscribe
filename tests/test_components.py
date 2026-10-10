"""Gain, VAD, Arabic keys and the segment gates."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.engine import Transcription, compression_ratio, gate_window, is_phantom, rejection
from src.audio.gain import AutoGain
from src.core.arabic import compare_key, normalize

SR = 16000


def tone(seconds, peak, freq=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (peak * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ── AutoGain ───────────────────────────────────────────────────────────

def test_quiet_playback_is_brought_up_to_the_target():
    # Local-Live-Captions' failing case: a normal recording at 0.05x, peak ~0.01.
    out = AutoGain()(tone(3.0, 0.0096))
    assert np.abs(out[-SR:]).max() == pytest.approx(0.0096 * 40, rel=0.02)   # capped at 40x
    out = AutoGain()(tone(3.0, 0.05))
    assert np.abs(out[-SR:]).max() == pytest.approx(0.5, rel=0.02)


def test_loud_playback_is_left_alone():
    audio = tone(2.0, 0.8)
    out = AutoGain()(audio)
    assert np.allclose(out, audio[: len(out)])


def test_digital_silence_and_noise_floor_are_not_amplified():
    gain = AutoGain()
    out = gain(tone(2.0, 0.001))
    assert gain.gain == 1.0
    assert np.abs(out).max() <= 0.001 + 1e-6


def test_a_pause_does_not_pump_the_gain():
    gain = AutoGain()
    gain(tone(2.0, 0.05))
    before = gain.gain
    gain(tone(1.0, 0.0005))            # one second of near-silence between sentences
    assert gain.gain == pytest.approx(before)


def test_the_result_does_not_depend_on_how_the_stream_is_sliced():
    audio = np.concatenate([tone(1.5, 0.02), tone(2.0, 0.3), tone(1.0, 0.01)])
    whole = AutoGain()(audio)
    g = AutoGain()
    pieces = []
    for i in range(0, len(audio), 777):
        pieces.append(g(audio[i:i + 777]))
    pieces.append(g.flush())
    sliced = np.concatenate(pieces)
    assert np.allclose(sliced[: len(whole)], whole)


# ── VAD ────────────────────────────────────────────────────────────────

def test_vad_scores_a_stream_in_pieces_exactly_as_in_one_go():
    from src.audio.vad import StreamingVad

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)

    one = StreamingVad()
    one.feed(audio)
    pieces = StreamingVad()
    for i in range(0, len(audio), 3001):
        pieces.feed(audio[i:i + 3001])

    assert one.scored_until == pieces.scored_until
    assert np.allclose(one._probs, pieces._probs, atol=1e-5)


def test_vad_hears_no_speech_in_silence():
    from src.audio.vad import StreamingVad

    vad = StreamingVad()
    vad.feed(np.zeros(SR * 2, dtype=np.float32))
    assert vad.speech_samples(0, vad.scored_until) == 0
    assert vad.last_speech_end(0, vad.scored_until) is None


def test_vad_forgetting_keeps_later_frames_on_their_own_index():
    from src.audio.vad import FRAME, StreamingVad

    vad = StreamingVad()
    vad.feed(np.zeros(FRAME * 10, dtype=np.float32))
    vad.forget_before(FRAME * 50)                 # past everything scored
    vad.feed(np.zeros(FRAME * 5, dtype=np.float32))
    assert vad.scored_until == FRAME * 15


# ── Arabic keys ────────────────────────────────────────────────────────

@pytest.mark.parametrize("a, b", [
    ("إلى", "الى"),
    ("مدرسة", "مدرسه"),
    ("على", "علي"),
    ("القناة.", "القناة"),
    ("، وقال", "وقال"),
    ("كِتَابٌ", "كتاب"),
    ("الســلام", "السلام"),
])
def test_spellings_of_one_word_share_a_key(a, b):
    assert compare_key(a) == compare_key(b)


def test_different_words_keep_different_keys():
    assert compare_key("فلهم") != compare_key("ولهم")


def test_normalize_keeps_punctuation_for_display():
    assert normalize("قال: نعم.") == "قال: نعم."


# ── Gates ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "اشتركوا في القناة",
    " ترجمة نانسي قنقر",
    "شكراً على المشاهدة!",
    "اشتركوا في القناة. شكرا للمشاهدة",
    "موسيقى",
])
def test_whisper_phantom_lines_are_recognised(text):
    assert is_phantom(text)


@pytest.mark.parametrize("text", [
    "قال إن الحكومة ستعلن القرار غدا",
    "اشتركوا في القناة وتابعوا البرنامج",   # contains a phantom, but is more than one
    "",
])
def test_real_speech_is_not_a_phantom(text):
    assert not is_phantom(text)


SPEECH = "قال رئيس الوزراء في مؤتمر صحفي إن الحكومة ستعلن القرار عام ألفين وستة وعشرين"


def test_gates():
    assert rejection(SPEECH, -0.3, 0.1) is None
    assert rejection("كتبك " * 20, -0.3, 0.1)[0] == "repetitive"
    assert rejection(SPEECH, -1.4, 0.1)[0] == "low_logprob"
    assert rejection("ترجمة نانسي قنقر", -0.3, 0.1)[0] == "phantom"
    # High no_speech_prob alone is not a reason: speech over music scores high.
    assert rejection(SPEECH, -0.3, 0.9) is None


def test_real_arabic_is_far_from_the_repetition_limit():
    # 36 words of news reading: a whole 8 s buffer of fast speech.
    long_speech = ("قال رئيس الوزراء في مؤتمر صحفي عقده اليوم في العاصمة إن الحكومة ستعلن خلال "
                   "الأسبوع المقبل عن حزمة من القرارات الاقتصادية التي تهدف إلى خفض الأسعار "
                   "ودعم الأسر ذات الدخل المحدود وتشجيع الاستثمار في القطاعات الإنتاجية")
    assert compression_ratio(long_speech) < 2.4 < compression_ratio("ايه " * 30)


def window(*texts, cr=1.2, lp=-0.3, nsp=0.1):
    """One decode window as faster-whisper yields it: every segment carries the window's scores."""
    segs, t = [], 0.0
    for text in texts:
        words = []
        for token in text.split():
            words.append(SimpleNamespace(word=" " + token, start=t, end=t + 0.3, probability=0.9))
            t += 0.4
        segs.append(SimpleNamespace(text=" " + text, words=words, seek=0, compression_ratio=cr,
                                    avg_logprob=lp, no_speech_prob=nsp))
    return segs


def gated(segs):
    result = Transcription()
    gate_window(segs, result)
    return " ".join(w.text for w in result.words), result.dropped


def test_one_looping_segment_does_not_take_the_windows_sentences_with_it():
    # Replayed 2026-09-27 (al-Sanani): a window scored 3.75 because one of its
    # segments looped; its other sentences were right and were dropped with it.
    text, dropped = gated(window("اللي ما يهاجر اللي بلادن فيها كفر", "وين يروح المناطق",
                                 "انا " * 12, cr=3.75))
    assert text == "اللي ما يهاجر اللي بلادن فيها كفر وين يروح المناطق انا"
    assert dropped[0][0] == "loop_cut"


def test_a_loop_spread_over_segments_is_cut_to_one_copy():
    # "ايه صحيح" thirteen times, one segment each: no segment repeats itself.
    text, _ = gated(window("وين هي المناطق", *["ايه صحيح"] * 13, cr=7.38))
    assert text == "وين هي المناطق ايه صحيح"


def test_a_repetitive_window_without_a_loop_keeps_everything():
    # Live 2026-09-27: fast speech with its own repeats scored 2.50 as a window,
    # with no loop in it, and all of its segments were dropped.
    segs = window("اللي ما يهاجر اللي بلادن فيها كفر", "ايه صحيح", "وين يروح",
                  "المناطق اللي يسيطر عليهم جاهدون", cr=2.50)
    text, dropped = gated(segs)
    assert text.split() == " ".join(s.text for s in segs).split()
    assert dropped == []


def test_a_low_confidence_window_is_still_dropped_whole():
    text, dropped = gated(window("كلام غير واضح", "ابدا", lp=-1.3))
    assert text == "" and [d[0] for d in dropped] == ["low_logprob", "low_logprob"]


# ── Loops and the token budget ─────────────────────────────────────────

from src.audio.engine import WhisperEngine, Word, cut_loop   # noqa: E402


def words_of(text):
    return [Word(t, i * 0.4, i * 0.4 + 0.3) for i, t in enumerate(text.split())]


def test_a_decode_loop_keeps_the_words_before_it_and_one_copy():
    # Measured live: this ayah's tail looped until the 448-token limit.
    looped = words_of("وذلك الفوز العظيم ومن يعص الله ورسوله " + "يدخله نارا خالدا فيها " * 5)
    kept = cut_loop(looped)
    assert " ".join(w.text for w in kept) == "وذلك الفوز العظيم ومن يعص الله ورسوله يدخله نارا خالدا فيها"


def test_a_phrase_said_twice_is_not_a_loop():
    assert cut_loop(words_of("يدخله نارا يدخله نارا خالدا فيها")) is None
    assert cut_loop(words_of("قال الرئيس إن الحكومة ستعلن القرار")) is None


def test_the_token_budget_grows_with_the_buffer_and_leaves_room_for_the_prompt():
    class Tok:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=list(range(len(text.split()) * 3)))

    engine = WhisperEngine.__new__(WhisperEngine)
    engine.model = SimpleNamespace(hf_tokenizer=Tok())
    assert engine._token_budget(2.0, None) < engine._token_budget(10.0, None)
    assert engine._token_budget(10.0, None) == 15 * 10 + 24
    # A long prompt must never push prompt + budget past Whisper's 448 tokens.
    long_prompt = " ".join(["كلمة"] * 200)          # 600 tokens, cut to 223 by faster-whisper
    assert engine._token_budget(25.0, long_prompt) + 1 + 223 + 3 <= 448


# ── References from human captions ─────────────────────────────────────

from src.evaluation import load_cues, load_reference, score_text   # noqa: E402

SRT = """1
00:00:01,000 --> 00:00:03,500
[موسيقى]

2
00:00:04,000 --> 00:00:06,000
قال رئيس الوزراء
في مؤتمر صحفي

3
00:00:06,500 --> 00:00:09,000
- إن الحكومة ستعلن القرار عام ٢٠٢٦.

4
00:01:10,000 --> 00:01:12,000
<i>وانتهى المؤتمر</i>
"""


def test_srt_cues_keep_their_times_and_lose_annotations_and_tags(tmp_path):
    path = tmp_path / "ref.srt"
    path.write_text(SRT, encoding="utf-8")
    cues = load_cues(path)
    assert [c[:2] for c in cues] == [(1.0, 3.5), (4.0, 6.0), (6.5, 9.0), (70.0, 72.0)]
    assert cues[0][2] == ""                                  # [موسيقى] is not speech
    assert cues[1][2] == "قال رئيس الوزراء في مؤتمر صحفي"   # a two-line cue is one text
    assert cues[3][2] == "وانتهى المؤتمر"


def test_a_window_takes_only_whole_cues_and_reports_their_span(tmp_path):
    path = tmp_path / "ref.srt"
    path.write_text(SRT, encoding="utf-8")
    text, start, end = load_reference(path, 3.8, 10.0)
    assert (start, end) == (4.0, 9.0)
    assert text.startswith("قال رئيس الوزراء") and "المؤتمر" not in text


def test_digits_score_the_same_in_either_script():
    assert score_text("ستعلن القرار عام 2026", "ستعلن القرار عام ٢٠٢٦").cer == 0.0


def test_swallowed_counts_grey_words_the_final_text_ends_without_once_each():
    from src.audio.engine import Word
    from src.audio.streamer import Line, Update
    from src.evaluation import SessionRecord

    def shown(tentative, finished=()):
        words = tuple(Word(t, a, b) for t, a, b in tentative)
        lines = tuple(Line(" ".join(w.text for w in ws), ws[0].start, ws[-1].end, ws)
                      for ws in (tuple(Word(t, a, b) for t, a, b in line) for line in finished))
        return Update(lines, "", "", (), words, None, 0.0)

    record = SessionRecord(duration_s=60.0)
    record.add(1.0, shown([("قال", 0.0, 0.4), ("الأظي", 0.5, 0.9), ("في", 1.0, 1.2)]))
    record.add(2.0, shown([("قال", 0.0, 0.4), ("الأظي", 0.5, 0.9), ("في", 1.0, 1.2)]))
    record.add(3.0, shown([], [[("قال", 0.0, 0.4), ("العظيم", 0.5, 0.9)]]))
    # الأظي: replaced by العظيم at its time, counted once over two passes; في: gone.
    assert record.swallowed() == (1, 1)
    assert record.summary()["swallowed_per_min"] == 2.0


# ── Session log ────────────────────────────────────────────────────────

def test_a_new_session_log_counts_only_its_own_session(tmp_path):
    # A settings change reloads the model and opens a new log in the same
    # process; its summary once repeated the last session's drops.
    from src.core.debug import SessionLog

    log = SessionLog()
    log.open(tmp_path / "first.log")
    log.count("drop_repetitive", 63)
    log.pass_latencies.append(0.4)
    log.close()
    log.open(tmp_path / "second.log")
    assert "repetitive=0" in log.summary(1.0)
    assert "passes 0" in log.summary(1.0)
    log.close()
