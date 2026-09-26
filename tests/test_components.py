"""Gain, VAD, Arabic keys and the segment gates."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.engine import is_phantom, rejection
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


def seg(text="نص", cr=1.2, lp=-0.3, nsp=0.1):
    return SimpleNamespace(text=text, compression_ratio=cr, avg_logprob=lp, no_speech_prob=nsp)


def test_gates():
    assert rejection(seg()) is None
    assert rejection(seg(cr=3.1))[0] == "repetitive"
    assert rejection(seg(lp=-1.4))[0] == "low_logprob"
    assert rejection(seg(text="ترجمة نانسي قنقر"))[0] == "phantom"
    # High no_speech_prob alone is not a reason: speech over music scores high.
    assert rejection(seg(nsp=0.9)) is None


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
