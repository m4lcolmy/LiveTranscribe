"""Deepgram's streamer against a fake socket: what is sent, what is shown, what survives a drop.

No network. The fake socket records what the streamer sends and hands back
the Deepgram messages a test queues; the scripted VAD from test_streamer says
where the speech is. Deepgram's times count only the audio it was sent, so
every test that checks a time checks the map back to the stream's.
"""

import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.deepgram import (
    Closed, DeepgramEngine, DeepgramError, DeepgramStreamer, check_key, refusal,
)
from src.audio.worker import Pipeline
from tests.test_streamer import FakeVad

SR = 16000
CHUNK = 1600          # 0.1 s, as pw-record delivers it
KEY = "dg-secret-key-1234"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeSocket:
    def __init__(self, on_finalize=()):
        self.audio = bytearray()
        self.messages: list[str] = []
        self.inbox: deque = deque()
        self.on_finalize = list(on_finalize)
        self.broken = False
        self.closed = False

    def send(self, data):
        if self.broken:
            raise Closed("connection lost")
        if isinstance(data, bytes):
            self.audio += data
        else:
            kind = json.loads(data)["type"]
            self.messages.append(kind)
            if kind == "Finalize":
                self.inbox.extend(self.on_finalize)

    def recv(self, timeout=0.0):
        if self.broken:
            raise Closed("connection lost")
        return self.inbox.popleft() if self.inbox else None

    def close(self):
        self.closed = True

    @property
    def seconds(self):
        return len(self.audio) / 2 / SR


class Connector:
    """Hands out a new FakeSocket per connection, or raises what it is told to."""

    def __init__(self, error=None, **socket_args):
        self.sockets: list[FakeSocket] = []
        self.calls = []
        self.error = error
        self.socket_args = socket_args

    def __call__(self, url, api_key):
        self.calls.append((url, api_key))
        if self.error is not None:
            raise self.error
        self.sockets.append(FakeSocket(**self.socket_args))
        return self.sockets[-1]

    @property
    def socket(self):
        return self.sockets[-1]


def results(words, final, from_finalize=False, start=0.0):
    """A Deepgram Results message; words are (text, start, end) on the connection's clock."""
    end = max((b for _, _, b in words), default=start)
    return json.dumps({
        "type": "Results", "is_final": final, "speech_final": False, "from_finalize": from_finalize,
        "start": start, "duration": end - start,
        "channel": {"alternatives": [{
            "transcript": " ".join(t for t, _, _ in words),
            "words": [{"word": t, "punctuated_word": t, "start": a, "end": b, "confidence": 0.9}
                      for t, a, b in words],
        }]},
    })


def make(spans, connector=None, **kw):
    clock = Clock()
    connector = connector or Connector()
    streamer = DeepgramStreamer(DeepgramEngine(KEY), FakeVad(spans), connect=connector,
                                clock=clock, **kw)
    return streamer, connector, clock


def play(streamer, clock, until_s, updates=None):
    """Feed silence-valued audio up to `until_s` of stream, stepping after every chunk."""
    while streamer.fed < int(until_s * SR):
        streamer.feed(np.zeros(CHUNK, dtype=np.float32))
        clock.t += CHUNK / SR
        update = streamer.step()
        if updates is not None and update is not None:
            updates.append(update)


# ── What is sent ───────────────────────────────────────────────────────

def test_only_speech_is_sent_with_a_lead_in_and_then_finalized():
    s, conn, clock = make([(1.0, 3.0)])
    play(s, clock, 0.9)
    assert conn.sockets == []                          # silence: not even a connection
    play(s, clock, 5.0)
    sock = conn.socket
    # From 0.3 s before the speech to END_SILENCE_S after it: 0.7 .. 3.7 s.
    assert sock.seconds == pytest.approx(3.0, abs=0.01)
    assert sock.messages == ["Finalize"]
    assert s._spans == [(0, int(0.7 * SR))]


def test_everything_is_sent_when_asked_to():
    s, conn, clock = make([(1.0, 2.0)], send_silence=True)
    play(s, clock, 4.0)
    assert conn.socket.seconds == pytest.approx(4.0, abs=0.01)


def test_the_key_goes_in_the_header_only():
    engine = DeepgramEngine(KEY, "ar-EG")
    assert KEY not in engine.url() and KEY not in repr(engine)
    assert "language=ar-EG" in engine.url() and "model=nova-3" in engine.url()
    s, conn, clock = make([(0.5, 1.5)])
    play(s, clock, 1.0)
    assert conn.calls[0][1] == KEY and KEY not in conn.calls[0][0]


# ── What is shown ──────────────────────────────────────────────────────

def test_interim_words_are_tentative_final_ones_committed_and_finalize_ends_the_line():
    s, conn, clock = make([(1.0, 4.0)])
    play(s, clock, 2.0)
    sock = conn.socket
    sock.inbox.append(results([("بسم", 0.3, 0.7)], final=False))
    update = s.step()
    assert (update.committed, update.tentative) == ("", "بسم")

    sock.inbox.append(results([("بسم", 0.3, 0.7), ("الله", 0.8, 1.2)], final=True))
    update = s.step()
    assert (update.committed, update.tentative, update.finished) == ("بسم الله", "", ())
    # Deepgram's 0.3 s is 0.3 s after the first audio it got, which began at 0.7 s.
    assert update.line_words[0].start == pytest.approx(1.0)

    sock.on_finalize = [results([("الرحمن", 1.3, 1.9)], final=True, from_finalize=True)]
    updates = []
    play(s, clock, 5.0, updates)
    lines = [line for u in updates for line in u.finished]
    assert [line.text for line in lines] == ["بسم الله الرحمن"]
    assert lines[0].start == pytest.approx(1.0) and lines[0].end == pytest.approx(2.6)


def test_a_finalize_never_answered_closes_the_line_with_what_was_heard():
    s, conn, clock = make([(1.0, 2.0)])
    play(s, clock, 1.5)
    conn.socket.inbox.append(results([("مرحبا", 0.3, 0.8)], final=False))
    updates = []
    play(s, clock, 3.0, updates)
    assert "Finalize" in conn.socket.messages
    assert not any(u.finished for u in updates)
    play(s, clock, 5.0, updates)                        # DEEPGRAM_FINALIZE_WAIT_S later
    assert [line.text for u in updates for line in u.finished] == ["مرحبا"]


def test_pause_asks_for_the_last_words_then_closes_the_connection():
    reply = results([("السلام", 0.3, 0.9)], final=True, from_finalize=True)
    s, conn, clock = make([(1.0, 9.0)], Connector(on_finalize=[reply]))
    play(s, clock, 2.0)
    update = s.finish()
    assert [line.text for line in update.finished] == ["السلام"]
    assert conn.socket.messages[-2:] == ["Finalize", "CloseStream"] and conn.socket.closed
    assert s.finish() is None                           # nothing left to say


def test_long_lines_are_broken_like_the_whisper_ones():
    s, conn, clock = make([(1.0, 9.0)], line_max_chars=9)
    play(s, clock, 2.0)
    conn.socket.inbox.append(results([("كلمة", 0.1, 0.2), ("أخرى", 0.3, 0.4),
                                      ("ثالثة", 0.5, 0.6)], final=True))
    update = s.step()
    assert [line.text for line in update.finished] == ["كلمة أخرى"]
    assert update.committed == "ثالثة"


# ── The connection ─────────────────────────────────────────────────────

def test_a_drop_resends_the_utterance_and_final_words_are_not_repeated():
    s, conn, clock = make([(1.0, 9.0)])
    play(s, clock, 3.0)
    first = conn.socket
    first.inbox.append(results([("أ", 0.3, 0.8), ("ب", 0.9, 1.5)], final=True))
    assert s.step().committed == "أ ب"

    first.broken = True
    play(s, clock, 3.1)
    assert first.closed and s.problem == "connection lost — trying again" and not s.fatal
    play(s, clock, 3.8)                                 # past the first retry wait
    second = conn.socket
    assert second is not first and s.problem is None
    # Sent again from the utterance's start, 0.7 s.
    assert second.seconds == pytest.approx(s.now - 0.7, abs=0.01)

    second.inbox.append(results([("أ", 0.3, 0.8), ("ب", 0.9, 1.5), ("ج", 1.6, 2.0)], final=True))
    assert s.step().committed == "أ ب ج"


def test_a_refused_key_stops_with_a_plain_reason_and_is_not_retried():
    s, conn, clock = make([(1.0, 9.0)], Connector(error=refusal(401)))
    play(s, clock, 5.0)
    assert (s.problem, s.fatal) == ("Deepgram rejected the API key", True)
    assert len(conn.calls) == 1


def test_an_unreachable_server_is_retried_with_backoff():
    connector = Connector(error=DeepgramError("Cannot reach Deepgram: no route"))
    s, conn, clock = make([(1.0, 9.0)], connector)
    play(s, clock, 1.2)
    assert len(conn.calls) == 1 and s.problem.startswith("Cannot reach Deepgram")
    play(s, clock, 1.6)                                 # 0.5 s later: the second try
    assert len(conn.calls) == 2
    connector.error = None
    play(s, clock, 3.0)                                 # 1 s after that, it works
    assert s.problem is None and conn.socket.seconds > 2.0


def test_silence_keeps_the_connection_alive_then_closes_it():
    reply = results([("نعم", 0.3, 0.6)], final=True, from_finalize=True)
    s, conn, clock = make([(1.0, 1.5), (40.0, 41.0)], Connector(on_finalize=[reply]))
    play(s, clock, 10.0)
    first = conn.socket
    assert "KeepAlive" in first.messages and not first.closed
    play(s, clock, 35.0)
    assert first.messages[-1] == "CloseStream" and first.closed
    play(s, clock, 40.5)                                # the next speech connects again
    assert len(conn.sockets) == 2 and conn.socket.seconds > 0


def test_the_status_line_says_what_is_wrong():
    s, _, _ = make([], Connector(error=refusal(402)))
    s._connected()
    said = []
    sink = SimpleNamespace(status=said.append, update=lambda u: None, close=lambda: None)
    Pipeline(None, s, [sink], 0.1)._status()
    assert said == ["✕ The Deepgram account is out of credit"]


# ── The key ────────────────────────────────────────────────────────────

def test_an_empty_key_is_refused_without_asking_deepgram(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("no request"))
    with pytest.raises(DeepgramError, match="No Deepgram API key") as e:
        check_key("  ")
    assert e.value.fatal


def test_a_401_from_the_key_check_means_a_rejected_key(monkeypatch):
    import urllib.error
    import urllib.request

    def refuse(request, timeout):
        assert request.get_header("Authorization") == f"Token {KEY}"
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(DeepgramError, match="rejected the API key"):
        DeepgramEngine(KEY).load()


def test_the_session_log_never_holds_the_key(tmp_path):
    from src.core.debug import log
    log.open(tmp_path / "live.log")
    try:
        reply = results([("نعم", 0.3, 0.6)], final=True, from_finalize=True)
        s, conn, clock = make([(1.0, 2.0)], Connector(on_finalize=[reply]))
        play(s, clock, 4.0)
        s.finish()
        assert "deepgram finals 1 " in log.summary(0.1)
    finally:
        log.close()
    assert KEY not in (tmp_path / "live.log").read_text()


# ── Choosing it ────────────────────────────────────────────────────────

def test_the_model_setting_picks_the_engine_and_the_streamer():
    from src.audio.streamer import Streamer
    from src.core import keystore
    from src.session import effective_args, make_engine, make_streamer
    from src.ui.settings import AppSettings

    keystore.save(KEY)
    cli = SimpleNamespace(model=None, device=None, beam=None, step=None, sink=None)
    args = effective_args(cli, AppSettings(model="deepgram", deepgram_language="ar-SA"))
    assert not hasattr(args, "deepgram_key")            # read only when connecting
    engine = make_engine(args)
    assert isinstance(engine, DeepgramEngine) and (engine.api_key, engine.language) == (KEY, "ar-SA")
    assert isinstance(make_streamer(engine, args), DeepgramStreamer)

    whisper = make_engine(effective_args(cli, AppSettings(model="small")))
    whisper.device = "cpu"
    assert isinstance(make_streamer(whisper, args), Streamer)

