"""Deepgram: live Arabic transcription in the cloud, with the user's own API key.

**Why a streamer, not an engine.** The local seam is "audio in, words out",
and the Streamer above it hands Whisper everything since the last commit,
every second. Deepgram bills every second it is sent: through that seam each
second of speech would be billed eight times over, and every pass would wait
for a round trip. Its live API instead takes the audio once, as it plays, and
answers with interim words, which may still change, and final ones — the
tentative and committed words the window already shows. So Deepgram replaces
the Streamer: `DeepgramStreamer` has the same feed()/step()/finish() and
returns the same `Update`s, and the pipeline, the window and the sinks do not
know which of the two they drive.

**A final can end before the interim words shown so far.** Deepgram
finalises the audio up to a point it chooses; the interim words past that
point come back in the next interim. They stay on screen, grey, until then —
cleared with the final, they vanished on fast speech (2026-10-07).

**What is sent.** Only speech. Silero decides, as it does for the Streamer:
an utterance is sent from LEAD_IN_S before its first speech frame, and when
END_SILENCE_S of silence follows it, a Finalize asks Deepgram for its last
words; the line is closed when they come. Deepgram's times count only the
audio it was sent, so each connection keeps a map from its time back to the
stream's — the transcript's times are the stream's, as with Whisper.

**When the connection drops**, it is opened again and the utterance in
progress is sent again from its start. Words already final are recognised by
their stream time and not repeated — on that audio only: otherwise Deepgram
never sends a word twice, and a word overlapping the last final one is new.

The API key goes in a request header and nowhere else: not in the URL, the
log, the transcript, the recording or an error message.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import numpy as np

from src.audio.engine import Engine, Word
from src.audio.streamer import Line, Update
from src.config import (
    COMMIT_TIME_TOLERANCE_S, DEEPGRAM_FINALIZE_WAIT_S, DEEPGRAM_IDLE_CLOSE_S,
    DEEPGRAM_KEEPALIVE_S, DEEPGRAM_KEY_URL, DEEPGRAM_LANGUAGE, DEEPGRAM_MODEL,
    DEEPGRAM_RESEND_MAX_S, DEEPGRAM_RETRY_BACKOFF_S, DEEPGRAM_SEND_SILENCE,
    DEEPGRAM_TIMEOUT_S, DEEPGRAM_URL, END_SILENCE_S, LEAD_IN_S, LINE_MAX_CHARS, SAMPLE_RATE,
)
from src.core.debug import log


class DeepgramError(Exception):
    """Deepgram refused, or could not be reached. `fatal`: trying again will not help."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


class Closed(Exception):
    """The connection ended."""


def refusal(status: int) -> DeepgramError:
    """What an HTTP status from Deepgram means, in words."""
    if status == 401:
        return DeepgramError("Deepgram rejected the API key", fatal=True)
    if status == 402:
        return DeepgramError("The Deepgram account is out of credit", fatal=True)
    if status == 403:
        return DeepgramError("This Deepgram key is not allowed to transcribe", fatal=True)
    if status == 429:
        return DeepgramError("Deepgram is busy (too many requests)")
    # Any other 4xx is a request Deepgram will refuse again; a 5xx may pass.
    return DeepgramError(f"Deepgram answered HTTP {status}", fatal=400 <= status < 500)


def check_key(api_key: str, timeout: float = DEEPGRAM_TIMEOUT_S):
    """Raises DeepgramError unless Deepgram accepts the key."""
    if not api_key.strip():
        raise DeepgramError("No Deepgram API key yet", fatal=True)
    request = urllib.request.Request(DEEPGRAM_KEY_URL,
                                     headers={"Authorization": f"Token {api_key.strip()}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return      # a key with fewer permissions is still a key; streaming will say more
        raise refusal(e.code) from None
    except (urllib.error.URLError, OSError) as e:
        raise DeepgramError(f"Cannot reach Deepgram: {getattr(e, 'reason', e)}") from None


# ── The connection ─────────────────────────────────────────────────────

class Socket:
    """websockets' sync client, behind the three calls the streamer makes."""

    def __init__(self, connection):
        self._conn = connection

    def send(self, data):
        from websockets.exceptions import ConnectionClosed
        try:
            self._conn.send(data)
        except ConnectionClosed as e:
            raise Closed(_why(e)) from None

    def recv(self, timeout: float = 0.0):
        """The next message, or None when none comes within `timeout` seconds."""
        from websockets.exceptions import ConnectionClosed
        try:
            return self._conn.recv(timeout=timeout)
        except TimeoutError:
            return None
        except ConnectionClosed as e:
            raise Closed(_why(e)) from None

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def _why(e) -> str:
    frame = getattr(e, "rcvd", None)
    if frame is None:
        return "connection lost"
    return f"connection closed ({frame.code}{f' {frame.reason}' if frame.reason else ''})"


def connect(url: str, api_key: str, timeout: float = DEEPGRAM_TIMEOUT_S) -> Socket:
    from websockets.exceptions import InvalidHandshake, InvalidStatus
    from websockets.sync.client import connect as ws_connect
    try:
        conn = ws_connect(url, additional_headers={"Authorization": f"Token {api_key}"},
                          open_timeout=timeout)
    except InvalidStatus as e:
        raise refusal(e.response.status_code) from None
    except (InvalidHandshake, OSError, TimeoutError) as e:
        raise DeepgramError(f"Cannot reach Deepgram: {e}") from None
    return Socket(conn)


# ── The engine: what to connect with ───────────────────────────────────

class DeepgramEngine(Engine):
    """The key, model and dialect. load() checks the key, so a bad one fails like a missing model."""

    name = "deepgram"
    device = "cloud"

    def __init__(self, api_key: str, language: str = DEEPGRAM_LANGUAGE, model: str = DEEPGRAM_MODEL):
        self.api_key = api_key.strip()
        self.language = language or DEEPGRAM_LANGUAGE
        self.model = model
        self.compute_type = f"{model} {self.language}"
        self.load_seconds = 0.0

    def __repr__(self):                       # never the key
        return f"DeepgramEngine({self.model}, {self.language})"

    def load(self) -> "DeepgramEngine":
        t0 = time.monotonic()
        check_key(self.api_key)
        self.load_seconds = time.monotonic() - t0
        return self

    def url(self, sample_rate: int = SAMPLE_RATE) -> str:
        return DEEPGRAM_URL + "?" + urllib.parse.urlencode({
            "model": self.model, "language": self.language,
            "encoding": "linear16", "sample_rate": sample_rate, "channels": 1,
            "interim_results": "true", "punctuate": "true",
        })

    def transcribe(self, audio, prompt=None):
        raise NotImplementedError("Deepgram streams: it runs through DeepgramStreamer")


# ── The streamer ───────────────────────────────────────────────────────

class DeepgramStreamer:
    def __init__(self, engine: DeepgramEngine, vad=None, *, connect=connect, clock=time.monotonic,
                 sample_rate=SAMPLE_RATE, lead_in_s=LEAD_IN_S, end_silence_s=END_SILENCE_S,
                 line_max_chars=LINE_MAX_CHARS, commit_tolerance_s=COMMIT_TIME_TOLERANCE_S,
                 send_silence=DEEPGRAM_SEND_SILENCE, keepalive_s=DEEPGRAM_KEEPALIVE_S,
                 idle_close_s=DEEPGRAM_IDLE_CLOSE_S, finalize_wait_s=DEEPGRAM_FINALIZE_WAIT_S,
                 retry_backoff_s=DEEPGRAM_RETRY_BACKOFF_S, resend_max_s=DEEPGRAM_RESEND_MAX_S):
        if vad is None:
            from src.audio.vad import StreamingVad
            vad = StreamingVad()
        self.engine = engine
        self.vad = vad
        self.sr = sample_rate
        self._connect = connect
        self._clock = clock
        self.lead_in_s = lead_in_s
        self.end_silence_s = end_silence_s
        self.line_max_chars = line_max_chars
        self.commit_tolerance_s = commit_tolerance_s
        self.send_silence = send_silence
        self.keepalive_s = keepalive_s
        self.idle_close_s = idle_close_s
        self.finalize_wait_s = finalize_wait_s
        self.retry_backoff_s = retry_backoff_s
        self.resend_max_s = resend_max_s

        # The audio not yet safe to forget: a lead-in, or the utterance under way.
        self._audio = np.zeros(0, dtype=np.float32)
        self._start = 0                   # stream sample of _audio[0]
        self._scored = 0                  # how far the VAD had scored at the last look
        self._speaking = False
        self._utterance_start = 0         # stream sample its audio starts at
        self._utterance_end: int | None = None
        self._sent_until = 0              # stream sample; before it, sent on this connection
        self._finalize_needed = False
        self._finalizing_since: float | None = None

        self._socket = None
        self._spans: list[tuple[int, int]] = []   # (Deepgram sample, stream sample) of each run sent
        self._sent = 0                    # samples sent on this connection
        self._span_next: int | None = None
        self._last_sent_at = 0.0
        self._last_audio_at = 0.0
        self._retry_at = 0.0
        self._failures = 0
        self.problem: str | None = None   # what is wrong now, for the status line
        self.fatal = False                # ...and whether trying again would help

        self._line: list[Word] = []       # final words of the current line
        self._interim: list[Word] = []    # Deepgram's latest guess past them
        self._finished: list[Line] = []   # lines closed since the last update
        self._last_committed_end = 0.0
        self._resent_until: float | None = None   # stream time: before it, final already, sent again
        self._changed = False
        self.stats: Counter = Counter()

    # ── Time ───────────────────────────────────────────────────────────

    @property
    def fed(self) -> int:
        return self._start + len(self._audio)

    @property
    def now(self) -> float:
        return self.fed / self.sr

    def _to_stream(self, t: float) -> float:
        """A time on this connection's clock, as seconds of the stream."""
        sample = t * self.sr
        base_sent, base_stream = 0, 0
        for sent, stream in self._spans:
            if sent > sample:
                break
            base_sent, base_stream = sent, stream
        return (base_stream + sample - base_sent) / self.sr

    # ── Driving ────────────────────────────────────────────────────────

    def feed(self, audio: np.ndarray):
        if audio.size == 0:
            return
        self._audio = np.concatenate([self._audio, audio.astype(np.float32, copy=False)])
        self.vad.feed(audio)
        self._follow_speech()
        self._send()
        self._trim()

    def step(self) -> Update | None:
        """Collect what Deepgram has sent. None when nothing changed."""
        self._send()
        self._receive()
        if (self._finalizing_since is not None
                and self._clock() - self._finalizing_since >= self.finalize_wait_s):
            # No answer to the Finalize: what was heard is all there is.
            self.stats["finalize_timeouts"] += 1
            log.event("DG", "no answer to Finalize; closing the line")
            self._finalizing_since = None
            self._close_line()
        self._keep_alive()
        self._trim()
        return self._update()

    def finish(self) -> Update | None:
        """End of stream, or pause: Deepgram's last words, then the line and the connection close."""
        if self._speaking:
            self._end_utterance()
        self._send()
        deadline = self._clock() + self.finalize_wait_s
        while self._finalizing_since is not None and self._socket is not None:
            left = deadline - self._clock()
            if left <= 0:
                break
            self._receive(left)
        self._finalize_needed, self._finalizing_since = False, None
        self._close_line()
        self._disconnect("finished")
        self.problem, self.fatal, self._failures, self._retry_at = None, False, 0, 0.0
        self._trim()
        return self._update()

    # ── Speech: when to send ───────────────────────────────────────────

    @property
    def _pending(self) -> bool:
        """An utterance is under way, or its last words are still to come."""
        return self._speaking or self._finalize_needed or self._finalizing_since is not None

    def _follow_speech(self):
        scored = min(self.fed, self.vad.scored_until)
        if not self._speaking:
            if self.vad.last_speech_end(self._scored, scored) is not None:
                self._speaking = True
                self._utterance_start = max(self._start, self._scored - int(self.lead_in_s * self.sr))
                self._utterance_end = None
                self.stats["utterances"] += 1
        else:
            last = self.vad.last_speech_end(self._utterance_start, scored)
            if last is None or (scored - last) / self.sr >= self.end_silence_s:
                self._end_utterance()
        self._scored = scored

    def _end_utterance(self):
        """Its trailing silence is sent too — Deepgram places the last word's end in it."""
        self._speaking = False
        self._utterance_end = self.fed
        self._finalize_needed = True

    def _send(self):
        if self.send_silence:
            end = self.fed
        elif self._speaking:
            end = self.fed
        elif self._utterance_end is not None and self._pending:
            end = self._utterance_end
        else:
            end = 0
        begin = max(self._sent_until, self._start)
        if end > begin and self._connected():
            chunk = self._audio[begin - self._start: end - self._start]
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            if self._span_next != begin:
                self._spans.append((self._sent, begin))
            try:
                self._socket.send(pcm)
            except Closed as e:
                self._lost(str(e))
                return
            self._sent += len(chunk)
            self._span_next = self._sent_until = end
            self._last_sent_at = self._last_audio_at = self._clock()
        if (self._finalize_needed and self._socket is not None
                and self._utterance_end is not None and self._sent_until >= self._utterance_end):
            if self._message({"type": "Finalize"}):
                self._finalize_needed = False
                self._finalizing_since = self._clock()

    def _trim(self):
        """Forget audio no longer needed: all but a lead-in, or, under way, all before the utterance."""
        keep_from = self._utterance_start if self._pending else self.fed - int(self.lead_in_s * self.sr)
        keep_from = max(keep_from, self.fed - int(self.resend_max_s * self.sr))
        keep_from = min(max(keep_from, self._start), self.fed)
        if keep_from > self._start:
            self._audio = self._audio[keep_from - self._start:]
            self._start = keep_from
            self._utterance_start = max(self._utterance_start, keep_from)
            self.vad.forget_before(keep_from)

    # ── The connection ─────────────────────────────────────────────────

    def _connected(self) -> bool:
        """Connected, or connected now; False while waiting to try again."""
        if self._socket is not None:
            return True
        if self.fatal or self._clock() < self._retry_at:
            return False
        try:
            self._socket = self._connect(self.engine.url(self.sr), self.engine.api_key)
        except DeepgramError as e:
            self._failed(str(e), e.fatal)
            return False
        self._spans, self._sent, self._span_next = [], 0, None
        self._failures, self.problem = 0, None
        self._last_sent_at = self._last_audio_at = self._clock()
        self.stats["connections"] += 1
        log.event("DG", f"connected: {self.engine.model} {self.engine.language}")
        return True

    def _message(self, message: dict) -> bool:
        try:
            self._socket.send(json.dumps(message))
            self._last_sent_at = self._clock()
            return True
        except Closed as e:
            self._lost(str(e))
            return False

    def _keep_alive(self):
        if self._socket is None:
            return
        now = self._clock()
        if not self._pending and not self.send_silence and now - self._last_audio_at >= self.idle_close_s:
            self._disconnect("idle")
        elif now - self._last_sent_at >= self.keepalive_s:
            self._message({"type": "KeepAlive"})

    def _disconnect(self, why: str):
        if self._socket is None:
            return
        try:
            self._socket.send(json.dumps({"type": "CloseStream"}))
        except Closed:
            pass
        self._socket.close()
        self._socket = None
        log.event("DG", f"disconnected: {why}")

    def _lost(self, why: str):
        self._socket.close()
        self._socket = None
        log.count("deepgram_reconnects")
        log.event("DG", f"lost: {why}")
        if not self._pending:
            return               # nothing was waiting; the next speech connects again
        # Guesses about an earlier utterance cannot be heard again: they stay.
        before = self._utterance_start / self.sr
        heard = [w for w in self._interim if w.end <= before]
        if heard:
            self._commit(heard)
            self._end_line()
        # Send the utterance again from its start, and ask for its end again.
        # Its words final already will come back; they are not committed twice.
        self._sent_until = self._utterance_start
        self._resent_until = self._last_committed_end
        if self._finalizing_since is not None:
            self._finalizing_since, self._finalize_needed = None, True
        self._interim, self._changed = [], True
        self._failed(why, fatal=False)

    def _failed(self, why: str, fatal: bool):
        if fatal:
            self.problem, self.fatal = why, True
            log.event("DG", f"stopped: {why}")
            self._finalize_needed, self._finalizing_since = False, None
            self._close_line()          # what was heard stays heard
            return
        wait = self.retry_backoff_s[min(self._failures, len(self.retry_backoff_s) - 1)]
        self._failures += 1
        self._retry_at = self._clock() + wait
        self.problem = f"{why} — trying again"
        log.event("DG", f"{why}; again in {wait:.1f}s")

    # ── What comes back ────────────────────────────────────────────────

    def _receive(self, wait: float = 0.0):
        while self._socket is not None:
            try:
                raw = self._socket.recv(wait)
            except Closed as e:
                self._lost(str(e))
                return
            if raw is None:
                return
            self._handle(raw)
            wait = 0.0

    def _handle(self, raw):
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return
        kind = message.get("type")
        if kind == "Results":
            self._results(message)
        elif kind == "Metadata":
            log.event("DG", f"request {message.get('request_id', '?')}")
        elif kind == "Error":
            self.stats["errors"] += 1
            log.event("DG", f"error {message.get('code', '')}: {message.get('description', '')}")

    def _results(self, message: dict):
        alternatives = (message.get("channel") or {}).get("alternatives") or [{}]
        words = [w for w in (self._word(raw) for raw in alternatives[0].get("words") or [])
                 if w is not None]
        if self._resent_until is not None:
            # Sent again after a drop: the words already final come back too.
            # Only then — Deepgram never sends a final word twice otherwise,
            # and a word whose time overlaps the last final one is a new word.
            cutoff = self._resent_until - self.commit_tolerance_s
            words = [w for w in words if w.start > cutoff]
        if not message.get("is_final"):
            text = " ".join(w.text for w in words)
            if text != " ".join(w.text for w in self._interim):
                log.event("DG", f'interim "{text}"')
            self._interim, self._changed = words, True
            return
        end = self._to_stream(float(message.get("start", 0.0)) + float(message.get("duration", 0.0)))
        lag = max(0.0, self.now - end)
        log.result_lags.append(lag)
        finalize = bool(message.get("from_finalize"))
        log.event("DG", f"final lag={lag * 1000:.0f}ms{' FINALIZE' if finalize else ''}")
        if words:
            log.detail("commit", '"' + " ".join(w.text for w in words) + '"')
        self._commit(words)
        # A final covers the audio up to a point Deepgram chose, which can come
        # before the end of the interim words shown so far: those after it come
        # back in the next interim. Clearing them made words shown grey vanish
        # on fast speech until then (2026-10-07); they stay until it comes.
        # A Finalize's answer covers everything sent: nothing is left to come.
        floor = self._last_committed_end - self.commit_tolerance_s
        self._interim = [] if finalize else [w for w in self._interim
                                             if w.end > end and w.start >= floor]
        if self._interim:
            self.stats["interim_kept"] += len(self._interim)
            log.detail("kept", '"' + " ".join(w.text for w in self._interim) + '"')
        if self._resent_until is not None and end >= self._resent_until:
            self._resent_until = None
        self._changed = True
        if finalize:
            self._finalizing_since = None
            self._end_line()

    def _word(self, raw: dict) -> Word | None:
        text = (raw.get("punctuated_word") or raw.get("word") or "").strip()
        if not text:
            return None
        return Word(text, self._to_stream(float(raw.get("start", 0.0))),
                    self._to_stream(float(raw.get("end", 0.0))), float(raw.get("confidence", 1.0)))

    # ── Lines ──────────────────────────────────────────────────────────

    def _commit(self, words: list[Word]):
        for w in words:
            self._line.append(w)
            self._last_committed_end = max(self._last_committed_end, w.end)
            self.stats["words"] += 1
            if len(" ".join(x.text for x in self._line)) >= self.line_max_chars:
                self._end_line()

    def _close_line(self):
        """Commit the interim words, then end the line."""
        if self._interim:
            self._commit(self._interim)
            self._interim, self._changed = [], True
        self._end_line()

    def _end_line(self):
        if not self._line:
            return
        self._finished.append(Line(" ".join(w.text for w in self._line),
                                   self._line[0].start, self._line[-1].end, tuple(self._line)))
        self._line = []
        self.stats["lines"] += 1
        self._changed = True

    def _update(self) -> Update | None:
        if not self._changed and not self._finished:
            return None
        finished, self._finished, self._changed = tuple(self._finished), [], False
        return Update(
            finished=finished,
            committed=" ".join(w.text for w in self._line),
            tentative=" ".join(w.text for w in self._interim),
            line_words=tuple(self._line),
            tentative_words=tuple(self._interim),
            report=None,
            now=self.now,
        )
