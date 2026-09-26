"""Speech engines, behind one seam, so a second one can be measured.

An engine takes a stretch of float32 mono audio and returns the words it
heard, each with a start and end time relative to that audio. Everything
downstream — agreement, trimming, lines — is engine-agnostic. Adding one means
subclassing `Engine`; swapping the default means a benchmark run first
(hifz: "bigger is not better here, and nothing about that is visible from
reading code").
"""

import time
from dataclasses import dataclass, field, replace

import numpy as np

from src.config import (
    WHISPER_MODEL, DEVICE, USE_FP16, CPU_THREADS, SAMPLE_RATE,
    ASR_BEAM_SIZE, ASR_NO_SPEECH_THRESHOLD, ASR_MIN_AVG_LOGPROB,
    ASR_MAX_COMPRESSION_RATIO, PHANTOM_LINES, ASR_MAX_TOKENS_PER_S, ASR_MIN_TOKENS,
    LOOP_COPIES,
)
from src.core.arabic import compare_key
from src.core.device import resolve_device


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0

    def shifted(self, seconds: float) -> "Word":
        return replace(self, start=self.start + seconds, end=self.end + seconds)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def shifted(self, seconds: float) -> "Segment":
        return replace(self, start=self.start + seconds, end=self.end + seconds)


@dataclass
class Transcription:
    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    # (reason, detail, text) for every segment a gate threw away.
    dropped: list[tuple[str, str, str]] = field(default_factory=list)
    latency_s: float = 0.0


class Engine:
    """Audio in, words with times out."""

    name = "engine"
    device = "cpu"
    compute_type = ""

    def load(self) -> "Engine":
        return self

    def arabic_probability(self, audio: np.ndarray) -> float:
        """How sure the engine is that this audio is Arabic. 1.0 when it cannot tell."""
        return 1.0

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcription:
        raise NotImplementedError


# ── Gates ──────────────────────────────────────────────────────────────

_PHANTOM_KEYS = tuple(sorted({compare_key(p) for p in PHANTOM_LINES}, key=len, reverse=True))


def is_phantom(text: str) -> bool:
    """True when the text is nothing but lines Whisper invents out of silence."""
    rest = f" {compare_key(text)} "
    if not rest.strip():
        return False
    for phrase in _PHANTOM_KEYS:
        rest = rest.replace(f" {phrase} ", " ")
    return not rest.strip()


def cut_loop(words: list[Word], copies: int = LOOP_COPIES, longest: int = 8) -> list[Word] | None:
    """The words before a loop, plus one copy of the looped run; None if nothing loops."""
    keys = [compare_key(w.text) for w in words]
    for i in range(len(keys)):
        for n in range(1, longest + 1):
            if i + copies * n > len(keys):
                break
            run = keys[i:i + n]
            if all(keys[i + k * n: i + (k + 1) * n] == run for k in range(1, copies)):
                return words[: i + n]
    return None


def rejection(seg) -> tuple[str, str] | None:
    """(reason, detail) when a segment should be dropped, else None."""
    if seg.compression_ratio > ASR_MAX_COMPRESSION_RATIO:
        return "repetitive", f"cr={seg.compression_ratio:.2f}"
    if seg.avg_logprob < ASR_MIN_AVG_LOGPROB:
        return "low_logprob", f"lp={seg.avg_logprob:.2f}"
    if is_phantom(seg.text):
        return "phantom", f"nsp={seg.no_speech_prob:.2f}"
    return None


# ── Whisper ────────────────────────────────────────────────────────────

class WhisperEngine(Engine):
    """faster-whisper, forced to Arabic, with word times."""

    name = "whisper"

    def __init__(self, model: str = WHISPER_MODEL, device: str = DEVICE,
                 beam_size: int = ASR_BEAM_SIZE, precision: str = "auto"):
        self.model_name = model
        self.device_preference = device
        self.beam_size = beam_size
        self.precision = precision
        self.model = None
        self.load_seconds = 0.0
        # faster-whisper's own guards; overridable per run (replay.py --engine-set).
        self.options = {
            "no_speech_threshold": ASR_NO_SPEECH_THRESHOLD,
            "log_prob_threshold": ASR_MIN_AVG_LOGPROB,
            "compression_ratio_threshold": ASR_MAX_COMPRESSION_RATIO,
        }

    def load(self) -> "WhisperEngine":
        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        self.device, self.compute_type = resolve_device(self.device_preference, USE_FP16)
        if self.precision != "auto":
            # float16 has no CPU kernel; anything asked of a CPU becomes int8.
            gpu_only = self.precision in ("float16", "int8_float16")
            self.compute_type = "int8" if self.device == "cpu" and gpu_only else self.precision
        elif self.device == "cuda" and "large" in str(self.model_name):
            # large-v3 in float16 needs ~4.4 GB with its working memory — more
            # than a 4 GB laptop GPU has (CaptionForge measured the same model
            # on this machine). int8 weights halve it.
            self.compute_type = "int8_float16"
        # local_files_only: the app never reaches the network. A model that is
        # not in the cache is an error to fix once, not a silent download.
        self.model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type,
            cpu_threads=CPU_THREADS, local_files_only=True,
        )
        # The first pass pays for CUDA kernel selection and memory pools;
        # pay it here instead of on the first sentence.
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.load_seconds = time.monotonic() - t0
        return self

    # faster-whisper refuses prompt + max_new_tokens above the model's 448;
    # its prompt is <|startofprev|>, at most 223 prompt tokens, and 3 more.
    _MAX_LENGTH = 448

    def _token_budget(self, seconds: float, prompt: str | None) -> int:
        budget = int(ASR_MAX_TOKENS_PER_S * seconds) + ASR_MIN_TOKENS
        prompt_tokens = 0
        if prompt:
            ids = self.model.hf_tokenizer.encode(" " + prompt.strip(), add_special_tokens=False).ids
            prompt_tokens = 1 + min(len(ids), self._MAX_LENGTH // 2 - 1)
        return max(16, min(budget, self._MAX_LENGTH - prompt_tokens - 4))

    def arabic_probability(self, audio: np.ndarray) -> float:
        """Whisper's own language detection on this audio: P(Arabic)."""
        _lang, _p, probs = self.model.detect_language(audio=audio)
        return dict(probs).get("ar", 0.0)

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcription:
        t0 = time.monotonic()
        segments, _info = self.model.transcribe(
            audio,
            language="ar",                 # v1 is Arabic only; never let it switch
            task="transcribe",
            beam_size=self.beam_size,
            # One temperature: no fallback re-decodes, so no latency spikes.
            # A segment that would have needed one is caught by the gates.
            temperature=0.0,
            word_timestamps=True,          # the streamer cuts the buffer at word ends
            vad_filter=False,              # Silero has already run upstream
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            max_new_tokens=self._token_budget(len(audio) / SAMPLE_RATE, prompt),
            **self.options,
        )
        result = Transcription()
        for seg in segments:               # decoding happens while iterating
            words = [
                Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                for w in (seg.words or []) if w.word.strip()
            ]
            reason = rejection(seg)
            if reason is not None and reason[0] == "repetitive":
                kept = cut_loop(words)
                if kept:
                    result.dropped.append(("loop_cut", f"{reason[1]} kept {len(kept)}/{len(words)}",
                                           seg.text.strip()))
                    words = kept
                    reason = (("phantom", "after loop cut")
                              if is_phantom(" ".join(w.text for w in kept)) else None)
            if reason is not None:
                result.dropped.append((reason[0], reason[1], seg.text.strip()))
                continue
            if not words:
                continue
            result.words.extend(words)
            result.segments.append(Segment(words[0].start, words[-1].end,
                                           " ".join(w.text for w in words)))
        result.latency_s = time.monotonic() - t0
        return result

    def transcribe_long(self, audio: np.ndarray) -> Transcription:
        """A whole recording at once, as an offline tool would: the accuracy ceiling.

        Same model, beam and gates as the live passes; faster-whisper's own
        VAD and 30 s windows instead of the streamer. The difference between
        this and the streamed text is what streaming costs (STREAM GAP).
        """
        t0 = time.monotonic()
        segments, _info = self.model.transcribe(
            audio, language="ar", task="transcribe", beam_size=self.beam_size,
            vad_filter=True, condition_on_previous_text=False,
            no_speech_threshold=ASR_NO_SPEECH_THRESHOLD,
            log_prob_threshold=ASR_MIN_AVG_LOGPROB,
            compression_ratio_threshold=ASR_MAX_COMPRESSION_RATIO,
        )
        result = Transcription()
        for seg in segments:
            reason = rejection(seg)
            if reason is not None:
                result.dropped.append((reason[0], reason[1], seg.text.strip()))
                continue
            result.segments.append(Segment(float(seg.start), float(seg.end), seg.text.strip()))
        result.latency_s = time.monotonic() - t0
        return result
