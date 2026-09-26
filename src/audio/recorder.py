"""--record: keep the session so it can be replayed through the pipeline.

hifz's lesson: every bug was found live and then could not become a test,
because the audio was gone. With --record a session leaves behind

    logs/session-<timestamp>/
        audio.flac      what was captured, before gain — replay applies it again
        passes.jsonl    every Whisper pass: what it heard, committed, left tentative, dropped
        transcript.txt  the finished lines
        meta.json       model, device, step, source

and `python scripts/replay.py logs/session-<timestamp>/` runs it again.
"""

import json
from pathlib import Path

import numpy as np

from src.audio.streamer import Update
from src.config import SAMPLE_RATE
from src.sinks import TranscriptSink


def _words(words) -> list:
    return [[w.text, round(w.start, 3), round(w.end, 3)] for w in words]


class SessionRecorder:
    def __init__(self, directory: Path, meta: dict):
        import soundfile as sf

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._audio = sf.SoundFile(
            self.directory / "audio.flac", "w", samplerate=SAMPLE_RATE,
            channels=1, format="FLAC", subtype="PCM_16",
        )
        self._passes = open(self.directory / "passes.jsonl", "w", encoding="utf-8", buffering=1)
        self._transcript = TranscriptSink(self.directory / "transcript.txt", meta.get("source", ""))
        (self.directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def audio(self, pcm: np.ndarray):
        self._audio.write(pcm)

    def update(self, upd: Update):
        self._transcript.update(upd)
        r = upd.report
        if r is None:
            return
        self._passes.write(json.dumps({
            "at": round(r.at, 3),
            "buffer_start": round(r.buffer_start, 3),
            "latency_s": round(r.latency_s, 3),
            "final": r.final,
            "heard": r.heard,
            "committed": _words(r.committed),
            "tentative": _words(r.tentative),
            "dropped": [list(d) for d in r.dropped],
            "prompt": r.prompt,
        }, ensure_ascii=False) + "\n")

    def close(self):
        self._audio.close()
        self._passes.close()
        self._transcript.close()
