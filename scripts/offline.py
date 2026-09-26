"""Transcribe a whole file at once: the accuracy ceiling, and a reference draft.

    python scripts/offline.py clip.mp4                 # prints [mm:ss] lines
    python scripts/offline.py clip.mp4 --out clip.txt  # a draft to correct by hand

Correcting this by hand is far quicker than typing a reference from nothing.
The corrected file is what `replay.py --ref` scores against.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.engine import WhisperEngine                       # noqa: E402
from src.config import SAMPLE_RATE, WHISPER_MODEL, DEVICE, ASR_BEAM_SIZE   # noqa: E402
from src.sinks import clock                                      # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("clip")
    p.add_argument("--out", help="write the lines here as well")
    p.add_argument("--model", default=WHISPER_MODEL)
    p.add_argument("--device", default=DEVICE, choices=["auto", "cuda", "cpu"])
    p.add_argument("--beam", type=int, default=ASR_BEAM_SIZE)
    args = p.parse_args()

    from faster_whisper.audio import decode_audio

    engine = WhisperEngine(args.model, args.device, args.beam).load()
    audio = decode_audio(args.clip, sampling_rate=SAMPLE_RATE)
    result = engine.transcribe_long(audio)

    lines = [f"[{clock(s.start)}] {s.text}" for s in result.segments]
    print("\n".join(lines))
    for reason, detail, text in result.dropped:
        print(f"# dropped {reason} {detail}: {text}", file=sys.stderr)
    print(f"# {len(audio) / SAMPLE_RATE:.0f}s of audio in {result.latency_s:.1f}s "
          f"on {engine.device}/{engine.compute_type}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
