"""How long one pass takes on this machine, by buffer length and settings.

    python scripts/bench_latency.py speech.mp3
    python scripts/bench_latency.py speech.mp3 --device cpu --threads 4,8 --beams 1,5
    python scripts/bench_latency.py speech.mp3 --compute float16,int8_float16

Use real speech at least as long as the longest buffer: decoding time depends
on how many tokens come out, and silence produces almost none. The results set
ASR_STEP_S and ASR_BEAM_SIZE: a pass at MAX_BUFFER_S must stay well under the
step (PLAN.md step 0 gate: p90 ≤ 60% of it).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SAMPLE_RATE, WHISPER_MODEL, USE_FP16      # noqa: E402
from src.core.device import resolve_device                       # noqa: E402


def csv(kind):
    return lambda s: [kind(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("clip", help="real Arabic speech, at least as long as the longest buffer")
    p.add_argument("--model", default=WHISPER_MODEL)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute", type=csv(str), help="compute types (default: the app's)")
    p.add_argument("--threads", type=csv(int), default=[0], help="CPU threads, 0 = ctranslate2 default")
    p.add_argument("--beams", type=csv(int), default=[1, 5])
    p.add_argument("--lengths", type=csv(float), default=[2, 5, 10, 15, 25])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--no-words", action="store_true", help="also time without word timestamps")
    args = p.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    audio = decode_audio(args.clip, sampling_rate=SAMPLE_RATE)
    device, default_compute = resolve_device(args.device, USE_FP16)
    print(f"{args.model} on {device}; clip {len(audio) / SAMPLE_RATE:.0f}s; {args.runs} runs each")
    print(f"{'compute':<13}{'thr':>4}{'beam':>5}{'words':>6}{'buffer':>8}{'p50':>9}{'p90':>9}  tokens≈")

    for compute in args.compute or [default_compute]:
        for threads in args.threads:
            model = WhisperModel(args.model, device=device, compute_type=compute,
                                 cpu_threads=threads, local_files_only=True)
            for beam in args.beams:
                for words in ([True, False] if args.no_words else [True]):
                    for length in args.lengths:
                        chunk = audio[: int(length * SAMPLE_RATE)]
                        times, chars = [], 0
                        for i in range(args.runs + 1):
                            t0 = time.perf_counter()
                            segments, _ = model.transcribe(
                                chunk, language="ar", beam_size=beam, temperature=0.0,
                                word_timestamps=words, vad_filter=False,
                                condition_on_previous_text=False,
                            )
                            text = " ".join(s.text for s in segments)
                            if i:                      # the first run is warm-up
                                times.append(time.perf_counter() - t0)
                            chars = len(text)
                        times.sort()
                        p50 = times[len(times) // 2]
                        p90 = times[min(len(times) - 1, int(len(times) * 0.9))]
                        print(f"{compute:<13}{threads:>4}{beam:>5}{'on' if words else 'off':>6}"
                              f"{length:>7.0f}s{p50 * 1000:>7.0f}ms{p90 * 1000:>7.0f}ms  {chars}ch")
            del model


if __name__ == "__main__":
    main()
