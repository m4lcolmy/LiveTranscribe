"""Replay a clip or a recorded session through the streamer, on a simulated clock.

    python scripts/replay.py clip.mp4
    python scripts/replay.py logs/session-20260926-101500/
    python scripts/replay.py clip.mp4 --ref clip.txt      # CER / WER / coverage (.txt or .srt)
    python scripts/replay.py clip.mp4 --offline           # plus the whole-file ceiling → STREAM GAP
    python scripts/replay.py music.mp3 --silent           # a control clip: must produce nothing
    python scripts/replay.py --manifest tests/clips/manifest.json

The clock is simulated the way the live loop behaves: a pass starts at t with
the audio up to t, takes as long as it really takes here (measured), and the
next starts at t + max(step, that). A slow machine drops nothing, it updates
less often, exactly as live, and the latencies are this machine's. It is the
same Streamer and AutoGain the app runs, so the numbers describe the app.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.engine import WhisperEngine                       # noqa: E402
from src.audio.gain import AutoGain                              # noqa: E402
from src.audio.streamer import Streamer                          # noqa: E402
from src.config import (                                         # noqa: E402
    SAMPLE_RATE, WHISPER_MODEL, DEVICE, ASR_BEAM_SIZE, ASR_STEP_S_GPU, ASR_STEP_S_CPU,
)
from src.evaluation import SessionRecord, load_reference, score_text   # noqa: E402
from src.sinks import clock                                      # noqa: E402


def load_audio(path: Path):
    from faster_whisper.audio import decode_audio

    if path.is_dir():                                            # a --record session
        meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
        return decode_audio(str(path / "audio.flac"), sampling_rate=SAMPLE_RATE), meta
    return decode_audio(str(path), sampling_rate=SAMPLE_RATE), {}


def stream(engine, audio, step_s: float) -> SessionRecord:
    gain, streamer = AutoGain(), Streamer(engine)
    record = SessionRecord(duration_s=len(audio) / SAMPLE_RATE, step_s=step_s)
    t, fed = step_s, 0
    while True:
        target = min(len(audio), int(t * SAMPLE_RATE))
        if target > fed:
            streamer.feed(gain(audio[fed:target]))
            fed = target
        t0 = time.perf_counter()
        update = streamer.step()
        took = time.perf_counter() - t0
        if update is not None:
            record.add(t + took, update)
        if fed >= len(audio):
            break
        t += max(step_s, took)
    streamer.feed(gain.flush())
    t0 = time.perf_counter()
    update = streamer.finish()
    if update is not None:
        record.add(t + time.perf_counter() - t0, update)
    return record


def evaluate(engine, path: Path, step_s: float | None, ref: Path | None,
             offline: bool, silent: bool, quiet: bool) -> dict:
    audio, meta = load_audio(path)
    step = step_s or meta.get("step_s") or (ASR_STEP_S_GPU if engine.device == "cuda" else ASR_STEP_S_CPU)
    record = stream(engine, audio, step)
    result = {"clip": path.name, "seconds": round(len(audio) / SAMPLE_RATE, 1), "step": step}
    result.update(record.summary())

    if not quiet:
        print(f"\n── {path.name}  ({result['seconds']}s, step {step}s)")
        for _, u in record.shown:
            for line in u.finished:
                print(f"  [{clock(line.start)}] {line.text}")

    text = record.text
    if silent:
        result["stayed_silent"] = not text.strip()
    reference = load_reference(ref) if ref else None
    if reference:
        s = score_text(text, reference)
        result.update(cer=s.cer, wer=s.wer, coverage=s.coverage, ref_words=s.ref_words)
    if offline:
        long = engine.transcribe_long(audio)
        offline_text = " ".join(seg.text for seg in long.segments)
        result["offline_seconds"] = round(long.latency_s, 1)
        if reference:
            result["offline_cer"] = score_text(offline_text, reference).cer
            result["stream_gap"] = result["cer"] - result["offline_cer"]
        else:
            # No reference: how far the stream strays from the whole-file text.
            result["vs_offline_cer"] = score_text(text, offline_text).cer
    return result


def report(r: dict):
    def ms(x):
        return "-" if x != x else f"{x * 1000:.0f}ms"

    def s(x):
        return "-" if x != x else f"{x:.2f}s"

    lines = [
        f"  words {r['words']}"
        + (f"   COVERAGE {r['coverage']:.1%}  CER {r['cer']:.1%}  WER {r['wer']:.1%}"
           f"  (ref {r['ref_words']} words)" if "cer" in r else ""),
    ]
    if "stream_gap" in r:
        lines.append(f"  offline CER {r['offline_cer']:.1%}   STREAM GAP {r['stream_gap'] * 100:+.1f} pts")
    if "vs_offline_cer" in r:
        lines.append(f"  stream vs offline text: {r['vs_offline_cer']:.1%} CER apart (no reference given)")
    if "stayed_silent" in r:
        lines.append(f"  STAYED SILENT {'yes' if r['stayed_silent'] else 'NO — ' + str(r['words']) + ' words'}")
    lines.append(f"  LATENCY shown p50 {s(r['shown_p50'])} p90 {s(r['shown_p90'])}   "
                 f"committed p50 {s(r['commit_p50'])} p90 {s(r['commit_p90'])}")
    lines.append(f"  FLICKER {r['flicker_per_min']:.1f}/min   "
                 f"REAL-TIME passes {r['passes']}  p50 {ms(r['pass_p50'])} p90 {ms(r['pass_p90'])}"
                 f"  overran step {r['overran']}")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("clip", nargs="?", help="audio/video file, or a logs/session-* directory")
    p.add_argument("--manifest", help="JSON corpus: {\"clips\": [{\"file\", \"type\", \"ref\"?, \"silent\"?}]}")
    p.add_argument("--ref", help="reference transcript (.txt, .srt, or a LiveTranscribe transcript)")
    p.add_argument("--offline", action="store_true", help="also transcribe the whole file: STREAM GAP")
    p.add_argument("--silent", action="store_true", help="control clip: must produce no words")
    p.add_argument("--step", type=float, help="seconds between passes")
    p.add_argument("--model", default=WHISPER_MODEL)
    p.add_argument("--device", default=DEVICE, choices=["auto", "cuda", "cpu"])
    p.add_argument("--beam", type=int, default=ASR_BEAM_SIZE)
    p.add_argument("--quiet", action="store_true", help="numbers only, no transcript")
    p.add_argument("--json", help="also write the results here")
    args = p.parse_args()
    if not args.clip and not args.manifest:
        p.error("give a clip or --manifest")

    engine = WhisperEngine(args.model, args.device, args.beam).load()
    print(f"{args.model} on {engine.device}/{engine.compute_type}, beam {args.beam}")

    results = []
    if args.manifest:
        manifest = Path(args.manifest)
        for clip in json.loads(manifest.read_text())["clips"]:
            base = manifest.parent
            r = evaluate(engine, base / clip["file"], args.step,
                         base / clip["ref"] if clip.get("ref") else None,
                         args.offline, clip.get("silent", False), args.quiet)
            r["type"] = clip.get("type", "")
            report(r)
            results.append(r)
    else:
        r = evaluate(engine, Path(args.clip), args.step, Path(args.ref) if args.ref else None,
                     args.offline, args.silent, args.quiet)
        report(r)
        results.append(r)

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
