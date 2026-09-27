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

import numpy as np

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


# Streamer keyword arguments from --set name=value, e.g. --set prompt_chars=0.
OVERRIDES: dict = {}
TRACE = False
# --fixed-clock: every pass exactly one step after the last, however long it
# took. The measured clock (the default) is what live does, but a slow pass
# shifts every later buffer boundary, and on a hard clip that alone moved
# coverage by 10 points between two runs of the same settings (DW
# Fckec5jxCZk, 2026-09-26). A/B comparisons need the fixed clock.
FIXED_CLOCK = False


def trace(t: float, update, streamer):
    """One line per pass: when, how much audio, how long, what it did."""
    r = update.report
    if r is None:
        return
    print(f"  t={t:6.1f} buf={r.at - r.buffer_start:5.1f}s {r.latency_s * 1000:5.0f}ms"
          f"{' FINAL' if r.final else '      '} +{len(r.committed):2d} ~{len(r.tentative):2d} "
          f"| {' '.join(w.text for w in r.tentative)[:60]}")
    for reason, detail, text in r.dropped:
        print(f"         DROP {reason} {detail}: {text[:90]}")
    for line in update.finished:
        print(f"         LINE {line.text[:100]}")


def stream(engine, audio, step_s: float) -> SessionRecord:
    gain, streamer = AutoGain(), Streamer(engine, **OVERRIDES)
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
            if TRACE:
                trace(t, update, streamer)
        if fed >= len(audio):
            break
        t += step_s if FIXED_CLOCK else max(step_s, took)
    streamer.feed(gain.flush())
    t0 = time.perf_counter()
    update = streamer.finish()
    if update is not None:
        record.add(t + time.perf_counter() - t0, update)
    return record


def evaluate(engine, path: Path, step_s: float | None, ref: Path | None,
             offline: bool, silent: bool, quiet: bool,
             start: float | None = None, end: float | None = None, name: str = "") -> dict:
    audio, meta = load_audio(path)
    reference, span_start, span_end = load_reference(ref, start, end) if ref else ("", None, None)
    # A window: the audio is cut to the span its reference cues cover (with a
    # little air), so both sides of the comparison hold the same speech.
    if span_start is not None:
        start, end = max(0.0, span_start - 0.5), span_end + 0.5
    if start is not None or end is not None:
        lo = int((start or 0.0) * SAMPLE_RATE)
        hi = int(end * SAMPLE_RATE) if end is not None else len(audio)
        audio = audio[lo:hi]

    step = step_s or meta.get("step_s") or (ASR_STEP_S_GPU if engine.device == "cuda" else ASR_STEP_S_CPU)
    record = stream(engine, audio, step)
    label = name or path.name
    result = {"clip": label, "seconds": round(len(audio) / SAMPLE_RATE, 1), "step": step}
    result.update(record.summary())

    if not quiet:
        print(f"\n── {label}  ({result['seconds']}s, step {step}s)")
        for _, u in record.shown:
            for line in u.finished:
                print(f"  [{clock(line.start)}] {line.text}")

    text = record.text
    if silent:
        result["stayed_silent"] = not text.strip()
    if reference:
        sc = score_text(text, reference)
        result.update(cer=sc.cer, wer=sc.wer, coverage=sc.coverage,
                      ref_words=sc.ref_words, ref_chars=sc.ref_chars)
    if offline:
        long = engine.transcribe_long(audio)
        offline_text = " ".join(seg.text for seg in long.segments)
        result["offline_seconds"] = round(long.latency_s, 1)
        if reference:
            off = score_text(offline_text, reference)
            result.update(offline_cer=off.cer, offline_coverage=off.coverage,
                          stream_gap=result["cer"] - off.cer)
        else:
            # No reference: how far the stream strays from the whole-file text.
            result["vs_offline_cer"] = score_text(text, offline_text).cer
    return result


def summarize(results: list[dict]):
    """Per clip type and overall: CER weighted by reference length, so a long clip counts more."""
    scored = [r for r in results if "cer" in r]
    groups: dict[str, list[dict]] = {}
    for r in sorted(scored, key=lambda r: (r.get("reference", ""), r.get("type", ""))):
        key = r.get("type", "") + (f"·{r['reference']}" if r.get("reference") else "")
        groups.setdefault(key, []).append(r)
    if scored:
        groups["ALL"] = scored

    def weighted(rs, key, weight):
        total = sum(r[weight] for r in rs)
        return sum(r[key] * r[weight] for r in rs) / max(1, total)

    print("\n══ corpus " + "═" * 70)
    print(f"  {'type·reference':<24}{'clips':>6}{'words':>7}{'COVER':>8}{'CER':>7}{'WER':>7}"
          f"{'offCER':>8}{'GAP':>7}{'shown p90':>11}{'final p90':>11}{'flick/min':>10}{'vanish/min':>11}")
    for name, rs in groups.items():
        off = (f"{weighted(rs, 'offline_cer', 'ref_chars'):>7.1%}"
               f"{(weighted(rs, 'cer', 'ref_chars') - weighted(rs, 'offline_cer', 'ref_chars')) * 100:>+6.1f}"
               if all("offline_cer" in r for r in rs) else f"{'-':>7}{'-':>7}")
        print(f"  {name:<24}{len(rs):>6}{sum(r['ref_words'] for r in rs):>7}"
              f"{weighted(rs, 'coverage', 'ref_words'):>8.1%}{weighted(rs, 'cer', 'ref_chars'):>7.1%}"
              f"{weighted(rs, 'wer', 'ref_words'):>7.1%} {off}"
              f"{np.nanmean([r['shown_p90'] for r in rs]):>10.2f}s"
              f"{np.nanmean([r['commit_p90'] for r in rs]):>10.2f}s"
              f"{np.mean([r['flicker_per_min'] for r in rs]):>10.1f}"
              f"{np.mean([r['vanished_per_min'] for r in rs]):>11.1f}")
    silent = [r for r in results if "stayed_silent" in r]
    if silent:
        ok = sum(r["stayed_silent"] for r in silent)
        failed = ", ".join(f"{r['clip']} ({r['words']} words)" for r in silent if not r["stayed_silent"])
        print(f"  STAYED SILENT {ok}/{len(silent)}" + (f"   failed: {failed}" if failed else ""))


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
    lines.append(f"  FLICKER {r['flicker_per_min']:.1f}/min (vanished {r['vanished_per_min']:.1f})   "
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
    p.add_argument("--start", type=float, help="score only from this second (snapped to SRT cues)")
    p.add_argument("--end", type=float, help="...up to this second")
    p.add_argument("--step", type=float, help="seconds between passes")
    p.add_argument("--model", default=WHISPER_MODEL)
    p.add_argument("--device", default=DEVICE, choices=["auto", "cuda", "cpu"])
    p.add_argument("--beam", type=int, default=ASR_BEAM_SIZE)
    p.add_argument("--quiet", action="store_true", help="numbers only, no transcript")
    p.add_argument("--json", help="also write the results here")
    p.add_argument("--only", nargs="*", help="manifest mode: only these clip ids or types")
    p.add_argument("--engine-set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a faster-whisper guard, e.g. --engine-set no_speech_threshold=none")
    p.add_argument("--fixed-clock", action="store_true",
                   help="schedule passes exactly one step apart: reproducible text for A/B tuning")
    p.add_argument("--trace", action="store_true", help="print every pass: buffer, time, commits, drops")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a Streamer setting, e.g. --set prompt_chars=0 --set end_silence_s=0.5")
    args = p.parse_args()
    global TRACE, FIXED_CLOCK
    TRACE, FIXED_CLOCK = args.trace, args.fixed_clock
    for item in args.set:
        name, _, value = item.partition("=")
        OVERRIDES[name] = type(getattr(Streamer(None, vad=object()), name))(value)
    if not args.clip and not args.manifest:
        p.error("give a clip or --manifest")

    engine = WhisperEngine(args.model, args.device, args.beam)
    for item in args.engine_set:
        name, _, value = item.partition("=")
        engine.options[name] = None if value.lower() == "none" else float(value)
    engine.load()
    print(f"{args.model} on {engine.device}/{engine.compute_type}, beam {args.beam}")

    results = []
    if args.manifest:
        manifest = Path(args.manifest)
        base = manifest.parent
        for clip in json.loads(manifest.read_text(encoding="utf-8"))["clips"]:
            if args.only and clip.get("id") not in args.only and clip.get("type") not in args.only:
                continue
            if "id" in clip:        # fetch_corpus.py layout: <id>/audio.mp3 + <id>/ref.srt
                audio, ref = base / clip["id"] / "audio.mp3", base / clip["id"] / "ref.srt"
            else:
                audio, ref = base / clip["file"], base / clip["ref"] if clip.get("ref") else None
            if not audio.exists():
                print(f"\n── {clip.get('id', audio.name)}: not fetched (scripts/fetch_corpus.py --fetch)")
                continue
            r = evaluate(engine, audio, args.step,
                         ref if ref is not None and ref.exists() and not clip.get("silent") else None,
                         args.offline, clip.get("silent", False), args.quiet,
                         clip.get("start"), clip.get("end"), clip.get("id", ""))
            r["type"] = clip.get("type", "")
            r["reference"] = clip.get("reference", "")
            report(r)
            results.append(r)
        summarize(results)
    else:
        r = evaluate(engine, Path(args.clip), args.step, Path(args.ref) if args.ref else None,
                     args.offline, args.silent, args.quiet, args.start, args.end)
        report(r)
        results.append(r)

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
