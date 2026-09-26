"""LiveTranscribe — live Arabic transcription of whatever the computer plays.

Step 1 (PLAN.md): transcription only, in the terminal. No window yet.

    ./run.sh                          # listen to the system's audio output
    ./run.sh --file lecture.mp4       # a file, at the pace it would play
    ./run.sh --record                 # also keep the session for scripts/replay.py
    ./run.sh --sink <node.name>       # a specific output instead of the default
"""

import argparse
import os
import sys
import time
import warnings

# Offline, always: a model missing from the cache is an error, not a download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import (
    WHISPER_MODEL, DEVICE, ASR_BEAM_SIZE, ASR_STEP_S_GPU, ASR_STEP_S_CPU, LOGS_DIR,
)
from src.core.debug import log, new_log_path


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="livetranscribe",
        description="Live Arabic transcription of the system's audio output.",
    )
    p.add_argument("--file", help="transcribe this audio/video file at playback pace")
    p.add_argument("--sink", help="capture this output (node.name, see `wpctl status`) "
                                  "instead of following the default one")
    p.add_argument("--model", default=WHISPER_MODEL, help=f"model size or path (default {WHISPER_MODEL})")
    p.add_argument("--device", default=DEVICE, choices=["auto", "cuda", "cpu"])
    p.add_argument("--beam", type=int, default=ASR_BEAM_SIZE, help="beam size")
    p.add_argument("--step", type=float, help="seconds between passes "
                                              f"(default {ASR_STEP_S_GPU} GPU / {ASR_STEP_S_CPU} CPU)")
    p.add_argument("--record", action="store_true",
                   help="keep audio + every pass under logs/session-<time>/ for replay")
    p.add_argument("--no-debug", action="store_true", help="no session log")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    from src.audio.capture import CaptureError, SystemAudioSource
    from src.audio.engine import WhisperEngine
    from src.audio.filesource import FileSource
    from src.audio.streamer import Streamer
    from src.audio.worker import Pipeline
    from src.sinks import ConsoleSink, TranscriptSink

    print(f"Loading Whisper {args.model}…", end="", flush=True)
    try:
        engine = WhisperEngine(args.model, args.device, args.beam).load()
    except Exception as e:  # a missing model or a broken CUDA install: say which
        print(f"\nCould not load the model: {e}")
        return 1
    print(f"\r\033[K", end="")

    try:
        source = FileSource(args.file) if args.file else SystemAudioSource(args.sink)
    except (CaptureError, OSError, RuntimeError) as e:
        print(f"Cannot read audio: {e}")
        return 1

    step = args.step or (ASR_STEP_S_GPU if engine.device == "cuda" else ASR_STEP_S_CPU)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    transcript = LOGS_DIR / f"transcript-{stamp}.txt"
    model_line = (f"{args.model} on {engine.device}/{engine.compute_type}, beam {args.beam}, "
                  f"loaded in {engine.load_seconds:.1f}s")

    if not args.no_debug:
        log.open(new_log_path("live"), header=f"model {model_line}\nstep {step}s")

    recorder = None
    if args.record:
        from src.audio.recorder import SessionRecorder
        recorder = SessionRecorder(LOGS_DIR / f"session-{stamp}", {
            "source": args.file or f"system audio ({args.sink or 'default output'})",
            "model": args.model, "device": engine.device, "compute_type": engine.compute_type,
            "beam": args.beam, "step_s": step,
        })

    print("LiveTranscribe — step 1 (terminal)")
    print(f"  model    {model_line}")
    if engine.device != "cuda":
        print("           no GPU in use — passes are slow. `nvidia-smi` must work for CUDA.")
    print(f"  source   {source.label}")
    print(f"  step     {step:.1f}s between passes")
    print(f"  text     {transcript.relative_to(LOGS_DIR.parent)}")
    if log.enabled:
        print(f"  log      {log.path.relative_to(LOGS_DIR.parent)}")
    if recorder is not None:
        print(f"  record   {recorder.directory.relative_to(LOGS_DIR.parent)}/")
    print("Bold = final, grey = may still change.  Ctrl+C to stop.\n")

    pipeline = Pipeline(
        source, Streamer(engine),
        [ConsoleSink(), TranscriptSink(transcript, source.label)],
        step, recorder,
    )
    pipeline.run()

    summary = log.summary(step)
    log.close()
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
