"""What both front ends set up the same way: source, step, log, transcript, recording.

The terminal mode (--console) and the overlay differ only in where the text
goes and which thread runs the loop; everything a session needs besides that
is built here, so the two cannot drift apart.
"""

import copy
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import ASR_BEAM_SIZE, ASR_STEP_S_CPU, ASR_STEP_S_GPU, LOGS_DIR
from src.core.debug import log, new_log_path


@dataclass
class Session:
    source: object
    step_s: float
    transcript: Path
    recorder: object | None
    model_line: str

    def header(self) -> list[str]:
        lines = [f"  model    {self.model_line}"]
        if "cuda" not in self.model_line:
            lines.append("           no GPU in use — passes are slow. `nvidia-smi` must work for CUDA.")
        lines += [
            f"  source   {self.source.label}",
            f"  step     {self.step_s:.1f}s between passes",
            f"  text     {self.transcript.relative_to(LOGS_DIR.parent)}",
        ]
        if log.enabled:
            lines.append(f"  log      {log.path.relative_to(LOGS_DIR.parent)}")
        if self.recorder is not None:
            lines.append(f"  record   {self.recorder.directory.relative_to(LOGS_DIR.parent)}/")
        return lines


def effective_args(cli, prefs):
    """The saved settings, with this run's command-line flags on top (never saved)."""
    args = copy.copy(cli)
    args.model = cli.model or prefs.model
    args.device = cli.device or prefs.device
    args.beam = cli.beam or ASR_BEAM_SIZE
    args.step = cli.step or (prefs.step_s or None)
    args.sink = cli.sink if cli.sink is not None else (prefs.sink or None)
    args.precision = prefs.precision
    args.arabic_only = prefs.arabic_only
    return args


def open_session(args, engine) -> Session:
    """Raises CaptureError / OSError / RuntimeError when there is no audio to read."""
    from src.audio.capture import SystemAudioSource
    from src.audio.filesource import FileSource

    source = FileSource(args.file) if args.file else SystemAudioSource(args.sink)
    step = args.step or (ASR_STEP_S_GPU if engine.device == "cuda" else ASR_STEP_S_CPU)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_line = (f"{args.model} on {engine.device}/{engine.compute_type}, beam {args.beam}, "
                  f"loaded in {engine.load_seconds:.1f}s")

    if log.enabled:
        log.close()
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
    return Session(source, step, LOGS_DIR / f"transcript-{stamp}.txt", recorder, model_line)
