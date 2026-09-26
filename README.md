# LiveTranscribe

Live Arabic subtitles for whatever the computer is playing: a browser video,
a lecture in a media player, a call. It listens to the speaker output, not the
microphone, and transcribes it with Whisper, entirely offline.

**This is step 1 of the plan (PLAN.md): transcription only, in the terminal.**
The subtitle window at the bottom of the screen is step 2, and waits until the
text is measurably right.

## Running it

```bash
./run.sh
```

Then play anything with Arabic speech. Each finished line scrolls up with its
time; the line being spoken is redrawn in place:

```
[00:08.2] ولكم نصف ما ترك أزواجكم إن لم يكن لهن ولد فإن كان لهن ولد فلكم الربع
بها أو دين ولهن الربع مما تركتم إن لم يكن لكم ولد       ← bold: final
فإن كان لكم ولد فلهن                                     ← grey: may still change
```

With nothing to show, a dim status line says whether audio is arriving:
`● listening — level -23 dB, gain 1.0×, speech…`. If the level says `silent`
while something is playing, the audio is going to a different output than the
default one (see `--sink`).

Ctrl+C stops it and prints whether the machine kept up:

```
passes 37  latency p50=1573ms p90=1910ms max=2064ms  step=2500ms  overran=0 (0%)
lines 6  words 116  dropped segments: phantom=0 low_logprob=0 repetitive=0 loops cut=0  short blips=0
```

**Read Arabic in GNOME Terminal**, not the VS Code terminal — the latter does
not lay right-to-left text out reliably. Every run also writes the finished
lines to `logs/transcript-<time>.txt`, which any editor shows correctly.

```bash
./run.sh --file lecture.mp4      # a file, at the pace it would play
./run.sh --record                # keep the session: logs/session-<time>/
./run.sh --sink <node.name>      # a specific output (names: wpctl status)
./run.sh --device cpu            # or cuda; default picks the GPU when usable
./run.sh --step 1.5 --beam 5     # override the tuned values
```

## Installing

```bash
conda create -n livetranscribe python=3.12 -y && conda activate livetranscribe
pip install -r requirements.txt
```

From the system: PipeWire's `pw-record` and `pw-metadata` (package
`pipewire-bin`, present on Ubuntu 24.04), and the model in the Hugging Face
cache — `Systran/faster-whisper-small`. The app never downloads anything; a
missing model is an error.

**GPU.** The app uses CUDA when `nvidia-smi` works and falls back to CPU
otherwise, saying so at start. On Ubuntu's HWE kernels the NVIDIA module
package can lag a kernel update, leaving the GPU invisible:

```bash
sudo apt install linux-modules-nvidia-595-open-generic-hwe-24.04 && sudo reboot
```

On CPU (Ryzen 7 6800H) a pass takes ~1.5 s, so the step is 2.5 s and text
appears ~3 s after it is spoken and is final ~6 s after. The GPU step is 1 s.

## How it works

```
pw-record (speaker monitor) → AutoGain → Silero VAD → Streamer ⇄ Whisper → terminal + transcript
```

- **Capture** — `pw-record` on the default output's monitor, 16 kHz mono.
  Restarted if it dies, and moved when the default output changes.
- **Gain** — quiet playback is brought up to a level Whisper hears well; loud
  playback is left alone.
- **Streamer** (`src/audio/streamer.py`) — every step, Whisper transcribes
  *all audio since the last commit*, not a short window: a small model is much
  worse on short windows (measured in hifz). A word becomes final when two
  passes agree on it; one word spelled differently each time does not hold up
  the rest. When the speaker pauses (0.7 s), one closing pass hears the whole
  utterance and the line ends. Past 15 s the buffer is cut at a pause.
- **Gates** — segments that loop, have very low confidence, or are one of
  Whisper's invented lines (`اشتركوا في القناة`, `ترجمة نانسي قنقر`…) are
  dropped; a looping segment keeps the words before the loop.

Every constant is in `src/config.py`, with the reason for its value.

## Measuring

```bash
python -m pytest tests/ -q                               # the streaming policy, gain, VAD, gates
python scripts/replay.py clip.mp4 --ref clip.txt         # CER/WER/coverage, latency, flicker
python scripts/replay.py logs/session-<time>/            # a --record session, again
python scripts/replay.py clip.mp4 --offline              # vs the whole file transcribed at once
python scripts/replay.py music.mp3 --silent              # a control clip must produce nothing
python scripts/offline.py clip.mp4 --out clip.txt        # a reference draft to correct by hand
python scripts/bench_latency.py clip.mp4                 # pass time by buffer / beam / threads
```

`replay.py` runs the same Streamer and gain as the app, on a simulated clock
that behaves like the live loop, so its latencies are this machine's. Results
so far are in MEASUREMENTS.md.

## Debugging a session

Every run logs to `logs/live-<time>.log` (newest linked as
`logs/live-session.log`, 20 kept; `--no-debug` turns it off): every pass with
its buffer and latency, what Whisper heard, what was committed, what stayed
tentative, and every dropped segment with its reason and text.

## Layout

```
app.py                  entry point (step 1: terminal)
run.sh                  launcher (conda env livetranscribe)
src/
  config.py             every tunable constant, with the reasoning
  evaluation.py         CER/WER/coverage, latency, flicker
  sinks.py              terminal and transcript file
  core/  device.py      GPU detection, CUDA libraries from pip (from hifz)
         arabic.py      comparison keys for Arabic words
         debug.py       the session log
  audio/ capture.py     pw-record on the output monitor
         filesource.py  --file
         gain.py        make-up gain
         vad.py         streaming Silero VAD
         engine.py      Whisper, gates, loop cutting
         streamer.py    the streaming policy
         worker.py      the live loop
         recorder.py    --record
scripts/                replay.py, offline.py, bench_latency.py
tests/                  test_streamer.py (scripted engine), test_components.py
```
