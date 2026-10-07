# LiveTranscribe — for developers

How it works, how it is measured, and where things are. For installing and
using the app, see the [README](../README.md). The plan and its history are in
[PLAN.md](../PLAN.md); every measurement and what it changed is in
[MEASUREMENTS.md](../MEASUREMENTS.md).

## How it works

```
pw-record (speaker monitor) → AutoGain → Silero VAD → Streamer ⇄ Whisper → window / terminal + transcript file
```

- **Capture** (`src/audio/capture.py`) — `pw-record` on the default output's
  monitor, 16 kHz mono. Restarted if it dies, and moved when the default output
  changes. Pause stops the process; nothing is captured while paused.
- **Gain** (`gain.py`) — quiet playback is brought up to a level Whisper hears
  well (measured: at ~1% volume Whisper loses first words); loud playback is
  left alone.
- **Voice activity** (`vad.py`) — Silero, scored once per 32 ms frame, with
  its recurrent state carried between pieces so streaming equals one pass.
- **Streamer** (`streamer.py`) — every step, Whisper transcribes *all audio
  since the last commit* (a small model is much worse on short windows). A word
  is final when two passes agree on it; one word spelled differently each time
  does not hold up the rest. When the speaker pauses (0.7 s) a closing pass
  hears the whole utterance and the line ends. Past 8 s the buffer is cut at a
  pause (Whisper stops after the first sentence of a longer buffer). A pass
  that stops short of the last one leaves the grey words after its end on
  screen instead of erasing them.
- **Language** — once the buffer holds 6 s, Whisper's language detection is
  asked; two "not Arabic" verdicts in a row drop the speech (shorter audio
  misjudges Arabic too often to act on).
- **Gates** (`engine.py`) — segments that loop, have very low confidence, or
  are one of Whisper's invented lines (`اشتركوا في القناة`, `ترجمة نانسي قنقر`…)
  are dropped; a loop keeps the words before it, also when it runs across
  segments; each segment is judged on its own text, not on the decode window
  it came in. A pass may generate at most 15 tokens per second of audio.

### Deepgram

```
pw-record → AutoGain → Silero VAD → DeepgramStreamer ⇄ wss://api.deepgram.com (Nova-3) → window / terminal + transcript file
```

Chosen as the model `deepgram` (`src/audio/deepgram.py`). It replaces the
**Streamer**, not the engine: the Streamer re-sends everything since the last
commit every second, which Deepgram would bill eight times over. Its live API
takes the audio once and answers with interim words (shown grey) and final
ones (white), so `DeepgramStreamer` maps those onto the same `Update`s, behind
the same `feed()/step()/finish()`. `session.make_engine()` and
`make_streamer()` pick the pair.

- **Only speech is sent**: Silero opens an utterance (with the 0.3 s lead-in)
  and closes it after 0.7 s of silence with a `Finalize`; the answer marked
  `from_finalize` ends the line, or 2 s without one does. `KeepAlive` every
  5 s holds the socket in a pause; after 30 s without speech it is closed and
  reopened at the next speech. `DEEPGRAM_SEND_SILENCE` sends everything.
- **Times**: Deepgram counts only the audio it was sent, so each connection
  keeps a map from its clock back to the stream's.
- **Drops**: reconnect with backoff, send the utterance again from its start,
  drop words already final by their stream time. 401/402/403 stop with the
  reason in red; there is nothing to retry.
- **The key** (`src/core/keystore.py`): in the system keyring through
  `keyring` (Secret Service — GNOME Keyring), never in the settings file; a
  key the first version left there is moved on start. No keyring, no saving:
  `DEEPGRAM_API_KEY` instead, which always wins and is never saved. The
  dialog's field is write-only — a saved key shows as its last four
  characters and is read only to connect or to **Test**. It goes in the
  `Authorization` header only — never the URL, the log, the transcript or
  `--record`'s metadata (`tests/test_deepgram.py` checks). `load()` checks it
  against `/v1/auth/token`, so a bad key fails the way a missing model does.
  The tests run on an in-memory keyring (`tests/conftest.py`).
- **Not available**: the "not Arabic" filter (it is Whisper's own language
  detection), and `scripts/replay.py` — every replay would cost credit.

Every constant is in `src/config.py`, with the measurement behind its value.
The look — colours, radii, stylesheets — is in `src/ui/theme.py`: greyscale
and one accent, and no colour typed anywhere else.

### The window on GNOME

GNOME on Wayland lets no application place its own window or keep it above
others, and has no layer-shell. The window runs through XWayland
(`QT_QPA_PLATFORM=xcb`): frameless, always on top, re-raised every 2 s, on
every workspace (an EWMH hint Qt has no API for, `src/ui/x11.py`).
Click-through sets the X Shape *input region* to the header alone, so the
transcript passes clicks while ⏸ ⚙ ✕ and 🔒 stay clickable — Qt's own flag is
all or nothing, and was a trap. `--bypass-wm` makes the window
override-redirect (above fullscreen video, outside the window manager).

## Running from source

```bash
./run.sh                          # the window
./run.sh --console                # no window: the text in the terminal (GNOME Terminal for Arabic)
./run.sh --file lecture.mp4       # a file, at the pace it would play
./run.sh --record                 # keep the session: logs/session-<time>/
./run.sh --sink <node.name>       # a specific output (names: wpctl status)
./run.sh --model large-v3 --device cuda --step 1.5   # this run only; not saved
DEEPGRAM_API_KEY=… ./run.sh --model deepgram         # Deepgram, this run only
./run.sh --reset-settings         # forget saved settings first
./run.sh --bypass-wm              # outside the window manager
```

`run.sh` uses `.venv` (made by `install.sh`), else the conda environment
`livetranscribe`, else `python3`. Dependencies: `requirements.txt`, plus
`requirements-gpu.txt` for NVIDIA's CUDA libraries, which
`src/core/device.py` loads from the pip packages itself.

## Measuring

```bash
python -m pytest tests/ -q                               # streaming policy, gain, VAD, gates, window, settings
python scripts/replay.py clip.mp4 --ref clip.txt         # CER/WER/coverage, latency, flicker
python scripts/replay.py logs/session-<time>/            # a --record session, again
python scripts/replay.py clip.mp4 --offline              # vs the whole file transcribed at once
python scripts/replay.py music.mp3 --silent              # a control clip must produce nothing
python scripts/offline.py clip.mp4 --out clip.txt        # a reference draft to correct by hand
python scripts/bench_latency.py clip.mp4                 # pass time by buffer / beam / threads
```

`replay.py` runs the same Streamer and gain as the app, on a simulated clock
that behaves like the live loop. `--fixed-clock` schedules passes exactly one
step apart so the text is reproducible for A/B tuning; `--set name=value`
changes a Streamer setting and `--engine-set` a faster-whisper guard for one
run; `--trace` prints every pass.

### The accuracy corpus

```bash
python scripts/fetch_corpus.py --discover     # YouTube videos with uploader-provided Arabic captions
python scripts/fetch_corpus.py --fetch        # the clips in tests/clips/manifest.json, via CaptionForge
python scripts/fetch_corpus.py --inspect      # what each caption track turned out to be
python scripts/replay.py --manifest tests/clips/manifest.json --fixed-clock --offline
```

12 four-minute clips (Al Jazeera, Al Jazeera Mubasher, DW Arabic, Sky News
Arabia, TEDx) and two controls that must stay silent (instrumental music,
English speech). Each clip's `reference` says what its captions are, which
decides what a score means: `human` (a person's verbatim transcript),
`asr-style` (a broadcaster's recogniser — agreement, not truth) or
`msa-rendering` (dialect captioned in standard Arabic — only the streaming
cost means anything). Audio and captions are fetched, not committed. YouTube
rate-limits bursts; fetching is paced and stops at the first refusal.

## Debugging a session

Every run logs to `logs/live-<time>.log` (newest linked as
`logs/live-session.log`, 20 kept; `--no-debug` turns it off): every pass with
its buffer and latency, what Whisper heard, what was committed, what stayed
tentative, and every dropped segment with its reason and text. The summary
at the end says whether the machine kept up. With Deepgram, `DG` lines show
each connection, every final result and how far behind the stream it came,
and the summary gives that lag's p50/p90.

## Pictures and icons

```bash
python packaging/make_icons.py        # packaging/icons/*.png — the app icon at every size, and the tray's
python packaging/make_screenshots.py   # docs/images/*.png — rendered from the real widgets
```

The app menu and the dock show a *copy* of the icon, made by `install.sh` in
`~/.local/share/icons/hicolor`: after redrawing it, run `./install.sh` again
(it only redoes what is missing, and always refreshes the icon).

## Layout

```
app.py                  entry point: the window, or --console
run.sh                  launcher: .venv, else conda env livetranscribe, else python3
install.sh              environment, model, app menu entry (--uninstall, --purge)
requirements.txt        + requirements-gpu.txt (CUDA libraries)
packaging/              icons, the .desktop template, make_icons.py, make_screenshots.py
docs/                   this file, images/ for the README
src/
  config.py             every tunable constant, with the reasoning
  evaluation.py         CER/WER/coverage, latency, flicker
  session.py            what both front ends set up the same way; saved settings + flags
  sinks.py              terminal and transcript file
  ui/    overlay.py     the transcript window: fixed-size, scrollable, copyable, translate button
         settings.py    saved settings and the ⚙ dialog
         theme.py       every colour, radius and stylesheet the app draws with
         icons.py       the line icons, drawn with QPainter
         translate.py   Google Translate for selected text: request, popup
         controller.py  model loading, the pipeline thread, applying settings, the menu
         single.py      one instance at a time
         tray.py        the app / tray icon
         x11.py         every workspace, and the click-through input region
  core/  device.py      GPU detection, CUDA libraries from pip
         keystore.py    the Deepgram key in the system keyring
         arabic.py      comparison keys for Arabic words
         debug.py       the session log
  audio/ capture.py     pw-record on the output monitor
         filesource.py  --file
         gain.py        make-up gain
         vad.py         streaming Silero VAD
         engine.py      Whisper, gates, loop cutting, language detection
         deepgram.py    Deepgram in the cloud: key check, WebSocket, its own streamer
         streamer.py    the streaming policy
         worker.py      the live loop
         recorder.py    --record
scripts/                replay.py, offline.py, bench_latency.py, fetch_corpus.py
tests/                  test_streamer.py (scripted engine), test_deepgram.py (fake socket),
                        test_components.py, test_overlay.py
```
