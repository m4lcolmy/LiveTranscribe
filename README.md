# LiveTranscribe

Live Arabic subtitles for whatever the computer is playing: a browser video,
a lecture in a media player, a call. It listens to the speaker output, not the
microphone, transcribes it with Whisper, and shows it at the bottom of the
screen — entirely offline, like Xiaomi's AI Subtitles.

## Running it

```bash
./run.sh
```

A dark panel opens at the bottom centre of the screen. Its header says what it
is doing — `Loading Whisper small…`, then `● small · GPU — level -21 dB,
speech…` — and holds three buttons: **⏸** pause, **⚙** settings, **✕** quit.
Play anything with Arabic speech and the transcript fills the panel: each
finished line a paragraph, the line being spoken last, with words that may
still change dimmed.

- **The panel has a fixed size.** It does not grow or shrink with the text.
  Older lines move up; **scroll up** to read them. While you are scrolled up
  it stays where you are; scroll back to the bottom and it follows the speech
  again. Drag the corner grip to change the size.
- **Select and copy** anything in it — with the mouse, then Ctrl+C or
  right-click → Copy. Right-click → *Copy whole transcript* copies everything
  final (never the dimmed words). The window keeps the last 2000 lines;
  `logs/transcript-<time>.txt` has every line of the session.
- **Move it** by dragging the header. Size and position are remembered.
- **It stays on top** of other windows and shows on every workspace. It
  appears without taking the keyboard, and takes it only when you click into
  it (so Ctrl+C works).
- The **tray icon** has the same menu, plus Click-through (the panel ignores
  the mouse, so a video's controls under it keep working — turn it back off
  from the tray).

### Settings (⚙)

| Setting | What it does |
|---|---|
| Whisper model | The models in your Hugging Face cache (the app never downloads): `small` (0.5 GB, the tuned default) or `large-v3` (3.1 GB, most accurate, several times slower), or *Other model folder…* for any CTranslate2 Whisper model |
| Run on | Automatic (GPU when usable) · GPU · CPU |
| Precision | Automatic · float16 · int8 + float16 · int8. Automatic loads large models in int8 + float16 on the GPU: in float16 they need more than a 4 GB card has |
| Update every | How often Whisper runs: Automatic (1 s GPU, 2.5 s CPU) or 0.5–3 s |
| Listen to | The default output (and follow it when it changes), or one specific output |
| Ignore speech that is not Arabic | Whisper's language check per utterance; English speech is dropped instead of being written out in Arabic letters |
| Text size, Background | Font size, and how opaque the panel is |
| Show words that may still change | Off shows only final words — a little later, never changing |
| Click-through | As in the tray menu |

**Save** writes them to `~/.config/LiveTranscribe/LiveTranscribe.conf`, and
the app starts with them next time. A new model, device or precision reloads
the model in place (the panel notes it); a new source or interval restarts
listening; the rest applies at once. A command-line flag (`--model`,
`--device`, `--step`, `--sink`) overrides the saved value for that run only.

### Why it runs through XWayland

GNOME on Wayland lets no application place its own window or keep it above
others — deliberately — and it does not implement the layer-shell protocol
real overlays use elsewhere. So the window runs through XWayland, where both
are allowed: the same approach as Local-Live-Captions on GNOME (PLAN.md §6).
If a fullscreen video still covers the panel, try

```bash
./run.sh --bypass-wm     # outside the window manager: above everything; drag to move
```

and, as a last resort, Super + right-click the panel's header → *Always on Top*.

### Terminal mode

```bash
./run.sh --console
```

Step 1's mode: no window, the text in the terminal —

```
[00:08.2] ولكم نصف ما ترك أزواجكم إن لم يكن لهن ولد فإن كان لهن ولد فلكم الربع
بها أو دين ولهن الربع مما تركتم إن لم يكن لكم ولد       ← bold: final
فإن كان لكم ولد فلهن                                     ← grey: may still change
```

With nothing to show, a dim status line says whether audio is arriving:
`● listening — level -23 dB, gain 1.0×, speech…`. If the level says `silent`
while something is playing, the audio is going to a different output than the
default one (see `--sink`). Read Arabic in GNOME Terminal, not the VS Code
terminal, which does not lay right-to-left text out reliably.

Ctrl+C prints whether the machine kept up:

```
passes 37  latency p50=1573ms p90=1910ms max=2064ms  step=2500ms  overran=0 (0%)
lines 6  words 116  dropped segments: phantom=0 low_logprob=0 repetitive=0 loops cut=0  short blips=0
```

### Options

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
pw-record (speaker monitor) → AutoGain → Silero VAD → Streamer ⇄ Whisper → overlay / terminal + transcript
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
python -m pytest tests/ -q                               # streaming policy, gain, VAD, gates, overlay
python scripts/replay.py clip.mp4 --ref clip.txt         # CER/WER/coverage, latency, flicker
python scripts/replay.py logs/session-<time>/            # a --record session, again
python scripts/replay.py clip.mp4 --offline              # vs the whole file transcribed at once
python scripts/replay.py music.mp3 --silent              # a control clip must produce nothing
python scripts/offline.py clip.mp4 --out clip.txt        # a reference draft to correct by hand
python scripts/bench_latency.py clip.mp4                 # pass time by buffer / beam / threads
```

### The accuracy corpus

```bash
python scripts/fetch_corpus.py --discover     # YouTube videos with *human* Arabic captions
python scripts/fetch_corpus.py --fetch        # the clips in tests/clips/manifest.json, via CaptionForge
python scripts/fetch_corpus.py --inspect      # what each caption track turned out to be
python scripts/replay.py --manifest tests/clips/manifest.json --fixed-clock --offline
python scripts/replay.py --manifest tests/clips/manifest.json --fixed-clock --set max_buffer_s=10
```

12 four-minute clips from Al Jazeera, Al Jazeera Mubasher, DW Arabic, Sky News
Arabia and TEDx, plus two controls that must stay silent (instrumental music,
English speech). Each clip's `reference` says what its captions are, because
that decides what a score means: `human` (a person's verbatim transcript),
`asr-style` (a broadcaster's recogniser — agreement, not truth) or
`msa-rendering` (dialect speech captioned in standard Arabic — only the
streaming cost means anything). Audio and captions are fetched, not committed.
`--fixed-clock` makes the text reproducible for A/B tuning; `--set` and
`--engine-set` change one setting for the run; `--trace` prints every pass.

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
app.py                  entry point: transcript window, or --console
run.sh                  launcher (conda env livetranscribe)
src/
  config.py             every tunable constant, with the reasoning
  evaluation.py         CER/WER/coverage, latency, flicker
  session.py            what both front ends set up the same way
  sinks.py              terminal and transcript file
  ui/    overlay.py     the transcript window: fixed-size, scrollable, copyable
         settings.py    saved settings and the ⚙ dialog
         controller.py  model loading, the pipeline thread, applying settings, the menu
         tray.py        the tray icon
         x11.py         "on every workspace", which Qt has no API for
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
tests/                  test_streamer.py (scripted engine), test_components.py, test_overlay.py
```
