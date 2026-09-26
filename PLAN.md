# LiveTranscribe — v1 plan

Live Arabic subtitles for whatever the computer is playing. The idea comes from
Xiaomi/Redmi AI Subtitles, but everything runs offline on the local GPU. It
listens to the speaker output rather than the microphone, transcribes it with
Whisper small, and shows the text in a small translucent box at the bottom
centre of the screen.

v1 is deliberately narrow: Arabic only, system audio only, one model, one
screen. It is built in three steps, and **step 1 is transcription and nothing
else**:

| Step | Goal | Output |
|---|---|---|
| **1. Transcription** | The text is right, it arrives fast, and it stays silent when nobody speaks | Terminal + live transcript file, with measured numbers |
| 2. Overlay | The step 1 text in a subtitle box at the bottom centre, above other windows | The app as a user sees it |
| 3. Packaging | Launcher, icon, README | Installable |

**Status 2026-09-26:** steps 0, 1A, 1B (first pass) and 2 are built. The GPU
is measured (float16, 1 s step). The accuracy corpus exists — 12 YouTube clips
with uploader-provided Arabic captions plus two silent controls, fetched through
CaptionForge — and changed the design: a shorter buffer (15 → 8 s, because
Whisper stops after the first sentence of a buffer), a per-utterance language
check (English speech was written out in Arabic letters), and a reproducible
fixed-clock replay for tuning (MEASUREMENTS.md). Step 2's two-line subtitle box
became, at the user's request, a fixed-size scrollable transcript panel with
selectable text and a saved-settings dialog (model, device, precision,
interval, source, text size, opacity). `src/config.py` holds the current
values, which supersede §5.5. Still open: the structural fix for the stalled
sentence (re-run a pass when speech is left untranscribed), a human-verbatim
reference for more than one clip, and step 3.

Step 2 does not start until step 1 passes its exit gate (§7). A good-looking
box around wrong text is worth nothing. Transcription is measured first, and
the overlay is fitted around it afterwards.

---

## 1. What the machine looks like (checked 2026-09-26)

| Thing | Found | Consequence |
|---|---|---|
| GPU | RTX 3050 Ti Mobile, 4 GB (plus AMD 680M iGPU) | Whisper small fp16 (~1 GB VRAM) fits easily |
| NVIDIA driver | **Not loaded.** Running kernel is `7.0.0-34`; the `nvidia-595-open` modules are installed only up to `7.0.0-31` | Fix first (§7, step 0). Until then everything runs on CPU |
| Session | GNOME 46 on **Wayland** | Matters only for step 2 (§6) |
| XWayland + Qt xcb | Works: `QT_QPA_PLATFORM=xcb` gives screen `eDP-1` 1920×1080, usable area from y=32; `libxcb-cursor0` installed | Step 2 path |
| Audio | PipeWire 1.0.5 + WirePlumber, `pw-record` present (no `--raw` flag in 1.0.5; writing to `-` gives raw PCM) | Capture with `pw-record` (§3) |
| System-audio capture | **Verified**: `pw-record -P '{ stream.capture.sink=true }' --rate 16000 --channels 1 --format s16 -` links to `alsa_output…C-Media…:monitor_FL/FR`, the sink Chrome plays into, and delivers raw 16 kHz mono | No PulseAudio tools, no loopback module, no Python audio package |
| Tray | `ubuntu-appindicators` extension enabled | Step 2 |
| Arabic fonts | Noto Sans Arabic, Noto Naskh Arabic, Noto Kufi Arabic, Vazirmatn, Amiri Quran | Step 2 |
| Model | `Systran/faster-whisper-small` already in `~/.cache/huggingface/hub` (464 MB, complete) | Load with `local_files_only=True`, no download needed |
| Python | conda at `~/anaconda3`; `hifz` env has faster-whisper 1.2.1, ctranslate2 4.7.1, PyQt6 6.11, cuBLAS/cuDNN cu12 wheels | Same versions for the new env |

---

## 2. What carries over from hifz and from similar projects

### From hifz

| Taken | Why |
|---|---|
| `src/core/device.py`: preloads cuBLAS/cuDNN from pip wheels, `resolve_device("auto")` | No `LD_LIBRARY_PATH`, and it falls back to CPU on its own. Copy verbatim |
| Whisper gates: `no_speech_threshold=0.6`, `log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4`, plus `_rejected_segment()` | Whisper invents text out of silence |
| `Engine` seam (`load()` / `transcribe()`) | Swapping the model becomes a setting that gets benchmarked, never a silent change |
| Session log on by default, newest 20 kept, every ASR pass logged with latency | The sessions worth investigating are the ones nobody planned to record |
| `--record` + replay through the real pipeline | hifz's bugs were found live and could not become tests once the audio was gone |
| Benchmark = the app's own code path at simulated real-time pace | The numbers describe the app, not an idealised offline run |
| `config.py`: every constant commented with why | Tuning without that history becomes guesswork |
| `numpy<2` pin, `HF_HUB_OFFLINE=1`, conda `run.sh` | Same environment discipline |

**Not taken: short fixed windows.** hifz fed Whisper 3–4 s windows. On those,
`whisper-small-quran` measured **23.4% false alarms against base's 4.6%**,
because *"the small model degrades badly on partial context"* (hifz
MEASUREMENTS.md). hifz could live with short windows because the Quran text
reconciles many noisy looks at every word. Free speech has no reference text.
→ This project uses a **growing buffer with local agreement** (§5).

### From Local-Live-Captions (the closest existing project)

[gitlon/Local-Live-Captions](https://github.com/gitlon/Local-Live-Captions) does
on-device live captions for Linux: Python, PyQt6, `pw-record`, Silero VAD, a
click-through overlay on Wayland and X11. It is useful proof that this stack
works, and three of its measured findings apply directly to step 1:

- **Auto gain.** They measured that system playback at an ordinary low volume
  (peak ≈ 0.01) *"loses the first word of each utterance and garbles the rest"*
  in Whisper, and that make-up gain recovers the sentences exactly. →
  `AutoGain` stage: target peak 0.5, max gain 40×, peak measured over the last
  10 s, so the gain holds steady across pauses.
- **Silero VAD, not a level threshold.** Music and noise have speech-level
  energy. Their utterance timing matches ours: 0.3 s lead-in, 0.7 s silence
  ends an utterance.
- **What their Whisper mode costs.** It transcribes (translates, in fact, with
  large-v3) each utterance only once it has *ended*, which gives 2–6 s latency.
  Our design also produces a whole-utterance pass when speech stops, and
  additionally shows the words while the utterance is still being spoken (§5).

### Rules carried forward

1. **Benchmark before swapping models.** "Bigger is not better" was measured in
   hifz, not assumed.
2. **Score the session, not only the final text.** hifz's "retracted reds"
   become *flicker* here: words shown and then changed.
3. **Coverage first.** A setting that shows less text makes fewer mistakes.
   Accuracy is always reported next to how much speech was shown at all.
4. **Silence controls must stay silent.** Music-only and silence clips must
   produce zero words.

---

## 3. Architecture (step 1: no UI at all)

```
 ┌──────────── capture thread ────────────┐
 │ pw-record (subprocess)                 │  -P { stream.capture.sink=true }
 │   stdout: s16le 16 kHz mono            │  --rate 16000 --channels 1 --format s16 -
 │ read 100 ms chunks → AudioRing         │  restart with backoff if it exits
 └───────────────┬────────────────────────┘
                 │ float32 + absolute sample index
 ┌───────────────▼──────── ASR worker thread ─────────────────────────┐
 │ every ASR_STEP:                                                     │
 │   AutoGain → Streamer.step(audio)     ← pure Python, unit-tested    │
 │     ├─ Silero VAD (speech? trailing silence?)                       │
 │     ├─ WhisperEngine.transcribe(buffer, prompt) → words + times     │
 │     ├─ LocalAgreement-2 → committed / tentative                     │
 │     └─ trim buffer; on silence: final pass, end the line            │
 │   → SubtitleState(lines, committed, tentative)                      │
 └───────────────┬─────────────────────────────────────────────────────┘
                 │
     step 1: ConsoleSink (terminal) + TranscriptSink (logs/transcript-*.txt)
     step 2: OverlayWindow (the same SubtitleState over a Qt signal)
```

The `Streamer` holds all the transcription logic. It knows nothing about Qt,
PipeWire or real time. The app, the unit tests (with a scripted fake engine)
and the replay benchmark all drive that same class. Step 2 adds only a new
consumer of `SubtitleState` and changes nothing upstream.

**Input sources in step 1** (all feed the same `Streamer`):

- `app.py`: live system audio.
- `app.py --file clip.mp4`: a local file, decoded with `av`, fed at real-time
  pace. You can watch it being transcribed exactly as it would be live.
- `scripts/replay.py`: the same, headless, with a simulated clock and metrics.

**Why a transcript file as well as the terminal:** the VS Code terminal does
not lay out Arabic right-to-left reliably. GNOME Terminal does. The live
`logs/transcript-*.txt` (committed lines with timestamps) opens correctly in
any editor, and it is what you read to judge whether the transcription is
*right*.

**`pw-record` details.** With `stream.capture.sink=true` and no `--target`,
WirePlumber links it to the default output's monitor. Still to check in step 0:
whether it follows a change of output (e.g. Bluetooth headphones). If not, the
capture thread polls `wpctl inspect @DEFAULT_AUDIO_SINK@` every 2 s and restarts
`pw-record`. Capturing a single app (`--target <serial>` from `pw-dump`, as
Local-Live-Captions does) is post-v1.

---

## 4. Components of step 1

```
src/
  config.py            every constant, with the reason for its value
  core/device.py       copied from hifz
  core/debug.py        session log, adapted from hifz
  core/arabic.py       normalize() for comparing hypotheses and for CER
  audio/capture.py     pw-record → AudioRing; restart; default-sink watch
  audio/filesource.py  --file: decode with av, pace in real time
  audio/gain.py        AutoGain
  audio/vad.py         Silero: speech spans, trailing silence
  audio/engine.py      Engine + WhisperEngine + gates + phantom blocklist
  audio/streamer.py    §5: buffer, agreement, trim, echo guard, finalise
  audio/worker.py      thread: source → gain → streamer → sinks
  audio/recorder.py    --record: session.flac + passes.jsonl
  sinks.py             ConsoleSink, TranscriptSink
app.py                 CLI entry (step 1 has no window)
scripts/
  bench_latency.py     pass latency by buffer length / beam / device
  replay.py            file or recorded session → streamer → metrics
  offline.py           whole-file transcription: the accuracy ceiling
tests/
  test_streamer.py  test_gain.py  test_arabic.py  test_engine_gates.py
  clips/  (audio gitignored, manifest.json committed)
```

`requirements.txt` (step 1):

```
faster-whisper>=1.2.1   # includes Silero VAD via onnxruntime
ctranslate2>=4.7.1
numpy>=1.26.0,<2
av>=17.0.1              # --file and replay decoding
soundfile               # --record (FLAC)
pytest>=9.0.0
# GPU: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
# PyQt6 arrives in step 2
```

---

## 5. The streaming algorithm

### 5.1 Why a growing buffer

faster-whisper pads every input to 30 s, so the encoder costs the same for 2 s
of audio as for 15 s. Short windows add no speed. They only take context away,
and hifz measured what that does to the small model. Each pass therefore
transcribes **all audio since the last commit point** (~1 s up to
`MAX_BUFFER_S`). A word becomes final when two consecutive passes agree on it
(LocalAgreement-2, from Macháček et al., *whisper_streaming*).

If inference is slower than `ASR_STEP`, the next pass simply covers more audio.
No audio is ever dropped. Updates arrive less often, that is all.

**The accuracy argument:** when speech stops, the final pass sees the whole
utterance (up to 15 s) in one go. That is the same input the offline,
whole-utterance approach sees. So the committed text of a short utterance
should match offline quality, and step 1 measures whether it does (STREAM GAP,
§7).

### 5.2 One step

```
step(new_audio):
    buffer.append(gain(new_audio))
    vad = silero(buffer)

    if no speech in buffer:
        buffer.keep_last(LEAD_IN_S)
        return state

    if speech_duration < MIN_FIRST_PASS_S and not vad.trailing_silence_long:
        return state

    words = engine.transcribe(buffer.audio,
                              prompt = committed text whose audio is NOT in the buffer)
    words = [w for w in words if w.end > committed_until]
    words = drop_boundary_echo(words, last 3 committed words)

    if vad.trailing_silence ≥ END_SILENCE_MS:          # the speaker stopped
        commit(words); end_line(); buffer.clear(); prev = []
        return state

    stable = longest_common_prefix(prev, words, key = arabic.normalize)
    commit(stable); prev = words[len(stable):]         # prev = tentative

    if buffer.duration > MAX_BUFFER_S:
        buffer.trim_to(end of last committed segment, else last committed word)
    if buffer.duration > HARD_MAX_BUFFER_S:            # nothing agreed for 25 s
        commit(prev); prev = []; buffer.trim_to(their end)

    return state(lines, committed, tentative = prev)
```

`arabic.normalize` (from hifz's `arabic.py`) strips tashkeel and tatweel,
unifies أ/إ/آ→ا, ى→ي and ة→ه, and removes punctuation. It is used only for
comparing text and for scoring. The text shown is Whisper's own.

### 5.3 Guards

- **Echo at the trim point.** A word cut at the boundary can be heard again at
  the start of the next buffer (hifz's "echoed tail"). If the first 1–3 new
  words equal the last 1–3 committed ones (normalised), they are dropped.
- **Prompt contamination.** Committed text goes back as `initial_prompt` for
  continuity, but only text whose audio has already been trimmed away.
  Otherwise the model repeats it, which is why hifz disables
  `condition_on_previous_text`. `PROMPT_CHARS` (200, 0 = off) is still
  A/B-tested.
- **Phantom lines.** On silence and music, Whisper produces lines copied from
  YouTube subtitle credits: `اشتركوا في القناة`, `ترجمة نانسي قنقر`,
  `شكرا على المشاهدة`, `موسيقى`… A segment matching the blocklist (normalised)
  is dropped and logged as `PHANTOM`. The list lives in `config.py` and grows
  from the logs.

### 5.4 Whisper call

```python
model.transcribe(audio, language="ar", task="transcribe",
                 beam_size=ASR_BEAM_SIZE,        # 5 vs 1–2, measured in step 0
                 temperature=0.0,                # no fallback re-decodes → no latency spikes
                 word_timestamps=True,           # needed for trimming
                 vad_filter=False,               # our Silero already ran
                 condition_on_previous_text=False,
                 initial_prompt=prompt or None,
                 no_speech_threshold=0.6, log_prob_threshold=-1.0,
                 compression_ratio_threshold=2.4)
```

`language="ar"` is forced, so the model cannot switch languages on music or
English loan words.

### 5.5 Starting values (step 1 replaces them with measured ones)

| Constant | Start | Meaning |
|---|---|---|
| `CAPTURE_CHUNK_MS` | 100 | pw-record read size |
| `GAIN_TARGET_PEAK` / `GAIN_MAX` / `GAIN_WINDOW_S` | 0.5 / 40 / 10 | from Local-Live-Captions' measurements |
| `ASR_STEP_S` | 1.0 GPU / 3.0 CPU | how often a pass runs |
| `MIN_FIRST_PASS_S` | 1.2 | speech needed before the first attempt |
| `LEAD_IN_S` | 0.3 | audio kept before speech onset |
| `END_SILENCE_MS` | 700 | trailing silence that ends a line |
| `MAX_BUFFER_S` / `HARD_MAX_BUFFER_S` | 15 / 25 | trim / force-commit (Whisper's limit is 30) |
| `AGREEMENT_N` | 2 | passes that must agree |
| `PROMPT_CHARS` | 200 | 0 = off |
| `ASR_BEAM_SIZE` | 5 | cut if latency requires |
| `VAD_THRESHOLD` | 0.5 | Silero's own default |

---

## 6. Step 2 preview: the overlay on GNOME Wayland (researched)

**The constraint, from GNOME itself.** An app cannot set always-on-top on
GNOME Wayland: *"there isn't a programmatic way to manipulate the window stack,
by design"* (Emmanuele Bassi, GNOME Discourse, 2025-09). GNOME also does not
implement `wlr-layer-shell`, the protocol real overlays use on KDE, Sway and
Hyprland. Existing apps get around this in four ways:

| # | How | Who does it | Result on GNOME 46 |
|---|---|---|---|
| **A** | **Run the window through XWayland** (`QT_QPA_PLATFORM=xcb`): `Tool \| WindowStaysOnTopHint \| FramelessWindowHint`, `WA_TranslucentBackground`, `WA_ShowWithoutActivating`, `setGeometry()` to bottom centre, **plus a `raise_()` timer every 2 s** against windows that raise themselves | Local-Live-Captions (its automatic fallback whenever layer-shell is missing, which on GNOME is always) | Works with nothing extra to install. X11 lets the app position itself, and mutter honours "above" for X11 windows. Already checked to start here |
| **B** | **A small GNOME Shell extension**: the app publishes a `KeepAbove` property on D-Bus; the extension finds the window by WM class and calls mutter's `window.make_above()` + `window.stick()` (and could `move_frame()` it) | Live Captions (abb128), via [gnome-live-captions-assistant](https://github.com/abb128/gnome-live-captions-assistant), ~180 lines, supports shell 45/46/47 | Native Wayland, crisp text with fractional scaling, visible on every workspace. Costs a one-time extension install and a re-login |
| C | `layer-shell-qt`: a true overlay layer above fullscreen | Local-Live-Captions, when it is installed | **Not possible on GNOME** (mutter lacks the protocol). Detect it, so KDE/Hyprland users get it for free |
| D | The user sets it by hand: Super + right-click → "Always on Top" / "Always on Visible Workspace" | Live Captions, Show Me The Key (documented as their GNOME answer) | Always works, but manual and forgotten after every restart |

**Step 2 plan:** A as the default, exactly the recipe Local-Live-Captions uses
on GNOME. Click-through via `WindowTransparentForInput`, toggled from the tray.
Their README notes that a click-through window cannot be grabbed, so an
"Adjust position" toggle temporarily makes it grabbable. Step 2's first task is
a spike that tests A over fullscreen Chrome and mpv video, across workspaces,
and for focus stealing. If A fails any of these (most likely: hidden under
fullscreen, or not following workspace switches), B comes next. That is ~100
lines of extension, modelled on abb128's, adding `move_frame()` for the bottom
centre position. D goes in the README either way.

Look (unchanged): translucent black rounded box, white Noto Sans Arabic ~28 px,
right-to-left, 2 lines, committed words full white, tentative words ~55%
opacity, fades after 5 s of silence, `Loading model…` / `⏸` / `CPU` states,
position remembered, 60% of the screen width, bottom edge 8% above the screen
bottom.

---

## 7. Phases

### Step 0: Ground (≈ half a day)

**0.1 GPU driver.** The kernel meta package moved to 7.0.0-34, but the NVIDIA
module meta package stayed at 7.0.0-31:

```bash
sudo apt install linux-modules-nvidia-595-open-generic-hwe-24.04   # → 7.0.0-34
sudo reboot
nvidia-smi
```

(Short term: boot 7.0.0-31 from GRUB → Advanced options.)

**0.2 Environment.**

```bash
conda create -n livetranscribe python=3.12 -y && conda activate livetranscribe
pip install -r requirements.txt nvidia-cublas-cu12 nvidia-cudnn-cu12
```

**0.3 Audio check.** While Arabic speech plays: capture 10 s with `pw-record`
and listen back. Record the peak level at 100%, 30% and 10% volume, and while
muted; this tells us how much work AutoGain has to do. Switch output to
Bluetooth mid-capture and see whether the stream follows.

**0.4 Speed** (`scripts/bench_latency.py`). Whisper small on the GPU, fp16 vs
int8_float16, beam 1/2/5, `word_timestamps` on/off, buffers of 2/5/10/15/25 s,
20 runs each → p50/p90 into MEASUREMENTS.md. CPU int8 the same, for the
fallback. This sets `ASR_STEP_S` and the beam size.

**Gate:** `nvidia-smi` works · captured speech is intelligible · p90 pass
latency at a 15 s buffer ≤ 60% of `ASR_STEP_S`.

### Step 1A: The transcription core (≈ 1–1.5 days)

The components in §4, in this order, each with its tests before the next one
starts:

1. `arabic.py`, `gain.py`, `vad.py` + tests (quiet fixture: speech at 0.05× must
   come back at target level; silence must not be amplified into "speech").
2. `engine.py` + gates + blocklist + tests.
3. `streamer.py` + `test_streamer.py` with a scripted fake engine: agreement
   commits only the stable prefix, silence finalises a line, trimming never
   loses or duplicates a word, the echo guard drops boundary repeats, the hard
   limit force-commits, and no-speech audio produces nothing.
4. `capture.py`, `filesource.py`, `worker.py`, sinks, `app.py`.
   `python app.py` prints committed text white and tentative text grey, and
   writes `logs/transcript-*.txt`.
5. `recorder.py` (`--record`), `replay.py`, `offline.py`.

### Step 1B: Measure and tune (≈ 1–2 days)

**Corpus** (`tests/clips/`, audio gitignored, manifest committed):

| Clip type | Why |
|---|---|
| MSA news (Al Jazeera style) | the best case, which should be good |
| Lecture / khutbah | long sentences, one speaker |
| Egyptian, Levantine, Gulf podcast | dialects, where small is expected to be weaker |
| Quran recitation | common in this user's listening |
| Speech over music | gates vs coverage |
| Quiet playback (10% volume) | AutoGain |
| **Music only, silence, non-Arabic speech** | **must produce 0 words** |

References: run `offline.py` on a 1–2 minute clip and correct its text by hand.
That is much faster than typing it from scratch.

**Metrics**, always reported together:

```
COVERAGE          committed words / reference words
CER / WER         vs the corrected reference (after normalize(); CER leads,
                  since WER is harsh on Arabic clitics)
STREAM GAP        CER(streaming) − CER(offline whole-file, same model): what streaming costs
LATENCY           word end → first shown (tentative) and → committed: p50 / p90
FLICKER           tentative words shown then changed, per minute
STAYED SILENT     control clips with 0 words
REAL-TIME         pass p90 / step; passes that ran late
```

Tune in this order: `ASR_STEP_S`, then `END_SILENCE_MS`, then `PROMPT_CHARS`,
then beam, then `MAX_BUFFER_S`, then the VAD threshold. Each change goes into
MEASUREMENTS.md with the numbers before and after.

### Step 1 exit gate: "it truly transcribes"

1. **STREAM GAP ≤ 2 CER points** on the MSA and lecture clips. Streaming must
   cost almost nothing against the same model run on the whole file.
2. **COVERAGE ≥ 95%** of the reference on every speech clip, including quiet
   playback.
3. **STAYED SILENT** on every control clip: 0 words.
4. GPU latency p90: **tentative ≤ 1.5 s, committed ≤ 3 s**.
5. 30 minutes of a live Arabic stream without a stall, with flat memory, and
   with `pw-record` surviving an output change.
6. `pytest` green. MEASUREMENTS.md holds the numbers, including honest per-type
   results for dialects.

If gate 1 fails, the streaming policy needs fixing (step, agreement, trimming).
If the offline ceiling itself is poor on some clip type, that is the model's
limit. It is recorded, not tuned around, and it becomes the input to a later
model decision (benchmark first, per hifz).

### Step 2: Overlay (≈ 1 day)

Spike A (§6) → overlay window + tray + `--console` kept for debugging. It
fades, never takes focus, remembers its position, has a click-through toggle,
and stops `pw-record` cleanly on quit. Extension B only if the spike requires
it.

**Gate:** 20 minutes of a fullscreen Arabic video with the subtitles visible
throughout and nothing getting in the way.

**Built 2026-09-26, checked from outside the app** (xprop/xwininfo on the live
window, through XWayland): `_NET_WM_STATE_ABOVE` and `_NET_WM_STATE_STICKY`
set, `_NET_WM_DESKTOP` = all, WM_HINTS input = False (never focused),
skip-taskbar/pager, geometry 1100×136 at x=409 — centred on 1920 — with the
bottom edge 8% above the usable area, input shape = the drawn box only, tray
item registered with the AppIndicator host, Ctrl+C exits with no pw-record
left behind. mutter marks the window DEMANDS_ATTENTION when it maps; GNOME
Shell 46 ignores that for skip-taskbar windows (windowAttentionHandler.js), so
no "is ready" notification. `--bypass-wm` gives an override-redirect window
(fallback if fullscreen video covers the managed one). Not checkable from
outside, so the gate stays a human test: stacking over fullscreen Chrome/mpv,
dragging, the menu, click-through in use.

### Step 3: Packaging (≈ half a day)

`run.sh`, `.desktop`, icon, README (install, run, troubleshooting: GPU driver
trap, overlay fallbacks A→B→D, audio).

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Whisper small is weak on dialects and noisy audio | High for dialects | MSA is the v1 target. Per-type numbers in step 1B. A later model swap is benchmarked, not assumed |
| Hallucinations on music and silence | High without guards | Silero VAD, three gates, phantom blocklist, silence controls in the gate |
| Low playback volume garbles words | Measured by Local-Live-Captions | AutoGain, plus a quiet clip in the corpus |
| Flicker of tentative words | Medium | LocalAgreement-2, dimmed tentative text, `MIN_FIRST_PASS_S`, FLICKER metric |
| Overlay hidden under fullscreen video on GNOME | Medium | A → B → D (§6) |
| Driver breaks again on the next kernel update | Already happened | CPU fallback, visible as a `CPU` badge. README documents the fix |
| Capture does not follow an output change | Unknown | Step 0.3, then restart on default-sink change |

---

## 9. Not in v1

Translation · other languages and auto-detection · microphone input · capturing
a single app · choosing a monitor · global hotkeys · saving transcripts beyond
the log file · speaker labels · bigger models.

---

## 10. Sources

- Local-Live-Captions: <https://github.com/gitlon/Local-Live-Captions> (`overlay.py` backend
  selection, re-raise timer; `pipeline.py` AutoGain and utterance timing; `audio.py` pw-record/pw-dump)
- Live Captions GNOME extension: <https://github.com/abb128/gnome-live-captions-assistant> (`extension.js`)
- GNOME Discourse, "Any way to set window always on top programmatically?":
  <https://discourse.gnome.org/t/any-way-to-set-window-always-on-top-programmatically/31579>
- Show Me The Key README (GNOME Wayland: manual Always on Top): <https://github.com/AlynxZhou/showmethekey>
- whisper_streaming (LocalAgreement policy): Macháček, Dabre, Bojar, *Turning Whisper into Real-Time Transcription System*, 2023
