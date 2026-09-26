"""Every tunable constant, with the reason it has its value.

These are step 1's starting values: from PLAN.md §5.5, from hifz, and from
Local-Live-Captions' measurements. None has been measured on this project's
own clips yet. When one is, the numbers go in its comment and in
MEASUREMENTS.md — a value without its reason is how tuning turns into guessing.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
# The sessions worth investigating are the ones nobody expected, so every run
# is logged; only the newest few are kept.
LOG_KEEP = 20

# ── Model ──────────────────────────────────────────────────────────────
# A faster-whisper size name, resolved from the Hugging Face cache (the app
# never downloads), or a path to a CTranslate2 model directory.
#
# hifz measured a small model degrading badly on 3-second windows. That is
# why the streamer below hands Whisper everything since the last commit
# (up to MAX_BUFFER_S) rather than short fixed windows. Swapping the model
# is a benchmark run (scripts/replay.py), never a silent change.
WHISPER_MODEL = "small"
DEVICE = "auto"              # "auto" | "cuda" | "cpu"
USE_FP16 = True              # ignored on CPU, which runs int8
# ctranslate2 threads on CPU (ignored on GPU). Its default is 4. Measured on
# the Ryzen 7 6800H (8 cores), small/int8, beam 5, 2026-09-26:
#   buffer   4 threads   8 threads
#    5 s      1507 ms     1135 ms
#   10 s      1734 ms     1353 ms
#   15 s      2004 ms     1508 ms
# One thread per physical core: a quarter off every CPU pass.
CPU_THREADS = 8

# ── Audio ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # what Whisper and Silero expect
CAPTURE_CHUNK_MS = 100       # pw-record read size, and AutoGain's block
# pw-record is a child process. When it dies (PipeWire restarts, the device
# vanishes) it is started again after these waits, the last one repeating.
CAPTURE_RESTART_BACKOFF_S = (0.5, 1.0, 2.0, 5.0)
# How often the default output is checked. When it changes (headphones
# plugged in, Bluetooth connected) capture is restarted on the new one, so it
# never keeps listening to a device nothing plays through any more.
SINK_WATCH_S = 2.0

# ── Gain ───────────────────────────────────────────────────────────────
# Local-Live-Captions measured that playback at an ordinary low volume
# (peak ~0.01) makes Whisper lose the first word of each utterance and
# garble the rest, and that make-up gain recovers the sentences exactly.
# The peak is taken over the last GAIN_WINDOW_S, so a pause between
# sentences — two orders of magnitude quieter — does not pump the gain up.
GAIN_TARGET_PEAK = 0.5
GAIN_MAX = 40.0
GAIN_WINDOW_S = 10.0
# Below this peak nothing is playing at all. Amplifying it 40x would turn the
# noise floor into something the VAD might call speech.
GAIN_FLOOR_PEAK = 0.002

# ── Voice activity ─────────────────────────────────────────────────────
# Silero, bundled with faster-whisper. A level threshold cannot tell music
# from a voice; Silero can. 0.5 is Silero's own default.
VAD_THRESHOLD = 0.5

# ── Streaming ──────────────────────────────────────────────────────────
# How often a Whisper pass runs. Each pass re-reads everything since the last
# commit, so a pass slower than the step loses no audio — the next one simply
# covers more. The step only has to stay above the pass time, or passes run
# back to back. CPU passes take seconds, hence the longer step there.
# CPU: a pass at the full 15 s buffer takes p90 1.6 s with CPU_THREADS=8
# (bench_latency.py, 2026-09-26), so 2.5 s keeps it at ~65% of the step with
# a video player competing for the same cores. GPU: not measured yet — the
# driver is not loaded (PLAN.md step 0.1).
ASR_STEP_S_GPU = 1.0
ASR_STEP_S_CPU = 2.5
# Speech needed before an utterance's first pass. Whisper on one second of
# audio mostly guesses, and a guess shown and then withdrawn is flicker.
MIN_FIRST_PASS_S = 1.2
# An utterance with less speech than this is a cough or a click. It is
# dropped without a pass, because Whisper would transcribe it as something.
MIN_UTTERANCE_SPEECH_S = 0.25
# Audio kept in front of the first speech frame. Silero misses a soft
# consonant onset, and Whisper needs it to get the first word.
LEAD_IN_S = 0.3
# Trailing silence that ends an utterance: its words are all committed and
# the line is closed. Local-Live-Captions settled on the same 0.7 s.
END_SILENCE_S = 0.7
# Audio kept after the last speech frame in an utterance's closing pass —
# enough for a word's tail, too little for Whisper to fill with a phantom.
TAIL_PAD_S = 0.2
# Once the buffer is longer than this, it is cut at the last committed point.
# 15 s keeps plenty of context and stays well inside Whisper's 30 s input.
MAX_BUFFER_S = 15.0
# Nothing agreed for this long (music under speech, a model that keeps
# changing its mind): commit what is there and cut, before Whisper's 30 s.
HARD_MAX_BUFFER_S = 25.0
# ...and if even then there was nothing to commit, keep only this much.
KEEP_AFTER_HARD_TRIM_S = 5.0
# Committed text fed back as initial_prompt, for continuity of names and
# topic. Only text whose audio has already left the buffer is ever used —
# text still in the buffer would be heard twice and repeated (hifz's
# echoed-tail problem). 0 turns it off.
PROMPT_CHARS = 200
# A committed line is closed once it reaches this length, even mid-sentence.
# Subtitles are read a line at a time; a line that never ends is not one.
LINE_MAX_CHARS = 90
# Whisper's word times wobble by about this much between passes over the same
# audio. A new word starting before the last committed word's end minus this
# is that word heard again, not a new one.
COMMIT_TIME_TOLERANCE_S = 0.1
# Agreement is a common prefix, so one word Whisper spells differently on every
# pass holds back every word after it. Measured on a 97 s room-mic recording
# (hifz's An-Nisa page), one hard word — الربع, heard as نرغبوا / ربع / نربع /
# إن الربع on four passes — held back 24 correct words for 12 s, and put the
# p90 commit latency at 17 s. So a disagreement of at most AGREE_MAX_GAP words
# is passed over when at least AGREE_MIN_RUN words agree right after it; the
# newest pass's spelling is committed, since it heard the most audio.
AGREE_MAX_GAP = 2
AGREE_MIN_RUN = 3
# The buffer is cut at a pause the VAD found (at least this many silent 32 ms
# frames), so a word is never sliced in half at the cut — half a word at the
# head of the next buffer is transcribed differently every pass, which is the
# unstable-word stall above. Without a pause, at the last committed segment end.
CUT_PAUSE_FRAMES = 2
# A word cut at the trim point is heard again at the head of the next buffer.
# Up to this many new leading words are checked against the committed tail.
ECHO_MAX_WORDS = 5

# ── Whisper decoding ───────────────────────────────────────────────────
# On CPU the encoder dominates a pass: beam 5 costs only 10-15% over beam 1
# (15 s buffer, 8 threads: 1508 vs 1320 ms), so the accuracy is kept.
ASR_BEAM_SIZE = 5
# Segment gates, from hifz, where they were tuned against real recordings.
# faster-whisper already drops a segment with no_speech_prob above the
# threshold AND avg_logprob below the floor (Whisper's own rule). hifz also
# dropped on no_speech_prob alone. That is not done here: speech over music —
# the commonest thing in system audio — can score high no_speech_prob and
# still be transcribed right. Every drop is logged with its text, so step 1B
# can see what each gate costs.
ASR_NO_SPEECH_THRESHOLD = 0.6
ASR_MIN_AVG_LOGPROB = -1.0
ASR_MAX_COMPRESSION_RATIO = 2.4

# How many tokens one pass may generate, per second of buffer. When Whisper
# loops — "يدخله نارا خالدا فيها" over and over — it decodes until the model's
# 448-token limit: an 8 s pass on CPU, measured live 2026-09-26. Arabic runs
# ~3 tokens a word in Whisper's vocabulary: 3.0 tokens/s for measured
# recitation, ~9 for fast news-reading. 15/s is 1.7x the fast case.
ASR_MAX_TOKENS_PER_S = 15
ASR_MIN_TOKENS = 24
# A segment that fails the compression gate because it loops is not thrown
# away whole: the words before the loop were usually right (the live case
# above lost a whole correct ayah to its repeated tail). A run of words that
# comes this many times in a row is the loop; one copy is kept.
LOOP_COPIES = 3

# Lines Whisper produces out of silence and music, copied from the credits of
# the YouTube subtitles it was trained on. A segment made of nothing but
# these is dropped and logged as PHANTOM. Compared after arabic.compare_key(),
# so diacritics, alef forms and punctuation do not matter. Grow this from the
# logs, the way hifz found its تَعْمَى.
PHANTOM_LINES = (
    "اشتركوا في القناة",
    "اشترك في القناة",
    "لا تنسوا الاشتراك في القناة",
    "ترجمة نانسي قنقر",
    "نانسي قنقر",
    "شكرا للمشاهدة",
    "شكرا على المشاهدة",
    "شكرا لكم على المشاهدة",
    "شكرا لمشاهدتكم",
    "موسيقى",
)
