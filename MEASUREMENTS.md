# Measurements

What the benchmarks found, and what each finding changed. Newest first.
Machine: Ryzen 7 6800H (8 cores), RTX 3050 Ti Mobile 4 GB (driver 595.91).

## 2026-09-26 — the accuracy corpus (step 1B)

### What the references are

`scripts/fetch_corpus.py --discover` searched YouTube with the Subtitles/CC
filter and kept a video only if it has a *manual* Arabic track and YouTube
says it is spoken in Arabic: 76 of ~300. Most Al Jazeera "subtitled" videos
failed that — their manual tracks were English, Persian, French and Turkish
translations. CaptionForge's own output folder was no help either: those SRTs
were written by Whisper large-v3 whenever a video had no track.

12 clips, 4 minutes each (48 min), fetched through CaptionForge. Reading the
tracks (`--inspect`) showed three kinds, which decide what a score means:

```
reference       clips   what it is                                         CER against it is
human             1     DW: punctuated, full spelling, a person's work     accuracy
asr-style         8     Al Jazeera ×6, Sky News, DW ×1: no hamza, ه for ة,  agreement with a strong
                        no punctuation, disfluencies kept, 7 s cues        in-domain recogniser
msa-rendering     3     TEDx dialect talks: human, but rendered into MSA   nothing — only STREAM GAP
                        (Beiruti "ما بتعرفي شو بيسموها" captioned
                        "ألا تعرفين ماذا يسمونها؟")
```

Two controls: an hour of instrumental music (3 min scored) and an Al Jazeera
English report — both must produce no words.

YouTube answered discovery's ~300 metadata requests with HTTP 429 / "confirm
you're not a bot" for ~10 minutes; fetching is paced and stops at the first
refusal.

### Baseline (measured clock, 1 s step, GPU)

```
type·reference          clips  words   COVER    CER   offCER    GAP   final p90
report·asr-style            2    772   99.4%   8.3%    7.0%   +1.4      2.81 s
talk·asr-style              5   1980   97.0%   9.6%    8.7%   +0.9      3.16 s
dialect·asr-style           1    503   90.7%  16.8%   16.5%   +0.3      3.41 s
talk·human                  1    449   79.7%  26.6%   16.1%  +10.5      8.97 s
dialect·msa-rendering       3   1306   92.0%  31.2%   30.4%   +0.8      2.99 s
```

On news in standard Arabic, streaming costs about a point over the same
model on the whole file, and words are final within ~3 s. The outlier is the
one clip with a human transcript.

### Whisper stops after the first sentence

On the DW clip, whole passes came back empty for up to 8 s while speech
played. Not faster-whisper's window-level no-speech skip (switching it off
changed nothing on a fixed clock). Replaying one of those buffers — 8.5 s,
88% speech by Silero — Whisper transcribed the first sentence (0.0–5.5 s,
already committed) and ended there; the next sentence, in the last 3 s, never
came out. The buffer keeps committed sentences until MAX_BUFFER_S (15 s), so
for seconds on end every pass re-heard the old sentence and dropped the new.

### The measured clock is not reproducible

With the replay clock driven by measured pass times, one slow pass shifts
every later buffer boundary; the DW clip gave 79.7% and 91.3% coverage on
two runs of identical settings. `replay.py --fixed-clock` schedules passes
exactly one step apart, so the text is reproducible (checked: two runs,
identical); latencies are still measured. All tuning below is fixed-clock.

### MAX_BUFFER_S: 15 → 8

```
max   coverage   CER     DW (human ref)    report CER   asr-style talk CER
 15     92.8%    18.4%    69.3% / 39.8%       8.6%           9.6%
 10     94.1%    17.6%    83.3% / 28.5%       8.0%          10.5%
  8     94.6%    16.9%    92.2% / 19.5%       8.7%          10.1%
  6     92.6%    18.9%    69.5% / 37.1%       9.5%          10.6%
```

A shorter buffer drops committed sentences sooner, so Whisper starts where the
new speech is. 8 s wins overall; on the clean clips a longer buffer is half a
point better, which the DW loss outweighs. 6 s starves the model of context.
The DW column is not monotonic — 69 / 92 / 83 — so this tunes a symptom. The
cure: when Silero hears speech after the last word a pass returned, cut at
the last committed word and run again at once. Not built yet.

Latencies in these four runs are inflated: they shared the GPU (4 processes).

### Controls, and the language check

Instrumental music: silent — also with faster-whisper's window-level
no-speech skip turned off. English speech: 379 words in Arabic letters, with
passes up to 1.5 s (forced Arabic on English decodes badly).

A language check per utterance (Whisper's detect_language, P(Arabic) < 0.2 →
drop), made at the utterance's start, cut the English to 3 words — and cost
the Arabic clips 3 points of coverage (94.6% → 91.7%; MSA talk CER 10.1% →
12.0%). Whisper's language detection on short audio, measured over every
clip (share of speech windows with P(Arabic) < 0.2):

```
window   Arabic clips (12)                     English control
1.5 s    5-30%  (DW 30%, Sky 21%, Mourinho 19%)     94%
3 s      0-17%  (DW 17%, Sky 5%)                    96%
6 s      0-6%   (DW 6%, the rest <= 1%)             96%
```

An utterance's start is ~1.5 s — exactly where the detector confuses Arabic.
Now the check waits until the buffer holds 6 s (LANGUAGE_MIN_AUDIO_S) and
rechecks every 5 s; shorter utterances are not judged.

That first cut still wrecked the DW clip (coverage 25%): once judged foreign
the buffer was cut back to a lead-in, so it never again held 6 s to judge on,
and the verdict stuck until a pause — rare in DW's talk over a music bed. Now
foreign speech keeps its last 7 s (untranscribed) for the next verdict, and it
takes two "not Arabic" verdicts in a row, 2 s apart, to drop anything.

```
                          DW (human ref)      11 other Arabic clips      English control
no check                  92.2% / 19.5%       as in the table above        379 words
check at utterance start  80.2% / 25.6%       coverage -3 points             3 words
6 s, stuck verdict        25.2% / 74.9%       identical to no check         42 words
6 s, two verdicts, kept   92.2% / 19.5%       identical to no check *       48 words
```

\* not re-run: with a single verdict the check never fired on them, and two
verdicts in a row can only fire less. P(Arabic) < 0.2 and < 0.05 gave
identical results. The English that still gets through is each utterance's
first 6-8 s, before there is enough audio to judge — the price of not
misjudging short Arabic.

## 2026-09-26 — the GPU (step 0)

`bench_latency.py`, Whisper small, the 97 s room-mic recitation, p50 of 5 runs:

```
compute        beam   2 s     5 s    10 s    15 s    25 s
float16          1   129 ms  152 ms  203 ms  256 ms  552 ms
float16          5     *     185 ms  237 ms  292 ms  449 ms
int8_float16     1   124 ms  148 ms  195 ms  245 ms  433 ms
int8_float16     5   354 ms  183 ms  233 ms  282 ms  534 ms
```

\* 1352 ms: on 2 s of audio Whisper looped to 240 characters. The benchmark
has no token cap; the app's (15 tokens per second of buffer) would have
stopped it at 54 tokens. int8_float16 buys nothing worth its accuracy risk:
**float16, beam 5** stays. A 15 s pass is 29% of the 1 s step — the step 0
gate (≤ 60%) passes.

Replayed through the streamer on the same clip:

```
step    first shown p50 / p90    final p50 / p90    flicker    overran
1.0 s       0.88 / 1.64 s          2.00 / 3.19 s    45.8/min      0
0.5 s       0.50 / 1.10 s          1.07 / 2.00 s    62.5/min      4
(CPU 2.5 s  2.99 / 5.08 s          5.72 / 7.57 s    23.5/min)
```

At 1 s the plan's latency targets (≤ 1.5 s shown, ≤ 3 s final, p90) are met
within a few hundredths. Whether 0.5 s is worth its flicker is a question for
the accuracy corpus, not for this one clip.

## 2026-09-26 — step 1A, first real audio

The only Arabic audio on this machine was hifz's recordings: Quran recitation,
not the target content, but real speech. The one used throughout is
`surah nisa 12-14 whole page room mic.flac` — 97 s, a room microphone, dense
text, the reciter repeating himself once. No reference transcript yet, so
there is no CER; see "What is not measured" below.

### CPU pass time (`bench_latency.py`)

```
threads beam   5 s buffer  10 s buffer  15 s buffer   (p50)
   4      1      1406 ms      1561 ms      1721 ms
   4      5      1507 ms      1734 ms      2004 ms
   8      1      1096 ms      1216 ms      1320 ms
   8      5      1135 ms      1353 ms      1508 ms
```

The encoder dominates: beam 5 costs 10–15% over beam 1, so beam 5 stays.
Eight threads take a quarter off every pass → `CPU_THREADS = 8`, and the CPU
step went from 3.0 s to 2.5 s (a 15 s pass at p90 1.6 s is ~65% of it).

### One unstable word held back the sentence

Strict LocalAgreement (a common prefix) committed nothing from t=24 to t=36 s
while the tentative text grew to 24 words: Whisper spelled الربع as نرغبوا,
ربع, نربع, إن الربع on four passes, and every correct word after it waited.
Tolerant agreement — a disagreement of ≤ 2 words is passed over when ≥ 3 words
agree right after it (`AGREE_MAX_GAP`, `AGREE_MIN_RUN`) — plus cutting the
buffer at VAD pauses instead of Whisper segment ends:

```
                          strict prefix   tolerant + pause cut   + 8 threads, 2.5 s step
committed latency p50        7.59 s            6.83 s                5.72 s
committed latency p90       17.38 s            8.76 s                7.57 s
first shown p50 / p90   3.93 / 6.69 s     3.89 / 6.09 s         2.99 / 5.08 s
flicker                   23.5/min          23.5/min              23.5/min
```

Flicker is untouched by either change: it is the half-heard last word of each
buffer (تجريم → تجري, الأظي → الأظيم) shown grey and then corrected. Whether
to hide it is a step 1B question — hiding it costs responsiveness.

### A decode loop cost 8 s and a correct ayah

Live, one pass looped يدخله نارا خالدا فيها until the 448-token limit: 8004 ms
on CPU, and the compression gate then dropped the whole segment, correct words
included. Now a pass may generate at most 15 tokens per second of buffer
(fast MSA is ~9 tokens/s: Whisper spends ~3 tokens per Arabic word; this
recitation was 3.0), and a looping segment keeps the words before the loop
plus one copy (`LOOP_COPIES = 3`). Next live run: max pass 2064 ms, no overruns.

### The phantom gate caught a real one

Live: after a correct ayah, Whisper appended `اشتركوا في القناة` as its own
segment (no_speech_prob 0.61). Dropped; the ayah kept.

### "Different from the offline text" is not "wrong"

Streaming and the whole-file transcription differed by 9–12% CER, and the
biggest difference looked like a streaming duplicate: يدخله نارا خالدا فيها
twice. It is not. The VAD hears continuous speech 87.9–94.2 s, a pass over
82–97 s jumps from فيها at 87.8 s to وله at 92.6 s — skipping 4.7 s of speech —
and a pass with more context fills exactly that gap with the phrase again. The
reciter repeated it; the offline pass silently dropped the repeat. So the
stream-vs-offline number is a divergence, not an accuracy measure, and STREAM
GAP needs hand-checked references.

## What is not measured yet

- **Accuracy**: no reference transcripts exist. Step 1B builds `tests/clips/`
  (MSA news, lecture, dialects, speech over music, quiet playback, and
  music-only / silence / non-Arabic controls) with references drafted by
  `offline.py` and corrected by hand.
- **General Arabic speech** at all — only recitation so far.
