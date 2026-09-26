# Measurements

What the benchmarks found, and what each finding changed. Newest first.
Machine: Ryzen 7 6800H (8 cores), RTX 3050 Ti Mobile 4 GB — **GPU not yet
measured**: its driver module was missing for the running kernel (PLAN.md step
0.1), so everything below is CPU, small/int8.

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

- **The GPU**: step 0.1, then `bench_latency.py` with `--compute float16,int8_float16`.
- **Accuracy**: no reference transcripts exist. Step 1B builds `tests/clips/`
  (MSA news, lecture, dialects, speech over music, quiet playback, and
  music-only / silence / non-Arabic controls) with references drafted by
  `offline.py` and corrected by hand.
- **General Arabic speech** at all — only recitation so far.
