"""Build the accuracy corpus from YouTube videos that carry human Arabic captions.

    python scripts/fetch_corpus.py --discover            # search; list videos with manual Arabic captions
    python scripts/fetch_corpus.py --plan                # what --fetch would download
    python scripts/fetch_corpus.py --fetch               # download captions + MP3 through CaptionForge

The reference must be written by a person. YouTube offers three kinds of
Arabic track, and only one of them measures anything:

  * manual, on a video *spoken* in Arabic — a transcript by a person: the reference;
  * manual, on a video spoken in another language — a translation, not a transcript;
  * automatic — Google's speech recognition. Scoring against it compares one
    recogniser with another. (CaptionForge's own output folder is the same
    trap: its SRTs were made by Whisper large-v3 whenever no track existed.)

So discovery searches with YouTube's Subtitles/CC filter, then keeps a video
only if it has a *manual* track in Arabic and YouTube says it is spoken in
Arabic. Measured on the first Al Jazeera search: 1 of 8 CC-filtered videos
qualified — the rest had human English, Persian, French or Turkish translations.

Fetching goes through CaptionForge (`extract --no-postprocess`, which ranks a
manual track first and keeps its segmentation, and `download --quality mp3`),
into tests/clips/<id>/. Audio and captions are gitignored; the manifest, which
says what each clip is and why it is there, is committed.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "tests" / "clips"
MANIFEST = CLIPS / "manifest.json"
CAPTIONFORGE = Path.home() / "code" / "CaptionForge"
YTDLP = CAPTIONFORGE / ".venv" / "bin" / "yt-dlp"
CF = CAPTIONFORGE / ".venv" / "bin" / "captionforge"

# YouTube's "Subtitles/CC" search filter: videos with uploaded (not automatic) captions.
CC_FILTER = "EgIoAQ%253D%253D"

# YouTube answers a burst of requests with HTTP 429 and "confirm you're not a
# bot" — discovery's ~300 metadata requests in a few minutes earned a block of
# over an hour (2026-09-26). Fetching is paced, and stops at the first refusal
# rather than hammering a block into a longer one.
PAUSE_BETWEEN_CLIPS_S = 20
BLOCKED = ("429", "Too Many Requests", "not a bot")

QUERIES = [
    "الجزيرة تقرير", "الجزيرة نشرة الأخبار", "الجزيرة الوثائقية", "الجزيرة مباشر",
    "بي بي سي عربي", "DW عربية", "سكاي نيوز عربية", "فرانس 24 عربي", "العربية تقرير",
    "TEDx عربي", "محاضرة عربية", "بودكاست عربي",
]


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def search(query: str, limit: int) -> list[dict]:
    url = (f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
           f"&sp={CC_FILTER}")
    out = subprocess.run(
        [str(YTDLP), "--flat-playlist", "--playlist-end", str(limit), "-J", url],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        return []
    return [e for e in json.loads(out.stdout).get("entries", []) if e.get("id")]


def tracks(video_id: str) -> dict | None:
    out = subprocess.run([str(YTDLP), "-J", "--skip-download", watch_url(video_id)],
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        return None
    d = json.loads(out.stdout)
    manual = sorted(k for k in (d.get("subtitles") or {}) if k != "live_chat")
    return {
        "id": d["id"], "title": d.get("title", ""), "channel": d.get("channel", ""),
        "duration": d.get("duration") or 0, "language": d.get("language"),
        "manual": manual,
        "auto_ar": any(k in (d.get("automatic_captions") or {}) for k in ("ar", "ar-orig")),
    }


def is_reference(t: dict) -> bool:
    """A human transcript: a manual Arabic track on a video spoken in Arabic."""
    return t["language"] == "ar" and any(k == "ar" or k.startswith("ar-") for k in t["manual"])


def discover(limit: int, min_s: int, max_s: int):
    seen, candidates = set(), []
    for query in QUERIES:
        for e in search(query, limit):
            d = e.get("duration") or 0
            if e["id"] not in seen and min_s <= d <= max_s:
                seen.add(e["id"])
                candidates.append(e["id"])
        print(f"  searched {query!r}: {len(candidates)} candidates so far", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = [t for t in pool.map(tracks, candidates) if t]
    good = [t for t in checked if is_reference(t)]
    print(f"\n{len(good)} of {len(checked)} CC-filtered videos have human Arabic captions:\n")
    for t in sorted(good, key=lambda t: (t["channel"], t["duration"])):
        print(f"  {t['id']}  {t['duration']:>5}s  {t['channel'][:28]:<28}  {t['title'][:70]}")
    out = CLIPS / "candidates.json"
    CLIPS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out.relative_to(ROOT)}  (pick clips from here into manifest.json)")


# ── Fetching ───────────────────────────────────────────────────────────

def clip_dir(clip: dict) -> Path:
    return CLIPS / clip["id"]


def fetch_one(clip: dict) -> str:
    target = clip_dir(clip)
    audio, ref = target / "audio.mp3", target / "ref.srt"
    needs_ref = not clip.get("silent")
    if audio.exists() and (ref.exists() or not needs_ref):
        return "already here"
    target.mkdir(parents=True, exist_ok=True)
    url = watch_url(clip["id"])

    with tempfile.TemporaryDirectory(dir=CLIPS) as tmp:
        if needs_ref and not ref.exists():
            r = subprocess.run([str(CF), "extract", url, "--language", "ar", "--format", "srt",
                                "--no-postprocess", "--output", tmp],
                               capture_output=True, text=True, cwd=CAPTIONFORGE)
            srts = list(Path(tmp).glob("*.srt"))
            if r.returncode != 0 or not srts:
                return f"captions failed: {(r.stdout + r.stderr).strip()[-200:]}"
            shutil.move(str(srts[0]), ref)
        if not audio.exists():
            r = subprocess.run([str(CF), "download", url, "--quality", "mp3", "--output", tmp],
                               capture_output=True, text=True, cwd=CAPTIONFORGE)
            mp3s = list(Path(tmp).glob("*.mp3"))
            if r.returncode != 0 or not mp3s:
                return f"audio failed: {(r.stdout + r.stderr).strip()[-200:]}"
            shutil.move(str(mp3s[0]), audio)
    return "fetched"


def inspect_clip(clip: dict):
    """What a person reading the captions needs in order to judge who wrote them.

    Machine captions re-uploaded as "manual" have no punctuation and cut the
    speech into short fixed-length cues; a person's captions carry ، . ؟ and
    break at phrases. The numbers only point; the sample lines decide.
    """
    sys.path.insert(0, str(ROOT))
    from src.evaluation import load_cues

    ref = clip_dir(clip) / "ref.srt"
    if not ref.exists():
        print(f"{clip['id']}: no captions fetched")
        return
    cues = [c for c in load_cues(ref) if c[2]]
    text = " ".join(c[2] for c in cues)
    words = len(text.split())
    punct = sum(text.count(ch) for ch in "،.؟?!:؛")
    lengths = sorted(c[1] - c[0] for c in cues)
    window = [c for c in cues if clip.get("start", 0) <= c[0] <= clip.get("end", 1e9)]
    print(f"── {clip['id']}  {clip['channel']}  {clip['title'][:60]}")
    print(f"   {len(cues)} cues, {words} words, {100 * punct / max(1, words):.1f} punctuation marks "
          f"per 100 words, cue length median {lengths[len(lengths) // 2]:.1f}s")
    for c in window[3:9]:
        print(f"   [{c[0]:6.1f}] {c[2]}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--discover", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--inspect", action="store_true", help="show what each fetched caption track looks like")
    p.add_argument("--limit", type=int, default=20, help="search results per query")
    p.add_argument("--min", type=int, default=120, help="shortest video, seconds")
    p.add_argument("--max", type=int, default=1500, help="longest video, seconds")
    args = p.parse_args()

    if not YTDLP.exists() or not CF.exists():
        sys.exit(f"CaptionForge with its venv is expected at {CAPTIONFORGE}")
    if args.discover:
        discover(args.limit, args.min, args.max)
        return

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.inspect:
        for clip in manifest["clips"]:
            inspect_clip(clip)
        return
    for i, clip in enumerate(manifest["clips"]):
        name = f"{clip['id']}  {clip['type']:<10} {clip.get('title', '')[:60]}"
        if args.fetch:
            result = fetch_one(clip)
            print(f"{name}  → {result}", flush=True)
            # CaptionForge words a 429 as "could not be retrieved", so any
            # failure stops the run: a refusal answered with more requests
            # only lengthens the block.
            if "failed" in result:
                blocked = any(b in result for b in BLOCKED) or "retrieved" in result
                sys.exit(("YouTube is refusing requests (rate limit)." if blocked else "Stopped.")
                         + " Clips already fetched are kept; run --fetch again later.")
            if result == "fetched" and i + 1 < len(manifest["clips"]):
                time.sleep(PAUSE_BETWEEN_CLIPS_S)
        else:
            what = "audio" if clip.get("silent") else "audio + human captions"
            print(f"{name}  ({what}, {watch_url(clip['id'])})")
    if not args.fetch:
        print("\nNothing downloaded. Run with --fetch to download through CaptionForge.")


if __name__ == "__main__":
    main()
