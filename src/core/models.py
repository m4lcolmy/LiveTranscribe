"""The models the app can run, whether each is on this computer, and fetching one.

The app itself never reaches the network for a model: it runs with
HF_HUB_OFFLINE=1 and loads only what is in the Hugging Face cache. install.sh
fetches Whisper small, the default. Any other model is downloaded when the
user asks for it in the settings, by a separate process —

    python -m src.core.models medium

— which the settings follow by the bytes arriving in the cache, and can stop
at any moment (src/ui/downloads.py). The files it finished stay; the one it
was in the middle of starts again next time: huggingface_hub (1.x) writes
each download to a file of its own process and never resumes another's, so
those are removed (remove_partial) rather than left to pile up.

That process says what it is doing on stdout, one JSON object a line:
{"total": bytes} once the Hub has told it the size, then {"error": "…"} if
it fails; it exits 0 only when every file is in the cache.
"""

import fnmatch
import json
import os
import shutil
import sys
from dataclasses import dataclass

from src.config import NLLB_REPO

# What faster-whisper's own download_model() fetches.
WHISPER_FILES = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json",
                 "vocabulary.*")
NLLB_FILES = ("config.json", "model.bin", "shared_vocabulary.txt", "tokenizer.json")
NLLB_NAME = "nllb-600m"


@dataclass(frozen=True)
class Model:
    name: str                   # a Whisper size name, or NLLB_NAME
    repo: str
    files: tuple[str, ...]      # patterns, as snapshot_download's allow_patterns
    size_gb: float              # to show before asking; the download asks the Hub
    note: str = ""
    # Without these the model cannot load; a download stopped part way can
    # leave the small files without model.bin, or the other way round.
    required: tuple[str, ...] = ("config.json", "model.bin", "tokenizer.json")

    @property
    def size(self) -> str:
        return size_text(self.size_gb * 1e9)


def _whisper(name: str, size_gb: float, note: str) -> Model:
    try:
        from faster_whisper.utils import _MODELS
        repo = _MODELS[name]
    except (ImportError, KeyError):
        repo = f"Systran/faster-whisper-{name}"
    return Model(name, repo, WHISPER_FILES, size_gb, note)


# Arabic needs at least base; tiny and the .en and distil (English-only)
# models are not offered. Sizes from the Hub, 2026-10-07.
WHISPER = [
    _whisper("base", 0.15, "fastest, least accurate"),
    _whisper("small", 0.49, "fast; the tuned default"),
    _whisper("medium", 1.53, "between small and large"),
    _whisper("large-v3-turbo", 1.62, "near large-v3 accuracy at a fraction of its cost"),
    _whisper("large-v3", 3.09, "most accurate; several times slower, int8 on a 4 GB GPU"),
]
NLLB = Model(NLLB_NAME, NLLB_REPO, NLLB_FILES, 0.64,
             "Meta's NLLB-200 (600M) — Arabic into 200 languages, on this computer",
             required=NLLB_FILES)
CATALOG = {m.name: m for m in [*WHISPER, NLLB]}


def size_text(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def find(name: str) -> Model | None:
    return CATALOG.get(name)


def _cached(repo: str, filename: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(repo, filename)
    return path if isinstance(path, str) else None


def is_downloaded(model: Model) -> bool:
    return all(_cached(model.repo, f) for f in model.required)


def local_path(model: Model) -> str | None:
    """The snapshot folder that holds the model, when all of it is there."""
    if not is_downloaded(model):
        return None
    return os.path.dirname(_cached(model.repo, "model.bin"))


def cache_root() -> str:
    from huggingface_hub import constants
    return constants.HF_HUB_CACHE


def _blobs(model: Model) -> str:
    return os.path.join(cache_root(), "models--" + model.repo.replace("/", "--"), "blobs")


def bytes_in_cache(model: Model) -> int:
    """Everything of this model in the cache so far, finished files and partial ones."""
    try:
        return sum(e.stat().st_size for e in os.scandir(_blobs(model)) if e.is_file())
    except OSError:
        return 0


def remove_partial(model: Model):
    """The half-written files a stopped download leaves; no later download reads them."""
    try:
        entries = list(os.scandir(_blobs(model)))
    except OSError:
        return
    for e in entries:
        if e.name.endswith(".incomplete"):
            try:
                os.remove(e.path)
            except OSError:
                pass


# ── The download process ───────────────────────────────────────────────

def _say(**message):
    print(json.dumps(message), flush=True)


def _reason(e: Exception) -> str:
    """A failure, in words the settings can show under the download button."""
    text = str(e)
    name = type(e).__name__
    if "RepositoryNotFound" in name or "404" in text:
        return "The model is no longer on Hugging Face"
    if any(s in name for s in ("Connection", "Timeout")) or "Name or service" in text \
            or "Temporary failure" in text or "Max retries" in text:
        return "Could not reach Hugging Face — offline?"
    if isinstance(e, OSError) and e.errno == 28:
        return "The disk is full"
    return f"Download failed: {text.splitlines()[0][:160] if text else name}"


def download(name: str) -> int:
    model = find(name)
    if model is None:
        _say(error=f"Unknown model: {name}")
        return 2
    try:
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().model_info(model.repo, files_metadata=True)
        total = sum(s.size or 0 for s in info.siblings
                    if any(fnmatch.fnmatch(s.rfilename, p) for p in model.files))
        _say(total=total)
        os.makedirs(cache_root(), exist_ok=True)
        needed = total - bytes_in_cache(model)
        free = shutil.disk_usage(cache_root()).free
        if needed > free:
            _say(error=f"Not enough disk space ({size_text(needed)} needed)")
            return 1
        snapshot_download(model.repo, revision=info.sha, allow_patterns=list(model.files))
    except Exception as e:
        _say(error=_reason(e))
        return 1
    if not is_downloaded(model):
        _say(error="Download ended without every file")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(download(sys.argv[1]) if len(sys.argv) == 2 else 2)
