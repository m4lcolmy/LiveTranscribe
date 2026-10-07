"""Offline translation: Arabic into the chosen language with NLLB-200, on this computer.

Nothing leaves the machine and nothing can be refused, which is the point of
it next to Google. The model (src/core/models.py: NLLB) is loaded the first
time it is asked for and kept until the user turns offline translation off
(unload), on the GPU when there is one to spare — 0.7 GB beside Whisper —
else on the CPU.

NLLB reads and writes SentencePiece pieces. The tokenizer is faster-whisper's
own dependency, `tokenizers`, reading the model's tokenizer.json: no
SentencePiece package and no transformers. A source is
[source language] + pieces + </s>; the answer starts with the target
language's token, which is dropped.
"""

import re
import threading

from src.config import NLLB_BEAM, NLLB_MAX_TOKENS, NLLB_SOURCE
from src.core import models

# The app's language codes (src/ui/translate.py: LANGUAGES) in NLLB's.
CODES = {"en": "eng_Latn", "tr": "tur_Latn", "fr": "fra_Latn", "de": "deu_Latn",
         "es": "spa_Latn", "it": "ita_Latn", "ru": "rus_Cyrl", "fa": "pes_Arab",
         "ur": "urd_Arab", "id": "ind_Latn", "ms": "zsm_Latn", "zh-CN": "zho_Hans",
         "ja": "jpn_Jpan"}

# The end of a sentence, Arabic or Latin punctuation, and what follows it.
_SENTENCE_END = re.compile(r"(?<=[.!?؟؛…])\s+")


class OfflineError(Exception):
    """Something to show the user instead of a translation."""


_lock = threading.Lock()
_model = None               # (ctranslate2.Translator, tokenizers.Tokenizer)
_device = "auto"


def available() -> bool:
    return models.is_downloaded(models.NLLB)


def loaded() -> bool:
    return _model is not None


def set_device(preference: str):
    """The app's device setting (auto | cuda | cpu); applies from the next load."""
    global _device
    _device = preference


def unload():
    global _model
    with _lock:
        _model = None


def _load():
    global _model
    with _lock:
        if _model is not None:
            return _model
        path = models.local_path(models.NLLB)
        if path is None:
            raise OfflineError("The offline translation model is not downloaded — "
                               "download it in Settings → Translation.")
        import ctranslate2
        import tokenizers
        from src.core.device import resolve_device
        device, _ = resolve_device(_device)
        tokenizer = tokenizers.Tokenizer.from_file(f"{path}/tokenizer.json")
        try:
            translator = ctranslate2.Translator(
                path, device=device, compute_type="int8_float16" if device == "cuda" else "int8")
        except (RuntimeError, ValueError):
            if device != "cuda":
                raise
            # No room on the GPU (a large Whisper model): the CPU is slower, not wrong.
            translator = ctranslate2.Translator(path, device="cpu", compute_type="int8")
        _model = translator, tokenizer
        return _model


def pieces(tokenizer, text: str) -> list[list[str]]:
    """Each sentence's pieces, a long sentence cut every NLLB_MAX_TOKENS."""
    out = []
    for sentence in _SENTENCE_END.split(text):
        tokens = tokenizer.encode(sentence, add_special_tokens=False).tokens
        for i in range(0, len(tokens), NLLB_MAX_TOKENS):
            out.append(tokens[i:i + NLLB_MAX_TOKENS])
    return [p for p in out if p]


def translate(text: str, target: str) -> str:
    code = CODES.get(target)
    if code is None:
        raise OfflineError(f"Offline translation into “{target}” is not offered")
    translator, tokenizer = _load()
    # Line by line, so the translation keeps the transcript's lines.
    lines = text.split("\n")
    chunks = [pieces(tokenizer, line) for line in lines]
    flat = [[NLLB_SOURCE, *p, "</s>"] for line in chunks for p in line]
    if not flat:
        return ""
    try:
        results = translator.translate_batch(
            flat, target_prefix=[[code]] * len(flat), beam_size=NLLB_BEAM,
            max_decoding_length=2 * NLLB_MAX_TOKENS)
    except RuntimeError as e:
        raise OfflineError(f"Offline translation failed ({e})")
    decoded = iter(tokenizer.decode([tokenizer.token_to_id(t) for t in r.hypotheses[0][1:]
                                     if tokenizer.token_to_id(t) is not None])
                   for r in results)
    joint = "" if target in ("zh-CN", "ja") else " "     # no spaces between sentences there
    return "\n".join(joint.join(next(decoded) for _ in line) for line in chunks).strip()
