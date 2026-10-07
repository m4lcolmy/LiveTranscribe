"""Translation behind the popup — Google's two endpoints and its refusals, NLLB offline —
and the model downloads the settings start."""

import json
import sys
import time

import pytest
from PyQt6.QtWidgets import QApplication

from src.core import models, nllb
from src.ui import translate as tr
from src.ui.settings import AppSettings


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(condition, app, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    return condition()


# ── Google ─────────────────────────────────────────────────────────────

def endpoints(monkeypatch, single, extension):
    """Stand-ins for the two endpoints; each a list of answers, or RateLimited to refuse."""
    calls = []

    def fake(name, answers):
        def endpoint(text, target, source, timeout):
            calls.append(name)
            answer = answers.pop(0) if len(answers) > 1 else answers[0]
            if answer is tr.RateLimited:
                raise tr.RateLimited(tr.RATE_LIMITED)
            return answer
        return endpoint
    tr._google.endpoints = [fake("single", single), fake("extension", extension)]
    return calls


def test_the_fallback_endpoints_answer_is_read_with_and_without_a_detected_language():
    assert tr.parse_fallback('[["Welcome.\\nHow are you?","ar"]]') == "Welcome.\nHow are you?"
    assert tr.parse_fallback('["Welcome."]') == "Welcome."
    with pytest.raises(tr.RateLimited):
        tr.parse_fallback("<html>Sorry…</html>")


def test_a_refused_endpoint_hands_over_to_the_other_which_is_then_asked_first(monkeypatch):
    calls = endpoints(monkeypatch, [tr.RateLimited], ["from the extension"])
    assert tr.translate("مرحبا", "en") == "from the extension"
    assert tr.translate("مرحبا بكم", "en") == "from the extension"
    assert calls == ["single", "extension", "extension"]


def test_when_both_refuse_nothing_is_sent_until_the_wait_is_over(monkeypatch):
    calls = endpoints(monkeypatch, [tr.RateLimited], [tr.RateLimited])
    with pytest.raises(tr.RateLimited, match="2 min"):
        tr.translate("مرحبا", "en")
    assert calls == ["single", "extension"]
    with pytest.raises(tr.RateLimited, match="Offline"):
        tr.translate("مرحبا", "en")
    assert calls == ["single", "extension"]                 # not asked again

    # Waited out, refused again: the next wait is longer. A success starts over.
    tr._google.until = 0
    with pytest.raises(tr.RateLimited, match="5 min"):
        tr.translate("مرحبا", "en")
    tr._google.until = 0
    endpoints(monkeypatch, ["hello"], ["hello"])
    assert tr.translate("مرحبا", "en") == "hello"
    assert tr._google.refusals == 0


def test_answers_are_kept_and_the_same_text_is_not_asked_twice(app, monkeypatch):
    asked = []
    monkeypatch.setattr(tr, "translate", lambda text, target, engine="google", **_:
                        asked.append((engine, target)) or f"{engine}:{target}")
    translator = tr.Translator()
    got = []
    translator.finished.connect(lambda _t, text, error: got.append(text))
    translator.request("نص", "en")
    assert wait_until(lambda: got == ["google:en"], app)
    translator.request(" نص ", "en")
    assert got == ["google:en", "google:en"]               # at once, from the cache
    translator.request("نص", "tr")
    translator.request("نص", "en", "offline")
    # The "tr" answer may or may not be overtaken by the newer request; the newest always arrives.
    assert wait_until(lambda: got[-1:] == ["offline:en"], app)
    assert wait_until(lambda: len(asked) == 3, app)
    assert asked == [("google", "en"), ("google", "tr"), ("offline", "en")]


def test_a_refusal_is_not_kept(app, monkeypatch):
    def refuse(*_a, **_k):
        raise tr.TranslationError("no")
    monkeypatch.setattr(tr, "translate", refuse)
    translator = tr.Translator()
    translator.request("نص", "en")
    wait_until(lambda: False, app, timeout=0.2)
    assert translator.cached("نص", "en", "google") is None


# ── Offline ────────────────────────────────────────────────────────────

class WordTokenizer:
    """NLLB's tokenizer, as far as nllb.py uses it: words for pieces."""

    class Encoding:
        def __init__(self, tokens):
            self.tokens = tokens

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return self.Encoding(text.split())

    def token_to_id(self, token):
        return self.vocab.setdefault(token, len(self.vocab))

    def decode(self, ids):
        words = {i: t for t, i in self.vocab.items()}
        return " ".join(words[i] for i in ids)


class EchoTranslator:
    """Answers each sentence with its words upper-cased, after the language token."""

    def __init__(self):
        self.batches = []

    def translate_batch(self, batch, target_prefix, **_):
        self.batches.append(batch)

        class Result:
            def __init__(self, hypothesis):
                self.hypotheses = [hypothesis]
        return [Result(prefix + [t.upper() for t in source[1:-1]])
                for source, prefix in zip(batch, target_prefix)]


def test_nllb_gets_one_sentence_at_a_time_and_the_lines_come_back_as_they_were(monkeypatch):
    echo, tokenizer = EchoTranslator(), WordTokenizer()
    monkeypatch.setattr(nllb, "_load", lambda: (echo, tokenizer))
    text = "a b. c d؟ e\nf g"
    assert nllb.translate(text, "en") == "A B. C D؟ E\nF G"
    sources = echo.batches[0]
    assert sources == [["arb_Arab", "a", "b.", "</s>"], ["arb_Arab", "c", "d؟", "</s>"],
                       ["arb_Arab", "e", "</s>"], ["arb_Arab", "f", "g", "</s>"]]


def test_a_sentence_longer_than_nllb_takes_is_cut_into_pieces(monkeypatch):
    monkeypatch.setattr(nllb, "NLLB_MAX_TOKENS", 3)
    assert [len(p) for p in nllb.pieces(WordTokenizer(), "a b c d e f g")] == [3, 3, 1]


def test_offline_into_a_language_nllb_is_not_given_says_so(monkeypatch):
    monkeypatch.setattr(nllb, "_load", lambda: (EchoTranslator(), WordTokenizer()))
    with pytest.raises(tr.TranslationError, match="not offered"):
        tr.translate("مرحبا", "xx", engine="offline")


def test_every_language_offered_has_an_nllb_code():
    assert {code for code, _ in tr.LANGUAGES} <= set(nllb.CODES)


def test_offline_without_the_model_points_to_the_settings(monkeypatch):
    monkeypatch.setattr(models, "is_downloaded", lambda model: False)
    nllb.unload()
    with pytest.raises(tr.TranslationError, match="Settings"):
        tr.translate("مرحبا", "en", engine="offline")


# ── Downloads ──────────────────────────────────────────────────────────

def test_the_download_process_says_the_size_then_fails_plainly_when_the_disk_is_full(
        monkeypatch, capsys, tmp_path):
    import shutil
    from types import SimpleNamespace
    import huggingface_hub
    info = SimpleNamespace(sha="abc", siblings=[
        SimpleNamespace(rfilename="model.bin", size=3_000_000_000),
        SimpleNamespace(rfilename="config.json", size=1_000),
        SimpleNamespace(rfilename="README.md", size=50_000)])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda repo, files_metadata: info))
    monkeypatch.setattr(models, "cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(free=1_000_000_000))
    assert models.download("medium") == 1
    said = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert said[0] == {"total": 3_000_001_000}
    assert said[1]["error"] == "Not enough disk space (3.0 GB needed)"


def test_the_download_process_names_a_network_failure():
    assert models._reason(ConnectionError("Max retries exceeded")).startswith("Could not reach")
    assert models.download("not-a-model") == 2


def test_what_a_stopped_download_half_wrote_is_removed_and_finished_files_stay(tmp_path):
    blobs = tmp_path / "hf-cache" / "models--Systran--faster-whisper-medium" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc").write_bytes(b"x" * 10)
    (blobs / "def.1a2b3c4d.incomplete").write_bytes(b"x" * 5)
    medium = models.find("medium")
    assert models.bytes_in_cache(medium) == 15
    models.remove_partial(medium)
    assert [p.name for p in blobs.iterdir()] == ["abc"] and models.bytes_in_cache(medium) == 10


def stand_in(monkeypatch, script):
    """The download process replaced by a few lines of Python."""
    from src.ui import downloads
    monkeypatch.setattr(downloads, "command", lambda name: (sys.executable, ["-c", script]))
    manager = downloads.Downloads()
    return manager


def test_a_download_that_works_is_announced_and_leaves_no_job(app, monkeypatch):
    manager = stand_in(monkeypatch, "import json; print(json.dumps({'total': 10}))")
    ended = []
    manager.finished.connect(lambda name, error: ended.append((name, error)))
    manager.start("medium")
    assert manager.running("medium")
    assert wait_until(lambda: ended, app)
    assert ended == [("medium", "")] and manager.job("medium") is None


def test_a_failed_download_keeps_its_reason_and_can_be_tried_again(app, monkeypatch):
    manager = stand_in(monkeypatch, "import json, sys; print(json.dumps({'total': 10}));"
                                    "print(json.dumps({'error': 'The disk is full'})); sys.exit(1)")
    monkeypatch.setattr(models, "is_downloaded", lambda model: False)
    ended = []
    manager.finished.connect(lambda name, error: ended.append(error))
    manager.start("medium")
    assert wait_until(lambda: ended, app)
    assert ended == ["The disk is full"] and manager.job("medium").total == 10
    manager.start("medium")                                    # Try again
    assert wait_until(lambda: len(ended) == 2, app)


def test_cancel_stops_the_download_at_once_and_says_nothing(app, monkeypatch):
    manager = stand_in(monkeypatch, "import time; time.sleep(30)")
    ended = []
    manager.finished.connect(lambda *a: ended.append(a))
    manager.start("medium")
    assert manager.running("medium")
    manager.cancel("medium")
    assert wait_until(lambda: manager.job("medium") is None, app)
    assert ended == []


# ── The settings ───────────────────────────────────────────────────────

def test_a_model_not_downloaded_shows_its_size_a_download_button_and_holds_save(app, monkeypatch):
    from src.ui.settings import SettingsDialog
    monkeypatch.setattr(models, "is_downloaded", lambda model: model.name == "small")
    dialog = SettingsDialog(AppSettings())
    dialog.show()
    assert not dialog.model_download.isVisible() and dialog.save_button.isEnabled()
    i = dialog.model.findData("medium")
    assert dialog.model.itemText(i) == "medium  ·  download 1.5 GB"
    dialog.model.setCurrentIndex(i)
    assert dialog.model_download.isVisible()
    assert dialog.model_download.button.text() == "Download"
    assert "1.5 GB" in dialog.model_download.label.text()
    assert not dialog.save_button.isEnabled()
    assert dialog.note.text() == "Download medium to save, or choose another"

    # It arrives: the label loses its size, the row says so, and Save is back.
    monkeypatch.setattr(models, "is_downloaded", lambda model: True)
    dialog.model_download.manager.changed.emit("medium")
    assert dialog.model.itemText(i) == "medium"
    assert dialog.model_download.label.text().startswith("✓ Downloaded")
    assert dialog.save_button.isEnabled() and dialog.note.text() == "Saving reloads the model"
    dialog.close()


def test_offline_translation_offers_its_download_only_when_chosen(app, monkeypatch):
    from src.ui.settings import SettingsDialog
    monkeypatch.setattr(models, "is_downloaded", lambda model: model is not models.NLLB)
    dialog = SettingsDialog(AppSettings())
    dialog.show()
    assert not dialog.translate_download.isVisible()
    dialog.translate_with.setCurrentIndex(dialog.translate_with.findData("offline"))
    assert dialog.translate_download.isVisible() and not dialog.save_button.isEnabled()
    assert "download 640 MB" in dialog.translate_with.currentText()
    dialog.translate.setCurrentIndex(dialog.translate.findData("off"))   # no translation at all
    assert not dialog.translate_download.isVisible() and dialog.save_button.isEnabled()
    assert not dialog.translate_with.isEnabled()
    dialog.close()


def test_a_download_in_progress_shows_its_bar_and_a_cancel_button(app, monkeypatch):
    from src.ui.settings import SettingsDialog
    monkeypatch.setattr(models, "is_downloaded", lambda model: model.name == "small")
    from src.ui import downloads
    dialog = SettingsDialog(AppSettings(model="small"))
    dialog.show()
    dialog.model.setCurrentIndex(dialog.model.findData("large-v3"))
    row = dialog.model_download
    fake = downloads.Job(models.find("large-v3"), process=type("P", (), {
        "state": lambda self: downloads.QProcess.ProcessState.Running})(),
        total=3_000_000_000, done=750_000_000, speed=5e6)
    row.manager.jobs["large-v3"] = fake
    row.manager.changed.emit("large-v3")
    assert row.bar.isVisible() and row.bar.value() == 250
    assert row.button.text() == "Cancel"
    assert row.label.text() == ("750 MB of 3.0 GB · 5.0 MB/s · 8 min left\n"
                                "Keeps going if you close Settings.")
    del row.manager.jobs["large-v3"]
    dialog.close()
