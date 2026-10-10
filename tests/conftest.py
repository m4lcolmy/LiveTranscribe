"""Every test runs against a keyring in memory, with no DEEPGRAM_API_KEY: the user's own are never touched.

And every model counts as downloaded, whatever this machine's Hugging Face
cache holds; a test about a missing one says so itself. The cache the
downloads read and tidy is an empty folder of the test's own, and so is the
translation history.
"""

import pytest


@pytest.fixture(autouse=True)
def memory_keyring(monkeypatch):
    import keyring
    from src.core.keystore import MemoryKeyring
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    yield backend
    keyring.set_keyring(previous)


@pytest.fixture(autouse=True)
def every_model_downloaded(monkeypatch, tmp_path):
    from src.core import models
    monkeypatch.setattr(models, "is_downloaded", lambda model: True)
    monkeypatch.setattr(models, "cache_root", lambda: str(tmp_path / "hf-cache"))


@pytest.fixture(autouse=True)
def google_not_refusing():
    from src.ui.translate import _google
    _google.reset()
    yield
    _google.reset()


@pytest.fixture(autouse=True)
def history_of_its_own(monkeypatch, tmp_path):
    """Translations a test makes are kept in its own folder, never in the user's history."""
    from src.core import history
    kept = history.History(tmp_path / "translations.jsonl")
    monkeypatch.setattr(history, "_shared", kept)
    return kept
