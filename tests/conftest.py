"""Every test runs against a keyring in memory, with no DEEPGRAM_API_KEY: the user's own are never touched."""

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
