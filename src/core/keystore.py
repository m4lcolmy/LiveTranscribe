"""The Deepgram API key, in the system's credential store — never in the settings file.

GNOME Keyring (any Secret Service, or KWallet) keeps it encrypted on disk and
unlocks it at login; the `keyring` package talks to it. The app writes the
key, replaces it, removes it, and reads it back only to connect. The settings
dialog shows its last four characters and never the key itself, so a key
once saved cannot be read off the screen.

Without a credential store nothing is saved — not in plain text either — and
the key has to come from DEEPGRAM_API_KEY. That variable, when set, always
wins over the saved key, and is never saved.
"""

import os

SERVICE = "LiveTranscribe"
ACCOUNT = "deepgram"


class KeystoreError(Exception):
    """The key could not be saved or removed. The message never holds the key."""


def from_environment() -> str:
    return os.environ.get("DEEPGRAM_API_KEY", "").strip()


def available() -> bool:
    """True when there is a credential store to keep the key in."""
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return False
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    return not isinstance(backend, fail.Keyring) and backend.priority > 0


def load() -> str:
    """The saved key, or "" when none is saved or there is no store to ask."""
    try:
        import keyring
        return keyring.get_password(SERVICE, ACCOUNT) or ""
    except Exception:        # no store, a locked one the user would not unlock, D-Bus gone
        return ""


def save(key: str):
    if not available():
        raise KeystoreError("No system keyring to keep the key in")
    try:
        import keyring
        keyring.set_password(SERVICE, ACCOUNT, key.strip())
    except Exception as e:
        raise KeystoreError(f"The system keyring refused the key ({type(e).__name__})") from None


def remove():
    try:
        import keyring
        from keyring.errors import PasswordDeleteError
    except ImportError:
        return
    try:
        keyring.delete_password(SERVICE, ACCOUNT)
    except PasswordDeleteError:
        pass                 # there was none
    except Exception as e:
        raise KeystoreError(f"The system keyring did not remove the key ({type(e).__name__})") from None


def key() -> str:
    """The key to connect with: DEEPGRAM_API_KEY, else the saved one."""
    return from_environment() or load()


def ending(key: str) -> str:
    """What the dialog may show of a key: its last four characters."""
    return f"…{key[-4:]}" if len(key) > 8 else "…"


try:
    from keyring.backend import KeyringBackend
except ImportError:          # keyring missing: the app runs, it just cannot save a key
    KeyringBackend = object


class MemoryKeyring(KeyringBackend):
    """A store in memory, for the tests and the README's pictures: the real one is never touched."""

    priority = 1

    def __init__(self):
        super().__init__()
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.passwords.get((service, username))

    def set_password(self, service, username, password):
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        from keyring.errors import PasswordDeleteError
        if self.passwords.pop((service, username), None) is None:
            raise PasswordDeleteError("not saved")
