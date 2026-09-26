"""One LiveTranscribe at a time.

Two would capture the same audio, run two models on one GPU and stack two
windows on the same spot. The first instance listens on a local socket; a
second one, started from the app menu say, asks it to show its window and
exits.
"""

import os

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket


class SingleInstance(QObject):
    activated = pyqtSignal()          # another start asked the running one to show itself

    def __init__(self, name: str = f"livetranscribe-{os.getuid()}"):
        super().__init__()
        self.name = name
        self.server: QLocalServer | None = None

    def claim(self) -> bool:
        """True when this is the only instance; otherwise wakes the running one."""
        probe = QLocalSocket()
        probe.connectToServer(self.name)
        if probe.waitForConnected(300):
            probe.write(b"show")
            probe.waitForBytesWritten(300)
            probe.disconnectFromServer()
            return False
        # Nobody answered: a socket left by a crashed run is stale, not a rival.
        QLocalServer.removeServer(self.name)
        self.server = QLocalServer(self)
        self.server.newConnection.connect(self._on_connection)
        return self.server.listen(self.name)

    def _on_connection(self):
        while self.server.hasPendingConnections():
            self.server.nextPendingConnection().readyRead.connect(self.activated.emit)
