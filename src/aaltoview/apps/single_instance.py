"""single_instance.py -- one AaltoView window per user: a file opened while a
viewer is running loads in THAT window instead of starting a new viewer.

Starting a viewer takes seconds (Python + PySide6 + pyqtgraph + xarray); handing
a path to a running one takes ~0.2 s. AaltoFlow's catalogue and the `aaltoview`
command both try the handover first:

    from aaltoview.apps.single_instance import send_to_running
    if not send_to_running(path):       # nobody listening (or no answer in 0.5 s)
        ...start a viewer as before...

The wire: a QLocalServer (a named pipe on Windows, a socket in the temp folder
elsewhere) called "AaltoView-<user name>", so two users on one PC never share a
window. One JSON line per request, one line back:

    {"open": "C:/data/2026-09-30/141500_scan.nc"}\\n   ->  ok\\n
    {"ping": true}\\n                                    ->  ok\\n
    anything else                                       ->  error\\n

The sender needs only QtCore + QtNetwork and no QApplication (a QLocalSocket's
blocking calls work without one, and a QApplication can still be made after),
so this module imports nothing heavy at the top: the `aaltoview` command checks
for a running viewer BEFORE importing pyqtgraph and the window (`main` below).

Tests set AALTOVIEW_INSTANCE_NAME so they never talk to a real running viewer.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path

#: tests (and a second, separate viewer family) set their own name here
NAME_ENV = "AALTOVIEW_INSTANCE_NAME"
#: the longest request taken (a path is a few hundred bytes): junk is cut off
MAX_LINE = 64 * 1024


def server_name() -> str:
    """'AaltoView-<user>': per Windows user, only letters/digits/_ in the name."""
    if os.environ.get(NAME_ENV):
        return os.environ[NAME_ENV]
    try:
        user = getpass.getuser()
    except Exception:                       # no user name in the environment
        user = "user"
    return "AaltoView-" + (re.sub(r"[^A-Za-z0-9]", "_", user) or "user")


SERVER_NAME = server_name()


def _request(message: dict, timeout_ms: int, name: str | None) -> bool:
    """One request to the running viewer; True if it answered 'ok'. The whole
    exchange shares ONE deadline of timeout_ms (the catalogue calls this on its
    GUI thread: a viewer that accepts but never answers must not hold it longer)."""
    import time
    from PySide6 import QtNetwork            # light: QtCore + QtNetwork only
    deadline = time.monotonic() + timeout_ms / 1000.0

    def left() -> int:
        return max(1, int((deadline - time.monotonic()) * 1000))

    sock = QtNetwork.QLocalSocket()
    sock.connectToServer(name or server_name())
    try:
        if not sock.waitForConnected(left()):
            return False                     # nobody listening (fails at once)
        sock.write((json.dumps(message) + "\n").encode("utf-8"))
        if not sock.waitForBytesWritten(left()):
            return False
        reply = b""
        while not reply.endswith(b"\n"):
            if time.monotonic() >= deadline or not sock.waitForReadyRead(left()):
                return False                 # listening but not answering: start anew
            reply += bytes(sock.readAll())
        return reply.strip() == b"ok"
    finally:
        sock.abort()


def send_to_running(path, timeout_ms: int = 500, name: str | None = None) -> bool:
    """Hand `path` to the running viewer of this user. True = it took it (it
    loads the file and comes to the front); False = no viewer listening, or it
    did not answer within `timeout_ms` -- then start one."""
    _allow_foreground()
    return _request({"open": str(Path(path).resolve())}, timeout_ms, name)


def is_running(timeout_ms: int = 300, name: str | None = None) -> bool:
    """Is a viewer of this user listening? (A start without a file asks this
    before it becomes the listener itself.)"""
    return _request({"ping": True}, timeout_ms, name)


def _allow_foreground() -> None:
    """Windows gives the foreground only to a process the CURRENT foreground
    process allows: the sender (the catalogue that was just double-clicked)
    lets the viewer come to the front."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.AllowSetForegroundWindow(-1)      # ASFW_ANY
        except Exception:
            pass


# ───────────────────────────── the listener ───────────────────────────────────

class Listener:
    """The running viewer's end: a QLocalServer that opens what it is sent.

    `window` needs load_file(path) and, optionally, say(text, error) for the
    status line. Start it only when no other viewer is listening (is_running()
    False): start() first removes a name a crashed viewer may have left behind.
    Nothing a client sends can stop it -- every request is caught and answered.
    """

    def __init__(self, window, name: str | None = None):
        from PySide6 import QtNetwork
        self.window = window
        self.name = name or server_name()
        self.server = QtNetwork.QLocalServer()
        self.server.setSocketOptions(QtNetwork.QLocalServer.UserAccessOption)
        self.server.newConnection.connect(self._new_connection)
        self._buffers: dict = {}

    def start(self) -> bool:
        from PySide6 import QtNetwork
        QtNetwork.QLocalServer.removeServer(self.name)   # a stale name must not block
        ok = self.server.listen(self.name)
        if not ok:
            self._say(f"single window: could not listen ({self.server.errorString()}); "
                      f"files opened elsewhere will start their own viewer", error=True)
        return ok

    def close(self) -> None:
        self.server.close()

    # each connection: read lines, answer each
    def _new_connection(self):
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            self._buffers[sock] = b""
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: self._drop(s))

    def _drop(self, sock):
        self._buffers.pop(sock, None)
        sock.deleteLater()

    def _read(self, sock):
        try:
            buf = self._buffers.get(sock, b"") + bytes(sock.readAll())
            if len(buf) > MAX_LINE and b"\n" not in buf:
                sock.write(b"error\n")
                sock.disconnectFromServer()
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                sock.write(self._answer(line))
            self._buffers[sock] = buf
            sock.flush()
        except Exception as exc:            # never let the listener die
            self._say(f"single window: bad request ({exc})", error=True)

    def _answer(self, line: bytes) -> bytes:
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception:
            return b"error\n"
        if not isinstance(msg, dict):
            return b"error\n"
        if msg.get("ping"):
            return b"ok\n"
        path = msg.get("open")
        if not isinstance(path, str) or not path:
            return b"error\n"
        # answer FIRST, load after: a big file must not keep the sender
        # waiting past its timeout (it would then start a second viewer)
        from PySide6 import QtCore
        QtCore.QTimer.singleShot(0, lambda p=path: self._open(p))
        return b"ok\n"

    def _open(self, path: str):
        try:
            self.window.load_file(Path(path))   # a bad file: the window says so
        except Exception as exc:
            self._say(f"could not open {path}: {exc}", error=True)
        self._bring_to_front()

    def _bring_to_front(self):
        try:
            w = self.window
            if w.isMinimized():
                w.showNormal()
            w.show()
            w.raise_()
            w.activateWindow()
            if sys.platform == "win32":
                import ctypes
                hwnd = int(w.winId())
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                if ctypes.windll.user32.GetForegroundWindow() != hwnd:
                    from PySide6 import QtWidgets
                    QtWidgets.QApplication.alert(w, 3000)   # flash the taskbar button
        except Exception:
            pass

    def _say(self, text: str, error: bool = False):
        try:
            say = getattr(self.window, "say", None)
            if say is not None:
                say(text, error)
        except Exception:
            pass


# ───────────────────────────── the command ────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AaltoView")
    ap.add_argument("file", nargs="?", help="a measurement (.nc) to open")
    ap.add_argument("--folder", default=None,
                    help="data folder to list (default: the one used last time)")
    ap.add_argument("--theme", choices=["dark", "light"], default=None)
    ap.add_argument("--new-window", action="store_true",
                    help="always open a new window (do not hand the file to the "
                         "viewer that is already running)")
    return ap


def main(argv=None) -> int:
    """The `aaltoview` command: a file goes to the running viewer if there is
    one -- decided before pyqtgraph and the window are imported, so the
    handover takes ~0.2 s instead of a viewer start."""
    args = build_parser().parse_args(argv)
    if args.file and not args.new_window and send_to_running(args.file):
        return 0
    from .viewer import main as viewer_main
    return viewer_main(argv, try_handover=False)   # just tried
