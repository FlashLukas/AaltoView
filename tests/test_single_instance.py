"""One viewer window per user: a file opened while a viewer runs loads THERE.

Every test listens on its own server name (AALTOVIEW_INSTANCE_NAME), so a
viewer running on this PC is never touched. The sender runs in a separate
Python process -- as AaltoFlow's catalogue does -- because its blocking wait
would otherwise stop this process's event loop from answering.
"""

import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from aaltoview.apps import single_instance as SI  # noqa: E402


@pytest.fixture
def name(monkeypatch):
    n = f"AaltoView-test-{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv(SI.NAME_ENV, n)
    return n


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _nc(path):
    f = np.linspace(1.0, 4.0, 4)
    ds = xr.Dataset({"kerr": (("freq", "x"), np.outer(f, [1.0, 2.0]), {"units": "mdeg"})},
                    coords={"freq": ("freq", f, {"units": "GHz"}),
                            "x": ("x", [0.0, 1.0], {"units": "um"})},
                    attrs={"name": "small", "dims": "freq,x"})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def _in_child(code: str, name: str, app, timeout=20.0):
    """Run `code` in another Python process while this one's events run (the
    listener answers); return its stdout."""
    env = {**os.environ, SI.NAME_ENV: name}
    p = subprocess.Popen([sys.executable, "-c", code], env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t = time.time()
    while p.poll() is None:
        app.processEvents()
        time.sleep(0.005)
        if time.time() - t > timeout:
            p.kill()
            raise AssertionError("the child process hung")
    out, err = p.communicate()
    assert p.returncode == 0, err
    return out


def _pump(app, seconds=0.3):
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture
def window(tmp_path, app):
    from aaltoview.apps import viewer as VW
    VW.configure_pyqtgraph()
    w = VW.ViewerWindow(tmp_path)
    yield w
    if w.listener:
        w.listener.close()
    w.close()


def test_nobody_listening_means_start_a_viewer(name, tmp_path):
    t = time.perf_counter()
    assert not SI.send_to_running(tmp_path / "a.nc")
    assert not SI.is_running()
    assert time.perf_counter() - t < 1.0          # no waiting when nobody is there


def test_the_server_name_is_per_user_and_plain(monkeypatch):
    monkeypatch.delenv(SI.NAME_ENV, raising=False)
    monkeypatch.setattr(SI.getpass, "getuser", lambda: "dr. Ann-Lee")
    assert SI.server_name() == "AaltoView-dr__Ann_Lee"


def test_a_file_sent_to_the_running_window_is_loaded_there(name, window, tmp_path, app):
    path = _nc(tmp_path / "2026-10-06" / "120000_small.nc")
    window.listener = SI.Listener(window)
    assert window.listener.start()
    out = _in_child("from aaltoview.apps.single_instance import send_to_running, is_running\n"
                    f"print(is_running(), send_to_running({str(path)!r}))", name, app)
    assert out.split() == ["True", "True"]
    _pump(app)
    assert window.viewer.path == path and window.viewer.ds is not None


def test_the_aaltoview_command_hands_over_and_exits(name, window, tmp_path, app):
    """`python -m aaltoview FILE` with a viewer running: handed over, exit 0,
    and pyqtgraph never imported in the sender."""
    path = _nc(tmp_path / "2026-10-06" / "130000_small.nc")
    window.listener = SI.Listener(window)
    assert window.listener.start()
    out = _in_child("import sys, time\n"
                    "t = time.perf_counter()\n"
                    "from aaltoview.apps.single_instance import main\n"
                    f"rc = main([{str(path)!r}])\n"
                    "print(rc, 'pyqtgraph' in sys.modules, 'xarray' in sys.modules,"
                    " round(time.perf_counter() - t, 3))", name, app)
    rc, pg, xr_, secs = out.split()
    assert (rc, pg, xr_) == ("0", "False", "False")
    assert float(secs) < 1.0
    _pump(app)
    assert window.viewer.path == path


def test_a_bad_file_still_answers_ok_and_the_window_says_why(name, window, tmp_path, app):
    window.listener = SI.Listener(window)
    window.listener.start()
    missing = tmp_path / "no_such_file.nc"
    out = _in_child("from aaltoview.apps.single_instance import send_to_running\n"
                    f"print(send_to_running({str(missing)!r}))", name, app)
    assert out.strip() == "True"
    _pump(app)
    assert "Could not read no_such_file.nc" in window.viewer.map.status.text()


def test_a_stale_name_does_not_block_the_next_viewer(name, window, tmp_path, app):
    first = SI.Listener(window)
    assert first.start()
    first.close()                                  # the viewer that "crashed"
    if sys.platform != "win32":                    # a socket file left behind
        from PySide6 import QtCore
        open(os.path.join(QtCore.QDir.tempPath(), name), "w").close()
    assert not SI.is_running()
    window.listener = SI.Listener(window)
    assert window.listener.start()
    path = _nc(tmp_path / "2026-10-06" / "140000_small.nc")
    out = _in_child("from aaltoview.apps.single_instance import send_to_running\n"
                    f"print(send_to_running({str(path)!r}))", name, app)
    assert out.strip() == "True"


def test_malformed_requests_do_not_kill_the_listener(name, window, app):
    window.listener = SI.Listener(window)
    window.listener.start()
    code = r"""
import json
from PySide6 import QtNetwork
from aaltoview.apps.single_instance import server_name, is_running

s = QtNetwork.QLocalSocket()
s.connectToServer(server_name())
s.waitForConnected(1000)

def ask(raw):
    s.write(raw)
    s.waitForBytesWritten(1000)
    got = b""
    while not got.endswith(b"\n") and s.waitForReadyRead(1000):
        got += bytes(s.readAll())
    return got.strip().decode()

replies = [ask(line) for line in (b"not json\n", b"[1, 2]\n", b"{}\n", b'{"open": 5}\n',
                                   b'{"open": "\xc3\xa9"\n', b"\xff\xfe\n")]
big = ask(b"x" * 70000)
s.abort()
print(json.dumps(replies), big, is_running())
"""
    out = _in_child(code, name, app)
    replies, big, alive = out.rsplit(" ", 2)
    assert json.loads(replies) == ["error"] * 6
    assert big == "error" and alive.strip() == "True"
    assert window.listener.server.isListening()


def test_a_viewer_that_never_answers_costs_one_timeout(name, app):
    """Listening but silent (busy, or hung): send_to_running gives up after its
    ONE timeout in total -- the catalogue calls it on its GUI thread."""
    from PySide6 import QtNetwork
    server = QtNetwork.QLocalServer()
    assert server.listen(name)                     # accepts, never answers
    try:
        out = _in_child("import time\n"
                        "from aaltoview.apps.single_instance import send_to_running\n"
                        "t = time.perf_counter()\n"
                        "ok = send_to_running('x.nc', timeout_ms=400)\n"
                        "print(ok, round(time.perf_counter() - t, 3))", name, app)
    finally:
        server.close()
    ok, secs = out.split()
    assert ok == "False" and float(secs) < 0.6
