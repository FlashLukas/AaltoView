"""analysis.py -- the Qt half of the analysis link (analysis_link.py is the rest).

Two ends:
* the VIEWER's "Analysis" menu (AnalysisMenu): lists the modules that are
  running (send to one) and the ones that are installed but closed (start it,
  then send when it has announced itself);
* a MODULE's start-up (run_module): theme, icon, the listener, and curves
  arriving on the GUI thread as a Qt signal. A new module is a window class
  with an `add_curves(list[Curve])` method plus three lines in __main__.py.
"""

from __future__ import annotations

import argparse
import time

from PySide6 import QtCore, QtWidgets

from .. import analysis_link as AL
from .theme import (DEFAULT_THEME, apply, apply_window_icon, refresh_taskbar_icon,
                    set_theme)


# ─────────────────────────────── the viewer's end ─────────────────────────────

class AnalysisMenu(QtWidgets.QMenu):
    """Built fresh each time it opens, so it shows what is running NOW.

    get_curves() -> list[Curve]   what to send (raises ValueError with a message
                                  for the status line when there is nothing);
                                  with kind="maps": list[MapData]
    say(text, error)              the panel's status line

    Maps go to every module: as maps to one whose module.toml accepts them,
    as one curve per row to one that takes only curves.
    """

    #: how long a started module gets to announce itself. Generous: the FIRST
    #: start of a newly dropped module installs its packages (uv run)
    START_TIMEOUT_S = 180.0

    def __init__(self, parent, get_curves, say, kind: str = "curves"):
        super().__init__(parent)
        self.get_curves = get_curves
        self.kind = kind
        self.say = say
        self._waiting: tuple | None = None       # (info, curves, t0, beacons before, proc)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._poll_started)
        self.aboutToShow.connect(self._fill)

    def _fill(self):
        self.clear()
        run = AL.running()
        inst = AL.installed()
        for r in run:
            a = self.addAction(f"Send to {r.title}")
            a.triggered.connect(lambda _=False, r=r: self.send_to(r))
        if run:
            self.addSeparator()
        for m in inst:
            how = "" if self.kind == "curves" or "maps" in m.accepts else " (as curves, one per row)"
            a = self.addAction(f"Start {m.name} and send{how}")
            a.setToolTip(m.description)
            a.triggered.connect(lambda _=False, m=m: self.start_and_send(m))
        if not inst:
            a = self.addAction("no analysis modules found")
            a.setEnabled(False)
            a2 = self.addAction(f"(drop one into {AL.home.repo() / AL.MODULES_DIR})")
            a2.setEnabled(False)
        self.setToolTipsVisible(True)

    def _curves(self):
        try:
            return self.get_curves()
        except Exception as exc:
            self.say(f"Analysis: {exc}", error=True)
            return None

    def send_to(self, r: AL.Running, curves=None):
        curves = curves if curves is not None else self._curves()
        if not curves:
            return
        what = "map" if self.kind == "maps" else "curve"
        try:
            if self.kind == "maps":
                AL.send_maps(r.port, curves, r.accepts)
            else:
                AL.send_curves(r.port, curves)
        except Exception as exc:
            self.say(f"Analysis: {r.title} did not take the {what}s: {exc}", error=True)
            return
        n = len(curves)
        how = "" if self.kind == "curves" or "maps" in r.accepts else " (as one curve per row)"
        self.say(f"sent {n} {what}{'s' if n != 1 else ''} to {r.title}{how}")

    def start_and_send(self, m: AL.ModuleInfo):
        curves = self._curves()
        if not curves:
            return
        if self._waiting is not None:
            self.say(f"Analysis: still waiting for {self._waiting[0].name} to start…")
            return
        before = {b.beacon for b in AL.beacons() if b.key == m.key}
        try:
            proc = AL.launch(m)
        except Exception as exc:
            self.say(f"Analysis: could not start {m.name}: {exc}", error=True)
            return
        self._waiting = (m, curves, time.monotonic(), before, proc)
        self._timer.start()
        self.say(f"starting {m.name}… the curves follow as soon as its window is up "
                 "(the first start of a new module installs its packages: up to a minute)")

    def _poll_started(self):
        m, curves, t0, before, proc = self._waiting
        new = [b for b in AL.beacons() if b.key == m.key and b.beacon not in before]
        if new:
            self._timer.stop()
            self._waiting = None
            new[0].accepts = tuple(m.accepts)
            self.send_to(new[0], curves)
            return
        died = proc.poll() is not None
        if died or time.monotonic() - t0 > self.START_TIMEOUT_S:
            self._timer.stop()
            self._waiting = None
            log = AL.launch_log(m)
            try:
                tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                last = tail[-1] if tail else ""
            except OSError:
                last = ""
            why = "stopped" if died else f"did not start within {self.START_TIMEOUT_S:.0f} s"
            self.say(f"Analysis: {m.name} {why}. {last}  (full output: {log})", error=True)


# ─────────────────────────────── a module's end ───────────────────────────────

class Inbox(QtCore.QObject):
    """Moves curves from the listener thread to the GUI thread (a queued signal)."""
    curvesArrived = QtCore.Signal(object)          # list[Curve]
    mapsArrived = QtCore.Signal(object)            # list[MapData]
    shutdownRequested = QtCore.Signal()


def run_module(info: dict, window_class, argv=None) -> int:
    """Start an analysis module: `window_class()` must have add_curves(curves);
    with add_maps(maps) too, whole maps arrive as maps (else one curve per row).

        # fmr_fit/__main__.py
        return run_module(INFO, FitWindow, argv)
    """
    from .viewer import configure_pyqtgraph      # same plotting defaults as the viewer

    ap = argparse.ArgumentParser(description=info.get("name", info["key"]))
    ap.add_argument("--theme", choices=["dark", "light"], default=None)
    args = ap.parse_args(argv)
    set_theme(args.theme or DEFAULT_THEME)       # BEFORE any widget is built
    configure_pyqtgraph()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply(app)
    # its own taskbar button, not merged with the viewer's
    apply_window_icon(app, app_id=f"Aalto.AaltoView.{info['key']}")

    win = window_class()
    inbox = Inbox()
    inbox.curvesArrived.connect(lambda curves: _deliver(win, curves))
    inbox.shutdownRequested.connect(win.close)
    on_maps = None
    if hasattr(win, "add_maps"):
        inbox.mapsArrived.connect(lambda maps: _deliver(win, maps, "add_maps"))
        on_maps = inbox.mapsArrived.emit
    lis = AL.Listener(info, on_curves=inbox.curvesArrived.emit,
                      on_shutdown=inbox.shutdownRequested.emit, on_maps=on_maps)
    lis.start()
    app.aboutToQuit.connect(lis.stop)
    win.setWindowTitle(lis.title)
    win.show()
    refresh_taskbar_icon(win)
    try:
        return app.exec()
    finally:
        lis.stop()


def _deliver(win, items, method: str = "add_curves"):
    getattr(win, method)(items)
    # Windows does not let a background program take the focus; it flashes the
    # taskbar button instead, which is what alert() asks for
    if win.isMinimized():
        win.showNormal()
    win.raise_()
    win.activateWindow()
    QtWidgets.QApplication.alert(win)
