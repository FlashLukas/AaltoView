"""analysis_link.py -- how AaltoView hands curves to analysis modules.

An analysis module (a fit, a correction, ...) is a SEPARATE PROGRAM with its own
window: `analysis/<name>/` in this repository, its own package. Separate so a
module that is busy or crashes cannot freeze the viewer, so several can run at
once, and so their heavy dependencies (lmfit, ...) stay out of the viewer.
docs/ANALYSIS_MODULES.md explains how to write one.

Three questions, three mechanisms:

* WHICH modules are there?  Every folder in AnalysisModules/ (next to this
  package's repository; more folders through AALTOVIEW_MODULES) with a
  module.toml: key, name, description, and the Python package to start.
  Dropping a folder in is all it takes -- the viewer never imports a module's
  code, it reads the manifest and starts the module (launch()).
* WHICH are running, and where?  A running module listens on a free TCP port
  of 127.0.0.1 (this PC only) and writes a "beacon" file saying so into
  beacon_dir(): {key, name, title, port, pid}. It deletes the file when it
  closes; a beacon left behind by a crash is removed as soon as its process is
  found dead. No fixed ports to configure, and two windows of one module can
  run side by side.
* HOW does data get there?  ZeroMQ request/reply with JSON, the same pattern as
  the AaltoFlow instrument services (its docs/DEVELOPER_NOTES.md section 4):
  every request is {"cmd": ...}, every reply {"ok": true, ...} or
  {"ok": false, "error": ...}. Verbs: describe, add_curves, add_maps, shutdown.

A curve travels complete (curve_to_dict): x, the shown y, the COMPLEX values if
the detector is complex, names, units, label, source file, the Selection and the
held coordinates -- the frozen Curve of export.py, nothing lost on the way.
A map travels the same way (map_to_dict): both axes, the shown values, the
complex map, the reference that was applied. A module that says `accepts =
["curves", "maps"]` but has no map handler of its own gets it as one curve per
row (export.map_to_curves); a module that takes only curves is sent those.

No Qt here: tests drive both ends in one process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .export import Curve, MapData, Selection, map_to_curves

PROTOCOL = 1
BEACON_ENV = "AALTOVIEW_ANALYSIS_DIR"      # tests point this at a temp folder
MODULES_ENV = "AALTOVIEW_MODULES"          # extra module folders, os.pathsep-separated
MODULES_DIR = "AnalysisModules"
#: the repository this package runs from (editable install); None when installed
#: as a plain dependency (scan-core), where there are no modules to offer
REPO = Path(__file__).resolve().parents[2]


# ──────────────────────────────── available modules ───────────────────────────

@dataclass
class ModuleInfo:
    key: str
    name: str
    description: str
    module: str                         # python -m <module>
    accepts: tuple[str, ...] = ("curves",)
    folder: Path | None = None          # AnalysisModules/<name>


def read_manifest(path: str | Path) -> dict:
    """module.toml -> {key, name, description, accepts, module}."""
    with open(path, "rb") as f:
        m = tomllib.load(f)["module"]
    return {"key": m["key"], "name": m.get("name", m["key"]),
            "description": m.get("description", ""),
            "accepts": list(m.get("accepts", ["curves"])),
            "module": m.get("python", m["key"])}


def info_from_dict(d: dict, folder: Path | None = None) -> ModuleInfo:
    return ModuleInfo(key=d["key"], name=d.get("name", d["key"]),
                      description=d.get("description", ""),
                      module=d.get("module", d["key"]),
                      accepts=tuple(d.get("accepts", ("curves",))), folder=folder)


def module_dirs() -> list[Path]:
    dirs = [Path(p) for p in os.environ.get(MODULES_ENV, "").split(os.pathsep) if p]
    dirs.append(REPO / MODULES_DIR)
    return [d for d in dirs if d.is_dir()]


def installed() -> list[ModuleInfo]:
    """Every module folder with a readable module.toml, by name. A broken one
    is skipped, not fatal: one bad folder must not take the menu down."""
    out, seen = [], set()
    for d in module_dirs():
        for manifest in sorted(d.glob("*/module.toml")):
            try:
                info = info_from_dict(read_manifest(manifest), manifest.parent)
            except Exception:
                continue
            if info.key not in seen:
                seen.add(info.key)
                out.append(info)
    return sorted(out, key=lambda m: m.name.lower())


def launch_command(info: ModuleInfo) -> tuple[list[str], dict]:
    """(argv, extra environment) to start a module.

    With uv and the module inside this repository's workspace: `uv run
    --all-packages` from the repository, which first installs whatever a newly
    dropped module needs (only the first start of that module takes long).
    Otherwise the viewer's own Python with the module's src/ on the path --
    enough when its packages are already there.
    """
    uv = shutil.which("uv")
    folder = info.folder
    in_repo = folder is not None and REPO / MODULES_DIR in folder.parents
    if uv and in_repo and (REPO / "pyproject.toml").exists():
        return ([uv, "run", "--project", str(REPO), "--all-packages", "--extra", "gui",
                 "python", "-m", info.module], {})
    env = {}
    if folder is not None and (folder / "src").is_dir():
        env["PYTHONPATH"] = os.pathsep.join(
            [str(folder / "src")] + [p for p in [os.environ.get("PYTHONPATH")] if p])
    return [sys.executable, "-m", info.module], env


def launch_log(info: ModuleInfo) -> Path:
    return beacon_dir() / f"{info.key}-start.log"


def launch(info: ModuleInfo) -> subprocess.Popen:
    """Start a module detached from the viewer (it keeps running when the viewer
    closes), with no console window; what it prints goes to launch_log()."""
    cmd, env = launch_command(info)
    flags = 0
    if sys.platform == "win32":
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                 | subprocess.DETACHED_PROCESS)
    log = open(launch_log(info), "w", encoding="utf-8")
    try:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env={**os.environ, **env},
                                cwd=str(REPO) if REPO.exists() else None,
                                creationflags=flags)
    finally:
        log.close()                     # the child has its own handle


# ──────────────────────────────── running modules ─────────────────────────────

def beacon_dir() -> Path:
    """Per Windows user (LOCALAPPDATA), so two people on one PC do not see each
    other's windows."""
    if os.environ.get(BEACON_ENV):
        d = Path(os.environ[BEACON_ENV])
    elif os.environ.get("LOCALAPPDATA"):
        d = Path(os.environ["LOCALAPPDATA"]) / "AaltoView" / "analysis"
    else:
        d = Path.home() / ".aaltoview" / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Running:
    key: str
    name: str
    title: str                          # "VNA-FMR fit" or "VNA-FMR fit 2": what the window says
    port: int
    pid: int
    beacon: Path
    accepts: tuple[str, ...] = ("curves",)      # from `describe`


def pid_alive(pid: int) -> bool:
    """Is process `pid` still there? NOT os.kill(pid, 0): on Windows signal 0 is
    CTRL_C_EVENT, which would interrupt the process instead of asking about it."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259      # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def beacons() -> list[Running]:
    """Every beacon of a live process; those of dead processes are deleted."""
    out = []
    for p in sorted(beacon_dir().glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            r = Running(key=d["key"], name=d["name"], title=d.get("title", d["name"]),
                        port=int(d["port"]), pid=int(d["pid"]), beacon=p)
        except Exception:
            continue                    # half-written right now, or not ours
        if not pid_alive(r.pid):
            p.unlink(missing_ok=True)
            continue
        out.append(r)
    return out


def running(timeout_ms: int = 400) -> list[Running]:
    """Beacons whose module actually answers `describe` -- what the menu offers."""
    out = []
    for r in beacons():
        try:
            d = request(r.port, {"cmd": "describe"}, timeout_ms)
        except Exception:
            continue
        if d.get("ok"):
            r.accepts = tuple(d.get("accepts", ("curves",)))
            out.append(r)
    return out


def request(port: int, msg: dict, timeout_ms: int = 2000) -> dict:
    """One request, one reply. A fresh socket each time: a REQ socket that has
    timed out is stuck and cannot be reused."""
    import zmq
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        s.connect(f"tcp://127.0.0.1:{port}")
        s.send_string(json.dumps(msg))
        return json.loads(s.recv_string())
    except zmq.Again:
        raise TimeoutError(f"no answer from port {port} within {timeout_ms} ms") from None
    finally:
        s.close()


def send_curves(port: int, curves: list[Curve], timeout_ms: int = 5000) -> dict:
    reply = request(port, {"cmd": "add_curves", "protocol": PROTOCOL,
                           "curves": [curve_to_dict(c) for c in curves]}, timeout_ms)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "the module refused the curves"))
    return reply


def send_maps(port: int, maps: list[MapData], accepts=("curves", "maps"),
              timeout_ms: int = 30000) -> dict:
    """Maps to a module -- as maps if it takes them, else one curve per row.
    A long timeout: a VNA map is 16 001 x 13 complex numbers of JSON."""
    if "maps" in accepts:
        reply = request(port, {"cmd": "add_maps", "protocol": PROTOCOL,
                               "maps": [map_to_dict(m) for m in maps]}, timeout_ms)
    else:
        return send_curves(port, [c for m in maps for c in map_to_curves(m)], timeout_ms)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "the module refused the maps"))
    return reply


# ────────────────────────────── curves on the wire ────────────────────────────

def _list(a) -> list | None:
    # json writes NaN as NaN (not strict JSON, but both ends are Python), so a
    # hole in a running scan stays a hole
    return None if a is None else [float(v) for v in np.asarray(a, dtype=float)]


def curve_to_dict(c: Curve) -> dict:
    z = None if c.z is None else np.asarray(c.z, dtype=complex)
    return {"x": _list(c.x), "y": _list(c.y),
            "z_real": _list(z.real) if z is not None else None,
            "z_imag": _list(z.imag) if z is not None else None,
            "label": c.label, "x_name": c.x_name, "x_unit": c.x_unit,
            "y_name": c.y_name, "y_unit": c.y_unit, "source": c.source,
            "selection": c.selection.to_dict(),
            "held": {k: [float(v), u] for k, (v, u) in c.held.items()}}


def curve_from_dict(d: dict) -> Curve:
    z = None
    if d.get("z_real") is not None:
        z = np.asarray(d["z_real"], dtype=float) + 1j * np.asarray(d["z_imag"], dtype=float)
    return Curve(x=np.asarray(d["x"], dtype=float), y=np.asarray(d["y"], dtype=float),
                 label=d["label"], x_name=d["x_name"], x_unit=d["x_unit"],
                 y_name=d["y_name"], y_unit=d["y_unit"],
                 selection=Selection.from_dict(d["selection"]), source=d.get("source"),
                 z=z, held={k: (float(v), str(u)) for k, (v, u) in d.get("held", {}).items()})


def _grid(a) -> list | None:
    return None if a is None else [_list(row) for row in np.atleast_2d(np.asarray(a, dtype=float))]


def map_to_dict(m: MapData) -> dict:
    z = None if m.z is None else np.asarray(m.z, dtype=complex)
    return {"x": _list(m.x), "y": _list(m.y), "values": _grid(m.values),
            "z_real": _grid(z.real) if z is not None else None,
            "z_imag": _grid(z.imag) if z is not None else None,
            "label": m.label, "x_name": m.x_name, "x_unit": m.x_unit,
            "y_name": m.y_name, "y_unit": m.y_unit, "z_name": m.z_name, "z_unit": m.z_unit,
            "source": m.source, "selection": m.selection.to_dict(), "ref": m.ref,
            "held": {k: [float(v), u] for k, (v, u) in m.held.items()}}


def map_from_dict(d: dict) -> MapData:
    z = None
    if d.get("z_real") is not None:
        z = np.asarray(d["z_real"], dtype=float) + 1j * np.asarray(d["z_imag"], dtype=float)
    return MapData(x=np.asarray(d["x"], dtype=float), y=np.asarray(d["y"], dtype=float),
                   values=np.asarray(d["values"], dtype=float), label=d["label"],
                   x_name=d["x_name"], x_unit=d["x_unit"], y_name=d["y_name"],
                   y_unit=d["y_unit"], z_name=d["z_name"], z_unit=d["z_unit"],
                   selection=Selection.from_dict(d["selection"]), source=d.get("source"),
                   z=z, ref=d.get("ref"),
                   held={k: (float(v), str(u)) for k, (v, u) in d.get("held", {}).items()})


# ─────────────────────────────── the module's end ─────────────────────────────

class Listener:
    """A module's mailbox: answers requests on a thread, announces itself with a
    beacon file.

        lis = Listener(info, on_curves=lambda curves: ...)
        lis.start()        # binds, writes the beacon
        ...
        lis.stop()         # deletes the beacon

    `on_curves` is called ON THE LISTENER THREAD with a list of Curve; a Qt
    module turns it into a signal (apps/analysis.py does). The reply goes out at
    once, so the viewer never waits for a fit. `on_maps` likewise with a list of
    MapData; without it, maps arrive through on_curves, one curve per row.
    """

    def __init__(self, info: dict | ModuleInfo, on_curves, on_shutdown=None, on_maps=None):
        self.info = info if isinstance(info, ModuleInfo) else info_from_dict(info)
        self.on_curves = on_curves
        self.on_maps = on_maps
        self.on_shutdown = on_shutdown
        self.port: int | None = None
        self.title = self.info.name
        self.beacon: Path | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock = None

    def start(self) -> None:
        import zmq
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        # "VNA-FMR fit 2" when one is already open, so the viewer's menu can tell
        # the windows apart
        others = [b for b in beacons() if b.key == self.info.key]
        if others:
            self.title = f"{self.info.name} {len(others) + 1}"
        # bound BEFORE announced: a viewer that reads the beacon can connect
        self._thread = threading.Thread(target=self._loop, name=f"{self.info.key}-listener",
                                        daemon=True)
        self._thread.start()
        self.beacon = beacon_dir() / f"{self.info.key}-{os.getpid()}-{self.port}.json"
        tmp = self.beacon.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": self.info.key, "name": self.info.name,
                                   "title": self.title, "port": self.port,
                                   "pid": os.getpid(), "protocol": PROTOCOL,
                                   "started": time.time()}), encoding="utf-8")
        tmp.replace(self.beacon)          # atomic: never a half-written beacon

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.beacon is not None:
            self.beacon.unlink(missing_ok=True)
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _loop(self) -> None:
        import zmq
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(200)):
                continue
            try:
                msg = json.loads(self._sock.recv_string())
            except Exception as exc:
                self._sock.send_string(json.dumps({"ok": False, "error": f"bad request: {exc}"}))
                continue
            try:
                reply = self.handle(msg)
            except Exception as exc:          # always reply, or the client hangs
                reply = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
            self._sock.send_string(json.dumps(reply))

    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        if cmd == "describe":
            return {"ok": True, "key": self.info.key, "name": self.info.name,
                    "title": self.title, "description": self.info.description,
                    "accepts": list(self.info.accepts), "protocol": PROTOCOL,
                    "pid": os.getpid()}
        if cmd == "add_curves":
            curves = [curve_from_dict(d) for d in msg.get("curves", [])]
            if not curves:
                return {"ok": False, "error": "no curves in the request"}
            self.on_curves(curves)
            return {"ok": True, "n": len(curves), "title": self.title}
        if cmd == "add_maps":
            maps = [map_from_dict(d) for d in msg.get("maps", [])]
            if not maps:
                return {"ok": False, "error": "no maps in the request"}
            if self.on_maps is not None:
                self.on_maps(maps)
            else:
                self.on_curves([c for m in maps for c in map_to_curves(m)])
            return {"ok": True, "n": len(maps), "title": self.title}
        if cmd == "shutdown":
            if self.on_shutdown is not None:
                self.on_shutdown()
            return {"ok": True}
        return {"ok": False, "error": f"unknown cmd '{cmd}' (have: describe, add_curves, "
                                      "add_maps, shutdown)"}
