"""The link to analysis modules: both ends in one process, over real sockets.

What must not go wrong:
* a curve arrives complete -- complex values, holes (NaN), units, the held
  coordinates -- or a fit is done on something other than what was on screen;
* the viewer only offers modules that are really there: a beacon left behind
  by a crashed module is cleaned up, not listed;
* the menu works from the viewer's side.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("zmq")

from aaltoview import analysis_link as AL  # noqa: E402
from aaltoview import export as E  # noqa: E402
from aaltoview.view import Slice  # noqa: E402

INFO = {"key": "test_mod", "name": "Test module", "description": "for tests",
        "module": "nothing_to_start"}


@pytest.fixture(autouse=True)
def beacon_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv(AL.BEACON_ENV, str(tmp_path / "beacons"))


def _complex_ds() -> xr.Dataset:
    f = np.array([6.0, 8.0])
    b = np.linspace(0, 100, 11)
    z = (f[:, None] + 1j * b[None, :]).astype(complex)
    z[1, 3] = np.nan
    return xr.Dataset(
        {"s21_real": (("rf_freq", "field"), z.real, {"units": "V", "complex_pair": "s21",
                                                     "complex_part": "real"}),
         "s21_imag": (("rf_freq", "field"), z.imag, {"units": "V", "complex_pair": "s21",
                                                     "complex_part": "imag"})},
        coords={"rf_freq": ("rf_freq", f, {"units": "GHz"}),
                "field": ("field", b, {"units": "mT"})},
        attrs={"dims": "rf_freq,field"})


def test_a_curve_keeps_its_complex_values_and_where_it_was_held():
    ds = _complex_ds()
    c = E.make_curve(ds, E.Selection("s21", x="field", slices={"rf_freq": Slice("at", 1)},
                                     part="abs"), "x.nc")
    assert c.held == {"rf_freq": (8.0, "GHz")}
    assert c.z is not None and c.z[2] == 8 + 20j
    assert c.y[2] == pytest.approx(abs(8 + 20j))       # y is still what was shown
    # averaged dims are not "held"
    c2 = E.make_curve(ds, E.Selection("s21", x="field", slices={"rf_freq": Slice("mean")}))
    assert c2.held == {}


def test_a_curve_survives_the_wire_holes_included():
    ds = _complex_ds()
    c = E.make_curve(ds, E.Selection("s21", x="field", slices={"rf_freq": Slice("at", 1)},
                                     part="real"), "x.nc")
    back = AL.curve_from_dict(__import__("json").loads(
        __import__("json").dumps(AL.curve_to_dict(c))))
    assert np.array_equal(back.x, c.x)
    assert np.isnan(back.z[3]) and np.isnan(back.y[3])
    np.testing.assert_array_equal(back.z[np.isfinite(back.z)], c.z[np.isfinite(c.z)])
    assert (back.label, back.x_unit, back.y_unit, back.source) == (c.label, "mT", "V", "x.nc")
    assert back.held == {"rf_freq": (8.0, "GHz")}
    assert back.selection.to_dict() == c.selection.to_dict()


def test_a_running_module_is_found_and_receives_curves():
    got = []
    lis = AL.Listener(INFO, on_curves=got.extend)
    lis.start()
    try:
        run = AL.running()
        assert [(r.key, r.port) for r in run] == [("test_mod", lis.port)]
        c = E.make_curve(_complex_ds(), E.Selection("s21", x="field"), "x.nc")
        reply = AL.send_curves(run[0].port, [c, c])
        assert reply["n"] == 2 and len(got) == 2
        assert got[0].z is not None
        bad = AL.request(lis.port, {"cmd": "fly"})
        assert not bad["ok"] and "unknown cmd" in bad["error"]
        # a second window of the same module gets a number
        lis2 = AL.Listener(INFO, on_curves=got.extend)
        lis2.start()
        try:
            assert lis2.title == "Test module 2"
            assert len(AL.running()) == 2
        finally:
            lis2.stop()
    finally:
        lis.stop()
    assert AL.running() == []                       # stop() removed the beacon


def test_the_beacon_of_a_crashed_module_is_cleaned_up():
    p = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                       capture_output=True, text=True)
    dead = int(p.stdout)
    assert not AL.pid_alive(dead)
    assert AL.pid_alive(os.getpid())
    f = AL.beacon_dir() / "test_mod-1.json"
    f.write_text('{"key": "test_mod", "name": "x", "port": 1, "pid": %d}' % dead,
                 encoding="utf-8")
    assert AL.beacons() == []
    assert not f.exists()


def test_the_viewer_menu_sends_to_a_running_module(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from aaltoview.apps import viewer as VW
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    got = []
    lis = AL.Listener(INFO, on_curves=got.extend)
    lis.start()
    w = VW.ViewerWidget()
    try:
        w.set_dataset(_complex_ds(), tmp_path / "s.nc")
        menu = w.lines.exports.analysis_menu
        menu._fill()
        sends = [a for a in menu.actions() if a.text() == "Send to Test module"]
        assert len(sends) == 1
        sends[0].trigger()                 # nothing frozen: the preview goes
        assert len(got) == 1 and got[0].z is not None
        assert "sent 1 curve to Test module" in w.lines.status.text()
    finally:
        lis.stop()
        w.close()


def test_a_dropped_module_folder_is_found(tmp_path, monkeypatch):
    """Drop-in: a folder with a module.toml in a module directory is listed --
    nothing installed, nothing registered. A broken manifest is skipped."""
    drop = tmp_path / "MyModules"
    (drop / "kerr-fit").mkdir(parents=True)
    (drop / "kerr-fit" / "module.toml").write_text(
        '[module]\nkey = "kerr_fit"\nname = "Kerr fit"\npython = "kerr_fit"\n',
        encoding="utf-8")
    (drop / "broken").mkdir()
    (drop / "broken" / "module.toml").write_text("[module\n", encoding="utf-8")
    (drop / "not-a-module").mkdir()
    monkeypatch.setenv(AL.MODULES_ENV, str(drop))
    found = {m.key: m for m in AL.installed()}
    assert "kerr_fit" in found and found["kerr_fit"].folder == drop / "kerr-fit"
    assert found["kerr_fit"].module == "kerr_fit"
    # outside this repository's workspace: the viewer's Python + its src/ on the path
    cmd, env = AL.launch_command(found["kerr_fit"])
    assert cmd[-2:] == ["-m", "kerr_fit"]


def test_the_fmr_fit_module_in_this_repository_is_found():
    m = next((m for m in AL.installed() if m.key == "fmr_fit"), None)
    assert m is not None and m.folder.name == "vna-fmr-fit"
    cmd, _ = AL.launch_command(m)
    assert cmd[-2:] == ["-m", "fmr_fit"]
    if "uv" in Path(cmd[0]).stem:          # uv run: installs a new module's packages first
        assert "--all-packages" in cmd


# ─────────────────────────────── whole maps ───────────────────────────────────

MAP_INFO = {**INFO, "accepts": ["curves", "maps"]}


def test_a_map_keeps_its_complex_values_and_its_reference():
    ds = _complex_ds()
    sel = E.Selection("s21", x="field", y="rf_freq", part="abs")
    m = E.make_map_data(ds, sel, E.MapStyle(ref="row", ref_i=0, norm="rows", log=True), "x.nc")
    z = E.detector(ds, "s21").values
    # divided by the 6 GHz row, on the complex values; norm and log NOT applied
    np.testing.assert_allclose(m.z[1, 5], z[1, 5] / z[0, 5])
    assert m.values[1, 5] == pytest.approx(abs(z[1, 5] / z[0, 5]))
    assert np.isnan(m.values[1, 3])                    # a hole stays a hole
    assert m.ref == {"mode": "row", "op": "divide", "i": 0}
    assert "÷ rf_freq = 6 GHz" in m.z_name and m.z_unit == ""
    back = AL.map_from_dict(__import__("json").loads(
        __import__("json").dumps(AL.map_to_dict(m))))
    assert back.values.shape == (2, 11) and np.isnan(back.z[1, 3])
    np.testing.assert_allclose(back.z[1, 5], m.z[1, 5])
    assert (back.x_name, back.y_name, back.y_unit, back.ref) == ("field", "rf_freq", "GHz", m.ref)


def test_a_map_becomes_one_curve_per_x_value_along_y():
    """Lukas (2026-10-14): "the x axis is the parameter, the y axis is plotted"."""
    ds = _complex_ds()
    m = E.make_map_data(ds, E.Selection("s21", x="rf_freq", y="field", part="real"),
                        source="x.nc")
    curves = E.map_to_curves(m)
    assert [c.held for c in curves] == [{"rf_freq": (6.0, "GHz")}, {"rf_freq": (8.0, "GHz")}]
    assert curves[1].label == "rf_freq = 8 GHz"
    assert curves[1].x_name == "field" and curves[1].x_unit == "mT"
    np.testing.assert_array_equal(curves[1].x, m.y)
    # the same numbers as a curve made from the file at that frequency
    c = E.make_curve(ds, E.Selection("s21", x="field", slices={"rf_freq": Slice("at", 1)},
                                     part="real"), "x.nc")
    np.testing.assert_array_equal(curves[1].y[np.isfinite(c.y)], c.y[np.isfinite(c.y)])
    np.testing.assert_array_equal(curves[1].z[np.isfinite(c.z)], c.z[np.isfinite(c.z)])
    sel = curves[1].selection
    assert sel.slices["rf_freq"].i0 == 1 and sel.x == "field" and sel.y is None
    # picked columns keep their index IN THE FILE
    (only,) = E.map_to_curves(m, columns=[1])
    assert only.selection.slices["rf_freq"].i0 == 1 and only.label == "rf_freq = 8 GHz"


def test_a_referenced_map_cut_per_x_value_is_the_files_referenced_line():
    """A cut keeps the map's reference (Selection.ref): regenerated from the
    file, the curve is the same line of the referenced map."""
    ds = _complex_ds()
    m = E.make_map_data(ds, E.Selection("s21", x="rf_freq", y="field"),
                        E.MapStyle(ref="column", ref_i=0), "x.nc")
    c = E.map_to_curves(m)[1]
    again = E.make_curve(ds, c.selection, "x.nc")
    np.testing.assert_allclose(again.z, c.z, equal_nan=True)


def test_a_vna_map_121_fields_x_1601_freqs_gives_121_frequency_sweeps():
    """His case: show vna.u |z|, X = clMag.field (121), Y = vna.freq (1601) ->
    121 sweeps along frequency, labelled by field. Sending 1601 field sweeps
    instead made a send take ~30 s; building these must be quick."""
    import time
    field = np.linspace(-50.0, 70.0, 121)
    freq = np.linspace(1.0, 17.0, 1601)
    rng = np.random.default_rng(0)
    u = rng.normal(size=(121, 1601)) + 1j * rng.normal(size=(121, 1601))
    pair = {"complex_pair": "vna.u"}
    ds = xr.Dataset(
        {"vna.u_real": (("clMag.field", "vna.freq"), u.real, {**pair, "complex_part": "real"}),
         "vna.u_imag": (("clMag.field", "vna.freq"), u.imag, {**pair, "complex_part": "imag"})},
        coords={"clMag.field": ("clMag.field", field, {"units": "mT"}),
                "vna.freq": ("vna.freq", freq, {"units": "GHz"})},
        attrs={"dims": "clMag.field,vna.freq"})
    m = E.make_map_data(ds, E.Selection("vna.u", x="clMag.field", y="vna.freq", part="abs"))
    assert m.values.shape == (1601, 121)
    t = time.perf_counter()
    curves = E.map_to_curves(m)
    took = time.perf_counter() - t
    assert len(curves) == 121
    assert curves[0].label == "clMag.field = -50 mT" and curves[-1].label == "clMag.field = 70 mT"
    assert all(c.x_name == "vna.freq" and c.x.size == 1601 for c in curves)
    np.testing.assert_array_equal(curves[60].x, freq)
    np.testing.assert_allclose(curves[60].z, u[60])                      # complex kept
    np.testing.assert_allclose(curves[60].y, np.abs(u[60]))
    assert curves[60].held["clMag.field"] == (pytest.approx(10.0), "mT")
    assert took < 0.5, f"building 121 curves took {took:.2f} s"


def test_maps_go_as_maps_or_as_curves_as_the_module_asks():
    got_maps, got_curves = [], []
    lis = AL.Listener(MAP_INFO, on_curves=got_curves.extend, on_maps=got_maps.extend)
    lis.start()
    plain = AL.Listener({**INFO, "key": "curves_only"}, on_curves=got_curves.extend)
    plain.start()
    m = E.make_map_data(_complex_ds(), E.Selection("s21", x="field", y="rf_freq"))
    try:
        run = {r.key: r for r in AL.running()}
        assert "maps" in run["test_mod"].accepts and run["curves_only"].accepts == ("curves",)
        AL.send_maps(lis.port, [m], run["test_mod"].accepts)
        assert len(got_maps) == 1 and got_maps[0].z.shape == (2, 11) and not got_curves
        AL.send_maps(plain.port, [m], run["curves_only"].accepts)
        assert len(got_curves) == 11                    # one per X value (field)
        # a module that says "maps" but has no handler: curves through on_curves
        got_curves.clear()
        lis.on_maps = None
        AL.send_maps(lis.port, [m])
        assert len(got_curves) == 11 and got_curves[0].x_name == "rf_freq"
    finally:
        lis.stop()
        plain.stop()


def test_the_map_tab_sends_the_whole_map(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from aaltoview.apps import viewer as VW
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    got = []
    lis = AL.Listener(MAP_INFO, on_curves=got.extend, on_maps=got.extend)
    lis.start()
    w = VW.ViewerWidget()
    try:
        w.set_dataset(_complex_ds(), tmp_path / "s.nc")
        menu = w.map.exports.analysis_menu
        menu._fill()
        sends = [a for a in menu.actions() if a.text() == "Send to Test module"]
        sends[0].trigger()
        assert len(got) == 1 and isinstance(got[0], E.MapData)
        assert got[0].values.shape == (w.map._map.y.size, w.map._map.x.size)
        assert "sent 1 map to Test module" in w.map.status.text()
    finally:
        lis.stop()
        w.close()
