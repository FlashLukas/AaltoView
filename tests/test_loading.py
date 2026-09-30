"""Loading scripts: a correction applied as a file is read (aaltoview/loading.py).

What must not go wrong:
* the TR-MOKE unfold gives back the signal the sample made -- tested on waves
  conjugated exactly the way an 80 MHz laser's aliasing conjugates them;
* a real detector, and the frequencies with no direction, are left alone;
* the viewer applies the chosen script to every file, re-reads the open one
  when the choice changes, and never shows a file uncorrected when its
  script fails;
* the notebook reads the files through the same script.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aaltoview import export as E
from aaltoview import loading as L
from aaltoview.data import as_complex


def _trmoke_ds(conjugated: bool = True):
    """Waves along pos_x at 1700-2690 MHz in 10 MHz steps, as the lock-in of an
    80 MHz TR-MOKE records them (conjugated where the alias is negative) or as
    the sample made them."""
    f = np.arange(1700.0, 2700.0, 10.0)                       # MHz, like the old files
    x = np.linspace(0, 40, 161)
    k = 1.0 + 3.0 * (f - 1700.0) / 1000.0
    true = np.exp(1j * (k[:, None] * x[None, :] + 0.3))
    d = L.alias(f / 1000.0, 80.0)
    seen = np.where((d < 0)[:, None], np.conj(true), true) if conjugated else true
    power = np.full(f.size, -10.0)
    return xr.Dataset(
        {"lockin_real": (("rf_freq", "pos_x"), seen.real, {"units": "V", "complex_pair":
                                                            "lockin", "complex_part": "real"}),
         "lockin_imag": (("rf_freq", "pos_x"), seen.imag, {"units": "V", "complex_pair":
                                                            "lockin", "complex_part": "imag"}),
         "rf_power": (("rf_freq",), power, {"units": "dBm"})},
        coords={"rf_freq": ("rf_freq", f, {"units": "MHz"}),
                "pos_x": ("pos_x", x, {"units": "um"})},
        attrs={"dims": "rf_freq,pos_x", "name": "trmoke"}), true, d


def test_the_repository_scripts_are_listed():
    names = [s.name for s in L.available()]
    assert "TR-MOKE unfold (80 MHz laser)" in names
    assert "TR-MOKE unfold (100 MHz laser)" in names
    sc = L.find("TR-MOKE unfold (80 MHz laser)")
    assert "80 MHz" in sc.description and sc.path.parent.name == "LoadingScripts"


def test_unfold_gives_back_what_the_sample_made():
    ds, true, d = _trmoke_ds()
    out = L.run(L.find("TR-MOKE unfold (80 MHz laser)"), ds)
    z = as_complex(out, "lockin").values
    has_sign = (np.abs(d) > 0) & (np.abs(d) < 40)
    np.testing.assert_allclose(z[has_sign], true[has_sign], atol=1e-12)
    # alias 0 or 40 MHz: no direction recorded, nothing to undo -- left as read
    np.testing.assert_array_equal(z[~has_sign], as_complex(ds, "lockin").values[~has_sign])
    np.testing.assert_array_equal(out["rf_power"].values, ds["rf_power"].values)
    assert out["lockin_imag"].attrs == ds["lockin_imag"].attrs      # still a pair
    assert out.attrs[L.ATTR] == "TR-MOKE unfold (80 MHz laser)"
    assert "80 MHz" in out.attrs["tr_moke_unfold"]
    # the file as read is not changed
    assert not np.array_equal(ds["lockin_imag"].values, out["lockin_imag"].values)


def test_the_other_half_and_the_100_mhz_laser():
    ds, true, d = _trmoke_ds()
    flipped = L.tr_moke_unfold(ds, 80.0, invert=True)
    z = as_complex(flipped, "lockin").values
    has_sign = (np.abs(d) > 0) & (np.abs(d) < 40)
    np.testing.assert_allclose(z[has_sign], np.conj(true[has_sign]), atol=1e-12)
    assert L.alias([2.03, 2.07, 2.05], 100.0) == pytest.approx([30, -30, 50])


def test_no_frequency_axis_is_an_error_not_a_guess():
    ds, _, _ = _trmoke_ds()
    ds = ds.assign_coords(rf_freq=ds["rf_freq"].assign_attrs(units=""))
    with pytest.raises(ValueError, match="no frequency axis"):
        L.tr_moke_unfold(ds, 80.0)


def test_dropped_scripts_helpers_and_broken_ones(tmp_path, monkeypatch):
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "double.py").write_text('"""Twice the signal."""\nNAME = "Double"\n'
                                 "def load(ds, path):\n"
                                 "    return ds * 2\n", encoding="utf-8")
    (d / "_shared.py").write_text("X = 1\n", encoding="utf-8")
    (d / "broken.py").write_text("def load(ds, path)\n", encoding="utf-8")
    (d / "wrong.py").write_text("def load(ds, path):\n    return 3\n", encoding="utf-8")
    monkeypatch.setenv(L.SCRIPTS_ENV, str(d))
    names = [s.name for s in L.available()]
    assert "Double" in names and "wrong" in names
    assert "_shared" not in names and "broken" not in names
    ds, _, _ = _trmoke_ds()
    assert L.run(L.find("Double"), ds)["rf_power"].values[0] == -20.0
    with pytest.raises(TypeError, match="not a Dataset"):
        L.run(L.find("wrong"), ds)


# ─────────────────────────────── in the viewer ────────────────────────────────

@pytest.fixture
def viewer(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from aaltoview.apps import viewer as VW
    monkeypatch.setattr(VW, "remember_script", lambda name: None)   # not the PC's settings
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = VW.ViewerWidget()
    yield w
    w.close()


def test_the_viewer_loads_with_the_chosen_script(viewer, tmp_path):
    ds, true, d = _trmoke_ds()
    path = tmp_path / "trmoke.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    viewer.load_file(path)
    assert L.ATTR not in viewer.ds.attrs
    combo = viewer.browser.script_combo
    combo.setCurrentIndex(combo.findData("TR-MOKE unfold (80 MHz laser)"))
    # the open file was read again, through the script
    assert viewer.ds.attrs[L.ATTR] == "TR-MOKE unfold (80 MHz laser)"
    assert "loaded with TR-MOKE unfold (80 MHz laser)" in viewer.current.text()
    assert "conjugated" in viewer.map.status.text()
    has_sign = (np.abs(d) > 0) & (np.abs(d) < 40)
    z = as_complex(viewer.ds, "lockin").values
    np.testing.assert_allclose(z[has_sign], true[has_sign], atol=1e-12)
    # back to none: as saved
    combo.setCurrentIndex(0)
    assert L.ATTR not in viewer.ds.attrs and viewer.current.text() == "trmoke.nc"


def test_a_failing_script_shows_nothing_rather_than_the_uncorrected_file(
        viewer, tmp_path, monkeypatch):
    d = tmp_path / "scripts"
    d.mkdir()
    (d / "boom.py").write_text("def load(ds, path):\n    raise RuntimeError('no')\n",
                               encoding="utf-8")
    monkeypatch.setenv(L.SCRIPTS_ENV, str(d))
    viewer.browser.fill_scripts()
    ds, _, _ = _trmoke_ds()
    path = tmp_path / "trmoke.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    combo = viewer.browser.script_combo
    combo.setCurrentIndex(combo.findData("boom"))
    viewer.load_file(path)
    assert viewer.ds is None
    assert "the loading script 'boom' failed" in viewer.map.status.text()


def test_the_notebook_reads_the_files_through_the_script(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    ds, true, d = _trmoke_ds()
    path = tmp_path / "data" / "trmoke.nc"
    path.parent.mkdir()
    ds.to_netcdf(path, engine="h5netcdf")
    sc = L.find("TR-MOKE unfold (80 MHz laser)")
    loaded = L.run(sc, ds, path)
    m, _ = E.make_map(loaded, E.Selection("lockin", x="pos_x", y="rf_freq", part="imag"),
                      source=path)
    (tmp_path / "nb").mkdir()
    nb = E.write_notebook(tmp_path / "nb" / "v.ipynb", m=m, loading_script=str(sc.path))
    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    ns: dict = {}
    here = os.getcwd()
    os.chdir(nb.parent)
    try:
        for c in cells:
            if c["cell_type"] == "code":
                exec("".join(c["source"]).replace("plt.show()", "plt.close('all')"), ns)
    finally:
        os.chdir(here)
    assert "80 MHz" in ns["data"]["f0"].attrs["tr_moke_unfold"]     # read through the script
    np.testing.assert_allclose(ns["z"], m.z, equal_nan=True)     # the map the viewer showed
    assert any("loading script" in "".join(c["source"]) for c in cells
               if c["cell_type"] == "markdown")
    assert Path(sc.path).exists()
