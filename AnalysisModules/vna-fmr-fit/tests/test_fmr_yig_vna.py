"""200 nm YIG measured the lab's way: field set, VNA frequency sweep.

tools/make_demo_data.py file 9: S21 at 25-300 mT, 2-18 GHz in 1 MHz steps,
with the VNA's delay, ripple and loss, the exact oscillator lineshape, the
uniform mode and PSSW n = 1-4, and a reference at 800 mT. Through the window,
the way it is used: curves in, reference, Fit all, roles by order, A.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lmfit")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from aaltoview import export as E  # noqa: E402
from aaltoview.data import load  # noqa: E402
from fmr_fit import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import make_demo_data as DEMO  # noqa: E402


@pytest.fixture(scope="module")
def curves(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    DEMO.main([str(out)])
    path = next(out.glob("*/*_yig_200nm_vna.nc"))
    ds = load(path).load()
    return E.curves_along(ds, E.Selection("s21", x="rf_freq"), "field",
                          range(ds.sizes["field"]), path)


@pytest.fixture
def win():
    from aaltoview.apps import viewer as VW
    from fmr_fit.app import FitWindow
    VW.configure_pyqtgraph()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = FitWindow()
    yield w
    w.close()


def test_the_lab_workflow_gives_the_exchange_stiffness(win, curves):
    win.add_curves(curves)
    e0 = win.entries[0]
    # a VNA frequency sweep arrives ready: exact lineshape, delay unwound
    assert e0.setup.lineshape == "oscillator" and e0.setup.delay
    ref = curves[-1]
    assert ref.held["field"][0] == 800.0
    win.ref_combo.setCurrentIndex(win.ref_combo.findText(ref.label))
    win.peaks.setValue(5)
    win.fit_all()
    assert win.entries[-1].result is None                   # the reference itself
    for e in win.entries[:-1]:
        b = e.curve.held["field"][0]
        want = sorted(f for f in (DEMO.yig_mode(np.array([b]), n)[0][0] for n in range(5))
                      if f > 2.0)
        got = sorted(e.result.values[f"p{k}_center"] for k in range(1, 6))
        if len(want) == 5:
            assert np.allclose(got, want, atol=3e-3), b
        else:
            # 25 mT: the uniform mode (1.98 GHz) is below the sweep -- the
            # fifth peak is made up and must be flagged
            assert M.suspicious(e.result), b
    tab = win.dispersion
    win.tabs.setCurrentWidget(tab)
    tab.assign_pssw()
    for p in tab.points:                                    # leave out the flagged sweep
        if p.H < 30:
            p.use = False
            tab.unused.add(tab._key(p))
    tab.pssw.setCurrentIndex(tab.pssw.findData("A"))
    tab.d_edit.setText("200")
    tab.ms_edit.setText("176")
    tab._model_changed()
    tab.fit()
    # without the 25 mT sweep the numbers come back exactly; WITH it (its made-
    # up peak, the roles one off) Meff was 179 and gamma 27.89 -- hence the ⚠
    v = tab.dres.values
    assert v["A"] == pytest.approx(DEMO.YIG["A"], rel=1e-3)
    assert v["Meff"] == pytest.approx(DEMO.YIG["Ms"], rel=1e-3)
    assert v["gamma"] == pytest.approx(DEMO.G, rel=1e-3)
    assert abs(tab.damp.alpha - DEMO.YIG["alpha"]) < 4 * tab.damp.alpha_err


def test_a_whole_map_arrives_as_one_sweep_per_field(win, tmp_path_factory):
    """Map tab -> Analysis: the field x frequency map, X = rf_freq, as one
    sweep per field, each held at its field -- the same curves 1D plots sends."""
    out = tmp_path_factory.mktemp("demo_map")
    DEMO.main([str(out)])
    path = next(out.glob("*/*_yig_200nm_vna.nc"))
    ds = load(path).load()
    m = E.make_map_data(ds, E.Selection("s21", x="rf_freq", y="field"), source=path)
    win.add_maps([m])
    assert len(win.entries) == 13
    c = win.entries[4].curve
    assert c.held["field"] == (125.0, "mT") and c.z is not None
    ref = E.curves_along(ds, E.Selection("s21", x="rf_freq"), "field", [4], path)[0]
    np.testing.assert_array_equal(c.z, ref.z)
    assert win.entries[0].setup.lineshape == "oscillator"      # still a VNA sweep
    assert "received a map" in win.status.text() and "one per field" in win.status.text()
