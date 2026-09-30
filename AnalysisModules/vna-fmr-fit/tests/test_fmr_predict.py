"""Fit a few curves by hand, fit the dispersion roughly, let it predict the rest.

On the lab-style YIG VNA sweeps (tools/make_demo_data.py file 9): from three
hand-fitted sweeps the dispersion places, bounds and labels the peaks of all
the others -- and leaves out modes that are not in a sweep (the uniform mode
is below 2 GHz at 25 mT), so nothing is made up and no role is shifted.
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


def _truth(b):
    return {n: DEMO.yig_mode(np.array([b]), n)[0][0] for n in range(5)}


def _three_by_hand(win, curves):
    """Reference, 5 peaks, fit 100 / 200 / 300 mT, roles by order, dispersion."""
    win.add_curves(curves)
    win.ref_combo.setCurrentIndex(win.ref_combo.findText(curves[-1].label))
    win.peaks.setValue(5)
    win.curve_list.clearSelection()
    for i, e in enumerate(win.entries):
        if e.curve.held["field"][0] in (100.0, 200.0, 300.0):
            win.curve_list.topLevelItem(i).setSelected(True)
    win.fit_all()                                      # "Fit selected (3)"
    tab = win.dispersion
    win.tabs.setCurrentWidget(tab)
    tab.assign_pssw()
    tab.fit()
    return tab


def test_three_by_hand_then_the_dispersion_does_the_rest(win, curves):
    tab = _three_by_hand(win, curves)
    assert len(tab.points) == 15                       # 3 sweeps x 5 modes
    tab.predict_and_fit()
    fitted = [e for e in win.entries if e.result is not None]
    assert len(fitted) == 12                           # all but the reference
    for e in fitted:
        b = e.curve.held["field"][0]
        truth = _truth(b)
        roles = [tab.roles[(id(e), k)] for k in range(1, e.setup.n_peaks + 1)]
        for k, role in enumerate(roles, 1):
            assert e.result.values[f"p{k}_center"] == pytest.approx(truth[role], abs=3e-3), b
        assert not M.suspicious(e.result), (b, M.suspicious(e.result))
        if b == 25.0:
            # the uniform mode is below the sweep: 4 peaks, PSSW 1-4, none made up
            assert sorted(roles) == [1, 2, 3, 4]
    # predicted curves carry the tight windows (+-3 linewidths)
    e = next(e for e in win.entries if e.predicted)
    lo, hi = e.result.windows["p1_center"]
    assert hi - lo == pytest.approx(2 * 3 * 2 * e.start["p1_hwhm"].value, rel=0.5)
    # the dispersion was refitted with all 59 resonances (4 + 11 x 5)
    assert len([p for p in tab.points if p.use]) == 59
    tab.pssw.setCurrentIndex(tab.pssw.findData("A"))
    tab.d_edit.setText("200")
    tab.ms_edit.setText("176")
    tab._model_changed()
    tab.fit()
    assert tab.dres.values["A"] == pytest.approx(DEMO.YIG["A"], rel=1e-3)
    assert tab.dres.values["Meff"] == pytest.approx(DEMO.YIG["Ms"], rel=1e-3)


def test_loose_predictions_and_nothing_left_to_predict(win, curves):
    tab = _three_by_hand(win, curves)
    tab.pred_bounds.setCurrentIndex(tab.pred_bounds.findData("loose"))
    assert not tab.pred_width.isEnabled()
    tab.predict_and_fit()
    e = next(e for e in win.entries if e.predicted)
    assert "p1_center" not in e.result.windows          # no tight window
    # hand-fitted curves are never replaced; predicted ones are redone
    by_hand = [e for e in win.entries if e.result is not None and not e.predicted]
    assert {e.curve.held["field"][0] for e in by_hand} == {100.0, 200.0, 300.0}


def test_predict_needs_a_dispersion_first(win, curves):
    win.add_curves(curves[:2])
    tab = win.dispersion
    tab.predict_and_fit()
    assert "Fit the dispersion first" in tab.status.text()


def test_ctrl_click_puts_a_peak_there(win, curves):
    win.add_curves(curves[4:5])                         # 100 mT
    win.ref_combo.setCurrentIndex(0)
    e = win.current()
    n = e.setup.n_peaks
    x = 6.2564                                          # PSSW 2 at 100 mT
    win.add_peak_at(x)
    assert e.setup.n_peaks == n + 1 and win.peaks.value() == n + 1
    assert e.start[f"p{n + 1}_center"].value == pytest.approx(x)
    assert "put at" in win.status.text()


def test_without_the_reference_the_fit_misses(win, curves):
    """docs/CASE_STUDY_FMR_PSSW.md step 2: the 200 mT sweep, 5 peaks, oscillator
    + delay but NOT divided by the 800 mT line -- the ripple is as large as the
    resonances and takes the peaks; divided, all five are within 2 MHz."""
    win.add_curves(curves)
    e = win.entries[7]
    assert e.curve.held["field"][0] == 200.0
    win.curve_list.setCurrentItem(win.curve_list.topLevelItem(7))
    win.peaks.setValue(5)
    truth = sorted(_truth(200.0).values())

    def miss():
        got = sorted(e.result.values[f"p{k}_center"] for k in range(1, 6))
        return max(abs(a - b) for a, b in zip(got, truth))

    win.ref_combo.setCurrentIndex(0)                    # none
    win.fit()
    assert miss() > 0.1 or M.suspicious(e.result)
    win.ref_combo.setCurrentIndex(win.ref_combo.findText(curves[-1].label))
    win.fit()
    assert miss() < 2e-3 and not M.suspicious(e.result)
